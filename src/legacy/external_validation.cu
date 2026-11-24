/*

    external_validation.cu

    Implementation of ExternalValidator (see include/cuadmm/external_validation.h). The residuals, objectives and
    norms are host code. The smallest eigenvalue of a PSD block comes from cusolverDnXsyevd on the device (double
    precision, eigenvalues only; ValidationGpuEig below) or from LAPACK dsyevd (LP64) on the CPU.

    Builds that must not reference LAPACK, such as the MATLAB MEX binding (MATLAB's own LAPACK uses 64-bit integers
    and would interpose the symbol), define CUADMM_NO_LAPACK: with EigDevice::Cpu evaluate() then throws, with
    EigDevice::Gpu a block whose device computation fails has no eigenvalue (NaN, the "eig" criterion fails closed).
    solve() without validation (the MEX interface) is unaffected.

*/

#include "cuadmm_legacy/external_validation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#ifndef CUADMM_NO_LAPACK
extern "C" void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a, const int *lda, double *w,
                        double *work, const int *lwork, int *iwork, const int *liwork, int *info);
#endif

namespace
{

double norm2(const std::vector<double> &v)
{
    long double s = 0;
    for (double x : v)
        s += (long double)x * x;
    return std::sqrt((double)s);
}

// Smallest eigenvalue of the n x n symmetric matrix stored in svec form at v[offset..] (upper triangle column by
// column, off-diagonal entries multiplied by sqrt(2)); returns NaN if LAPACK fails.
double svec_min_eig(const double *v, long offset, int n, std::vector<double> &A, std::vector<double> &w,
                    std::vector<double> &work, std::vector<int> &iwork)
{
    A.assign((size_t)n * n, 0.0);
    w.assign(n, 0.0);
    const double is2 = 1.0 / std::sqrt(2.0);
    for (int c = 0; c < n; c++)
        for (int r = 0; r <= c; r++)
        {
            double x = v[offset + (long)c * (c + 1) / 2 + r];
            if (r != c)
                x *= is2;
            A[(size_t)c * n + r] = x;
            A[(size_t)r * n + c] = x;
        }
#ifdef CUADMM_NO_LAPACK
    throw std::runtime_error("external validation is not available in this build (compiled with CUADMM_NO_LAPACK)");
#else
    const char jobz = 'N', uplo = 'U';
    int lwork = -1, liwork = -1, info = 0;
    double wq = 0;
    int iq = 0;
    dsyevd_(&jobz, &uplo, &n, A.data(), &n, w.data(), &wq, &lwork, &iq, &liwork, &info);
    if (info != 0)
        return std::numeric_limits<double>::quiet_NaN();
    lwork = std::max(1, (int)wq);
    liwork = std::max(1, iq);
    if ((int)work.size() < lwork)
        work.resize(lwork);
    if ((int)iwork.size() < liwork)
        iwork.resize(liwork);
    dsyevd_(&jobz, &uplo, &n, A.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0)
        return std::numeric_limits<double>::quiet_NaN();
    return w.front();
#endif
}

bool le(double x, double tol) { return x <= tol; } // false for NaN: the comparisons fail closed

// Dense n x n matrix (column-major, both triangles) of the svec block v (upper triangle column by column,
// off-diagonal entries multiplied by sqrt(2)). One thread per matrix entry.
__global__ void validation_unpack_svec(const double *v, double *A, int n)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n * n)
        return;
    const size_t r = idx % n, c = idx / n;
    const size_t lo = r < c ? r : c, hi = r < c ? c : r;
    const double x = v[hi * (hi + 1) / 2 + lo];
    A[idx] = r == c ? x : x * 0.70710678118654752440;
}

} // namespace

// Device side of ExternalValidator: lambda_min of one PSD block with cusolverDnXsyevd in double precision. The
// handle lives as long as the validator; the buffers of a call (the svec block, the dense matrix, the eigenvalues,
// the cuSOLVER workspace) are allocated for the call and released before it returns, so a validation leaves no
// device memory in use between two calls. Every failure is reported to the caller, which falls back to the CPU.
struct ValidationGpuEig
{
    std::mutex mutex;
    cusolverDnHandle_t handle = nullptr;
    cusolverDnParams_t params = nullptr;
    bool unusable = false; // the handle could not be created: no further attempt
    bool reported = false; // the fallback message is printed once

    ~ValidationGpuEig()
    {
        if (params != nullptr)
            cusolverDnDestroyParams(params);
        if (handle != nullptr)
            cusolverDnDestroy(handle);
    }

    // Smallest eigenvalue of the n x n block stored in svec form at v[0 .. n (n + 1) / 2). Returns false, with the
    // reason in `why`, if the device computation could not be done.
    bool min_eig(const double *v, int n, double &lmin, std::string &why)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (unusable)
        {
            why = "no cuSOLVER handle";
            return false;
        }
        if (handle == nullptr)
        {
            if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS)
                handle = nullptr;
            if (handle != nullptr && cusolverDnCreateParams(&params) != CUSOLVER_STATUS_SUCCESS)
                params = nullptr;
            if (handle == nullptr || params == nullptr)
            {
                unusable = true;
                cudaGetLastError(); // leave no pending error to the solver
                why = "the cuSOLVER handle could not be created";
                return false;
            }
        }
        const size_t len = (size_t)n * (n + 1) / 2, nn = (size_t)n * n;
        double *d_v = nullptr, *d_A = nullptr, *d_w = nullptr;
        void *d_work = nullptr;
        int *d_info = nullptr;
        auto release = [&]()
        {
            cudaFree(d_v);
            cudaFree(d_A);
            cudaFree(d_w);
            cudaFree(d_work);
            cudaFree(d_info);
        };
        auto fail = [&](const char *what)
        {
            why = what;
            release();
            cudaGetLastError(); // leave no pending error to the solver
            return false;
        };
        if (cudaMalloc(&d_A, nn * sizeof(double)) != cudaSuccess)
        {
            d_A = nullptr;
            return fail("not enough device memory for the dense block");
        }
        if (cudaMalloc(&d_w, (size_t)n * sizeof(double)) != cudaSuccess)
        {
            d_w = nullptr;
            return fail("not enough device memory for the eigenvalues");
        }
        if (cudaMalloc(&d_info, sizeof(int)) != cudaSuccess)
        {
            d_info = nullptr;
            return fail("device allocation failed");
        }
        size_t device_bytes = 0, host_bytes = 0;
        if (cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, (int64_t)n,
                                        CUDA_R_64F, d_A, (int64_t)n, CUDA_R_64F, d_w, CUDA_R_64F, &device_bytes,
                                        &host_bytes) != CUSOLVER_STATUS_SUCCESS)
            return fail("cusolverDnXsyevd_bufferSize failed");
        // the svec block is unpacked on the device and released before the eigensolver workspace is allocated
        if (cudaMalloc(&d_v, len * sizeof(double)) != cudaSuccess)
        {
            d_v = nullptr;
            return fail("not enough device memory for the svec block");
        }
        if (cudaMemcpy(d_v, v, len * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("host-to-device copy failed");
        const unsigned threads = 256;
        const size_t blocks = (nn + threads - 1) / threads;
        validation_unpack_svec<<<(unsigned)blocks, threads>>>(d_v, d_A, n);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
            return fail("the svec unpacking kernel failed");
        cudaFree(d_v);
        d_v = nullptr;
        if (cudaMalloc(&d_work, std::max<size_t>(device_bytes, 8)) != cudaSuccess)
        {
            d_work = nullptr;
            return fail("not enough device memory for the cuSOLVER workspace");
        }
        std::vector<char> h_work(std::max<size_t>(host_bytes, 8));
        if (cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, (int64_t)n, CUDA_R_64F, d_A,
                             (int64_t)n, CUDA_R_64F, d_w, CUDA_R_64F, d_work, device_bytes, h_work.data(), h_work.size(),
                             d_info) != CUSOLVER_STATUS_SUCCESS)
            return fail("cusolverDnXsyevd failed");
        int info = -1;
        double w0 = std::numeric_limits<double>::quiet_NaN();
        if (cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(&w0, d_w, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("device-to-host copy failed");
        release();
        if (info != 0 || !std::isfinite(w0))
        {
            why = "cusolverDnXsyevd did not converge";
            return false;
        }
        lmin = w0; // eigenvalues in increasing order
        return true;
    }

    void report_fallback(const std::string &why)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (reported)
            return;
        reported = true;
        fprintf(stderr, " [validation] eigenvalues on the device unavailable (%s): LAPACK on the CPU is used instead\n", why.c_str());
    }
};

bool ValidationResult::better_than(const ValidationResult &o) const
{
    if (validated != o.validated)
        return validated;
    // B23: validated or not, snapshots are ranked by the maximum DIMACS error (the measure of the strict stage, with
    // the DIMACS normalizations 1 + ||b||_inf and 1 + ||C||_inf); the practical merit, whose normalizers are
    // 1 + ||b||_2 and 1 + ||C||_2, decides only when neither snapshot has a finite DIMACS error
    const double a = std::isfinite(max_abs_dimacs) ? max_abs_dimacs : std::numeric_limits<double>::infinity();
    const double b = std::isfinite(o.max_abs_dimacs) ? o.max_abs_dimacs : std::numeric_limits<double>::infinity();
    if (std::isfinite(a) || std::isfinite(b))
        return a < b;
    return merit() < o.merit();
}

double ValidationResult::merit() const
{
    if (!finite)
        return std::numeric_limits<double>::infinity();
    const double v[6] = {primal_res, dual_res, relgap, X_cone_violation, dual_cone_violation, S_cone_violation};
    double m = 0.0;
    for (double x : v)
    {
        if (!std::isfinite(x))
            return std::numeric_limits<double>::infinity();
        m = std::max(m, x);
    }
    return m;
}

ExternalValidator::ExternalValidator(
    int vec_len, int con_num,
    const int *At_col_ptrs, const int *At_row_ids, const double *At_vals,
    int b_nnz, const int *b_indices, const double *b_vals,
    int C_nnz, const int *C_indices, const double *C_vals,
    const std::vector<char> &blk_types, const std::vector<int> &blk_sizes)
    : vec_len_(vec_len), con_num_(con_num), blk_types_(blk_types), blk_sizes_(blk_sizes)
{
    if (vec_len <= 0 || con_num <= 0 || blk_types.size() != blk_sizes.size())
        throw std::invalid_argument("ExternalValidator: invalid problem sizes");
    const int nnz = At_col_ptrs[con_num];
    At_col_ptrs_.assign(At_col_ptrs, At_col_ptrs + con_num + 1);
    At_row_ids_.assign(At_row_ids, At_row_ids + nnz);
    At_vals_.assign(At_vals, At_vals + nnz);
    b_.assign(con_num, 0.0);
    C_.assign(vec_len, 0.0);
    for (int k = 0; k < b_nnz; k++)
        b_[b_indices[k]] += b_vals[k];
    for (int k = 0; k < C_nnz; k++)
        C_[C_indices[k]] += C_vals[k];
    long offset = 0;
    for (size_t k = 0; k < blk_types_.size(); k++)
    {
        blk_offsets_.push_back(offset);
        const long n = blk_sizes_[k];
        offset += blk_types_[k] == 's' ? n * (n + 1) / 2 : n;
    }
    if (offset != vec_len)
        throw std::invalid_argument("ExternalValidator: the blocks do not match vec_len");
    norm_b_ = norm2(b_);
    norm_C_ = norm2(C_);
    // DIMACS: largest absolute component; for C over the matrix entries (svec off-diagonals carry a factor sqrt(2))
    norm_b_inf_ = 0.0;
    for (double v : b_)
        norm_b_inf_ = std::max(norm_b_inf_, std::fabs(v));
    norm_C_inf_ = 0.0;
    for (size_t k = 0; k < blk_types_.size(); k++)
    {
        const long off = blk_offsets_[k], n = blk_sizes_[k];
        if (blk_types_[k] == 's')
        {
            for (long c = 0; c < n; c++)
                for (long r = 0; r <= c; r++)
                {
                    const double v = C_[off + c * (c + 1) / 2 + r];
                    norm_C_inf_ = std::max(norm_C_inf_, std::fabs(r == c ? v : v / std::sqrt(2.0)));
                }
        }
        else
            for (long i = off; i < off + n; i++)
                norm_C_inf_ = std::max(norm_C_inf_, std::fabs(C_[i]));
    }
    dimacs_kinds_.resize(blk_types_.size());
    for (size_t k = 0; k < blk_types_.size(); k++)
        dimacs_kinds_[k] = blk_types_[k] == 's' ? 's' : 'l';
}

void ExternalValidator::set_dimacs_kinds(const std::vector<char> &kinds)
{
    if (kinds.size() != blk_types_.size())
        throw std::invalid_argument("ExternalValidator::set_dimacs_kinds: one kind per block expected");
    for (size_t k = 0; k < kinds.size(); k++)
        if ((kinds[k] != 's' && kinds[k] != 'l') || (kinds[k] == 'l' && blk_types_[k] == 's'))
            throw std::invalid_argument("ExternalValidator::set_dimacs_kinds: kind must be 's' or 'l' ('l' not for a PSD block)");
    dimacs_kinds_ = kinds;
}

ValidationResult ExternalValidator::evaluate(const double *X, const double *y, const double *S,
                                             const ValidationTolerances &tol, int threads) const
{
    ValidationResult r;
    r.finite = true;
    for (int i = 0; i < vec_len_ && r.finite; i++)
        r.finite = std::isfinite(X[i]) && std::isfinite(S[i]);
    for (int j = 0; j < con_num_ && r.finite; j++)
        r.finite = std::isfinite(y[j]);
    if (!r.finite)
    {
        r.validated = false;
        r.failed = "non-finite";
        return r;
    }

    // residuals and objectives
    std::vector<double> AX(con_num_, 0.0), Aty(vec_len_, 0.0);
    for (int j = 0; j < con_num_; j++)
        for (int k = At_col_ptrs_[j]; k < At_col_ptrs_[j + 1]; k++)
        {
            AX[j] += At_vals_[k] * X[At_row_ids_[k]];
            Aty[At_row_ids_[k]] += At_vals_[k] * y[j];
        }
    std::vector<double> rp(con_num_), rd(vec_len_), Z(vec_len_);
    for (int j = 0; j < con_num_; j++)
        rp[j] = AX[j] - b_[j];
    for (int i = 0; i < vec_len_; i++)
    {
        rd[i] = Aty[i] + S[i] - C_[i];
        Z[i] = C_[i] - Aty[i];
    }
    long double pobj = 0, dobj = 0;
    for (int i = 0; i < vec_len_; i++)
        pobj += (long double)C_[i] * X[i];
    for (int j = 0; j < con_num_; j++)
        dobj += (long double)b_[j] * y[j];
    r.pobj = (double)pobj;
    r.dobj = (double)dobj;
    r.primal_res = norm2(rp) / (1 + norm_b_);
    r.dual_res = norm2(rd) / (1 + norm_C_);
    r.relgap = std::fabs((double)(pobj - dobj)) / (1 + std::fabs((double)pobj) + std::fabs((double)dobj));
    r.kkt_full = std::max(r.primal_res, std::max(r.dual_res, r.relgap));

    // cones, per block (PSD blocks in parallel)
    const size_t nb = blk_types_.size();
    std::vector<double> lmin_X(nb, 0.0), lmin_Z(nb, 0.0), lmin_S(nb, 0.0), frob_X(nb, 0.0);
    std::vector<char> eig_ok(nb, 1);
    std::vector<int> eig_gpu(nb, 0), eig_cpu(nb, 0);
    const bool use_gpu = eig_device_ == EigDevice::Gpu;
    if (use_gpu && !gpu_)
        gpu_ = std::make_shared<ValidationGpuEig>();
    auto work_range = [&](size_t first, size_t last)
    {
        std::vector<double> A, w, work;
        std::vector<int> iwork;
        for (size_t k = first; k < last; k++)
        {
            const long off = blk_offsets_[k], n = blk_sizes_[k];
            if (blk_types_[k] == 's')
            {
                long double f = 0;
                for (long i = off; i < off + n * (n + 1) / 2; i++)
                    f += (long double)X[i] * X[i];
                frob_X[k] = std::sqrt((double)f);
                // lambda_min of the block of v: on the device, or with LAPACK (EigDevice::Cpu, or the fallback)
                auto min_eig = [&](const double *v) -> double
                {
                    if (use_gpu)
                    {
                        double lmin = std::numeric_limits<double>::quiet_NaN();
                        std::string why;
                        if (gpu_->min_eig(v + off, (int)n, lmin, why))
                        {
                            eig_gpu[k]++;
                            return lmin;
                        }
                        gpu_->report_fallback(why);
#ifdef CUADMM_NO_LAPACK
                        return std::numeric_limits<double>::quiet_NaN(); // no CPU eigensolver in this build
#endif
                    }
                    eig_cpu[k]++;
                    return svec_min_eig(v, off, (int)n, A, w, work, iwork);
                };
                lmin_X[k] = min_eig(X);
                lmin_Z[k] = min_eig(Z.data());
                lmin_S[k] = min_eig(S);
                eig_ok[k] = std::isfinite(lmin_X[k]) && std::isfinite(lmin_Z[k]) && std::isfinite(lmin_S[k]);
            }
            else if (blk_types_[k] == 'l')
            {
                long double f = 0;
                double mx = X[off], mz = Z[off], ms = S[off];
                for (long i = off; i < off + n; i++)
                {
                    f += (long double)X[i] * X[i];
                    mx = std::min(mx, X[i]);
                    mz = std::min(mz, Z[i]);
                    ms = std::min(ms, S[i]);
                }
                frob_X[k] = std::sqrt((double)f);
                lmin_X[k] = mx;
                lmin_Z[k] = mz;
                lmin_S[k] = ms;
            }
        }
    };
    threads = std::max(1, std::min<int>(threads, (int)nb));
    if (use_gpu)
        threads = 1; // the device computes one block at a time
    if (threads == 1)
        work_range(0, nb);
    else
    {
        // exceptions (e.g. std::bad_alloc) of a worker are rethrown here after every started thread has joined
        std::vector<std::exception_ptr> errors(threads);
        std::vector<std::thread> pool;
        std::exception_ptr spawn_error;
        try
        {
            for (int t = 0; t < threads; t++)
                pool.emplace_back([&, t]()
                                  {
                                      try
                                      {
                                          work_range(nb * t / threads, nb * (t + 1) / threads);
                                      }
                                      catch (...)
                                      {
                                          errors[t] = std::current_exception();
                                      } });
        }
        catch (...)
        {
            spawn_error = std::current_exception();
        }
        for (auto &th : pool)
            th.join();
        if (spawn_error)
            std::rethrow_exception(spawn_error);
        for (const std::exception_ptr &e : errors)
            if (e)
                std::rethrow_exception(e);
    }

    double worst_X = 0, worst_X_normb = 0, worst_Z = 0, worst_S = 0, min_X = INFINITY, min_Z = INFINITY, min_S = INFINITY, cert = 0;
    bool all_eig_ok = true;
    for (size_t k = 0; k < nb; k++)
    {
        r.gpu_eigs += eig_gpu[k];
        r.cpu_eigs += eig_cpu[k];
        if (blk_types_[k] == 'u')
            continue;
        all_eig_ok = all_eig_ok && eig_ok[k];
        min_X = std::min(min_X, lmin_X[k]);
        min_Z = std::min(min_Z, lmin_Z[k]);
        min_S = std::min(min_S, lmin_S[k]);
        worst_S = std::max(worst_S, std::max(0.0, -lmin_S[k]) / (1 + norm_C_));
        if (lmin_X[k] < 0)
            r.X_negative_blocks++;
        const double vX = std::max(0.0, -lmin_X[k]) / (1 + frob_X[k]);
        if (vX > worst_X)
        {
            worst_X = vX;
            r.X_worst_block = (int)k;
        }
        worst_X_normb = std::max(worst_X_normb, std::max(0.0, -lmin_X[k]) / (1 + norm_b_));
        worst_Z = std::max(worst_Z, std::max(0.0, -lmin_Z[k]) / (1 + norm_C_));
        if (blk_types_[k] == 's')
            cert += blk_sizes_[k] * std::min(0.0, lmin_Z[k]);
        else
            for (long i = blk_offsets_[k]; i < blk_offsets_[k] + blk_sizes_[k]; i++)
                cert += std::min(0.0, Z[i]); // entries of an optimal X in [0, 1]
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    r.X_cone_violation = all_eig_ok ? worst_X : nan;
    r.X_cone_violation_normb = all_eig_ok ? worst_X_normb : nan;
    r.dual_cone_violation = all_eig_ok ? worst_Z : nan;
    r.S_cone_violation = all_eig_ok ? worst_S : nan;
    r.min_eig_X = min_X;
    r.min_eig_Z = min_Z;
    r.min_eig_S = min_S;

    // DIMACS error measures (z = S), with the norms of the original block kinds
    {
        long double lin2 = 0, psd_sum = 0, xs = 0, xz = 0;
        for (size_t k = 0; k < nb; k++)
        {
            const long off = blk_offsets_[k], n = blk_sizes_[k];
            const long len = blk_types_[k] == 's' ? n * (n + 1) / 2 : n;
            long double bl = 0;
            for (long i = off; i < off + len; i++)
                bl += (long double)rd[i] * rd[i];
            if (dimacs_kinds_[k] == 's')
                psd_sum += std::sqrt((double)bl);
            else
                lin2 += bl;
        }
        for (int i = 0; i < vec_len_; i++)
        {
            xs += (long double)X[i] * S[i];
            xz += (long double)X[i] * Z[i];
        }
        const double rp2 = norm2(rp), rdK = (double)psd_sum + std::sqrt((double)lin2);
        const double den = 1 + std::fabs((double)pobj) + std::fabs((double)dobj);
        r.dimacs[0] = rp2 / (1 + norm_b_inf_);
        r.dimacs[1] = all_eig_ok ? std::max(0.0, -min_X) / (1 + norm_b_inf_) : nan;
        r.dimacs[2] = rdK / (1 + norm_C_inf_);
        r.dimacs[3] = all_eig_ok ? std::max(0.0, -min_S) / (1 + norm_C_inf_) : nan;
        r.dimacs[4] = (double)(pobj - dobj) / den;
        r.dimacs[5] = (double)xs / den;
        r.err4_z = all_eig_ok ? std::max(0.0, -min_Z) / (1 + norm_C_inf_) : nan;
        r.err6_z = (double)xz / den;
        r.err6_defined = r.dimacs[1] == 0.0 && r.dimacs[3] == 0.0;
        double mx = 0.0;
        for (double e : r.dimacs)
            mx = std::isfinite(e) && std::isfinite(mx) ? std::max(mx, std::fabs(e)) : nan;
        r.max_abs_dimacs = mx;
    }
    r.certificate_lower_bound_conditional = (double)dobj + cert;

    std::string failed;
    auto check = [&](bool ok, const char *name)
    {
        if (!ok)
            failed += (failed.empty() ? "" : ",") + std::string(name);
    };
    check(le(r.primal_res, tol.primal), "primal");
    check(le(r.dual_res, tol.dual), "dual");
    check(le(r.relgap, tol.gap), "gap");
    check(le(r.X_cone_violation, tol.cone), "X_cone");
    check(le(r.dual_cone_violation, tol.dual_cone), "dual_cone");
    check(le(r.S_cone_violation, tol.s_cone), "S_cone");
    if (!all_eig_ok)
        check(false, "eig");
    r.failed = failed;
    r.validated = failed.empty();
    return r;
}
