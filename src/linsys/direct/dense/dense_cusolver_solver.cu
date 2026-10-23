/*

    linsys/direct/dense/dense_cusolver_solver.cu

*/

#include "linsys/direct/dense/dense_cusolver_solver.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <stdexcept>

#include <cusolver_common.h>

#include "core/check.h"

namespace cuadmm
{
namespace linsys
{

namespace
{

// a[rowind[k] + colind[k] * lda] = val[k]: the lower triangle in column-major order
__global__ void scatter_lower_kernel(int nnz, const int *colind, const int *rowind, const double *val, double *a, int lda)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz)
        a[std::size_t(colind[k]) * std::size_t(lda) + std::size_t(rowind[k])] = val[k];
}

__global__ void ipiv_to_int64_kernel(int n, const int *ipiv32, std::int64_t *ipiv64)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n)
        ipiv64[k] = ipiv32[k];
}

double seconds_since(const std::chrono::steady_clock::time_point &t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string bytes_string(std::size_t b)
{
    char buf[64];
    if (b >= std::size_t(1) << 30)
        std::snprintf(buf, sizeof(buf), "%.2f GiB", double(b) / double(std::size_t(1) << 30));
    else
        std::snprintf(buf, sizeof(buf), "%.1f MiB", double(b) / double(std::size_t(1) << 20));
    return buf;
}

} // namespace

DenseCusolverSolver::DenseCusolverSolver() : DenseCusolverSolver(nullptr, Options()) {}

DenseCusolverSolver::DenseCusolverSolver(cusolverDnHandle_t handle) : DenseCusolverSolver(handle, Options()) {}

DenseCusolverSolver::DenseCusolverSolver(cusolverDnHandle_t handle, const Options &options) : handle_(handle), options_(options)
{
    if (handle_ == nullptr)
    {
        CUADMM_CHECK_CUSOLVER(cusolverDnCreate(&handle_));
        own_handle_ = true;
    }
    CUADMM_CHECK_CUSOLVER(cusolverDnCreateParams(&params_));
    info_.allocate(1);
}

DenseCusolverSolver::~DenseCusolverSolver()
{
    if (params_ != nullptr)
        CUADMM_CHECK_CUSOLVER_NOTHROW(cusolverDnDestroyParams(params_));
    if (own_handle_ && handle_ != nullptr)
        CUADMM_CHECK_CUSOLVER_NOTHROW(cusolverDnDestroy(handle_));
}

Capabilities DenseCusolverSolver::capabilities() const
{
    Capabilities c;
    c.exact = true;
    c.device_solves = true;
    c.saddle_point = false;
    c.sigma_dependent_ok = false;
    return c;
}

void DenseCusolverSolver::release_factor()
{
    a_.release();
    work_.release();
    host_work_.clear();
    host_work_.shrink_to_fit();
    ipiv32_.release();
    ipiv64_.release();
    factored_ = false;
    indefinite_ = false;
    device_bytes_ = 0;
    host_bytes_ = 0;
}

SymbolicInfo DenseCusolverSolver::analyze(const NormalSystem &sys)
{
    const auto t0 = std::chrono::steady_clock::now();
    release_factor();
    m_ = sys.m();
    const std::int64_t m = m_;
    SymbolicInfo info;
    info.nnz_L = 0.5 * double(m) * double(m + 1);
    info.ordering = "none (dense)";
    // the workspace queries do not touch the matrix: a null pointer is enough (the matrix is not allocated yet)
    potrf_ws_device_ = potrf_ws_host_ = 0;
    CUADMM_CHECK_CUSOLVER(cusolverDnXpotrf_bufferSize(handle_, params_, CUBLAS_FILL_MODE_LOWER, m, CUDA_R_64F, nullptr, m,
                                                      CUDA_R_64F, &potrf_ws_device_, &potrf_ws_host_));
    sytrf_ws_bytes_ = 0;
    if (double(m) * double(m) <= double(INT_MAX)) // Dsytrf is the 32-bit API
    {
        int lwork = 0;
        CUADMM_CHECK_CUSOLVER(cusolverDnDsytrf_bufferSize(handle_, int(m), nullptr, int(m), &lwork));
        sytrf_ws_bytes_ = sizeof(double) * std::size_t(std::max(lwork, 0));
    }
    const std::size_t pivots = sizeof(int) * std::size_t(m) + sizeof(std::int64_t) * std::size_t(m);
    info.device_bytes = matrix_bytes(m_) + std::max(potrf_ws_device_, sytrf_ws_bytes_) + pivots + sizeof(int);
    info.host_bytes = potrf_ws_host_;
    info.seconds = seconds_since(t0);
    symbolic_ = info;
    analyzed_ = true;
    return info;
}

void DenseCusolverSolver::scatter_lower(const NormalSystem &sys)
{
    const HostCscLower &L = sys.lower();
    const std::int64_t nnz = L.nnz();
    if (nnz > std::int64_t(INT_MAX))
        throw std::runtime_error("DenseCusolverSolver: the lower triangle has more than INT_MAX nonzeros");
    std::vector<int> colind(static_cast<std::size_t>(nnz));
    for (int j = 0; j < L.n; j++)
        for (int k = L.colptr[std::size_t(j)]; k < L.colptr[std::size_t(j) + 1]; k++)
            colind[std::size_t(k)] = j;
    a_.zero();
    if (nnz == 0)
        return;
    core::DeviceBuffer<int> d_colind(static_cast<std::size_t>(nnz)), d_rowind(static_cast<std::size_t>(nnz));
    core::DeviceBuffer<double> d_val(static_cast<std::size_t>(nnz));
    core::to_device(d_colind.span(), colind);
    core::to_device(d_rowind.span(), L.rowind);
    core::to_device(d_val.span(), L.val);
    const int threads = 256;
    const int blocks = int((nnz + threads - 1) / threads);
    scatter_lower_kernel<<<blocks, threads>>>(int(nnz), d_colind.data(), d_rowind.data(), d_val.data(), a_.data(), m_);
    CUADMM_CHECK_LAUNCH();
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize()); // the temporaries are freed on return
}

FactorInfo DenseCusolverSolver::factorize(const NormalSystem &sys)
{
    const auto t0 = std::chrono::steady_clock::now();
    if (!analyzed_ || m_ != sys.m())
        analyze(sys);
    release_factor();
    const int m = m_;
    factor_ = FactorInfo();

    // the memory check, before anything is allocated
    std::size_t free_bytes = 0, total_bytes = 0;
    CUADMM_CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t need = symbolic_.device_bytes;
    const std::size_t budget = std::size_t(double(free_bytes) * options_.memory_fraction);
    if (need > budget)
        throw std::runtime_error("DenseCusolverSolver: the dense factorization of the " + std::to_string(m) + " x " +
                                 std::to_string(m) + " system needs " + std::to_string(need) + " bytes (" + bytes_string(need) +
                                 ": 8 m^2 = " + std::to_string(matrix_bytes(m)) + " for the matrix plus the cuSOLVER workspace) but only " +
                                 std::to_string(free_bytes) + " bytes (" + bytes_string(free_bytes) + ") of " +
                                 std::to_string(total_bytes) + " are free on the device" +
                                 (options_.memory_fraction < 1.0 ? " (budget fraction " + std::to_string(options_.memory_fraction) + ")" : ""));

    a_.allocate(std::size_t(m) * std::size_t(m));
    scatter_lower(sys);
    CUADMM_CHECK_CUSOLVER(cusolverDnSetStream(handle_, nullptr));

    bool done = false;
    if (!options_.force_indefinite)
    {
        work_.allocate(std::max<std::size_t>(potrf_ws_device_, 1));
        host_work_.assign(std::max<std::size_t>(potrf_ws_host_, 1), 0);
        CUADMM_CHECK_CUSOLVER(cusolverDnXpotrf(handle_, params_, CUBLAS_FILL_MODE_LOWER, m, CUDA_R_64F, a_.data(), m, CUDA_R_64F,
                                               work_.data(), potrf_ws_device_, host_work_.data(), potrf_ws_host_, info_.data()));
        int info = 0;
        CUADMM_CHECK_CUDA(cudaMemcpy(&info, info_.data(), sizeof(int), cudaMemcpyDeviceToHost));
        if (info < 0)
            throw std::runtime_error("DenseCusolverSolver: cusolverDnXpotrf reports a wrong parameter " + std::to_string(-info));
        if (info == 0)
        {
            done = true;
            indefinite_ = false;
            factor_.kind = "dense_llt";
            pivot_statistics_llt();
        }
        else
        {
            // the leading minor of order info is not positive definite: A does not have full row rank (or eps is too
            // small for it); the symmetric indefinite factorization tolerates the tiny and negative pivots
            factor_.nonpositive_pivots = 1; // at least the one that stopped the Cholesky factorization (recounted below)
        }
    }
    if (!done)
    {
        if (double(m) * double(m) > double(INT_MAX))
            throw std::runtime_error("DenseCusolverSolver: the Cholesky factorization failed and the symmetric indefinite fallback "
                                     "(cusolverDnDsytrf, 32-bit API) cannot address a " + std::to_string(m) + " x " + std::to_string(m) +
                                     " matrix");
        scatter_lower(sys); // Xpotrf has overwritten the lower triangle
        work_.allocate(std::max<std::size_t>(sytrf_ws_bytes_, sizeof(double)));
        ipiv32_.allocate(static_cast<std::size_t>(m));
        CUADMM_CHECK_CUSOLVER(cusolverDnDsytrf(handle_, CUBLAS_FILL_MODE_LOWER, m, a_.data(), m, ipiv32_.data(),
                                               reinterpret_cast<double *>(work_.data()), int(sytrf_ws_bytes_ / sizeof(double)), info_.data()));
        int info = 0;
        CUADMM_CHECK_CUDA(cudaMemcpy(&info, info_.data(), sizeof(int), cudaMemcpyDeviceToHost));
        if (info < 0)
            throw std::runtime_error("DenseCusolverSolver: cusolverDnDsytrf reports a wrong parameter " + std::to_string(-info));
        if (info > 0)
            throw std::runtime_error("DenseCusolverSolver: A A^T + eps I is singular (cusolverDnDsytrf: D(" + std::to_string(info) +
                                     "," + std::to_string(info) + ") is exactly zero); A does not have full row rank and eps = " +
                                     std::to_string(sys.eps()) + " does not regularize it: use the presolve or a larger eps");
        ipiv64_.allocate(static_cast<std::size_t>(m));
        {
            const int threads = 256;
            const int blocks = (m + threads - 1) / threads;
            ipiv_to_int64_kernel<<<blocks, threads>>>(m, ipiv32_.data(), ipiv64_.data());
            CUADMM_CHECK_LAUNCH();
        }
        // the solve workspace of Xsytrs (the query does not touch the arrays)
        sytrs_ws_device_ = sytrs_ws_host_ = 0;
        CUADMM_CHECK_CUSOLVER(cusolverDnXsytrs_bufferSize(handle_, CUBLAS_FILL_MODE_LOWER, m, 1, CUDA_R_64F, a_.data(), m, ipiv64_.data(),
                                                          CUDA_R_64F, nullptr, m, &sytrs_ws_device_, &sytrs_ws_host_));
        work_.allocate(std::max<std::size_t>(sytrs_ws_device_, 1));
        host_work_.assign(std::max<std::size_t>(sytrs_ws_host_, 1), 0);
        indefinite_ = true;
        factor_.kind = "dense_ldlt";
        pivot_statistics_ldlt();
    }
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
    factored_ = true;
    device_bytes_ = sizeof(double) * a_.size() + work_.size() + sizeof(int) * ipiv32_.size() + sizeof(std::int64_t) * ipiv64_.size() + sizeof(int);
    host_bytes_ = host_work_.size();
    factor_.seconds = seconds_since(t0);
    return factor_;
}

// pivots L_jj^2 from the diagonal of the Cholesky factor
void DenseCusolverSolver::pivot_statistics_llt()
{
    const int m = m_;
    if (m == 0)
        return;
    std::vector<double> diag(static_cast<std::size_t>(m));
    CUADMM_CHECK_CUDA(cudaMemcpy2D(diag.data(), sizeof(double), a_.data(), sizeof(double) * (std::size_t(m) + 1), sizeof(double),
                                   std::size_t(m), cudaMemcpyDeviceToHost));
    std::vector<double> pivots(static_cast<std::size_t>(m));
    for (int j = 0; j < m; j++)
        pivots[std::size_t(j)] = diag[std::size_t(j)] * diag[std::size_t(j)];
    double max_abs = 0.0;
    factor_.min_pivot = pivots[0];
    factor_.max_pivot = pivots[0];
    factor_.nonpositive_pivots = 0;
    for (double d : pivots)
    {
        factor_.min_pivot = std::min(factor_.min_pivot, d);
        factor_.max_pivot = std::max(factor_.max_pivot, d);
        max_abs = std::max(max_abs, std::abs(d));
        if (d <= 0.0)
            factor_.nonpositive_pivots++;
    }
    factor_.tiny_pivots = 0;
    for (double d : pivots)
        if (std::abs(d) < 1e-12 * max_abs)
            factor_.tiny_pivots++;
}

// pivots: the eigenvalues of the 1 x 1 and 2 x 2 diagonal blocks of D (LAPACK convention for uplo = 'L': a 2 x 2
// block in rows k, k + 1 is flagged by ipiv[k] = ipiv[k + 1] < 0 and its off-diagonal entry is stored in A(k + 1, k))
void DenseCusolverSolver::pivot_statistics_ldlt()
{
    const int m = m_;
    if (m == 0)
        return;
    std::vector<double> diag(static_cast<std::size_t>(m)), sub(std::size_t(std::max(m - 1, 1)), 0.0);
    CUADMM_CHECK_CUDA(cudaMemcpy2D(diag.data(), sizeof(double), a_.data(), sizeof(double) * (std::size_t(m) + 1), sizeof(double),
                                   std::size_t(m), cudaMemcpyDeviceToHost));
    if (m > 1)
        CUADMM_CHECK_CUDA(cudaMemcpy2D(sub.data(), sizeof(double), a_.data() + 1, sizeof(double) * (std::size_t(m) + 1), sizeof(double),
                                       std::size_t(m - 1), cudaMemcpyDeviceToHost));
    const std::vector<int> ipiv = core::to_host(ipiv32_.data(), std::size_t(m));
    std::vector<double> pivots;
    pivots.reserve(static_cast<std::size_t>(m));
    int k = 0;
    while (k < m)
    {
        if (ipiv[std::size_t(k)] > 0 || k == m - 1)
        {
            pivots.push_back(diag[std::size_t(k)]);
            k++;
        }
        else
        {
            const double a = diag[std::size_t(k)], c = diag[std::size_t(k) + 1], b = sub[std::size_t(k)];
            const double tr = a + c, det = a * c - b * b;
            const double disc = std::sqrt(std::max(0.25 * tr * tr - det, 0.0));
            pivots.push_back(0.5 * tr - disc);
            pivots.push_back(0.5 * tr + disc);
            k += 2;
        }
    }
    double max_abs = 0.0;
    factor_.min_pivot = pivots[0];
    factor_.max_pivot = pivots[0];
    factor_.nonpositive_pivots = 0;
    for (double d : pivots)
    {
        factor_.min_pivot = std::min(factor_.min_pivot, d);
        factor_.max_pivot = std::max(factor_.max_pivot, d);
        max_abs = std::max(max_abs, std::abs(d));
        if (d <= 0.0)
            factor_.nonpositive_pivots++;
    }
    factor_.tiny_pivots = 0;
    for (double d : pivots)
        if (std::abs(d) < 1e-12 * max_abs)
            factor_.tiny_pivots++;
}

SolveReport DenseCusolverSolver::solve(const double *rhs, double *y, int m, const SolveRequest &req, cudaStream_t stream)
{
    (void)req; // exact solve: rel_tol, max_iter and the initial guess do not apply
    if (!factored_)
        throw std::logic_error("DenseCusolverSolver::solve called before factorize()");
    if (m != m_)
        throw std::invalid_argument("DenseCusolverSolver::solve: the right-hand side has " + std::to_string(m) + " entries, the system " +
                                    std::to_string(m_));
    if (rhs == nullptr || y == nullptr)
        throw std::invalid_argument("DenseCusolverSolver::solve: null device pointer");
    if (rhs != y)
        CUADMM_CHECK_CUDA(cudaMemcpyAsync(y, rhs, sizeof(double) * std::size_t(m), cudaMemcpyDeviceToDevice, stream));
    CUADMM_CHECK_CUSOLVER(cusolverDnSetStream(handle_, stream));
    if (!indefinite_)
        CUADMM_CHECK_CUSOLVER(cusolverDnXpotrs(handle_, params_, CUBLAS_FILL_MODE_LOWER, m, 1, CUDA_R_64F, a_.data(), m, CUDA_R_64F, y, m,
                                               info_.data()));
    else
        CUADMM_CHECK_CUSOLVER(cusolverDnXsytrs(handle_, CUBLAS_FILL_MODE_LOWER, m, 1, CUDA_R_64F, a_.data(), m, ipiv64_.data(), CUDA_R_64F,
                                               y, m, work_.data(), sytrs_ws_device_, host_work_.data(), sytrs_ws_host_, info_.data()));
    SolveReport report;
    report.iterations = 0;
    report.rel_residual = 0.0; // not computed here (relative_residual_host in the tests and the bench)
    report.converged = true;
    return report;
}

std::string DenseCusolverSolver::description() const
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), "dense_cusolver: cuSOLVER %d.%d.%d, m = %d, %s, %zu device bytes (8 m^2 = %zu), %zu host bytes",
                  CUSOLVER_VER_MAJOR, CUSOLVER_VER_MINOR, CUSOLVER_VER_PATCH, m_,
                  factored_ ? factor_.kind.c_str() : (analyzed_ ? "analyzed" : "not analyzed"), device_bytes_, matrix_bytes(m_),
                  host_bytes_);
    return buf;
}

} // namespace linsys
} // namespace cuadmm
