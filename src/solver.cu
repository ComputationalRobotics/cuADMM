/*

    solver.cu

    Main solver, works for any sizes of matrices.
    Uses the sGS-ADMM algorithm to solve an SDP problem.

*/

#include "cuadmm/solver.h"
#include "cuadmm/block_structure.h"
#include "cuadmm/kernels.h"
#include "cuadmm/rank.h"
#include "cuadmm/matrix_sizes.h"
#include "cuadmm/monitors.h"

#include "psd_projection/composite_FP32.h"
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
#include "psd_projection/composite_FP32_emulated.h"
#endif
#include "psd_projection/composite_FP16.h"
#include "psd_projection/lobpcg.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/utils.h"

#include <algorithm>
#include <stdio.h>
#include <limits>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>

#define LOBPCG_MAXIT 20
#define LOBPCG_TOL 1e-5
#define LOBPCG_WARMSTART true
#define LOBPCG_RATIO 0.03
#define LOBPCG_RELAXATION 1.2
#define LOBPCG_REEVALUATE 100
#define LOBPCG_RANK_RTOL 1e-8           // rank tolerance relative to max(|lambda_min|, |lambda_max|)
#define LOBPCG_RANK_STOP_TOL_FACTOR 1e-3 // lower bound of the rank tolerance, relative to the stopping tolerance

// Copies `count` cuSOLVER info values to the host and throws if an eigendecomposition failed.
// For cusolverDnDsyevjBatched (jacobi_n = matrix size), info == n + 1 only means that the Jacobi
// method did not reach its (1e-16) tolerance within the maximum number of sweeps, which still gives
// machine-precision results: such matrices are counted and returned instead.
static int check_eig_info(const int *d_info, const int count, const char *what, const int jacobi_n = -1)
{
    if (count == 0)
        return 0;
    std::vector<int> info(count);
    CHECK_CUDA(cudaMemcpy(info.data(), d_info, sizeof(int) * count, D2H));
    int not_converged = 0;
    for (int l = 0; l < count; l++)
    {
        if (info[l] == 0)
            continue;
        if (jacobi_n >= 0 && info[l] == jacobi_n + 1)
        {
            not_converged++;
            continue;
        }
        throw std::runtime_error(
            std::string(what) + " failed for matrix " + std::to_string(l) + ": info = " + std::to_string(info[l]));
    }
    return not_converged;
}

void SDPSolver::synchronize_gpu0_streams()
{
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[0].stream));
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[1].stream));
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[2].stream));
}

void SDPSolver::large_full_eig_project(int i, int j, int idx, bool analyze_rank, bool multiply)
{
    const int n = this->sizes.large_mat_sizes[i];
    const int mat_offset = this->sizes.large_mat_offset(i, j);
    const int W_offset = this->sizes.large_W_offset(i, j);
    const double one = 1.0;
    const double zero = 0.0;

    // compute the EVD using cuSOLVER: large_mat <- eigenvectors, large_W <- eigenvalues (increasing order)
    single_eig_cusolver(
        this->cusolverH_eig_large, eig_param_single,
        this->large_mat, this->large_W,
        this->eig_large_buffer, this->cpu_eig_large_buffer, this->large_info,
        n,
        this->eig_large_buffer_size[i], this->cpu_eig_large_buffer_size[i],
        mat_offset, W_offset,
        this->sizes.large_buffer_offset(i, j, this->eig_large_buffer_size),
        this->sizes.large_cpu_buffer_offset(i, j, this->cpu_eig_large_buffer_size),
        idx);
    check_eig_info(this->large_info.vals + idx, 1, "cusolverDnXsyevd (large matrix)");

    if (analyze_rank)
    {
        // rank tolerance: relative to the extreme eigenvalues, but not below a fraction of the stopping
        // tolerance (the negative part of C - A^T y - X/sigma is -X/sigma, tiny when sigma is large)
        double lambda_min, lambda_max;
        CHECK_CUDA(cudaMemcpy(&lambda_min, this->large_W.vals + W_offset, sizeof(double), D2H));
        CHECK_CUDA(cudaMemcpy(&lambda_max, this->large_W.vals + W_offset + n - 1, sizeof(double), D2H));
        const double rank_tol = std::max(
            LOBPCG_RANK_RTOL * std::max(std::abs(lambda_min), std::abs(lambda_max)),
            LOBPCG_RANK_STOP_TOL_FACTOR * this->stop_tol);
        this->large_rank_tol[idx] = rank_tol;

        // compute the ranks
        compute_ranks(
            this->large_W.vals + W_offset, n,
            this->positive_ranks.vals + idx,
            this->negative_ranks.vals + idx,
            rank_tol);

        // copy ranks to CPU
        CHECK_CUDA(cudaMemcpy(this->cpu_positive_ranks.vals + idx, this->positive_ranks.vals + idx, sizeof(int), D2H));
        CHECK_CUDA(cudaMemcpy(this->cpu_negative_ranks.vals + idx, this->negative_ranks.vals + idx, sizeof(int), D2H));
        const int positive_rank = this->cpu_positive_ranks.vals[idx];
        const int negative_rank = this->cpu_negative_ranks.vals[idx];

        // copy the extreme eigenpairs to use as a warmstart for LOBPCG
        if (positive_rank < LOBPCG_RATIO * n && positive_rank > 0)
        {
            const int k = std::min<int>(std::ceil(LOBPCG_RELAXATION * positive_rank), this->lobpcg_k_alloc[idx]);
            this->large_mode[idx] = LargeProjectionMode::LOBPCG_POSITIVE;
            this->large_k[idx] = k;

            // largest k eigenpairs, in decreasing order: copy the eigenvalues and reverse them
            reverse_vector(this->large_W.vals + W_offset + n - k, this->lobpcg_W[idx].vals, k);
            // copy the eigenvectors and reverse the columns
            reverse_columns(this->large_mat.vals + mat_offset + (n - k) * n, this->lobpcg_P[idx].vals, n, k);
        }
        else if (negative_rank < LOBPCG_RATIO * n && negative_rank > 0)
        {
            const int k = std::min<int>(std::ceil(LOBPCG_RELAXATION * negative_rank), this->lobpcg_k_alloc[idx]);
            this->large_mode[idx] = LargeProjectionMode::LOBPCG_NEGATIVE;
            this->large_k[idx] = k;

            // smallest k eigenpairs, in increasing order (i.e. decreasing order for -A): copy without reversing
            CHECK_CUDA(cudaMemcpy(
                this->lobpcg_W[idx].vals, this->large_W.vals + W_offset, sizeof(double) * k, D2D));
            CHECK_CUDA(cudaMemcpy(
                this->lobpcg_P[idx].vals, this->large_mat.vals + mat_offset, sizeof(double) * k * n, D2D));
            // LOBPCG runs on -A: store the eigenvalues of -A (the eigenvectors do not change)
            const double minus_one = -1.0;
            CHECK_CUBLAS(cublasDscal(this->cublasH_eig_large.cublas_handle, k, &minus_one, this->lobpcg_W[idx].vals, 1));
        }
        else
        {
            this->large_mode[idx] = LargeProjectionMode::FULL;
            this->large_k[idx] = 0;
        }
    }

    // P = Q max(W, 0) Q^T
    max_dense_vector_zero(this->large_W.vals + W_offset, n);
    dense_matrix_mul_diag_batch(
        this->large_mat_tmp, this->large_mat, this->large_W,
        n, 1, mat_offset, W_offset);
    if (!multiply)
        return; // the caller multiplies the whole size group with one batched GEMM
    CHECK_CUBLAS(cublasDgemm(
        this->cublasH_eig_large.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, n,
        &one,
        this->large_mat_tmp.vals + mat_offset, n,
        this->large_mat.vals + mat_offset, n,
        &zero,
        this->large_mat_P.vals + mat_offset, n));
}

bool SDPSolver::large_lobpcg_project(int i, int j, int idx)
{
    const int n = this->sizes.large_mat_sizes[i];
    const int k = this->large_k[idx];
    const int mat_offset = this->sizes.large_mat_offset(i, j);
    const bool negative = this->large_mode[idx] == LargeProjectionMode::LOBPCG_NEGATIVE;
    const double one = 1.0;
    const double minus_one = -1.0;

    cublasHandle_t cublasH = this->cublasH_eig_large.cublas_handle;
    double *A = this->large_mat.vals + mat_offset;
    double *P = this->large_mat_P.vals + mat_offset;
    double *eigenvectors = this->lobpcg_P[idx].vals;
    double *eigenvalues = this->lobpcg_W[idx].vals;
    double *relu_eigenvalues = this->lobpcg_W_relu.vals;
    double *scaled_eigenvectors = this->large_mat_tmp.vals + mat_offset; // n x k workspace

    if (negative)
    {
        // change the matrix sign to reuse LOBPCG code: A <- -A
        // (the warmstart eigenvalues in lobpcg_W are already those of -A)
        CHECK_CUBLAS(cublasDscal(cublasH, n * n, &minus_one, A, 1));
    }

    // compute the largest eigenpairs (of -A in the negative case), in decreasing order;
    // the pairs that were above the rank tolerance at the last rank analysis and the pairs currently
    // above it take part in the convergence test (the remaining relaxation pairs need not converge)
    const double rank_tol = this->large_rank_tol[idx];
    const int rank = negative ? this->cpu_negative_ranks.vals[idx] : this->cpu_positive_ranks.vals[idx];
    LobpcgInfo info;
    lobpcg(
        cublasH, this->cusolverH_eig_large.cusolver_dn_handle,
        A, eigenvectors, eigenvalues,
        n, k, LOBPCG_WARMSTART, LOBPCG_MAXIT, LOBPCG_TOL, false,
        &info, rank_tol, std::min(rank, k));
    this->lobpcg_calls++;

    if (negative)
    {
        // restore the sign of A
        CHECK_CUBLAS(cublasDscal(cublasH, n * n, &minus_one, A, 1));
    }

    // accept the result only if LOBPCG converged and the smallest computed Ritz value is below the
    // rank tolerance: otherwise an eigenvalue of the wanted sign may have been left out (e.g. the
    // rank grew since the last rank analysis), and the projection would not be PSD. This is necessary,
    // not sufficient: Ritz values are lower bounds, and an eigenvector (numerically) orthogonal to the
    // LOBPCG search subspace cannot be detected; the full EVD every LOBPCG_REEVALUATE iterations and the
    // exact PSD-violation check at termination bound this
    std::vector<double> ritz_values(k);
    CHECK_CUDA(cudaMemcpy(ritz_values.data(), eigenvalues, sizeof(double) * k, D2H));
    bool finite = true;
    for (int l = 0; l < k; l++)
        finite = finite && std::isfinite(ritz_values[l]);
    if (!(finite && info.converged && ritz_values[k - 1] <= rank_tol + info.residual_norm))
    {
        this->lobpcg_fallbacks++;
        return false;
    }

    // V diag(max(D, 0))
    CHECK_CUDA(cudaMemcpy(relu_eigenvalues, eigenvalues, sizeof(double) * k, D2D));
    max_dense_vector_zero(relu_eigenvalues, k);
    CHECK_CUBLAS(cublasDdgmm(
        cublasH, CUBLAS_SIDE_RIGHT, n, k,
        eigenvectors, n, relu_eigenvalues, 1,
        scaled_eigenvectors, n));

    // positive case: P = V max(D, 0) V^T
    // negative case: P = A + V max(-D', 0) V^T = A + proj(-A), where D' are the eigenvalues of -A
    double beta = 0.0;
    if (negative)
    {
        CHECK_CUDA(cudaMemcpy(P, A, sizeof(double) * n * n, D2D));
        beta = 1.0;
    }
    CHECK_CUBLAS(cublasDgemm(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, k,
        &one,
        scaled_eigenvectors, n,
        eigenvectors, n,
        &beta,
        P, n));

    return true;
}

bool SDPSolver::large_composite_project(int i, int j, int idx)
{
    const int n = this->sizes.large_mat_sizes[i];
    const int mat_offset = this->sizes.large_mat_offset(i, j);
    const int nn = n * n; // fits in an int since n <= 46340
    cublasHandle_t cublasH = this->cublasH_composite_proj.cublas_handle;
    double *A = this->large_mat.vals + mat_offset;
    double *A_copy = this->large_mat_tmp.vals + mat_offset; // large_mat_tmp is free here

    // keep a copy of the input to restore it if the polynomial filter overflows
    CHECK_CUDA(cudaMemcpy(A_copy, A, sizeof(double) * nn, D2D));

    // scale the spectrum into [-1, 1]: the Lanczos estimate 'up' is not a certified upper bound on
    // ||A||_2, and the filters overflow if ||A||/scale > ~1.01 (FP32) / ~1.02 (FP16), hence the margin
    double lo, up;
    approximate_two_norm(cublasH, this->cusolverH_composite_proj.cusolver_dn_handle, A, n, &lo, &up);
    const double margin = this->current_proj_method == ProjectionMethod::COMPOSITE_FP16 ? COMPOSITE_FP16_SCALE_MARGIN : COMPOSITE_FP32_SCALE_MARGIN;
    const double scale = up > 0.0 ? margin * up : 1.0;
    const double inv_scale = 1.0 / scale;
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &inv_scale, A, 1));

    if (this->current_proj_method == ProjectionMethod::COMPOSITE_FP32)
        composite_FP32(cublasH, A, n, this->float_proj_workspace.vals);
    else if (this->current_proj_method == ProjectionMethod::COMPOSITE_FP16)
        composite_FP16(cublasH, A, n, this->float_proj_workspace.vals, this->half_proj_workspace.vals);
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
    else if (this->current_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED)
        composite_FP32_emulated(cublasH, A, n, this->float_proj_workspace.vals);
#endif

    // rescale the result back to the original scale
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &scale, A, 1));

    // reject a non-finite result (overflow of the filter, or non-finite input)
    double norm;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A, 1, &norm));
    if (!std::isfinite(norm))
    {
        CHECK_CUDA(cudaMemcpy(A, A_copy, sizeof(double) * nn, D2D));
        this->composite_fallbacks++;
        return false;
    }

    // copy large_mat to large_mat_P
    CHECK_CUDA(cudaMemcpy(this->large_mat_P.vals + mat_offset, A, sizeof(double) * nn, D2D));

    return true;
}

void SDPSolver::cone_measures(DeviceDenseVector<double> &v, std::vector<double> &block_min, std::vector<double> &bound_coef)
{
    // matrices of the PSD blocks
    vector_to_matrices(v, this->large_mat, this->medium_mat, this->small_mat, this->map_B, this->map_M1, this->map_M2);
    CHECK_CUDA(cudaDeviceSynchronize());

    // eigenvalues of every PSD block, with the machinery of the projection (the eigenvectors overwrite the matrices)
    int counter = 0;
    for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++)
    {
        for (int j = 0; j < this->sizes.large_mat_nums[i]; j++)
        {
            single_eig_cusolver(
                this->cusolverH_eig_large, eig_param_single,
                this->large_mat, this->large_W,
                this->eig_large_buffer, this->cpu_eig_large_buffer, this->large_info,
                this->sizes.large_mat_sizes[i],
                this->eig_large_buffer_size[i], this->cpu_eig_large_buffer_size[i],
                this->sizes.large_mat_offset(i, j), this->sizes.large_W_offset(i, j),
                this->sizes.large_buffer_offset(i, j, this->eig_large_buffer_size),
                this->sizes.large_cpu_buffer_offset(i, j, this->cpu_eig_large_buffer_size),
                counter);
            check_eig_info(this->large_info.vals + counter, 1, "cusolverDnXsyevd (cone measure)");
            counter++;
        }
    }
    counter = 0;
    for (int i = 0; i < this->sizes.medium_mat_sizes.size(); i++)
    {
        for (int j = 0; j < this->sizes.medium_mat_nums[i]; j++)
        {
            single_eig_cusolver(
                this->cusolverH_eig_medium_arr[counter % this->eig_stream_num_per_gpu], eig_param_single,
                this->medium_mat, this->medium_W,
                this->eig_medium_buffer, this->cpu_eig_medium_buffer, this->medium_info,
                this->sizes.medium_mat_sizes[i],
                this->eig_medium_buffer_size[i], this->cpu_eig_medium_buffer_size[i],
                this->sizes.medium_mat_offset(i, j), this->sizes.medium_W_offset(i, j),
                this->sizes.medium_buffer_offset(i, j, this->eig_medium_buffer_size),
                this->sizes.medium_cpu_buffer_offset(i, j, this->cpu_eig_medium_buffer_size),
                counter);
            counter++;
        }
    }
    for (int i = 0; i < this->eig_stream_num_per_gpu; i++)
        CHECK_CUDA(cudaStreamSynchronize(this->eig_medium_stream_arr[i].stream));
    check_eig_info(this->medium_info.vals, this->sizes.medium_mat_num, "cusolverDnXsyevd (cone measure)");
    int info_offset = 0;
    for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++)
    {
        batch_eig_cusolver(
            this->cusolverH_eig_small, this->eig_param_batch,
            this->small_mat, this->small_W,
            this->eig_small_buffer, this->small_info,
            this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
            this->eig_small_buffer_size[i],
            this->sizes.small_mat_offset(i), this->sizes.small_W_offset(i),
            this->sizes.small_buffer_offset(i, this->eig_small_buffer_size),
            0, // buffer_host_offset (unused by the batched Jacobi EVD)
            info_offset);
        check_eig_info(
            this->small_info.vals + info_offset, this->sizes.small_mat_nums[i],
            "cusolverDnDsyevjBatched (cone measure)", this->sizes.small_mat_sizes[i]);
        info_offset += this->sizes.small_mat_nums[i];
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // host copies of the eigenvalues and of v
    std::vector<double> large_W_h(this->sizes.sum_large_mat_size);
    std::vector<double> medium_W_h(this->sizes.sum_medium_mat_size);
    std::vector<double> small_W_h(this->sizes.sum_small_mat_size);
    std::vector<double> v_h(this->vec_len);
    if (!large_W_h.empty())
        CHECK_CUDA(cudaMemcpy(large_W_h.data(), this->large_W.vals, sizeof(double) * large_W_h.size(), D2H));
    if (!medium_W_h.empty())
        CHECK_CUDA(cudaMemcpy(medium_W_h.data(), this->medium_W.vals, sizeof(double) * medium_W_h.size(), D2H));
    if (!small_W_h.empty())
        CHECK_CUDA(cudaMemcpy(small_W_h.data(), this->small_W.vals, sizeof(double) * small_W_h.size(), D2H));
    if (!v_h.empty())
        CHECK_CUDA(cudaMemcpy(v_h.data(), v.vals, sizeof(double) * v_h.size(), D2H));

    block_min.assign(this->blk_info.size(), 0.0);
    bound_coef.assign(this->blk_info.size(), 0.0);
    for (size_t b = 0; b < this->blk_info.size(); b++)
    {
        const BlockInfo &blk = this->blk_info[b];
        if (blk.type == 's')
        {
            const double *W = blk.category == MatrixSizeCategory::LARGE ? large_W_h.data() : (blk.category == MatrixSizeCategory::MEDIUM ? medium_W_h.data() : small_W_h.data());
            double lambda_min = W[blk.W_offset];
            for (int k = 1; k < blk.size; k++)
                lambda_min = std::min(lambda_min, W[blk.W_offset + k]);
            block_min[b] = lambda_min;
            bound_coef[b] = std::min(0.0, lambda_min);
        }
        else if (blk.type == 'l')
        {
            double entry_min = v_h[blk.vec_offset], negative_sum = 0.0;
            for (int k = 0; k < blk.size; k++)
            {
                entry_min = std::min(entry_min, v_h[blk.vec_offset + k]);
                negative_sum += std::min(0.0, v_h[blk.vec_offset + k]);
            }
            block_min[b] = entry_min;
            bound_coef[b] = negative_sum;
        }
        else // 'u'
        {
            double abs_max = 0.0, abs_sum = 0.0;
            for (int k = 0; k < blk.size; k++)
            {
                abs_max = std::max(abs_max, std::abs(v_h[blk.vec_offset + k]));
                abs_sum += std::abs(v_h[blk.vec_offset + k]);
            }
            block_min[b] = abs_max;
            bound_coef[b] = -abs_sum;
        }
    }
}

double SDPSolver::certified_lower_bound(const std::vector<double> &trace_bounds) const
{
    if (trace_bounds.size() != this->dual_cone_coef.size())
        throw std::invalid_argument(
            "certified_lower_bound: " + std::to_string(trace_bounds.size()) + " bounds given for " +
            std::to_string(this->dual_cone_coef.size()) + " blocks (call it after solve())");
    double bound = this->dobj;
    for (size_t b = 0; b < trace_bounds.size(); b++)
        bound += trace_bounds[b] * this->dual_cone_coef[b];
    return bound;
}

double SDPSolver::certified_lower_bound(double trace_bound) const
{
    return this->certified_lower_bound(std::vector<double>(this->dual_cone_coef.size(), trace_bound));
}

void SDPSolver::init(
    int eig_stream_num_per_gpu,
    int vec_len, int con_num,
    int *cpu_At_csc_col_ptrs, int *cpu_At_csc_row_ids, double *cpu_At_csc_vals, int At_nnz,
    int *cpu_b_indices, double *cpu_b_vals, int b_nnz,
    int *cpu_C_indices, double *cpu_C_vals, int C_nnz,
    char *cpu_blk_types, int *cpu_blk_sizes,
    int mat_num,
    ProjectionMethod initial_proj_method,
    ProjectionMethod final_proj_method,
    double *cpu_X_vals,
    double *cpu_y_vals,
    double *cpu_S_vals,
    double sig,
    bool use_lobpcg)
{
    // host copies of the original problem data, for the external validation of solve() (ValidationConfig); made
    // before the timing of init() starts
    this->host_At_col_ptrs.assign(cpu_At_csc_col_ptrs, cpu_At_csc_col_ptrs + con_num + 1);
    this->host_At_row_ids.assign(cpu_At_csc_row_ids, cpu_At_csc_row_ids + At_nnz);
    this->host_At_vals.assign(cpu_At_csc_vals, cpu_At_csc_vals + At_nnz);
    this->host_b_idx.assign(cpu_b_indices, cpu_b_indices + b_nnz);
    this->host_b_vals.assign(cpu_b_vals, cpu_b_vals + b_nnz);
    this->host_C_idx.assign(cpu_C_indices, cpu_C_indices + C_nnz);
    this->host_C_vals.assign(cpu_C_vals, cpu_C_vals + C_nnz);
    this->host_blk_types.assign(cpu_blk_types, cpu_blk_types + mat_num);
    this->host_blk_sizes.assign(cpu_blk_sizes, cpu_blk_sizes + mat_num);
    this->external_validator.reset();

    // start record time
    this->total_time = 0.0;
    if (this->start == nullptr)
        CHECK_CUDA(cudaEventCreate(&this->start));
    if (this->stop == nullptr)
        CHECK_CUDA(cudaEventCreate(&this->stop));
    cudaEventRecord(this->start);

    // prepare streams for copy data
    /*
    we create three flexible streams per GPU, corresponding to copy mom_mat, mom_W, mom_info
    they can also be used to parallelize kernel launches and cuda toolkit calls
    */
    this->stream_flex = std::vector<DeviceStream>(3);
    for (int stream_id = 0; stream_id < 3; stream_id++)
    {
        this->stream_flex[stream_id].set_gpu_id(GPU0);
        this->stream_flex[stream_id].activate();
    }

    // create handles for cuSPARSE and cuBLAS
    this->cusparseH.set_gpu_id(GPU0);
    this->cusparseH.activate();
    this->cublasH.set_gpu_id(GPU0);
    this->cublasH.activate();

    this->eig_stream_num_per_gpu = eig_stream_num_per_gpu;

    this->initial_proj_method = initial_proj_method;
    this->final_proj_method = final_proj_method;
    this->current_proj_method = initial_proj_method;
    this->switched_proj_method = false;
#if !(defined(CUDA_VERSION) && (CUDA_VERSION >= 12090))
    if (initial_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED || final_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED)
    {
        throw std::invalid_argument("the projection method 'COMPOSITE_FP32_EMULATED' was selected, but is not supported. BF16x9 emulation requires CUDA 12.9 or higher.");
    }
#endif

    /* Initialize the A matrix */
    this->vec_len = vec_len;
    this->con_num = con_num;
    this->At_csc.allocate(GPU0, vec_len, con_num, At_nnz);
    this->At_csr.allocate(GPU0, vec_len, con_num, At_nnz);
    this->A_csr.allocate(GPU0, con_num, vec_len, At_nnz);
    // first stream for col_ptrs
    CHECK_CUDA(cudaMemcpyAsync(this->At_csc.col_ptrs, cpu_At_csc_col_ptrs, sizeof(int) * (con_num + 1), H2D, this->stream_flex[0].stream));
    // second stream for row_ids
    CHECK_CUDA(cudaMemcpyAsync(this->At_csc.row_ids, cpu_At_csc_row_ids, sizeof(int) * At_nnz, H2D, this->stream_flex[1].stream));
    // third stream for vals
    CHECK_CUDA(cudaMemcpyAsync(this->At_csc.vals, cpu_At_csc_vals, sizeof(double) * At_nnz, H2D, this->stream_flex[2].stream));
    // wait for the streams to finish
    this->synchronize_gpu0_streams();

    // compute the norm of A
    this->normA.allocate(GPU0, con_num);
    get_normA(this->At_csc, this->normA);

    /* convert the At matrix from CSC to CSR format */
    this->CSCtoCSR_At2A_buffer_size = CSC_to_CSR_get_buffersize_cusparse(this->cusparseH, this->At_csc, this->At_csr);
    this->CSCtoCSR_At2A_buffer.allocate(GPU0, CSCtoCSR_At2A_buffer_size, true);
    CSC_to_CSR_cusparse(this->cusparseH, this->At_csc, this->At_csr, this->CSCtoCSR_At2A_buffer);
    CHECK_CUDA(cudaMemcpyAsync(this->A_csr.row_ptrs, this->At_csc.col_ptrs, sizeof(int) * (con_num + 1), D2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->A_csr.col_ids, this->At_csc.row_ids, sizeof(int) * At_nnz, D2D, this->stream_flex[1].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->A_csr.vals, this->At_csc.vals, sizeof(double) * At_nnz, D2D, this->stream_flex[2].stream));

    /* Initialize the AAt solver on CPU */
    const auto factorization_start = std::chrono::steady_clock::now();
    this->cpu_AAt_solver.get_A(
        this->At_csr.row_ptrs, this->At_csr.col_ids, this->At_csr.vals,
        this->At_csr.col_size, this->At_csr.row_size, this->At_csr.nnz,
        true, 1e-15);
    this->cpu_AAt_solver.factorize();
    printf("\n factorization of AA^T + 1e-15 I: %s, nnz(L) = %.3e, %.1f s\n",
           this->cpu_AAt_solver.supernodal_used ? "supernodal LL^T -> simplicial LDL^T" : "simplicial LDL^T",
           this->cpu_AAt_solver.cc.lnz,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - factorization_start).count());
    printf("   pivots in [%.2e, %.2e], %d tiny (< 1e-12 max) and %d nonpositive%s\n",
           this->cpu_AAt_solver.min_pivot, this->cpu_AAt_solver.max_pivot,
           this->cpu_AAt_solver.tiny_pivots, this->cpu_AAt_solver.nonpositive_pivots,
           this->cpu_AAt_solver.tiny_pivots > 0 ? ": A does not have full row rank (dependent constraints)" : "");
    // With a dense factor (few constraints, e.g. neosfrbr25: nnz(L) = m^2/2 = 1e8, 119 ms per CPU solve), the y-step
    // solves are two dense triangular solves on the GPU (4 ms), with the same factor. Sparse factors stay on the CPU:
    // cuSPARSE triangular solves were measured slower than CHOLMOD's there (0.1-1.3x).
    {
        size_t free_mem = 0, total_mem = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
        const double m = con_num;
        const cholmod_factor *F = this->cpu_AAt_solver.chol_fac_L;
        this->AAt_dense_gpu = con_num >= 2000 && m * m <= double(std::numeric_limits<int>::max()) &&
                              8.0 * m * m <= 0.25 * double(free_mem) &&
                              this->cpu_AAt_solver.cc.lnz > 0.3 * m * (m + 1) / 2 &&
                              !F->is_super && !F->is_ll;
        if (this->AAt_dense_gpu)
        {
            // simplicial LDL^T: column j holds D_jj first, then the entries L(i, j), i > j (Lnz[j] entries in all)
            const int *Lp = (const int *)F->p, *Li = (const int *)F->i, *Lnz = (const int *)F->nz;
            const double *Lx = (const double *)F->x;
            std::vector<double> L_dense(size_t(con_num) * con_num, 0.0), D(con_num);
            for (int j = 0; j < con_num; j++)
            {
                if (Lnz[j] < 1 || Li[Lp[j]] != j)
                    throw std::runtime_error("unexpected layout of the CHOLMOD factor: the diagonal is not first");
                D[j] = Lx[Lp[j]];
                L_dense[size_t(j) * con_num + j] = 1.0;
                for (int k = Lp[j] + 1; k < Lp[j] + Lnz[j]; k++)
                    L_dense[size_t(j) * con_num + Li[k]] = Lx[k];
            }
            this->AAt_L.allocate(GPU0, con_num * con_num);
            this->AAt_D.allocate(GPU0, con_num);
            CHECK_CUDA(cudaMemcpy(this->AAt_L.vals, L_dense.data(), sizeof(double) * L_dense.size(), H2D));
            CHECK_CUDA(cudaMemcpy(this->AAt_D.vals, D.data(), sizeof(double) * con_num, H2D));
        }
        if (this->AAt_dense_gpu)
            printf("   y-step solves: dense triangular solves on the GPU (%.2f GB)\n", 8.0 * m * m / 1e9);
        else
            printf("   y-step solves: CHOLMOD on the CPU\n");
    }
    // retrieve permutation of the L factor
    this->perm.allocate(GPU0, con_num);
    CHECK_CUDA(cudaMemcpyAsync(this->perm.vals, this->cpu_AAt_solver.chol_fac_L->Perm, sizeof(int) * con_num, H2D, this->stream_flex[0].stream));
    // allocate memory of right-hand side vector
    this->rhsy.allocate(GPU0, con_num);
    CHECK_CUDA(cudaMemset(this->rhsy.vals, 0, sizeof(double) * con_num)); // SpMV output, see Aty below
    this->rhsy_perm.allocate(GPU0, con_num);
    this->y_perm.allocate(GPU0, con_num);
    // compute inverse permutation
    std::vector<int> perm_tmp(con_num, 0);
    std::vector<int> perm_inv_tmp;
    memcpy(perm_tmp.data(), this->cpu_AAt_solver.chol_fac_L->Perm, sizeof(int) * con_num);
    this->perm_inv.allocate(GPU0, con_num);
    get_inverse_permutation(perm_inv_tmp, perm_tmp);
    CHECK_CUDA(cudaMemcpyAsync(this->perm_inv.vals, perm_inv_tmp.data(), sizeof(int) * con_num, H2D, this->stream_flex[1].stream));

    /* Initialize b, C, X, y, S, sig on GPU */
    this->b.allocate(GPU0, con_num, b_nnz);
    this->C.allocate(GPU0, vec_len, C_nnz);
    this->X.allocate(GPU0, vec_len);
    this->y.allocate(GPU0, con_num);
    this->S.allocate(GPU0, vec_len);
    CHECK_CUDA(cudaMemcpyAsync(this->b.indices, cpu_b_indices, sizeof(int) * b_nnz, H2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->b.vals, cpu_b_vals, sizeof(double) * b_nnz, H2D, this->stream_flex[1].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->C.indices, cpu_C_indices, sizeof(int) * C_nnz, H2D, this->stream_flex[2].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->C.vals, cpu_C_vals, sizeof(double) * C_nnz, H2D, this->stream_flex[0].stream));

    // copy X, y, and S from CPU to GPU
    // if the input is nullptr (no warm start), we will set them to 0
    if (cpu_X_vals != nullptr)
    {
        // copy from CPU to GPU
        CHECK_CUDA(cudaMemcpyAsync(this->X.vals, cpu_X_vals, sizeof(double) * vec_len, H2D, this->stream_flex[1].stream));
    }
    else
    {
        // set to 0
        CHECK_CUDA(cudaMemsetAsync(this->X.vals, 0, sizeof(double) * vec_len, this->stream_flex[1].stream));
    }
    if (cpu_y_vals != nullptr)
    {
        CHECK_CUDA(cudaMemcpyAsync(this->y.vals, cpu_y_vals, sizeof(double) * con_num, H2D, this->stream_flex[2].stream));
    }
    else
    {
        CHECK_CUDA(cudaMemsetAsync(this->y.vals, 0, sizeof(double) * con_num, this->stream_flex[2].stream));
    }
    if (cpu_S_vals != nullptr)
    {
        CHECK_CUDA(cudaMemcpyAsync(this->S.vals, cpu_S_vals, sizeof(double) * vec_len, H2D, this->stream_flex[0].stream));
    }
    else
    {
        CHECK_CUDA(cudaMemsetAsync(this->S.vals, 0, sizeof(double) * vec_len, this->stream_flex[0].stream));
    }
    this->sig = sig;

    /* Initialize blk and maps */
    // analyze the blk vector (validated, int-overflow checked): block sizes and numbers,
    // matrix layouts, and the maps for vectorization of matrices (computed on CPU)
    const BlockStructure blk_struct(cpu_blk_types, cpu_blk_sizes, mat_num);
    blk_struct.print();
    // get_maps-style maps have vec_len entries: the caller's vec_len must match the blocks
    if (blk_struct.vec_len != this->vec_len)
        throw std::invalid_argument("vec_len = " + std::to_string(this->vec_len) +
                                    " does not match the block sizes (expected " + std::to_string(blk_struct.vec_len) + ")");
    this->blk_info = blk_struct.blocks;
    this->psd_blk_sizes = blk_struct.psd_blk_sizes;
    this->psd_blk_nums = blk_struct.psd_blk_nums;
    this->sizes = blk_struct.sizes;
    const std::vector<int> &map_B_tmp = blk_struct.map_B;   // |
    const std::vector<int> &map_M1_tmp = blk_struct.map_M1; // |- CPU version
    const std::vector<int> &map_M2_tmp = blk_struct.map_M2; // |

    // copy to GPU
    this->map_B.allocate(GPU0, vec_len);  // |
    this->map_M1.allocate(GPU0, vec_len); // |- GPU version
    this->map_M2.allocate(GPU0, vec_len); // |
    CHECK_CUDA(cudaMemcpyAsync(this->map_B.vals, map_B_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->map_M1.vals, map_M1_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex[1].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->map_M2.vals, map_M2_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex[2].stream));

    /* Scale (A is already scaled) */
    // move b and C to GPU
    this->borg.allocate(GPU0, this->con_num, this->b.nnz);
    this->Corg.allocate(GPU0, this->vec_len, this->C.nnz);
    CHECK_CUDA(cudaMemcpyAsync(this->borg.indices, this->b.indices, sizeof(int) * this->b.nnz, D2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->borg.vals, this->b.vals, sizeof(double) * this->b.nnz, D2D, this->stream_flex[1].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->Corg.indices, this->C.indices, sizeof(int) * this->C.nnz, D2D, this->stream_flex[2].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->Corg.vals, this->C.vals, sizeof(double) * this->C.nnz, D2D, this->stream_flex[0].stream));
    this->synchronize_gpu0_streams();
    // compute the norms of b and C
    this->norm_borg = 1 + this->borg.get_norm(this->cublasH);
    this->norm_Corg = 1 + this->Corg.get_norm(this->cublasH);

    std::cout << std::endl
              << " ||C|| = " << norm_Corg << ", ||b|| = " << norm_borg << std::endl;

    // scale b and C by normA
    sparse_vector_div_dense_vector(this->b, this->normA);
    dense_vector_mul_dense_vector(this->y, this->normA);
    // divide b, C, X, y, and S by the corresponding norms
    this->bscale = 1 + this->b.get_norm(this->cublasH);
    this->Cscale = 1 + this->C.get_norm(this->cublasH);
    this->objscale = this->bscale * this->Cscale;
    sparse_vector_div_scalar(this->b, this->bscale);
    sparse_vector_div_scalar(this->C, this->Cscale);
    dense_vector_div_scalar(this->X, this->bscale);
    dense_vector_div_scalar(this->S, this->Cscale);
    dense_vector_div_scalar(this->y, this->Cscale);

    /* Initialize KKT residuals */
    // simple allocations
    this->Aty.allocate(GPU0, this->vec_len);
    this->Rp.allocate(GPU0, this->con_num);
    // cusparseSpMV reads its output vector even with beta = 0: never let it read uninitialized memory (0 * NaN = NaN)
    CHECK_CUDA(cudaMemset(this->Aty.vals, 0, sizeof(double) * this->vec_len));
    CHECK_CUDA(cudaMemset(this->Rp.vals, 0, sizeof(double) * this->con_num));
    this->SmC.allocate(GPU0, this->vec_len);
    this->Rd.allocate(GPU0, this->vec_len);
    this->Rporg.allocate(GPU0, this->con_num);
    this->Rdorg.allocate(GPU0, this->vec_len);

    // retrieve buffer sizes and allocate
    this->SpMV_Aty_buffer_size = SpMV_get_buffersize_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0);
    this->SpMV_Aty_buffer.allocate(GPU0, this->SpMV_Aty_buffer_size, true);
    SpMV_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0, this->SpMV_Aty_buffer);
    this->SpMV_AX_buffer_size = SpMV_get_buffersize_cusparse(this->cusparseH, this->A_csr, this->X, this->Rp, -1.0, 0.0);
    this->SpMV_AX_buffer.allocate(GPU0, this->SpMV_AX_buffer_size, true);
    SpMV_cusparse(this->cusparseH, this->A_csr, this->X, this->Rp, -1.0, 0.0, this->SpMV_AX_buffer);

    //
    axpby_cusparse(this->cusparseH, this->b, this->Rp, 1.0, 1.0);
    CHECK_CUDA(cudaMemcpy(this->SmC.vals, this->S.vals, sizeof(double) * this->vec_len, D2D));
    axpby_cusparse(this->cusparseH, this->C, this->SmC, -1.0, 1.0);
    dense_vector_add_dense_vector(this->Rd, this->Aty, this->SmC);
    dense_vector_mul_dense_vector_mul_scalar(this->Rporg, this->normA, this->Rp, this->bscale);
    dense_vector_mul_scalar(this->Rdorg, this->Rd, this->Cscale);

    // compute initial residuals
    this->errRp = this->Rporg.get_norm(this->cublasH) / this->norm_borg;
    this->errRd = this->Rdorg.get_norm(this->cublasH) / this->norm_Corg;
    this->maxfeas = max(this->errRp, this->errRd);
    this->SpVV_CtX_buffer_size = SparseVV_get_buffersize_cusparse(this->cusparseH, this->C, this->X);
    this->SpVV_CtX_buffer.allocate(GPU0, this->SpVV_CtX_buffer_size, true);
    this->pobj = SparseVV_cusparse(this->cusparseH, this->C, this->X, this->SpVV_CtX_buffer) * this->objscale;
    this->SpVV_bty_buffer_size = SparseVV_get_buffersize_cusparse(this->cusparseH, this->b, this->y);
    this->SpVV_bty_buffer.allocate(GPU0, this->SpVV_bty_buffer_size, true);
    this->dobj = SparseVV_cusparse(this->cusparseH, this->b, this->y, this->SpVV_bty_buffer) * this->objscale;
    this->relgap = abs(this->pobj - this->dobj) / (1 + abs(this->pobj) + abs(this->dobj));

    /* Eigen decomposition for medium matrices */
    this->medium_mat.allocate(GPU0, this->sizes.total_medium_mat_size);
    this->medium_W.allocate(GPU0, this->sizes.sum_medium_mat_size);
    this->medium_info.allocate(GPU0, this->sizes.medium_mat_num);

    // streams and handles for eigen decomposition
    this->eig_medium_stream_arr = std::vector<DeviceStream>(this->eig_stream_num_per_gpu);
    this->cusolverH_eig_medium_arr = std::vector<DeviceSolverDnHandle>(this->eig_stream_num_per_gpu);
    this->cublasH_eig_medium_arr = std::vector<DeviceBlasHandle>(this->eig_stream_num_per_gpu);
    for (int stream_id = 0; stream_id < this->eig_stream_num_per_gpu; stream_id++)
    {
        // ininitialize and activate the streams and handles
        this->eig_medium_stream_arr[stream_id].set_gpu_id(GPU0);
        this->eig_medium_stream_arr[stream_id].activate();
        this->cusolverH_eig_medium_arr[stream_id].set_gpu_id(GPU0);
        this->cusolverH_eig_medium_arr[stream_id].activate(this->eig_medium_stream_arr[stream_id]);
        this->cublasH_eig_medium_arr[stream_id].set_gpu_id(GPU0);
        this->cublasH_eig_medium_arr[stream_id].activate(this->eig_medium_stream_arr[stream_id]);
    }

    // compute the buffer sizes of the medium matrices eig decomposition
    this->eig_medium_buffer_size.assign(this->sizes.medium_mat_sizes.size(), 0);
    this->cpu_eig_medium_buffer_size.assign(this->sizes.medium_mat_sizes.size(), 0);

    this->sizes.medium_buffer_start_indices.push_back(0);
    this->sizes.medium_cpu_buffer_start_indices.push_back(0);
    size_t total_eig_medium_buffer_size = 0;
    size_t total_cpu_eig_medium_buffer_size = 0;
    for (int i = 0; i < this->sizes.medium_mat_sizes.size(); i++)
    {
        single_eig_get_buffersize_cusolver(
            this->cusolverH_eig_medium_arr[i % this->eig_stream_num_per_gpu], eig_param_single, this->medium_mat, this->medium_W,
            this->sizes.medium_mat_sizes[i],
            &this->eig_medium_buffer_size[i],
            &this->cpu_eig_medium_buffer_size[i],
            this->sizes.medium_mat_offset(i, 0), this->sizes.medium_W_offset(i, 0)); // buffer size per medium matrix of a given size

        // we need to multiply the buffer size by the number of matrices of this size
        total_eig_medium_buffer_size += this->eig_medium_buffer_size[i] * this->sizes.medium_mat_nums[i];
        total_cpu_eig_medium_buffer_size += this->cpu_eig_medium_buffer_size[i] * this->sizes.medium_mat_nums[i];

        this->sizes.medium_buffer_start_indices.push_back(
            this->sizes.medium_buffer_start_indices[i] + this->sizes.medium_mat_nums[i] * this->eig_medium_buffer_size[i]);
        this->sizes.medium_cpu_buffer_start_indices.push_back(
            this->sizes.medium_cpu_buffer_start_indices[i] + this->sizes.medium_mat_nums[i] * this->cpu_eig_medium_buffer_size[i]);
    }

    // batched EVD of the medium size groups with several matrices
    this->medium_group_batched.assign(this->sizes.medium_mat_sizes.size(), 0);
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
    {
        bool any_batched = false;
        for (int i = 0; i < this->sizes.medium_mat_sizes.size(); i++)
        {
            if (this->medium_batch_min_count > 0 && this->sizes.medium_mat_nums[i] >= this->medium_batch_min_count)
            {
                this->medium_group_batched[i] = 1;
                any_batched = true;
            }
        }
        if (any_batched)
        {
            this->cusolverH_eig_medium_batch.set_gpu_id(GPU0);
            this->cusolverH_eig_medium_batch.activate();
            for (int i = 0; i < this->sizes.medium_mat_sizes.size(); i++)
            {
                if (!this->medium_group_batched[i])
                    continue;
                const int n = this->sizes.medium_mat_sizes[i];
                size_t device_size = 0, host_size = 0;
                CHECK_CUSOLVER(cusolverDnXsyevBatched_bufferSize(
                    this->cusolverH_eig_medium_batch.cusolver_dn_handle, eig_param_single.param,
                    eig_param_single.jobz, eig_param_single.uplo,
                    n, CUDA_R_64F, this->medium_mat.vals + this->sizes.medium_mat_offset(i, 0), n,
                    CUDA_R_64F, this->medium_W.vals + this->sizes.medium_W_offset(i, 0), CUDA_R_64F,
                    &device_size, &host_size, this->sizes.medium_mat_nums[i]));
                this->medium_batch_buffer_size = std::max(this->medium_batch_buffer_size, device_size);
                this->medium_batch_cpu_buffer_size = std::max(this->medium_batch_cpu_buffer_size, host_size);
            }
            this->medium_batch_buffer.allocate_size_t(GPU0, std::max<size_t>(this->medium_batch_buffer_size, 8), true);
            this->medium_batch_cpu_buffer.allocate(int(std::max<size_t>(this->medium_batch_cpu_buffer_size, 8)), true);
        }
    }
#endif

    // allocate memory for the two buffers, host and device
    if (total_eig_medium_buffer_size != 0)
        this->eig_medium_buffer.allocate_size_t(GPU0, total_eig_medium_buffer_size, true);
    if (total_cpu_eig_medium_buffer_size != 0)
        this->cpu_eig_medium_buffer.allocate(total_cpu_eig_medium_buffer_size, true);

    /* Eigen decomposition for large matrices */
    // allocate GPU0 memory for large matrices
    this->large_mat.allocate(GPU0, this->sizes.total_large_mat_size);
    this->large_W.allocate(GPU0, this->sizes.sum_large_mat_size);
    this->large_info.allocate(GPU0, this->sizes.large_mat_num);

    // per large matrix: projection mode and number of LOBPCG eigenpairs (set by the rank analysis)
    this->large_mode = std::vector<int>(this->sizes.large_mat_num, LargeProjectionMode::FULL);
    this->large_k = std::vector<int>(this->sizes.large_mat_num, 0);
    this->large_rank_tol = std::vector<double>(this->sizes.large_mat_num, 0.0);
    this->stop_tol = 0.0; // set by solve()
    this->lobpcg_calls = 0;
    this->lobpcg_fallbacks = 0;
    this->composite_fallbacks = 0;
    this->jacobi_not_converged = 0;

    this->cusolverH_eig_large.set_gpu_id(GPU0);
    this->cusolverH_eig_large.activate();

    // compute the buffer sizes of the large matrices eig decomposition
    this->eig_large_buffer_size.assign(this->sizes.large_mat_sizes.size(), 0);
    this->cpu_eig_large_buffer_size.assign(this->sizes.large_mat_sizes.size(), 0);

    this->sizes.large_buffer_start_indices.push_back(0);
    this->sizes.large_cpu_buffer_start_indices.push_back(0);
    size_t total_eig_large_buffer_size = 0;
    size_t total_cpu_eig_large_buffer_size = 0;
    for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++)
    {
        // always needed: the exact cuSOLVER EVD is the fallback of every projection method
        single_eig_get_buffersize_cusolver(
            this->cusolverH_eig_large, eig_param_single, this->large_mat, this->large_W,
            this->sizes.large_mat_sizes[i],
            &this->eig_large_buffer_size[i],
            &this->cpu_eig_large_buffer_size[i],
            this->sizes.large_mat_offset(i, 0), this->sizes.large_W_offset(i, 0)); // buffer size per large matrix of a given size

        // we need to multiply the buffer size by the number of matrices of this size
        total_eig_large_buffer_size += this->eig_large_buffer_size[i] * this->sizes.large_mat_nums[i];
        total_cpu_eig_large_buffer_size += this->cpu_eig_large_buffer_size[i] * this->sizes.large_mat_nums[i];

        this->sizes.large_buffer_start_indices.push_back(
            this->sizes.large_buffer_start_indices[i] + this->sizes.large_mat_nums[i] * this->eig_large_buffer_size[i]);
        this->sizes.large_cpu_buffer_start_indices.push_back(
            this->sizes.large_cpu_buffer_start_indices[i] + this->sizes.large_mat_nums[i] * this->cpu_eig_large_buffer_size[i]);
    }

    // allocate memory for the two buffers, host and device
    if (total_eig_large_buffer_size != 0)
        this->eig_large_buffer.allocate_size_t(GPU0, total_eig_large_buffer_size, true);
    if (total_cpu_eig_large_buffer_size != 0)
        this->cpu_eig_large_buffer.allocate(total_cpu_eig_large_buffer_size, true);

    if (
        this->sizes.large_mat_sizes.size() > 0 && (this->initial_proj_method == ProjectionMethod::COMPOSITE_FP32 || this->initial_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED || this->initial_proj_method == ProjectionMethod::COMPOSITE_FP16

                                                   || this->final_proj_method == ProjectionMethod::COMPOSITE_FP32 || this->final_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED || this->final_proj_method == ProjectionMethod::COMPOSITE_FP16))
    {
        // create a workspace for the composite projection
        int largest_size = *std::max_element(this->sizes.large_mat_sizes.begin(), this->sizes.large_mat_sizes.end());
        size_t nn = (size_t)largest_size * largest_size;
        size_t stride = nn % 4 == 0 ? nn : nn + (4 - nn % 4); // we need to ensure proper memory alignment

        this->float_proj_workspace.allocate_size_t(GPU0, 3 * stride);
        if (this->initial_proj_method == ProjectionMethod::COMPOSITE_FP16 || this->final_proj_method == ProjectionMethod::COMPOSITE_FP16) // if FP16, we need a second workspace
            this->half_proj_workspace.allocate_size_t(GPU0, 3 * stride);

        // create a cuBLAS handle
        this->cublasH_composite_proj.set_gpu_id(GPU0);
        this->cublasH_composite_proj.activate();
        CHECK_CUBLAS(cublasSetMathMode(this->cublasH_composite_proj.cublas_handle, CUBLAS_TENSOR_OP_MATH));
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
        if (this->initial_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED || this->final_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED)
        {
            CHECK_CUBLAS(cublasSetEmulationStrategy(this->cublasH_composite_proj.cublas_handle, CUBLAS_EMULATION_STRATEGY_EAGER));
        }
#endif

        // create a cuSOLVER handle
        this->cusolverH_composite_proj.set_gpu_id(GPU0);
        this->cusolverH_composite_proj.activate();
    }

    this->cublasH_eig_large.set_gpu_id(GPU0);
    this->cublasH_eig_large.activate();

    /* Eigenvalue decomposition for small matrices */
    this->cusolverH_eig_small.set_gpu_id(GPU0);
    this->cusolverH_eig_small.activate();
    this->small_mat.allocate(GPU0, this->sizes.total_small_mat_size);
    this->small_W.allocate(GPU0, this->sizes.sum_small_mat_size);
    this->small_info.allocate(GPU0, this->sizes.small_mat_num);
    CHECK_CUDA(cudaMemset(this->small_info.vals, 0, sizeof(int) * this->sizes.small_mat_num));
    this->eig_small_buffer_size.reserve(this->sizes.small_mat_sizes.size());

    this->sizes.small_buffer_start_indices.push_back(0);
    for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++)
    {
        this->eig_small_buffer_size.push_back(
            batch_eig_get_buffersize_cusolver(
                this->cusolverH_eig_small, this->eig_param_batch,
                this->small_mat, this->small_W,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->sizes.small_mat_offset(i), this->sizes.small_W_offset(i)));

        this->sizes.small_buffer_start_indices.push_back(
            this->sizes.small_buffer_start_indices[i] + this->eig_small_buffer_size[i]);
    }

    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[0].stream));
    // we do not need to multiply the buffer size by the number of matrices,
    // since it is already done in the function
    this->eig_small_buffer.allocate_size_t(GPU0, this->sizes.small_buffer_start_indices.back(), true);

    /* For the computation of y, X, S */
    if (this->sizes.medium_mat_num > 0)
    {
        this->medium_mat_tmp.allocate(GPU0, this->sizes.total_medium_mat_size);
        this->medium_mat_P.allocate(GPU0, this->sizes.total_medium_mat_size);
    }
    if (this->sizes.large_mat_num > 0)
    {
        // large_mat_P is written by every projection method, large_mat_tmp by the exact EVD (the fallback of every method)
        this->large_mat_tmp.allocate(GPU0, this->sizes.total_large_mat_size);
        this->large_mat_P.allocate(GPU0, this->sizes.total_large_mat_size);
    }
    this->small_mat_tmp.allocate(GPU0, this->sizes.total_small_mat_size);
    this->small_mat_P.allocate(GPU0, this->sizes.total_small_mat_size);
    this->Rd1.allocate(GPU0, this->vec_len);
    this->Xinput.allocate(GPU0, this->vec_len);

    /* Application of LOBPCG to large eigenvalues */
    this->use_lobpcg = use_lobpcg;
    if (use_lobpcg)
    {
        if (final_proj_method != ProjectionMethod::EIG_FP64)
        {
            throw std::invalid_argument("when 'use_lobpcg' is enabled, the final projection method must be 'EIG_FP64'.");
        }

        this->positive_ranks.allocate(GPU0, this->sizes.large_mat_num);
        this->negative_ranks.allocate(GPU0, this->sizes.large_mat_num);
        this->cpu_positive_ranks.allocate(this->sizes.large_mat_num);
        this->cpu_negative_ranks.allocate(this->sizes.large_mat_num);

        // we only need the ReLU workspace for one matrix at a time
        int max_k = std::ceil(this->sizes.max_large_mat_size * LOBPCG_RATIO * LOBPCG_RELAXATION);
        this->lobpcg_W_relu.allocate(GPU0, max_k);

        // allocate one pointer for each large matrix
        this->lobpcg_W = std::vector<DeviceDenseVector<double>>(this->sizes.large_mat_num);
        this->lobpcg_P = std::vector<DeviceDenseVector<double>>(this->sizes.large_mat_num);
        this->lobpcg_k_alloc = std::vector<int>(this->sizes.large_mat_num, 0);
        int counter = 0;
        for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++)
        {
            for (int j = 0; j < this->sizes.large_mat_nums[i]; j++)
            {
                int n = this->sizes.large_mat_sizes[i];
                int k = std::ceil(n * LOBPCG_RATIO * LOBPCG_RELAXATION);
                this->lobpcg_k_alloc[counter] = k; // upper bound on the number of eigenpairs requested from LOBPCG
                // allocate space to store warmstarting eigenpairs
                this->lobpcg_W[counter] = DeviceDenseVector<double>();
                this->lobpcg_W[counter].allocate(GPU0, k);

                this->lobpcg_P[counter] = DeviceDenseVector<double>();
                this->lobpcg_P[counter].allocate(GPU0, k * n);
                counter++;
            }
        }
    }

    /* others */
    this->prim_win = 0;
    this->dual_win = 0;
    this->ratioconst = 1e0;
    this->sigmax = 1e6;
    this->sigmin = 1e-6;

    /* Main elements for the sGS-ADMM algorithm */
    this->X_best.allocate(GPU0, this->vec_len);
    this->y_best.allocate(GPU0, this->con_num);
    this->S_best.allocate(GPU0, this->vec_len);
    this->best_KKT = std::numeric_limits<double>::infinity(); // no best iterate saved yet
    this->errPSD_X = 0.0;
    this->errPSD_S = 0.0;
    this->converged = false;
    this->returned_best_iterate = false;
    this->time_limit_reached = false;
    this->solve_time = 0.0;

    // time spent in init()
    float init_milliseconds;
    CHECK_CUDA(cudaEventRecord(this->stop));
    CHECK_CUDA(cudaEventSynchronize(this->stop));
    CHECK_CUDA(cudaEventElapsedTime(&init_milliseconds, this->start, this->stop));
    this->init_time = init_milliseconds / 1000;

    return;
}

void SDPSolver::solve(
    int max_iter, double stop_tol,
    int sig_update_threshold,
    int sig_update_stage_1,
    int sig_update_stage_2,
    int switch_admm,
    int switch_proj_max_iter,
    double switch_proj_tol,
    double sigscale,
    bool if_first)
{
    // save parameters
    this->stop_tol = stop_tol;
    this->sig_update_threshold = sig_update_threshold;
    this->sig_update_stage_1 = sig_update_stage_1;
    this->sig_update_stage_2 = sig_update_stage_2;
    this->switch_admm = switch_admm;
    this->sigscale = sigscale;

#ifdef CUADMM_PLAIN_ADMM_ONLY
    // plain-ADMM-only build: sGS-ADMM (switch_admm > max_iter) and the sGS -> ADMM hybrid (1 < switch_admm) are rejected
    if (switch_admm > 1)
        throw std::invalid_argument("solve(): switch_admm = " + std::to_string(switch_admm) +
                                    " requests sGS-ADMM steps, which this plain-ADMM-only build (CUADMM_PLAIN_ADMM_ONLY) does "
                                    "not execute; use switch_admm = 0 (plain ADMM)");
    if (this->sigma_policy == SigmaPolicy::FIXED_SGS_LEGACY_ADMM)
        throw std::invalid_argument("solve(): the fixed_sgs_then_legacy_admm policy is for the sGS -> ADMM hybrid, which this "
                                    "plain-ADMM-only build does not execute; use fixed or legacy_adaptive");
#endif
    // validation-aware stopping (ValidationConfig): state of this call
    const bool validating = this->validation.interval > 0;
    const std::set<int> checkpoints(this->validation.checkpoint_iterations.begin(), this->validation.checkpoint_iterations.end());
    this->stop_reason.clear();
    this->externally_validated_converged = false;
    this->internal_converged_at_return = false;
    this->first_validated_iteration = -1;
    this->first_internal_below_1e_3 = -1;
    this->first_internal_below_tol = -1;
    this->validation_history.clear();
    this->best_validation = ValidationRecord();
    this->returned_best_external = false;
    this->validation_time_s = 0.0;
    this->y_solves = 0;
    this->sigma_log.clear();
    this->practical_validated = this->strict_validated = false;
    this->first_practical_iteration = this->first_strict_iteration = -1;
    this->first_practical_time_s = this->first_strict_time_s = -1.0;
    this->callback_time_s = 0.0;
    this->callback_failures = 0;
    this->peak_gpu_mem_bytes = 0;
    if (validating && !this->external_validator)
    {
        this->external_validator.reset(new ExternalValidator(
            this->vec_len, this->con_num,
            this->host_At_col_ptrs.data(), this->host_At_row_ids.data(), this->host_At_vals.data(),
            (int)this->host_b_idx.size(), this->host_b_idx.data(), this->host_b_vals.data(),
            (int)this->host_C_idx.size(), this->host_C_idx.data(), this->host_C_vals.data(),
            this->host_blk_types, this->host_blk_sizes));
        if (!this->dimacs_kinds.empty())
            this->external_validator->set_dimacs_kinds(this->dimacs_kinds);
        // the validator keeps its own copy of the problem data
        std::vector<int>().swap(this->host_At_col_ptrs);
        std::vector<int>().swap(this->host_At_row_ids);
        std::vector<double>().swap(this->host_At_vals);
        std::vector<int>().swap(this->host_b_idx);
        std::vector<double>().swap(this->host_b_vals);
        std::vector<int>().swap(this->host_C_idx);
        std::vector<double>().swap(this->host_C_vals);
    }
    bool prev_internal_ok = false;
    bool nonfinite = false; // (validation mode) the last iteration produced non-finite residuals
    auto sample_gpu_mem = [this]()
    {
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            this->peak_gpu_mem_bytes = std::max(this->peak_gpu_mem_bytes, total_mem - free_mem);
    };
    sample_gpu_mem();

    // sigma policy bookkeeping for this call
    this->sigma0 = this->sig;
    this->sigma_changes = 0;
    this->admm_phase_iterations = 0;
    this->sgs_phase_iterations = 0;
    this->switch_iteration = -1;
    this->sgs_phase_time_s = this->admm_phase_time_s = 0.0;
    this->sgs_phase_validation_s = this->admm_phase_validation_s = 0.0;
    this->kkt_at_switch = this->sigma_at_switch = this->tau_last_sgs = -1.0;
    this->sigma_changes_sgs_phase = this->sigma_changes_admm_phase = 0;

    // no best iterate saved yet for this call
    this->best_KKT = std::numeric_limits<double>::infinity();
    this->lobpcg_calls = 0;
    this->lobpcg_fallbacks = 0;
    this->composite_fallbacks = 0;
    this->jacobi_not_converged = 0;

    // declare variables
    bool breakyes = false; // for breaking out of the loop
    std::string final_msg; // output message
    this->info_iter_num = 0; // iteration number
    this->converged = false;
    this->returned_best_iterate = false;

    double one = 1.0;
    double zero = 0.0;
    double minus_one = -1.0;

    std::cout << std::endl;
    std::cout << " Algorithm: " << admm_mode_name(switch_admm, max_iter) << std::endl;
#ifdef CUADMM_PLAIN_ADMM_ONLY
    std::cout << " Build: plain ADMM only (CUADMM_PLAIN_ADMM_ONLY): sGS-ADMM and the sGS -> ADMM hybrid are disabled; "
                 "one y linear solve per iteration" << std::endl;
#endif
    if (validating && this->validation.strict_dimacs_tol > 0)
        std::cout << " Strict stage: after the practical criteria, continue until max_abs_DIMACS <= " << this->validation.strict_dimacs_tol
                  << " or a safety limit" << std::endl;
    if (switch_admm > 1 && switch_admm <= max_iter)
        std::cout << " Hybrid mode of the public code (not Algorithm 1): " << switch_admm - 1
                  << " sGS-ADMM iterations, then plain ADMM from iteration " << switch_admm << " (switch_admm = " << switch_admm
                  << "), continuing from the same X, y, S, sigma and workspaces" << std::endl;
    std::cout << " Sigma policy: " << sigma_policy_name(this->sigma_policy) << ", initial sigma " << this->sig
              << (this->sigma_policy == SigmaPolicy::FIXED ? " (kept for every iteration)"
                  : this->sigma_policy == SigmaPolicy::FIXED_SGS_LEGACY_ADMM
                      ? " (kept during the sGS phase; legacy plain-ADMM rules from the first plain-ADMM iteration, counted from there)"
                      : " (adaptive updates)")
              << std::endl;
    std::cout << " Maximum iterations: " << max_iter << ", internal KKT tolerance: " << stop_tol << std::endl;
    if (validating)
    {
        const ValidationTolerances &t = this->validation.tol;
        std::cout << " External validation: every " << this->validation.interval
                  << " iterations and at each crossing of the internal tolerance; stop only when eta_p <= " << t.primal
                  << ", eta_d <= " << t.dual << ", eta_g <= " << t.gap << ", X cone <= " << t.cone
                  << ", dual cone <= " << t.dual_cone << " (recomputed on the CPU, all finite)" << std::endl;
        if (!checkpoints.empty())
        {
            std::cout << " Checkpoints:";
            for (int c : checkpoints)
                std::cout << " " << c;
            std::cout << std::endl;
        }
    }
    else
        std::cout << " External validation: off (stopping on the internal KKT residual)" << std::endl;
    if (std::isfinite(this->time_limit))
        std::cout << " Time limit: " << this->time_limit << " s" << std::endl;
    std::cout << std::endl;
    std::cout << "Problem parameters:" << std::endl;
    std::cout << "              solver max iter: " << max_iter << std::endl;
    std::cout << "       KKT stopping tolerance: " << stop_tol << std::endl;
    std::cout << "       sigma update threshold: " << sig_update_threshold << std::endl;
    std::cout << "         sigma update stage 1: " << sig_update_stage_1 << std::endl;
    std::cout << "         sigma update stage 2: " << sig_update_stage_2 << std::endl;
    std::cout << "                  switch admm: " << switch_admm << std::endl;
    std::cout << "                    algorithm: " << admm_mode_name(switch_admm, max_iter) << std::endl;
    std::cout << "         switch proj max iter: " << switch_proj_max_iter << std::endl;
    std::cout << "              switch proj tol: " << switch_proj_tol << std::endl;
    std::cout << "                     sigscale: " << sigscale << std::endl;
    std::cout << "                 sigma policy: " << sigma_policy_name(this->sigma_policy) << std::endl;
    std::cout << "                initial sigma: " << this->sig << std::endl;
    std::cout << "    initial projection method: " << get_projection_method_name(this->initial_proj_method, false) << std::endl;
    std::cout << "      final projection method: " << get_projection_method_name(this->final_proj_method, false) << std::endl;
    std::cout << "                   use LOBPCG: " << (this->use_lobpcg ? "true" : "false") << std::endl;
    if (this->use_lobpcg)
    {
        std::cout << "              LOBPCG max iter: " << LOBPCG_MAXIT << std::endl;
        std::cout << "             LOBPCG tolerance: " << LOBPCG_TOL << std::endl;
        std::cout << "             LOBPCG warmstart: " << (LOBPCG_WARMSTART ? "true" : "false") << std::endl;
        std::cout << "                 LOBPCG ratio: " << LOBPCG_RATIO << std::endl;
        std::cout << "            LOBPCG relaxation: " << LOBPCG_RELAXATION << std::endl;
        std::cout << "        LOBPCG rank threshold: max(" << LOBPCG_RANK_RTOL << " * max|lambda|, " << LOBPCG_RANK_STOP_TOL_FACTOR << " * stop_tol)" << std::endl;
    }
    std::cout << "           small matrix limit: " << SMALL_MAT_LIMIT << std::endl;
    std::cout << "          medium matrix limit: " << MEDIUM_MAT_LIMIT << std::endl;

    /* Start the solver */
    std::cout << std::endl;
    std::cout << " ------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                   cuADMM" << std::endl;
    std::cout << " ------------------------------------------------------------------------------" << std::endl;
    float milliseconds;
    float seconds;

    if (!if_first)
    {
        // we suppose that for the second call, new X, y, S, sig are passed, but they are unscaled

        // scale X, y, S
        dense_vector_mul_dense_vector(this->y, this->normA);
        dense_vector_div_scalar(this->X, this->bscale);
        dense_vector_div_scalar(this->S, this->Cscale);
        dense_vector_div_scalar(this->y, this->Cscale);

        // SmC <-- S
        CHECK_CUDA(cudaMemcpy(this->SmC.vals, this->S.vals, sizeof(double) * this->vec_len, D2D));
        // hence Smc = S

        // SmC <-- -1.0 * C + 1.0 * SmC
        axpby_cusparse(this->cusparseH, this->C, this->SmC, -1.0, 1.0);
        // hence SmC = S - C

        // Rp <-- -1.0 * A * X + 0.0 * Rp
        SpMV_cusparse(this->cusparseH, this->A_csr, this->X, this->Rp, -1.0, 0.0, this->SpMV_AX_buffer);
        // hence Rp = - A X

        // Rp <-- 1.0 * b + 1.0 * Rp
        axpby_cusparse(this->cusparseH, this->b, this->Rp, 1.0, 1.0);
        // hence Rp = b - A X

        // residuals and objectives of the new starting point (the stored values describe the end of the
        // previous solve, and the stopping test of the first iteration uses them)
        SpMV_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0, this->SpMV_Aty_buffer);
        dense_vector_add_dense_vector(this->Rd, this->Aty, this->SmC);
        dense_vector_mul_dense_vector_mul_scalar(this->Rporg, this->normA, this->Rp, this->bscale);
        dense_vector_mul_scalar(this->Rdorg, this->Rd, this->Cscale);
        this->errRp = this->Rporg.get_norm(this->cublasH) / this->norm_borg;
        this->errRd = this->Rdorg.get_norm(this->cublasH) / this->norm_Corg;
        this->maxfeas = max(this->errRp, this->errRd);
        this->pobj = SparseVV_cusparse(this->cusparseH, this->C, this->X, this->SpVV_CtX_buffer) * this->objscale;
        this->dobj = SparseVV_cusparse(this->cusparseH, this->b, this->y, this->SpVV_bty_buffer) * this->objscale;
        this->relgap = abs(this->pobj - this->dobj) / (1 + abs(this->pobj) + abs(this->dobj));

        // per-solve state: primal/dual win counters of the sigma update, and the timer
        this->prim_win = 0;
        this->dual_win = 0;
        CHECK_CUDA(cudaEventRecord(this->start));
    }

    std::cout << "  it. | p infeas d infeas | primal obj.   dual obj. rel. gap |  time |   sigma " << std::endl;
    std::cout << " ------------------------------------------------------------------------------" << std::endl;
    std::cout << " --------------- Starting with projection method " << get_projection_method_name(this->current_proj_method) << "------" << std::endl;

    // for each iteration of the main solver
    Monitor1 monitor1;
    this->time_limit_reached = false;
    const auto solve_start = std::chrono::steady_clock::now();
    auto seconds_since_start = [&solve_start]()
    { return std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count(); };

    for (int iter = 1; iter <= max_iter + 1; iter++)
    {
        /*
            Step 0: Check if terminal conditions hold and log information
        */
        const double internal_kkt = max(this->maxfeas, this->relgap);
        const int done = iter - 1; // completed iterations: the current (X, y, S) is the iterate after `done` steps
        if (done > 0 && internal_kkt < 1e-3 && this->first_internal_below_1e_3 < 0)
            this->first_internal_below_1e_3 = done;
        if (done > 0 && internal_kkt < stop_tol && this->first_internal_below_tol < 0)
            this->first_internal_below_tol = done;
        if (!validating)
        {
            if (internal_kkt < stop_tol)
            {
                // stop if the stopping criterion is met
                breakyes = true;
                final_msg = "Solver ended: converged.";
                this->converged = true;
                this->stop_reason = "internal_kkt";
            }
            if (iter > max_iter && !breakyes)
            {
                // stop if the maximum number of iterations is reached
                breakyes = true;
                final_msg = "Solver ended: maximum iteration reached";
                this->stop_reason = "max_iter";
            }
        }
        const double elapsed = seconds_since_start();
        if (!validating && elapsed > this->time_limit && !breakyes)
        {
            // stop if the time limit is reached
            breakyes = true;
            final_msg = "Solver ended: time limit reached";
            this->time_limit_reached = true;
            this->stop_reason = "time_limit";
        }
        if (validating)
        {
            // the internal residual only triggers a validation; only a validated snapshot stops the solve
            const bool internal_ok = internal_kkt < stop_tol;
            const bool at_cap = iter > max_iter, at_time = elapsed > this->time_limit;
            std::string trigger;
            if (nonfinite)
            {
                // the iterate is not usable: stop and return the best externally evaluated one
                if (this->validation_history.empty())
                    throw std::runtime_error(
                        "non-finite KKT residuals at iteration " + std::to_string(done) + " before any external validation");
                breakyes = true;
                final_msg = "Solver ended: non-finite residuals (not validated)";
                this->stop_reason = "non_finite";
            }
            else if (done > 0 || at_cap || at_time || internal_ok)
            {
                // the starting point (done == 0) only when the solve ends at once or it already satisfies the
                // internal tolerance (a warm start); an iteration that is also a requested checkpoint is flagged
                if (at_cap)
                    trigger = "max_iter";
                else if (at_time)
                    trigger = "time_limit";
                else if (internal_ok && !prev_internal_ok)
                    trigger = "internal_crossing";
                else if (done > 0 && checkpoints.count(done))
                    trigger = "checkpoint";
                else if (done > 0 && done % this->validation.interval == 0)
                    trigger = "interval";
            }
            prev_internal_ok = internal_ok;
            if (!trigger.empty())
            {
                const ValidationRecord rec = this->validate_current_iterate(done, elapsed, trigger, done > 0 && checkpoints.count(done) > 0);
                if (rec.result.validated)
                {
                    const double strict_tol = this->validation.strict_dimacs_tol;
                    if (!this->practical_validated)
                    {
                        // the practical criteria: the first time (recorded; the solve stops here unless a strict stage is set)
                        this->practical_validated = true;
                        this->first_practical_iteration = done;
                        this->first_practical_time_s = seconds_since_start();
                        this->converged = true;
                        this->externally_validated_converged = true;
                        this->first_validated_iteration = done;
                        if (strict_tol > 0)
                            printf(" practical criteria validated at iteration %d (%.3f s, max_abs_DIMACS %.2e): continuing toward %.1e\n",
                                   done, this->first_practical_time_s, rec.result.max_abs_dimacs, strict_tol);
                    }
                    const bool strict_ok = strict_tol > 0 && std::isfinite(rec.result.max_abs_dimacs) && rec.result.max_abs_dimacs <= strict_tol;
                    if (strict_tol <= 0 || strict_ok)
                    {
                        breakyes = true;
                        final_msg = strict_ok ? "Solver ended: strict DIMACS target validated" : "Solver ended: externally validated";
                        this->stop_reason = strict_ok ? "strict_validated" : "externally_validated";
                        if (strict_ok)
                        {
                            this->strict_validated = true;
                            this->first_strict_iteration = done;
                            this->first_strict_time_s = seconds_since_start();
                        }
                    }
                }
                else if (trigger == "internal_crossing")
                    printf(" internal KKT %.2e < %.1e at iteration %d, but the external validation failed (%s): continuing\n",
                           internal_kkt, stop_tol, done, rec.result.failed.c_str());
            }
            if (!breakyes && at_cap)
            {
                breakyes = true;
                final_msg = "Solver ended: maximum iteration reached (safety cap, not validated)";
                this->stop_reason = "max_iter";
            }
            if (!breakyes && at_time)
            {
                breakyes = true;
                final_msg = "Solver ended: time limit reached (not validated)";
                this->time_limit_reached = true;
                this->stop_reason = "time_limit";
            }
        }
        if (
            // true ||
            (breakyes == true) ||
            ((iter <= 200) && ((iter % 50) == 1)) ||
            ((iter > 200) && ((iter % 100) == 1)))
        {
            // print the iteration number and the residuals
            cudaEventRecord(this->stop);
            cudaEventSynchronize(this->stop);
            cudaEventElapsedTime(&milliseconds, this->start, this->stop);
            seconds = milliseconds / 1000;
            printf(
                " %4d | %3.2e %3.2e | %- 5.4e %- 5.4e %3.2e | %5.1f | %2.1e ",
                iter - 1, this->errRp, this->errRd, this->pobj, this->dobj, this->relgap, seconds, this->sig);
            std::cout << std::endl;
        }
        if (breakyes == true)
        {
            // validation mode: the time after the final validation, which is part of the solve
            this->solve_time = validating ? seconds_since_start() : elapsed;
            sample_gpu_mem();
            if (this->switch_iteration < 0)
            {
                // no plain-ADMM step ran: the whole solve is the sGS phase
                this->sgs_phase_time_s = this->solve_time;
                this->sgs_phase_validation_s = this->validation_time_s;
            }
            else
            {
                this->admm_phase_time_s = this->solve_time - this->sgs_phase_time_s;
                this->admm_phase_validation_s = this->validation_time_s - this->sgs_phase_validation_s;
            }

            /* Validation-aware stopping without a validated snapshot: return the best externally evaluated
               iterate if it is strictly better than the last one (the last snapshot is the final iterate), or
               whenever the final iterate is non-finite */
            if (validating && !this->strict_validated && !(this->externally_validated_converged && this->validation.strict_dimacs_tol <= 0) &&
                !this->validation_history.empty())
            {
                const ValidationRecord &last = this->validation_history.back();
                const ValidationRecord &best = this->best_validation;
                if (nonfinite || (best.iter != last.iter && best.result.better_than(last.result)))
                {
                    this->returned_best_external = true;
                    if (nonfinite)
                        printf(" Returning the best externally evaluated iterate: iteration %d, merit %2.1e (iteration %d is non-finite)\n",
                               best.iter, best.result.merit(), done);
                    else
                        printf(" Returning the best externally evaluated iterate: iteration %d, merit %2.1e (final iterate: %2.1e)\n",
                               best.iter, best.result.merit(), last.result.merit());
                    CHECK_CUDA(cudaMemcpyAsync(this->X.vals, this->X_best.vals, sizeof(double) * this->vec_len, D2D, (cudaStream_t)0));
                    CHECK_CUDA(cudaMemcpyAsync(this->y.vals, this->y_best.vals, sizeof(double) * this->con_num, D2D, (cudaStream_t)0));
                    CHECK_CUDA(cudaMemcpyAsync(this->S.vals, this->S_best.vals, sizeof(double) * this->vec_len, D2D, (cudaStream_t)0));
                    // the solver's own residuals and objectives of that snapshot
                    this->errRp = best.internal_errRp;
                    this->errRd = best.internal_errRd;
                    this->relgap = best.internal_relgap;
                    this->pobj = best.internal_pobj;
                    this->dobj = best.internal_dobj;
                    this->maxfeas = max(this->errRp, this->errRd);
                }
            }

            /* Return the best iterate if it is strictly better than the last one */
            if (this->best_KKT < max(this->maxfeas, this->relgap))
            {
                this->returned_best_iterate = true;
                printf(" Returning the best iterate: max KKT residual = %2.1e (last iterate: %2.1e)\n",
                       this->best_KKT, max(this->maxfeas, this->relgap));
                CHECK_CUDA(cudaMemcpyAsync(this->X.vals, this->X_best.vals, sizeof(double) * this->vec_len, D2D, (cudaStream_t)0));
                CHECK_CUDA(cudaMemcpyAsync(this->y.vals, this->y_best.vals, sizeof(double) * this->con_num, D2D, (cudaStream_t)0));
                CHECK_CUDA(cudaMemcpyAsync(this->S.vals, this->S_best.vals, sizeof(double) * this->vec_len, D2D, (cudaStream_t)0));
                // the reported residuals and objectives describe the returned iterate
                this->errRp = this->best_errRp;
                this->errRd = this->best_errRd;
                this->relgap = this->best_relgap;
                this->pobj = this->best_pobj;
                this->dobj = this->best_dobj;
                this->maxfeas = max(this->errRp, this->errRd);
            }

            /* Cone violations of the returned X and S over all blocks, and certificate of C - A^T y */
            std::vector<double> block_min, bound_coef;
            double min_X = 0.0, min_S = 0.0;
            this->cone_measures(this->X, block_min, bound_coef);
            for (size_t b = 0; b < this->blk_info.size(); b++)
                if (this->blk_info[b].type != 'u')
                    min_X = std::min(min_X, block_min[b]);
            this->cone_measures(this->S, block_min, bound_coef);
            for (size_t b = 0; b < this->blk_info.size(); b++)
                if (this->blk_info[b].type != 'u')
                    min_S = std::min(min_S, block_min[b]);
            // scale them back and make them relative to 1 + ||b|| and 1 + ||C||
            double err_PSD_X = -min_X * this->bscale / this->norm_borg;
            double err_PSD_S = -min_S * this->Cscale / this->norm_Corg;
            this->errPSD_X = err_PSD_X;
            this->errPSD_S = err_PSD_S;

            this->internal_converged_at_return = max(this->maxfeas, this->relgap) < stop_tol;

            // Z = C - A^T y for the returned y (scaled units: C - A^T y = Cscale * (C_s - A_s^T y_s))
            SpMV_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0, this->SpMV_Aty_buffer);
            CHECK_CUDA(cudaMemcpy(this->Rd1.vals, this->Aty.vals, sizeof(double) * this->vec_len, D2D));
            dense_vector_negate(this->Rd1);
            axpby_cusparse(this->cusparseH, this->C, this->Rd1, 1.0, 1.0);
            this->cone_measures(this->Rd1, this->dual_block_min, this->dual_cone_coef);
            double dual_cone_coef_sum = 0.0;
            for (size_t b = 0; b < this->blk_info.size(); b++)
            {
                this->dual_block_min[b] *= this->Cscale;
                this->dual_cone_coef[b] *= this->Cscale;
                dual_cone_coef_sum += this->dual_cone_coef[b];
            }

            // print the final message
            printf(" ------------------------------------------------------------------------------\n\n");
            std::cout << final_msg << std::endl;
            printf(
                "\n primal infeasibility = %2.1e \n dual   infeasibility = %2.1e \n relative gap         = %2.1e",
                this->errRp, this->errRd, this->relgap);
            printf(
                "\n PSD violation X      = %2.1e \n PSD violation S      = %2.1e",
                err_PSD_X, err_PSD_S);
            printf(
                "\n certificate: <b,y> + R * %.6e is a lower bound on the optimal value if every block of an"
                "\n              optimal X has trace (or entries) bounded by R, see certified_lower_bound()",
                dual_cone_coef_sum);
            if (this->use_lobpcg)
                printf(
                    "\n\n LOBPCG calls = %d, certificate fallbacks to cuSOLVER = %d",
                    this->lobpcg_calls, this->lobpcg_fallbacks);
            if (this->jacobi_not_converged > 0)
                printf(
                    "\n batched Jacobi EVDs below their tolerance (warning only) = %d",
                    this->jacobi_not_converged);
            if (this->composite_fallbacks > 0)
                printf(
                    "\n non-finite composite projections replaced by cuSOLVER = %d",
                    this->composite_fallbacks);
            printf(
                "\n\n primal objective = %- 9.8e \n dual   objective = %- 9.8e",
                this->pobj, this->dobj);
            printf(
                "\n\n algorithm: %s, plain ADMM iterations = %d\n sigma policy: %s, initial sigma %.6e, final sigma %.6e, %d change(s)",
                admm_mode_name(switch_admm, max_iter).c_str(), this->admm_phase_iterations,
                sigma_policy_name(this->sigma_policy), this->sigma0, this->sig, this->sigma_changes);
            printf("\n phases: sGS-ADMM %d iterations (%.3f s, validation %.3f s) | plain ADMM %d iterations (%.3f s, validation %.3f s)",
                   this->sgs_phase_iterations, this->sgs_phase_time_s, this->sgs_phase_validation_s, this->admm_phase_iterations,
                   this->admm_phase_time_s, this->admm_phase_validation_s);
            printf("\n switch: switch_admm = %d, first plain-ADMM iteration %d", switch_admm, this->switch_iteration);
            if (this->switch_iteration > 1)
                printf(", internal KKT at the switch %.3e, sigma at the switch %.6e, last sGS tau %.4f; sigma changes sGS %d / ADMM %d",
                       this->kkt_at_switch, this->sigma_at_switch, this->tau_last_sgs, this->sigma_changes_sgs_phase,
                       this->sigma_changes_admm_phase);
            printf("\n stop reason: %s", this->stop_reason.c_str());
            if (validating)
            {
                printf("\n external validations: %zu (%.2f s); first validated iteration: %d; returned the %s",
                       this->validation_history.size(), this->validation_time_s, this->first_validated_iteration,
                       this->externally_validated_converged ? "validated iterate"
                                                            : (this->returned_best_external ? "best externally evaluated iterate" : "final iterate"));
                const ValidationRecord &b = this->best_validation;
                if (!this->validation_history.empty())
                    printf("\n best external iterate: iteration %d, KKT %.2e, X cone %.2e, dual cone %.2e, merit %.2e",
                           b.iter, b.result.kkt_full, b.result.X_cone_violation, b.result.dual_cone_violation, b.result.merit());
                if (this->callback_failures > 0)
                    printf("\n WARNING: %d validation callback(s) failed (see above)", this->callback_failures);
            }
            printf(
                "\n\n time per iteration = %2.4fs \n total time         = %2.1fs (init %2.1fs, iterations %2.1fs)",
                this->solve_time / std::max(1, this->info_iter_num), seconds, this->init_time, this->solve_time);
            printf("\n -------------------------------------------------------------------------------\n\n");

            cudaEventRecord(this->stop);
            cudaEventSynchronize(this->stop);
            cudaEventElapsedTime(&milliseconds, this->start, this->stop);
            this->total_time = milliseconds / 1000;
            break;
        }

        // hybrid mode: the first plain-ADMM iteration. Only bookkeeping: X, y, S, sigma, the factorization and every
        // workspace are carried over unchanged
        if (this->switch_iteration < 0 && iter >= std::max(this->switch_admm, 1))
        {
            this->switch_iteration = iter;
            this->sgs_phase_time_s = seconds_since_start();
            this->sgs_phase_validation_s = this->validation_time_s;
            this->kkt_at_switch = max(this->maxfeas, this->relgap);
            this->sigma_at_switch = this->sig;
            if (this->sigma_policy == SigmaPolicy::FIXED_SGS_LEGACY_ADMM && this->switch_admm > 1)
            {
                // the plain-ADMM sigma rules start as in a plain-ADMM solve: no primal/dual "wins" counted yet
                this->prim_win = 0;
                this->dual_win = 0;
            }
            if (this->switch_admm > 1)
                printf(" --- switching to plain ADMM at iteration %d (switch_admm = %d) after %d sGS iterations, %.3f s: internal KKT "
                       "%.3e, sigma %.6e, last sGS tau %.4f; X, y, S kept ---\n",
                       iter, this->switch_admm, this->sgs_phase_iterations, this->sgs_phase_time_s, this->kkt_at_switch,
                       this->sigma_at_switch, this->tau_last_sgs);
        }

        // check if the conditions to switch the projection method are met
        if (
            !this->switched_proj_method && (iter > switch_proj_max_iter || max(this->maxfeas, this->relgap) < switch_proj_tol) && iter > 1)
        {
            // switch the projection method
            if (this->final_proj_method != this->current_proj_method)
                std::cout << " ---------------- Switching projection method to " << get_projection_method_name(this->final_proj_method) << "------" << std::endl;
            this->current_proj_method = this->final_proj_method;
            this->switched_proj_method = true;
            this->switched_proj_method_iter = iter;
        }

        /*
            Step 1: Compute
                        r_s^{k+1/2} = 1/sigma b - A(X/sigma + S^k - C)
                                             and
                               y^{k+1/2} = (AA^T)^{-1} r_s^{k+1/2}
        */

        /* r_s^{k+1/2} = b/sigma - A(X/sigma + S - C) */
        // rhsy <-- -1.0 * A * SmC + 0.0 * rhsy
        SpMV_cusparse(this->cusparseH, this->A_csr, this->SmC, this->rhsy, -1.0, 0.0, this->SpMV_AX_buffer);
        // hence rhsy = - A S

        // rhsy <-- 1/sig * Rp + rhsy
        axpy_cublas(this->cublasH, this->Rp, this->rhsy, 1 / this->sig);
        // hence rhsy = 1/sig * Rp - A S

        /* y^{k+1/2} = (AA^T)^{-1} r_s^{k+1/2} */
        // y <-- linsys(rhsy)
        perform_permutation(this->rhsy_perm, this->rhsy, this->perm_inv);
        this->solve_AAt_perm();
        this->y_solves++;
        perform_permutation(this->y, this->y_perm, this->perm);
        // hence y = (AA^T)^{-1} r_s^{k+1/2}

        /*
            Step 2: Compute the optimization variables :

                    X_b^{k+1} = X^k + sigma(A^T y^{k+1/2} - C)
                                         and
                    S^{k+1} = 1/sigma (Pi(X_b^{k+1}) - X_b^{k+1})
        */

        /* Compute X^{k+1} */
        // Aty <-- 1.0 * At * y + 0.0 * Aty
        SpMV_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0, this->SpMV_Aty_buffer);
        // hence Aty = A^T y^{k+1/2}

        // Rd1 <-- Aty
        CHECK_CUDA(cudaMemcpy(this->Rd1.vals, this->Aty.vals, sizeof(double) * this->vec_len, D2D));
        // Rd1 <-- (-1.0) * C + 1.0 * Rd1
        axpby_cusparse(this->cusparseH, this->C, this->Rd1, -1.0, 1.0);
        // hence Rd1 = A^T y^{k+1/2} - C

        // Xinput <-- -(Rd1 + 1/sig * X)
        dense_vector_plus_dense_vector_mul_scalar(this->Xinput, this->Rd1, this->X, 1.0 / this->sig);
        dense_vector_negate(this->Xinput);

        /* Compute Pi(X^{k+1}) (this is long) */

        // first, we convert Xinput back to matrices
        vector_to_matrices(this->Xinput, this->large_mat, this->medium_mat, this->small_mat, this->map_B, this->map_M1, this->map_M2);
        CHECK_CUDA(cudaDeviceSynchronize());

        /*
            Step 3.1. PSD projection of the large matrices
            - if before the switch, use the initial projection method
            - if after the switch
                - if low rank, use LOBPCG
                - else, use cuSOLVER
        */
        bool is_composite =
            this->current_proj_method == ProjectionMethod::COMPOSITE_FP32 ||
            this->current_proj_method == ProjectionMethod::COMPOSITE_FP16 ||
            this->current_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED;
        bool is_lobpcg_phase = this->use_lobpcg && this->switched_proj_method;
        // every LOBPCG_REEVALUATE iterations of the LOBPCG phase, the full EVD is used to recompute the ranks
        bool reevaluate = is_lobpcg_phase && (iter - this->switched_proj_method_iter) % LOBPCG_REEVALUATE == 0;

        // when every large matrix takes the exact path in this iteration, the matrices of each size group are
        // multiplied by their eigenvectors with one batched GEMM (bit-identical to the code before the refactor:
        // a strided-batched GEMM with batch > 1 does not round like per-matrix GEMMs)
        bool any_lobpcg_mode = false;
        for (int idx = 0; idx < this->sizes.large_mat_num; idx++)
            any_lobpcg_mode = any_lobpcg_mode || this->large_mode[idx] != LargeProjectionMode::FULL;
        bool group_gemm = !is_composite && (!is_lobpcg_phase || reevaluate || !any_lobpcg_mode);

        int large_idx = 0; // flat index of the large matrix
        for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++)
        {
            for (int j = 0; j < this->sizes.large_mat_nums[i]; j++)
            {
                if (is_composite)
                {
                    if (!this->large_composite_project(i, j, large_idx))
                        this->large_full_eig_project(i, j, large_idx, false);
                }
                else if (!is_lobpcg_phase)
                    this->large_full_eig_project(i, j, large_idx, false, !group_gemm);
                else if (reevaluate)
                    this->large_full_eig_project(i, j, large_idx, true, !group_gemm);
                else if (this->large_mode[large_idx] == LargeProjectionMode::FULL)
                    this->large_full_eig_project(i, j, large_idx, false, !group_gemm);
                else if (!this->large_lobpcg_project(i, j, large_idx))
                    this->large_full_eig_project(i, j, large_idx, true);

                large_idx++;
            }
            if (group_gemm)
                dense_matrix_mul_trans_batch(
                    this->cublasH,
                    this->large_mat_P, this->large_mat_tmp, this->large_mat,
                    this->sizes.large_mat_sizes[i], this->sizes.large_mat_nums[i],
                    this->sizes.large_mat_offset(i, 0));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        if (reevaluate && this->sizes.large_mat_num > 0)
        {
            int number_low_rank_matrices = 0;
            for (int idx = 0; idx < this->sizes.large_mat_num; idx++)
                if (this->large_mode[idx] != LargeProjectionMode::FULL)
                    number_low_rank_matrices++;
            std::cout << " ------------------ Number of low rank matrices = " << std::setw(3) << number_low_rank_matrices << " / " << std::setw(3) << this->sizes.large_mat_num << " ------------------" << std::endl;
            for (int idx = 0; idx < this->sizes.large_mat_num; idx++)
                std::cout << " Ranks (matrix " << idx << "):   +" << cpu_positive_ranks.vals[idx] << "  -" << cpu_negative_ranks.vals[idx] << std::endl;
        }

        /* Step 3.2. Projection of the medium matrices */
        // one batched EVD per size group with several matrices, one cuSOLVER EVD per matrix on the streams otherwise
        int all_counter = 0; // serves as an info offset
        for (int i = 0; i < this->sizes.medium_mat_sizes.size(); i++)
        {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
            if (this->medium_group_batched[i])
            {
                const int n = this->sizes.medium_mat_sizes[i];
                CHECK_CUSOLVER(cusolverDnXsyevBatched(
                    this->cusolverH_eig_medium_batch.cusolver_dn_handle, eig_param_single.param,
                    eig_param_single.jobz, eig_param_single.uplo,
                    n, CUDA_R_64F, this->medium_mat.vals + this->sizes.medium_mat_offset(i, 0), n,
                    CUDA_R_64F, this->medium_W.vals + this->sizes.medium_W_offset(i, 0), CUDA_R_64F,
                    this->medium_batch_buffer.vals, this->medium_batch_buffer_size,
                    this->medium_batch_cpu_buffer.vals, this->medium_batch_cpu_buffer_size,
                    this->medium_info.vals + all_counter, this->sizes.medium_mat_nums[i]));
                all_counter += this->sizes.medium_mat_nums[i];
                continue;
            }
#endif
            for (int j = 0; j < this->sizes.medium_mat_nums[i]; j++)
            {
                int n = this->sizes.medium_mat_sizes[i];

                // compute the EVD using cuSOLVER
                single_eig_cusolver(
                    this->cusolverH_eig_medium_arr[all_counter % this->eig_stream_num_per_gpu], eig_param_single,
                    this->medium_mat, this->medium_W,
                    this->eig_medium_buffer, this->cpu_eig_medium_buffer, this->medium_info,
                    this->sizes.medium_mat_sizes[i],
                    this->eig_medium_buffer_size[i], this->cpu_eig_medium_buffer_size[i],
                    this->sizes.medium_mat_offset(i, j), this->sizes.medium_W_offset(i, j),
                    this->sizes.medium_buffer_offset(i, j, this->eig_medium_buffer_size),
                    this->sizes.medium_cpu_buffer_offset(i, j, this->cpu_eig_medium_buffer_size),
                    all_counter);

                all_counter++;
            }
        }

        // synchronize the streams used for the eigenvalue decomposition of medium matrices
        for (int i = 0; i < this->eig_stream_num_per_gpu; i++)
        {
            CHECK_CUDA(cudaStreamSynchronize(this->eig_medium_stream_arr[i].stream));
        }
        check_eig_info(this->medium_info.vals, this->sizes.medium_mat_num, "cusolverDnXsyevd (medium matrix)");

        if (this->sizes.medium_mat_num > 0)
        {
            max_dense_vector_zero(this->medium_W);
        }

        // multiply the medium matrices by their eigenvalues
        for (int i = 0; i < this->sizes.medium_mat_sizes.size(); i++)
        {
            dense_matrix_mul_diag_batch(
                this->medium_mat_tmp, this->medium_mat, this->medium_W,
                this->sizes.medium_mat_sizes[i], this->sizes.medium_mat_nums[i],
                this->sizes.medium_mat_offset(i, 0), this->sizes.medium_W_offset(i, 0));
        }

        // multiply the medium matrices by their eigenvectors
        for (int i = 0; i < this->sizes.medium_mat_sizes.size(); i++)
        {
            dense_matrix_mul_trans_batch(
                this->cublasH,
                this->medium_mat_P, this->medium_mat_tmp, this->medium_mat,
                this->sizes.medium_mat_sizes[i], this->sizes.medium_mat_nums[i],
                this->sizes.medium_mat_offset(i, 0));
        }

        /* Step 3.3. Projection of the small matrices */
        // always project with batched cuSOLVER
        int info_offset = 0;
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++)
        {
            batch_eig_cusolver(
                this->cusolverH_eig_small, this->eig_param_batch,
                this->small_mat, this->small_W,
                this->eig_small_buffer, this->small_info,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->eig_small_buffer_size[i],
                this->sizes.small_mat_offset(i), this->sizes.small_W_offset(i),
                this->sizes.small_buffer_offset(i, this->eig_small_buffer_size),
                0, // buffer_host_offset (unused by the batched Jacobi EVD)
                info_offset);
            info_offset += this->sizes.small_mat_nums[i];
        }
        info_offset = 0;
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++)
        {
            int not_converged = check_eig_info(
                this->small_info.vals + info_offset, this->sizes.small_mat_nums[i],
                "cusolverDnDsyevjBatched (small matrix)", this->sizes.small_mat_sizes[i]);
            if (not_converged > 0 && this->jacobi_not_converged == 0)
                std::cout << " WARNING: batched Jacobi EVD did not reach its tolerance for " << not_converged
                          << " matrices of size " << this->sizes.small_mat_sizes[i]
                          << " (results are still accurate to machine precision; reported once per solve)" << std::endl;
            this->jacobi_not_converged += not_converged;
            info_offset += this->sizes.small_mat_nums[i];
        }

        if (this->sizes.small_mat_num > 0)
        {
            max_dense_vector_zero(this->small_W);
        }

        // multiply the small matrices by their eigenvalues
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++)
        {
            dense_matrix_mul_diag_batch(
                this->small_mat_tmp, this->small_mat, this->small_W,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->sizes.small_mat_offset(i), this->sizes.small_W_offset(i));
        }

        // multiply the small matrices by their eigenvectors
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++)
        {
            dense_matrix_mul_trans_batch(
                this->cublasH,
                this->small_mat_P, this->small_mat_tmp, this->small_mat,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->sizes.small_mat_offset(i));
        }

        /* Step 3.4. Compute S */

        // convert the matrices back to vectorized format
        // at the same time, free variables and nonnegative variables are projected
        matrices_to_vector(this->S, this->Xinput, this->large_mat_P, this->medium_mat_P, this->small_mat_P, this->map_B, this->map_M1, this->map_M2);

        /*
            Step 3: Compute:
                        r_s^{k+1} = 1/sigma b - A(X^k/sigma + S^{k+1} - C)
                                              and
                                y^{k+1} = (AA^T)^{-1} r_s^{k+1}
        */

        /* Compute r_s^{k+1} */

        // SmC <-- S
        CHECK_CUDA(cudaMemcpy(this->SmC.vals, this->S.vals, sizeof(double) * this->vec_len, D2D));
        // SmC <-- -1.0 * C + 1.0 * SmC
        axpby_cusparse(this->cusparseH, this->C, this->SmC, -1.0, 1.0);
        // hence SmC = S^{k+1} - C

        /* Compute y^{k+1} */
        // If the number of iterations goes large but sGS-ADMM still fail to converge,
        // switch to ordinary ADMM
        if (iter == this->switch_admm)
        {
            // (historical: only the sGS sigma rule reads sig_update_stage_2 and the plain-ADMM rule overwrites sigscale
            // before using it, so these writes do not change the iteration)
            this->sig_update_stage_2 = this->sig_update_stage_2 / 2;
            this->sigscale = this->sigscale * 1.23;
            this->sgs_KKT = max(this->maxfeas, this->relgap);
        }

        // when before the switch, perform the special sGS-ADMM step
        if (iter >= this->switch_admm)
            this->admm_phase_iterations++;
        else
            this->sgs_phase_iterations++;
        if (iter < this->switch_admm)
        {
#ifdef CUADMM_PLAIN_ADMM_ONLY
            throw std::logic_error("sGS second y-update reached in a plain-ADMM-only build (iteration " + std::to_string(iter) + ")");
#endif
            // rhsy <-- -1.0 * A * SmC + 0.0 * rhsy
            SpMV_cusparse(this->cusparseH, this->A_csr, this->SmC, this->rhsy, -1.0, 0.0, this->SpMV_AX_buffer);
            // hence rhsy = - A(S - C)

            // rhsy <-- 1/sig * Rp + rhsy
            axpy_cublas(this->cublasH, this->Rp, this->rhsy, 1 / this->sig);
            // hence rhsy = 1/sigma Rp - A(S - C) = 1/sigma (b - A(X^k)) - A(S - C)
            // hence rhsy = 1/sigma b - A(X^k /sigma + S^{k+1} - C)

            // y <-- linsys(rhsy)
            perform_permutation(this->rhsy_perm, this->rhsy, this->perm_inv);
            this->solve_AAt_perm();
            this->y_solves++;
            perform_permutation(this->y, this->y_perm, this->perm);
            // hence y = (AA^T)^{-1} r_s^{k+1}

            // Aty <-- 1.0 * At * y + 0.0 * Aty
            SpMV_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0, this->SpMV_Aty_buffer);
            // hence Aty = A^T y^{k+1}

            // Rd1 <-- Aty
            CHECK_CUDA(cudaMemcpy(this->Rd1.vals, this->Aty.vals, sizeof(double) * this->vec_len, D2D));
            // Rd1 <-- (-1.0) * C + 1.0 * Rd1
            axpby_cusparse(this->cusparseH, this->C, this->Rd1, -1.0, 1.0);
            // hence Rd1 = A^T y^{k+1} - C
        }

        /* Step 4: Compute X^{k+1} = X^k + tau * sigma (S^{k+1} + A^T y^{k+1} - C) */
        // Rd <-- 1.0 * Rd1 + 1.0 * S
        dense_vector_add_dense_vector(this->Rd, this->Rd1, this->S, 1.0, 1.0);
        // hence Rd = Rd1 + S = A^T y^{k+1} - C + S

        // update tau
        if (iter < this->switch_admm)
        {
            this->tau = 1.95;
        }
        else
        {
            this->tau = 1.618; // (1 + sqrt(5)) / 2
        }
        if (this->errRd < stop_tol)
        {
            this->tau = max(1.618, this->tau / 1.1);
        }

        // X <-- X + (tau * sig) * Rd
        dense_vector_add_dense_vector(this->X, this->Rd, 1.0, this->tau * this->sig);
        // hence X = X^k + (tau * sig) * (A^T y^{k+1} - C + S)

        /* Step "5": Compute KKT residuals, update parameters */

        // Rp <-- -1.0 * A * X + 0.0 * Rp
        SpMV_cusparse(this->cusparseH, this->A_csr, this->X, this->Rp, -1.0, 0.0, this->SpMV_AX_buffer);
        // hence Rp = - A X

        // Rp <-- 1.0 * b + 1.0 * Rp
        axpby_cusparse(this->cusparseH, this->b, this->Rp, 1.0, 1.0);
        // hence Rp = b - A X

        /* Update errors and compute residuals */
        // compute primal error and objective
        dense_vector_mul_dense_vector_mul_scalar(this->Rporg, this->normA, this->Rp, this->bscale);
        this->errRp = this->Rporg.get_norm(this->cublasH) / this->norm_borg; // scale it
        this->pobj = SparseVV_cusparse(this->cusparseH, this->C, this->X, this->SpVV_CtX_buffer) * this->objscale;

        // compute dual error and objective
        dense_vector_mul_scalar(this->Rdorg, this->Rd, this->Cscale);
        this->errRd = this->Rdorg.get_norm(this->cublasH) / this->norm_Corg; // scale it
        this->dobj = SparseVV_cusparse(this->cusparseH, this->b, this->y, this->SpVV_bty_buffer) * this->objscale;

        // compute maximum feasibility violation and relative gap
        this->maxfeas = max(this->errRp, this->errRd);
        this->relgap = abs(this->pobj - this->dobj) / (1 + abs(this->pobj) + abs(this->dobj));
        if (validating && (!std::isfinite(this->errRp) || !std::isfinite(this->errRd) || !std::isfinite(this->relgap)))
            nonfinite = true; // Step 0 stops and returns the best externally evaluated iterate
        else if (!std::isfinite(this->errRp) || !std::isfinite(this->errRd) || !std::isfinite(this->relgap))
            throw std::runtime_error(
                "non-finite KKT residuals at iteration " + std::to_string(iter) +
                ": errRp = " + std::to_string(this->errRp) + ", errRd = " + std::to_string(this->errRd) +
                ", pobj = " + std::to_string(this->pobj) + ", dobj = " + std::to_string(this->dobj));

        // (legacy stopping only) after the switch to standard ADMM, save the best iterate so far; the residuals
        // above describe exactly the current (X, y, S), and the copies are issued on the
        // legacy default stream so that they are ordered with the kernels updating X, y, S
        if (!validating && iter >= this->switch_admm && max(this->maxfeas, this->relgap) < this->best_KKT)
        {
            this->best_KKT = max(this->maxfeas, this->relgap);
            this->best_errRp = this->errRp;
            this->best_errRd = this->errRd;
            this->best_relgap = this->relgap;
            this->best_pobj = this->pobj;
            this->best_dobj = this->dobj;
            CHECK_CUDA(cudaMemcpyAsync(this->X_best.vals, this->X.vals, sizeof(double) * this->vec_len, D2D, (cudaStream_t)0));
            CHECK_CUDA(cudaMemcpyAsync(this->y_best.vals, this->y.vals, sizeof(double) * this->con_num, D2D, (cudaStream_t)0));
            CHECK_CUDA(cudaMemcpyAsync(this->S_best.vals, this->S.vals, sizeof(double) * this->vec_len, D2D, (cudaStream_t)0));
        }

        // check whether primal or dual "wins" to schedule sigma update
        this->feasratio = this->ratioconst * this->errRp / this->errRd;
        if (this->feasratio < 1)
            this->prim_win += 1;
        else
            this->dual_win += 1;

        /* Update sigma (legacy adaptive rules only; with SigmaPolicy::FIXED sigma_k = sigma_0 for every k) */
        const double sig_before = this->sig;
        // every change is logged with the quantities the rule used (the counters before the rule resets them)
        const int prim_win_before = this->prim_win, dual_win_before = this->dual_win;
        auto log_sigma = [&](double old_sigma, const char *rule)
        {
            if (this->sig != old_sigma)
                this->sigma_log.push_back({iter, old_sigma, this->sig, rule, this->errRp, this->errRd, this->relgap, this->feasratio,
                                           prim_win_before, dual_win_before});
        };
        if (this->sigma_policy == SigmaPolicy::FIXED ||
            (this->sigma_policy == SigmaPolicy::FIXED_SGS_LEGACY_ADMM && iter < this->switch_admm))
        {
            // no update
        }
        else if (iter < this->switch_admm)
        {
            // sGS-ADMM update rule
            if (
                ((iter <= this->sig_update_threshold) && ((iter % this->sig_update_stage_1) == 1)) ||
                ((iter > this->sig_update_threshold) && ((iter % this->sig_update_stage_2) == 1)))
            {
                if (this->prim_win > 1.2 * this->dual_win)
                {
                    this->prim_win = 0;
                    this->sig = min(this->sigmax, this->sig * this->sigscale);
                }
                else if (this->dual_win > 1.2 * this->prim_win)
                {
                    this->dual_win = 0;
                    this->sig = max(this->sigmin, this->sig / this->sigscale);
                }
            }
            log_sigma(sig_before, "sgs_rule");
        }
        else
        {
            // this->sig = 1e2;

            // standard ADMM update rule; k counts the iterations of its schedule: the global iteration (LEGACY_ADAPTIVE,
            // as in the public code) or the plain-ADMM iterations since the switch (FIXED_SGS_LEGACY_ADMM, as in a
            // plain-ADMM solve started at the switch)
            const int k = (this->sigma_policy == SigmaPolicy::FIXED_SGS_LEGACY_ADMM && this->switch_admm > 1)
                              ? iter - this->switch_admm + 1
                              : iter;
            if (
                (k <= 200 && (k % 10) == 1) ||
                (k > 200 && k <= 1000 && (k % 25) == 1) ||
                (k > 1000 && k <= 5000 && (k % 50) == 1) ||
                (k > 5000 && k <= 10000 && (k % 200) == 1) ||
                (k > 10000 && (k % 1000) == 1))
            {
                this->sigscale = std::max(2.0 * std::exp(-k / 50000.0), 2.0);
                // std::printf("Current sigscale = %4.2f \n", this->sigscale);

                if (this->prim_win > 1.35 * this->dual_win)
                {
                    this->prim_win = 0;
                    this->sig = min(this->sigmax, this->sig * this->sigscale);
                }
                else if (this->dual_win > 1.35 * this->prim_win)
                {
                    this->dual_win = 0;
                    this->sig = max(this->sigmin, this->sig / this->sigscale);
                }
            }
            log_sigma(sig_before, "admm_schedule");

            // use monitor1 for the chasing phenomenon
            if (k > 5000 && k % monitor1.update_interval == 0)
            {
                const double sig_mon = this->sig;
                monitor1.push(this->pobj, this->dobj, this->errRp, this->errRd, this->relgap);
                if (monitor1.if_full())
                    this->sig = monitor1.chase_update_sig(this->sig);
                log_sigma(sig_mon, "monitor1");
            }
        }
        if (this->sig != sig_before)
        {
            this->sigma_changes++;
            (iter < this->switch_admm ? this->sigma_changes_sgs_phase : this->sigma_changes_admm_phase)++;
        }
        if ((this->sigma_policy == SigmaPolicy::FIXED ||
             (this->sigma_policy == SigmaPolicy::FIXED_SGS_LEGACY_ADMM && iter < this->switch_admm)) &&
            this->sig != this->sigma0)
            throw std::logic_error("fixed sigma policy violated at iteration " + std::to_string(iter) +
                                   ": sigma = " + std::to_string(this->sig) + ", sigma_0 = " + std::to_string(this->sigma0));

        /* Add info */
        this->info_pobj_arr.push_back(this->pobj);
        this->info_dobj_arr.push_back(this->dobj);
        this->info_errRp_arr.push_back(this->errRp);
        this->info_errRd_arr.push_back(this->errRd);
        this->info_relgap_arr.push_back(this->relgap);
        this->info_sig_arr.push_back(this->sig);
        this->info_tau_arr.push_back(this->tau);
        if (iter < this->switch_admm)
            this->tau_last_sgs = this->tau;
        this->info_bscale_arr.push_back(this->bscale);
        this->info_Cscale_arr.push_back(this->Cscale);
        this->info_time_arr.push_back(seconds_since_start());
        this->info_iter_num++;
    }

    // recover the original solution by unscaling
    dense_vector_mul_scalar(this->X, this->bscale);
    dense_vector_div_dense_vector_mul_scalar(this->y, this->normA, this->Cscale);
    dense_vector_mul_scalar(this->S, this->Cscale);

    return;
}

ValidationRecord SDPSolver::validate_current_iterate(int iter, double time_s, const std::string &trigger, bool checkpoint)
{
    const auto t0 = std::chrono::steady_clock::now();
    ValidationRecord rec;
    rec.iter = iter;
    rec.time_s = time_s;
    rec.sigma = this->sig;
    rec.internal_kkt = max(this->maxfeas, this->relgap);
    rec.internal_errRp = this->errRp;
    rec.internal_errRd = this->errRd;
    rec.internal_relgap = this->relgap;
    rec.internal_pobj = this->pobj;
    rec.internal_dobj = this->dobj;
    rec.trigger = trigger;
    rec.checkpoint = checkpoint;

    // snapshot of the scaled iterate, unscaled exactly like at the end of solve()
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<double> X(this->vec_len), y(this->con_num), S(this->vec_len), nA(this->con_num);
    CHECK_CUDA(cudaMemcpy(X.data(), this->X.vals, sizeof(double) * this->vec_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y.data(), this->y.vals, sizeof(double) * this->con_num, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(S.data(), this->S.vals, sizeof(double) * this->vec_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(nA.data(), this->normA.vals, sizeof(double) * this->con_num, cudaMemcpyDeviceToHost));
    for (int i = 0; i < this->vec_len; i++)
    {
        X[i] = X[i] * this->bscale;
        S[i] = S[i] * this->Cscale;
    }
    for (int j = 0; j < this->con_num; j++)
        y[j] = y[j] / nA[j] * this->Cscale;

    rec.result = this->external_validator->evaluate(X.data(), y.data(), S.data(), this->validation.tol, this->validation.threads);

    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
        this->peak_gpu_mem_bytes = std::max(this->peak_gpu_mem_bytes, total_mem - free_mem);

    // best externally evaluated iterate (scaled copies on the device, restored at termination if needed)
    if (this->validation_history.empty() || rec.result.better_than(this->best_validation.result))
    {
        CHECK_CUDA(cudaMemcpy(this->X_best.vals, this->X.vals, sizeof(double) * this->vec_len, D2D));
        CHECK_CUDA(cudaMemcpy(this->y_best.vals, this->y.vals, sizeof(double) * this->con_num, D2D));
        CHECK_CUDA(cudaMemcpy(this->S_best.vals, this->S.vals, sizeof(double) * this->vec_len, D2D));
        this->best_validation = rec; // validation_s is set below
    }
    rec.validation_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (this->best_validation.iter == rec.iter)
        this->best_validation.validation_s = rec.validation_s;
    this->validation_time_s += rec.validation_s;
    this->validation_history.push_back(rec);

    const ValidationResult &r = rec.result;
    printf(" [validation] it %7d %-17s | ext KKT %.2e (p %.1e d %.1e g %.1e) | X cone %.1e | dual cone %.1e | sigma %.3e | %s (%.0f ms)\n",
           iter, (trigger + (checkpoint && trigger != "checkpoint" ? "+checkpoint" : "")).c_str(), r.kkt_full, r.primal_res,
           r.dual_res, r.relgap, r.X_cone_violation, r.dual_cone_violation, rec.sigma,
           r.validated ? "VALIDATED" : ("failed: " + r.failed).c_str(), 1e3 * rec.validation_s);
    if (this->validation.on_validation)
    {
        // timed separately; a failure (e.g. a checkpoint that cannot be written) is reported, not fatal
        const auto t1 = std::chrono::steady_clock::now();
        try
        {
            this->validation.on_validation(rec, X, y, S);
        }
        catch (const std::exception &e)
        {
            this->callback_failures++;
            printf(" WARNING: validation callback failed at iteration %d: %s\n", iter, e.what());
        }
        rec.callback_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        this->callback_time_s += rec.callback_s;
        this->validation_history.back().callback_s = rec.callback_s;
    }
    return rec;
}

void SDPSolver::solve_AAt_perm()
{
    if (this->AAt_dense_gpu)
    {
        // y_perm <- L^{-T} D^{-1} L^{-1} rhsy_perm (cuBLAS and the kernels run on the default stream)
        CHECK_CUDA(cudaMemcpy(this->y_perm.vals, this->rhsy_perm.vals, sizeof(double) * this->con_num, D2D));
        CHECK_CUBLAS(cublasDtrsv(
            this->cublasH.cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
            this->con_num, this->AAt_L.vals, this->con_num, this->y_perm.vals, 1));
        dense_vector_div_dense_vector_mul_scalar(this->y_perm, this->AAt_D, 1.0);
        CHECK_CUBLAS(cublasDtrsv(
            this->cublasH.cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
            this->con_num, this->AAt_L.vals, this->con_num, this->y_perm.vals, 1));
        return;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpyAsync(
        this->cpu_AAt_solver.chol_dn_rhs->x, this->rhsy_perm.vals,
        sizeof(double) * this->con_num, D2H, this->stream_flex[0].stream));
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[0].stream));
    this->cpu_AAt_solver.solve();
    CHECK_CUDA(cudaMemcpyAsync(
        this->y_perm.vals, this->cpu_AAt_solver.chol_dn_res->x,
        sizeof(double) * this->con_num, H2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[0].stream));
}

SDPSolver::~SDPSolver()
{
    // the events live as long as the solver, so that solve() can be called again and nothing leaks if it throws
    if (this->start != nullptr)
        CHECK_CUDA_NOTHROW(cudaEventDestroy(this->start));
    if (this->stop != nullptr)
        CHECK_CUDA_NOTHROW(cudaEventDestroy(this->stop));
}