/*

    solver.cu

    Main solver, works for any sizes of matrices.
    Uses the sGS-ADMM algorithm to solve an SDP problem.

*/

#include "cuadmm_legacy/solver.h"
#include "cuadmm_legacy/block_structure.h"
#include "cuadmm_legacy/kernels.h"
#include "cuadmm_legacy/rank.h"
#include "cuadmm_legacy/matrix_sizes.h"
#include "cuadmm_legacy/monitors.h"

#include "admm/admm_solver.h"
#include "report/observers.h"
#include "termination/best_iterate_tracker.h"
#include "termination/stopping_rule.h"

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

SigmaPolicyParameters SDPSolver::sigma_policy_parameters() const
{
    SigmaPolicyParameters p;
    switch (this->sigma_policy)
    {
    case SigmaPolicy::FIXED:
        p.factor = 1.0;
        p.sigma_min = p.sigma_max = this->sig; // sigma_k = sigma_0
        p.guard = false;
        p.monitor1 = false;
        p.sgs_rule = "";
        break;
    case SigmaPolicy::LEGACY_ADAPTIVE: // 9bb4ba5
        p.factor = 2.0;
        p.sigma_min = 1e-6;
        p.sigma_max = 1e6;
        p.guard = false;
        p.monitor1 = true;
        p.sgs_rule = "legacy_sgs";
        break;
    case SigmaPolicy::RESIDUAL_BALANCING: // 2d8a755 with the factor as an option
        p.factor = this->penalty_factor;
        p.sigma_min = 1e-3;
        p.sigma_max = 1e3;
        p.guard = true;
        p.monitor1 = false;
        p.sgs_rule = "legacy_sgs";
        break;
    }
    if (this->sigma_policy != SigmaPolicy::FIXED)
    {
        if (this->penalty_sigma_min > 0)
            p.sigma_min = this->penalty_sigma_min;
        if (this->penalty_sigma_max > 0)
            p.sigma_max = this->penalty_sigma_max;
        if (!(p.sigma_min > 0) || !(p.sigma_max >= p.sigma_min) || !(p.factor > 0))
            throw std::invalid_argument("sigma policy parameters: need 0 < sigma_min <= sigma_max and factor > 0 (got " +
                                        std::to_string(p.sigma_min) + ", " + std::to_string(p.sigma_max) + ", " +
                                        std::to_string(p.factor) + ")");
    }
    p.win_ratio = 1.35;
    p.residual_weight = this->ratioconst;
    return p;
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

    // keep a copy of the input to restore it if the polynomial filter overflows or fails the norm test
    CHECK_CUDA(cudaMemcpy(A_copy, A, sizeof(double) * nn, D2D));
    double norm_in;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_copy, 1, &norm_in));

    // scale the spectrum into [-1, 1]: the Lanczos estimate 'up' is not a certified upper bound on
    // ||A||_2, and the filter overflows if ||A||/scale > ~1.02 (FP16), hence the margin
    double lo, up;
    approximate_two_norm(cublasH, this->cusolverH_composite_proj.cusolver_dn_handle, A, n, &lo, &up);
    const double margin = COMPOSITE_FP16_SCALE_MARGIN;
    const double scale = up > 0.0 ? margin * up : 1.0;
    const double inv_scale = 1.0 / scale;
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &inv_scale, A, 1));

    composite_FP16(cublasH, A, n, this->float_proj_workspace.vals, this->half_proj_workspace.vals);

    // rescale the result back to the original scale
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &scale, A, 1));

    // reject a non-finite result (overflow of the filter, or non-finite input) and, B22, a result whose norm exceeds
    // that of the input beyond the slack: the projection is nonexpansive with Pi(0) = 0, so ||P||_F <= ||Z||_F, and a
    // finite but wrong result from an underestimated scale fails this free necessary condition
    double norm;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A, 1, &norm));
    if (!std::isfinite(norm) || !std::isfinite(norm_in) || norm > (1.0 + COMPOSITE_NORM_SLACK) * norm_in)
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

    // B17: the medium-block EVDs are distributed over the streams with counter % eig_stream_num_per_gpu
    if (eig_stream_num_per_gpu < 1)
        throw std::invalid_argument("init(): eig_stream_num_per_gpu must be at least 1 (got " + std::to_string(eig_stream_num_per_gpu) + ")");
    this->eig_stream_num_per_gpu = eig_stream_num_per_gpu;

    this->initial_proj_method = initial_proj_method;
    this->final_proj_method = final_proj_method;
    this->current_proj_method = initial_proj_method;
    this->switched_proj_method = false;

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

    // compute the norm of A (get_normA rewrites At_csc.vals on the legacy default stream)
    this->normA.allocate(GPU0, con_num);
    get_normA(this->At_csc, this->normA);

    /* convert the At matrix from CSC to CSR format */
    this->CSCtoCSR_At2A_buffer_size = CSC_to_CSR_get_buffersize_cusparse(this->cusparseH, this->At_csc, this->At_csr);
    this->CSCtoCSR_At2A_buffer.allocate(GPU0, CSCtoCSR_At2A_buffer_size, true);
    CSC_to_CSR_cusparse(this->cusparseH, this->At_csc, this->At_csr, this->CSCtoCSR_At2A_buffer);
    // B2: the copies of At_csc into A_csr run on the non-blocking flex streams and must see the normalized values:
    // an event on the default stream orders them after get_normA (and after the CSC-to-CSR conversion)
    cudaEvent_t normA_done;
    CHECK_CUDA(cudaEventCreateWithFlags(&normA_done, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(normA_done, (cudaStream_t)0));
    for (int stream_id = 0; stream_id < 3; stream_id++)
        CHECK_CUDA(cudaStreamWaitEvent(this->stream_flex[stream_id].stream, normA_done, 0));
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
    // B21: the exact EVD (the fallback of every projection method) has a size limit on CUDA 12.9 (plan O5)
    if (this->sizes.large_mat_num > 0 && this->sizes.max_large_mat_size > CUADMM_MAX_EVD_BLOCK_SIZE)
        throw std::invalid_argument("PSD block of size " + std::to_string(this->sizes.max_large_mat_size) +
                                    " exceeds the exact-EVD limit of " + std::to_string(CUADMM_MAX_EVD_BLOCK_SIZE) +
                                    " (cusolverDnXsyevd workspace queries fail above it on CUDA 12.9)");
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
    CHECK_CUDA(cudaEventDestroy(normA_done)); // every stream that waited on it has completed
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

    if (this->sizes.large_mat_sizes.size() > 0 &&
        (this->initial_proj_method == ProjectionMethod::COMPOSITE_FP16 || this->final_proj_method == ProjectionMethod::COMPOSITE_FP16))
    {
        // create a workspace for the composite projection
        int largest_size = *std::max_element(this->sizes.large_mat_sizes.begin(), this->sizes.large_mat_sizes.end());
        size_t nn = (size_t)largest_size * largest_size;
        size_t stride = nn % 4 == 0 ? nn : nn + (4 - nn % 4); // we need to ensure proper memory alignment

        this->float_proj_workspace.allocate_size_t(GPU0, 3 * stride);
        this->half_proj_workspace.allocate_size_t(GPU0, 3 * stride);

        // create a cuBLAS handle; B1: the math mode is set explicitly to the default (the FP16 filter calls
        // cublasGemmEx with its own compute type; the plan measured its output bitwise identical under both modes)
        this->cublasH_composite_proj.set_gpu_id(GPU0);
        this->cublasH_composite_proj.activate();
        CHECK_CUBLAS(cublasSetMathMode(this->cublasH_composite_proj.cublas_handle, CUBLAS_DEFAULT_MATH));

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
    // sigmin / sigmax: set by solve() from the sigma policy (sigma_policy_parameters())

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
    // Phase 4: the loop lives in cuadmm::admm::AdmmSolver on the components of the algorithm layer; this method keeps
    // the signature of main for the copied tests and builds the components from the engine's fields.
    using namespace cuadmm;
    this->stop_tol = stop_tol;
    this->sig_update_threshold = sig_update_threshold;
    this->sig_update_stage_1 = sig_update_stage_1;
    this->sig_update_stage_2 = sig_update_stage_2;
    this->switch_admm = switch_admm;
    this->sigscale = sigscale;
    const SigmaPolicyParameters params = this->sigma_policy_parameters(); // validates the bounds and the factor
    this->sigmin = params.sigma_min;
    this->sigmax = params.sigma_max;

    PenaltyOptions po;
    po.policy = this->sigma_policy == SigmaPolicy::FIXED              ? PenaltyPolicyKind::Fixed
                : this->sigma_policy == SigmaPolicy::LEGACY_ADAPTIVE ? PenaltyPolicyKind::LegacyAdaptive
                                                                     : PenaltyPolicyKind::ResidualBalancing;
    po.sigma0 = this->sig;
    po.factor = this->penalty_factor;
    po.sigma_min = this->penalty_sigma_min;
    po.sigma_max = this->penalty_sigma_max;
    po.sgs_update_threshold = sig_update_threshold;
    po.sgs_update_stage_1 = sig_update_stage_1;
    po.sgs_update_stage_2 = sig_update_stage_2;
    po.sgs_factor = sigscale;
    std::unique_ptr<admm::PenaltyPolicy> penalty = admm::make_penalty_policy(po);

    admm::AdmmState state(*this);
    termination::BestIterateTracker tracker(*this);
    std::unique_ptr<termination::StoppingRule> rule;
    if (this->validation.interval > 0)
        rule.reset(new termination::ValidatedRule(*this, stop_tol, this->time_limit));
    else
        rule.reset(new termination::InternalKktRule(stop_tol, this->time_limit));
    report::ObserverSet observers;
    observers.add(std::make_shared<report::ConsoleObserver>(1));
    observers.add(std::make_shared<report::LegacyInfoObserver>(*this));

    admm::RunConfig cfg;
    cfg.max_iter = max_iter;
    cfg.tol = stop_tol;
    cfg.switch_admm = switch_admm;
    cfg.tau_admm = 1.618;
    cfg.tau_sgs = 1.95;
    cfg.legacy_sgs_tau_reduction = this->legacy_sgs_tau_reduction;
    cfg.switch_proj_max_iter = switch_proj_max_iter;
    cfg.switch_proj_tol = switch_proj_tol;
    cfg.verbose = true;
    admm::AdmmSolver loop(state, *penalty, *rule, tracker, observers, cfg);
    loop.run(this->sig, !if_first);
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