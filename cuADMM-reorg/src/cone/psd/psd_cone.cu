/*

    cone/psd/psd_cone.cu

    Phase 5: the three tiers of main (small: batched Jacobi; medium: batched or streamed cuSOLVER EVD; large: exact
    EVD, LOBPCG or the FP16 composite filter with the warm-up schedule) run unchanged on the buffers of the ported
    engine; project_tiers() is AdmmState::project_psd_tiers() of Phase 4 (itself Step 3 of SDPSolver::solve() of
    main) moved here without a change, and measure() is the PSD part of SDPSolver::cone_measures() of main. Phase 6a
    replaces the body with the projector classes; Phase 6b the memory layout.

*/

#include "cone/psd/psd_cone.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "cone/psd/svec_pack.h"
#include "cuadmm_legacy/solver.h"

#ifndef CUADMM_NO_LAPACK
extern "C" void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a, const int *lda, double *w, double *work, const int *lwork,
                        int *iwork, const int *liwork, int *info);
#endif

namespace cuadmm
{
namespace cone
{

namespace
{
constexpr int kLobpcgReevaluate = 100; // LOBPCG_REEVALUATE of main (psd.lowrank.reevaluate, fixed until Phase 6a)

// Copies `count` cuSOLVER info values to the host and throws if an eigendecomposition failed (check_eig_info of main).
int check_eig_info(const int *d_info, const int count, const char *what, const int jacobi_n = -1)
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
        throw std::runtime_error(std::string(what) + " failed for matrix " + std::to_string(l) + ": info = " + std::to_string(info[l]));
    }
    return not_converged;
}

double psd_block_min_host(const double *v, int n, std::vector<double> &work, std::vector<int> &iwork)
{
    std::vector<double> A(static_cast<std::size_t>(n) * n, 0.0), w(n, 0.0);
    const double is2 = 1.0 / std::sqrt(2.0);
    for (int c = 0; c < n; c++)
        for (int r = 0; r <= c; r++)
        {
            double x = v[static_cast<long>(c) * (c + 1) / 2 + r];
            if (r != c)
                x *= is2;
            A[static_cast<std::size_t>(c) * n + r] = x;
            A[static_cast<std::size_t>(r) * n + c] = x;
        }
#ifdef CUADMM_NO_LAPACK
    (void)work;
    (void)iwork;
    throw std::runtime_error("the PSD host reference measure needs LAPACK (built with CUADMM_NO_LAPACK)");
#else
    const char jobz = 'N', uplo = 'U';
    int lwork = -1, liwork = -1, info = 0;
    double wq = 0;
    int iq = 0;
    dsyevd_(&jobz, &uplo, &n, A.data(), &n, w.data(), &wq, &lwork, &iq, &liwork, &info);
    if (info != 0)
        return std::numeric_limits<double>::quiet_NaN();
    lwork = static_cast<int>(wq);
    liwork = iq;
    work.assign(lwork, 0.0);
    iwork.assign(liwork, 0);
    dsyevd_(&jobz, &uplo, &n, A.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0)
        return std::numeric_limits<double>::quiet_NaN();
    return w[0];
#endif
}
const HostReferenceMeasure kPsdReference{&psd_block_min_host};
} // namespace

PsdCone::PsdCone(std::vector<BlockSlice> blocks, SDPSolver &engine) : blocks_(std::move(blocks)), engine_(engine)
{
    for (const BlockSlice &b : blocks_)
        dimension_ += b.length;
}

void PsdCone::project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, core::ExecContext &)
{
    SDPSolver &e = engine_;
    // the matrices of the PSD blocks (svec -> dense, 1/sqrt 2 off the diagonal), as Step 3 of main
    psd_unpack(z.data, e.large_mat.vals, e.medium_mat.vals, e.small_mat.vals, e.map_B.vals, e.map_M1.vals, e.map_M2.vals, e.vec_len);
    CHECK_CUDA(cudaDeviceSynchronize());
    project_tiers(iteration_);
    // back to the vector (dense -> svec, sqrt 2 off the diagonal); the other cones write their own entries
    psd_pack(s.data, e.large_mat_P.vals, e.medium_mat_P.vals, e.small_mat_P.vals, e.map_B.vals, e.map_M1.vals, e.map_M2.vals, e.vec_len);
}

void PsdCone::project_tiers(int iter)
{
    SDPSolver &e = engine_;
    /*
        Step 3.1. PSD projection of the large matrices
        - if before the switch, use the initial projection method
        - if after the switch: if low rank, use LOBPCG, else cuSOLVER
    */
    bool is_composite = e.current_proj_method == ProjectionMethod::COMPOSITE_FP16;
    bool is_lobpcg_phase = e.use_lobpcg && e.switched_proj_method;
    // every LOBPCG_REEVALUATE iterations of the LOBPCG phase, the full EVD is used to recompute the ranks
    bool reevaluate = is_lobpcg_phase && (iter - e.switched_proj_method_iter) % kLobpcgReevaluate == 0;

    // when every large matrix takes the exact path in this iteration, the matrices of each size group are
    // multiplied by their eigenvectors with one batched GEMM (bit-identical to the code before the refactor:
    // a strided-batched GEMM with batch > 1 does not round like per-matrix GEMMs)
    bool any_lobpcg_mode = false;
    for (int idx = 0; idx < e.sizes.large_mat_num; idx++)
        any_lobpcg_mode = any_lobpcg_mode || e.large_mode[idx] != LargeProjectionMode::FULL;
    bool group_gemm = !is_composite && (!is_lobpcg_phase || reevaluate || !any_lobpcg_mode);

    int large_idx = 0; // flat index of the large matrix
    for (int i = 0; i < (int)e.sizes.large_mat_sizes.size(); i++)
    {
        for (int j = 0; j < e.sizes.large_mat_nums[i]; j++)
        {
            if (is_composite)
            {
                if (!e.large_composite_project(i, j, large_idx))
                    e.large_full_eig_project(i, j, large_idx, false);
            }
            else if (!is_lobpcg_phase)
                e.large_full_eig_project(i, j, large_idx, false, !group_gemm);
            else if (reevaluate)
                e.large_full_eig_project(i, j, large_idx, true, !group_gemm);
            else if (e.large_mode[large_idx] == LargeProjectionMode::FULL)
                e.large_full_eig_project(i, j, large_idx, false, !group_gemm);
            else if (!e.large_lobpcg_project(i, j, large_idx))
                e.large_full_eig_project(i, j, large_idx, true);

            large_idx++;
        }
        if (group_gemm)
            dense_matrix_mul_trans_batch(
                e.cublasH,
                e.large_mat_P, e.large_mat_tmp, e.large_mat,
                e.sizes.large_mat_sizes[i], e.sizes.large_mat_nums[i],
                e.sizes.large_mat_offset(i, 0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (reevaluate && e.sizes.large_mat_num > 0)
    {
        int number_low_rank_matrices = 0;
        for (int idx = 0; idx < e.sizes.large_mat_num; idx++)
            if (e.large_mode[idx] != LargeProjectionMode::FULL)
                number_low_rank_matrices++;
        std::cout << " ------------------ Number of low rank matrices = " << std::setw(3) << number_low_rank_matrices << " / " << std::setw(3) << e.sizes.large_mat_num << " ------------------" << std::endl;
        for (int idx = 0; idx < e.sizes.large_mat_num; idx++)
            std::cout << " Ranks (matrix " << idx << "):   +" << e.cpu_positive_ranks.vals[idx] << "  -" << e.cpu_negative_ranks.vals[idx] << std::endl;
    }

    /* Step 3.2. Projection of the medium matrices */
    // one batched EVD per size group with several matrices, one cuSOLVER EVD per matrix on the streams otherwise
    int all_counter = 0; // serves as an info offset
    for (int i = 0; i < (int)e.sizes.medium_mat_sizes.size(); i++)
    {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
        if (e.medium_group_batched[i])
        {
            const int n = e.sizes.medium_mat_sizes[i];
            CHECK_CUSOLVER(cusolverDnXsyevBatched(
                e.cusolverH_eig_medium_batch.cusolver_dn_handle, e.eig_param_single.param,
                e.eig_param_single.jobz, e.eig_param_single.uplo,
                n, CUDA_R_64F, e.medium_mat.vals + e.sizes.medium_mat_offset(i, 0), n,
                CUDA_R_64F, e.medium_W.vals + e.sizes.medium_W_offset(i, 0), CUDA_R_64F,
                e.medium_batch_buffer.vals, e.medium_batch_buffer_size,
                e.medium_batch_cpu_buffer.vals, e.medium_batch_cpu_buffer_size,
                e.medium_info.vals + all_counter, e.sizes.medium_mat_nums[i]));
            all_counter += e.sizes.medium_mat_nums[i];
            continue;
        }
#endif
        for (int j = 0; j < e.sizes.medium_mat_nums[i]; j++)
        {
            // compute the EVD using cuSOLVER
            single_eig_cusolver(
                e.cusolverH_eig_medium_arr[all_counter % e.eig_stream_num_per_gpu], e.eig_param_single,
                e.medium_mat, e.medium_W,
                e.eig_medium_buffer, e.cpu_eig_medium_buffer, e.medium_info,
                e.sizes.medium_mat_sizes[i],
                e.eig_medium_buffer_size[i], e.cpu_eig_medium_buffer_size[i],
                e.sizes.medium_mat_offset(i, j), e.sizes.medium_W_offset(i, j),
                e.sizes.medium_buffer_offset(i, j, e.eig_medium_buffer_size),
                e.sizes.medium_cpu_buffer_offset(i, j, e.cpu_eig_medium_buffer_size),
                all_counter);
            all_counter++;
        }
    }
    // synchronize the streams used for the eigenvalue decomposition of medium matrices
    for (int i = 0; i < e.eig_stream_num_per_gpu; i++)
        CHECK_CUDA(cudaStreamSynchronize(e.eig_medium_stream_arr[i].stream));
    check_eig_info(e.medium_info.vals, e.sizes.medium_mat_num, "cusolverDnXsyevd (medium matrix)");

    if (e.sizes.medium_mat_num > 0)
        max_dense_vector_zero(e.medium_W);
    // multiply the medium matrices by their eigenvalues
    for (int i = 0; i < (int)e.sizes.medium_mat_sizes.size(); i++)
        dense_matrix_mul_diag_batch(
            e.medium_mat_tmp, e.medium_mat, e.medium_W,
            e.sizes.medium_mat_sizes[i], e.sizes.medium_mat_nums[i],
            e.sizes.medium_mat_offset(i, 0), e.sizes.medium_W_offset(i, 0));
    // multiply the medium matrices by their eigenvectors
    for (int i = 0; i < (int)e.sizes.medium_mat_sizes.size(); i++)
        dense_matrix_mul_trans_batch(
            e.cublasH,
            e.medium_mat_P, e.medium_mat_tmp, e.medium_mat,
            e.sizes.medium_mat_sizes[i], e.sizes.medium_mat_nums[i],
            e.sizes.medium_mat_offset(i, 0));

    /* Step 3.3. Projection of the small matrices: always with the batched cuSOLVER Jacobi EVD */
    int info_offset = 0;
    for (int i = 0; i < (int)e.sizes.small_mat_sizes.size(); i++)
    {
        batch_eig_cusolver(
            e.cusolverH_eig_small, e.eig_param_batch,
            e.small_mat, e.small_W,
            e.eig_small_buffer, e.small_info,
            e.sizes.small_mat_sizes[i], e.sizes.small_mat_nums[i],
            e.eig_small_buffer_size[i],
            e.sizes.small_mat_offset(i), e.sizes.small_W_offset(i),
            e.sizes.small_buffer_offset(i, e.eig_small_buffer_size),
            0, // buffer_host_offset (unused by the batched Jacobi EVD)
            info_offset);
        info_offset += e.sizes.small_mat_nums[i];
    }
    info_offset = 0;
    for (int i = 0; i < (int)e.sizes.small_mat_sizes.size(); i++)
    {
        int not_converged = check_eig_info(
            e.small_info.vals + info_offset, e.sizes.small_mat_nums[i],
            "cusolverDnDsyevjBatched (small matrix)", e.sizes.small_mat_sizes[i]);
        if (not_converged > 0 && e.jacobi_not_converged == 0)
            std::cout << " WARNING: batched Jacobi EVD did not reach its tolerance for " << not_converged
                      << " matrices of size " << e.sizes.small_mat_sizes[i]
                      << " (results are still accurate to machine precision; reported once per solve)" << std::endl;
        e.jacobi_not_converged += not_converged;
        info_offset += e.sizes.small_mat_nums[i];
    }
    if (e.sizes.small_mat_num > 0)
        max_dense_vector_zero(e.small_W);
    for (int i = 0; i < (int)e.sizes.small_mat_sizes.size(); i++)
        dense_matrix_mul_diag_batch(
            e.small_mat_tmp, e.small_mat, e.small_W,
            e.sizes.small_mat_sizes[i], e.sizes.small_mat_nums[i],
            e.sizes.small_mat_offset(i), e.sizes.small_W_offset(i));
    for (int i = 0; i < (int)e.sizes.small_mat_sizes.size(); i++)
        dense_matrix_mul_trans_batch(
            e.cublasH,
            e.small_mat_P, e.small_mat_tmp, e.small_mat,
            e.sizes.small_mat_sizes[i], e.sizes.small_mat_nums[i],
            e.sizes.small_mat_offset(i));
}

ConeMeasures PsdCone::measure(core::DeviceSpan<const double> v, Side, core::ExecContext &)
{
    SDPSolver &e = engine_;
    // matrices of the PSD blocks
    psd_unpack(v.data, e.large_mat.vals, e.medium_mat.vals, e.small_mat.vals, e.map_B.vals, e.map_M1.vals, e.map_M2.vals, e.vec_len);
    CHECK_CUDA(cudaDeviceSynchronize());

    // eigenvalues of every PSD block, with the machinery of the projection (the eigenvectors overwrite the matrices)
    int counter = 0;
    for (int i = 0; i < e.sizes.large_mat_sizes.size(); i++)
    {
        for (int j = 0; j < e.sizes.large_mat_nums[i]; j++)
        {
            single_eig_cusolver(
                e.cusolverH_eig_large, e.eig_param_single,
                e.large_mat, e.large_W,
                e.eig_large_buffer, e.cpu_eig_large_buffer, e.large_info,
                e.sizes.large_mat_sizes[i],
                e.eig_large_buffer_size[i], e.cpu_eig_large_buffer_size[i],
                e.sizes.large_mat_offset(i, j), e.sizes.large_W_offset(i, j),
                e.sizes.large_buffer_offset(i, j, e.eig_large_buffer_size),
                e.sizes.large_cpu_buffer_offset(i, j, e.cpu_eig_large_buffer_size),
                counter);
            check_eig_info(e.large_info.vals + counter, 1, "cusolverDnXsyevd (cone measure)");
            counter++;
        }
    }
    counter = 0;
    for (int i = 0; i < e.sizes.medium_mat_sizes.size(); i++)
    {
        for (int j = 0; j < e.sizes.medium_mat_nums[i]; j++)
        {
            single_eig_cusolver(
                e.cusolverH_eig_medium_arr[counter % e.eig_stream_num_per_gpu], e.eig_param_single,
                e.medium_mat, e.medium_W,
                e.eig_medium_buffer, e.cpu_eig_medium_buffer, e.medium_info,
                e.sizes.medium_mat_sizes[i],
                e.eig_medium_buffer_size[i], e.cpu_eig_medium_buffer_size[i],
                e.sizes.medium_mat_offset(i, j), e.sizes.medium_W_offset(i, j),
                e.sizes.medium_buffer_offset(i, j, e.eig_medium_buffer_size),
                e.sizes.medium_cpu_buffer_offset(i, j, e.cpu_eig_medium_buffer_size),
                counter);
            counter++;
        }
    }
    for (int i = 0; i < e.eig_stream_num_per_gpu; i++)
        CHECK_CUDA(cudaStreamSynchronize(e.eig_medium_stream_arr[i].stream));
    check_eig_info(e.medium_info.vals, e.sizes.medium_mat_num, "cusolverDnXsyevd (cone measure)");
    int info_offset = 0;
    for (int i = 0; i < e.sizes.small_mat_sizes.size(); i++)
    {
        batch_eig_cusolver(
            e.cusolverH_eig_small, e.eig_param_batch,
            e.small_mat, e.small_W,
            e.eig_small_buffer, e.small_info,
            e.sizes.small_mat_sizes[i], e.sizes.small_mat_nums[i],
            e.eig_small_buffer_size[i],
            e.sizes.small_mat_offset(i), e.sizes.small_W_offset(i),
            e.sizes.small_buffer_offset(i, e.eig_small_buffer_size),
            0, // buffer_host_offset (unused by the batched Jacobi EVD)
            info_offset);
        check_eig_info(
            e.small_info.vals + info_offset, e.sizes.small_mat_nums[i],
            "cusolverDnDsyevjBatched (cone measure)", e.sizes.small_mat_sizes[i]);
        info_offset += e.sizes.small_mat_nums[i];
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // host copies of the eigenvalues
    std::vector<double> large_W_h(e.sizes.sum_large_mat_size);
    std::vector<double> medium_W_h(e.sizes.sum_medium_mat_size);
    std::vector<double> small_W_h(e.sizes.sum_small_mat_size);
    if (!large_W_h.empty())
        CHECK_CUDA(cudaMemcpy(large_W_h.data(), e.large_W.vals, sizeof(double) * large_W_h.size(), D2H));
    if (!medium_W_h.empty())
        CHECK_CUDA(cudaMemcpy(medium_W_h.data(), e.medium_W.vals, sizeof(double) * medium_W_h.size(), D2H));
    if (!small_W_h.empty())
        CHECK_CUDA(cudaMemcpy(small_W_h.data(), e.small_W.vals, sizeof(double) * small_W_h.size(), D2H));

    ConeMeasures m;
    m.block_min.assign(blocks_.size(), 0.0);
    m.bound_coef.assign(blocks_.size(), 0.0);
    m.violation.assign(blocks_.size(), 0.0);
    for (std::size_t i = 0; i < blocks_.size(); i++)
    {
        const BlockInfo &blk = e.blk_info[blocks_[i].block];
        const double *W = blk.category == MatrixSizeCategory::LARGE ? large_W_h.data() : (blk.category == MatrixSizeCategory::MEDIUM ? medium_W_h.data() : small_W_h.data());
        double lambda_min = W[blk.W_offset];
        for (int k = 1; k < blk.size; k++)
            lambda_min = std::min(lambda_min, W[blk.W_offset + k]);
        m.block_min[i] = lambda_min;
        m.bound_coef[i] = std::min(0.0, lambda_min);
        m.violation[i] = std::max(0.0, -lambda_min);
    }
    return m;
}

const HostReferenceMeasure &PsdCone::host_reference() const { return kPsdReference; }

} // namespace cone
} // namespace cuadmm
