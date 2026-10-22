/*

    admm/admm_state.cu

    The steps of SDPSolver::solve() of main (src/solver.cu, Steps 1-5), moved here unchanged: the same kernels in the
    same order on the same buffers, so that the iterates are bit for bit those of the references.

*/

#include "admm/admm_state.h"

#include <cmath>
#include <iomanip>
#include <iostream>

#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace admm
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
} // namespace

AdmmState::AdmmState(SDPSolver &engine) : engine_(engine) {}

void AdmmState::y_step(double sigma)
{
    SDPSolver &e = engine_;
    /* r_s^{k+1/2} = b/sigma - A(X/sigma + S - C) */
    // rhsy <-- -1.0 * A * SmC + 0.0 * rhsy
    SpMV_cusparse(e.cusparseH, e.A_csr, e.SmC, e.rhsy, -1.0, 0.0, e.SpMV_AX_buffer);
    // rhsy <-- 1/sig * Rp + rhsy
    axpy_cublas(e.cublasH, e.Rp, e.rhsy, 1 / sigma);
    /* y^{k+1/2} = (AA^T)^{-1} r_s^{k+1/2} */
    perform_permutation(e.rhsy_perm, e.rhsy, e.perm_inv);
    e.solve_AAt_perm();
    y_solves_++;
    perform_permutation(e.y, e.y_perm, e.perm);
}

void AdmmState::cone_step(double sigma, int k)
{
    SDPSolver &e = engine_;
    // Aty <-- 1.0 * At * y + 0.0 * Aty
    SpMV_cusparse(e.cusparseH, e.At_csr, e.y, e.Aty, 1.0, 0.0, e.SpMV_Aty_buffer);
    // Rd1 <-- Aty; Rd1 <-- (-1.0) * C + 1.0 * Rd1, hence Rd1 = A^T y^{k+1/2} - C
    CHECK_CUDA(cudaMemcpy(e.Rd1.vals, e.Aty.vals, sizeof(double) * e.vec_len, D2D));
    axpby_cusparse(e.cusparseH, e.C, e.Rd1, -1.0, 1.0);
    // Xinput <-- -(Rd1 + 1/sig * X)
    dense_vector_plus_dense_vector_mul_scalar(e.Xinput, e.Rd1, e.X, 1.0 / sigma);
    dense_vector_negate(e.Xinput);
    // the matrices of the PSD blocks
    vector_to_matrices(e.Xinput, e.large_mat, e.medium_mat, e.small_mat, e.map_B, e.map_M1, e.map_M2);
    CHECK_CUDA(cudaDeviceSynchronize());
    project_psd_tiers(k);
    // back to the vector; the free and nonnegative blocks are projected on the way
    matrices_to_vector(e.S, e.Xinput, e.large_mat_P, e.medium_mat_P, e.small_mat_P, e.map_B, e.map_M1, e.map_M2);
}

void AdmmState::project_psd_tiers(int iter)
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

void AdmmState::update_smc()
{
    SDPSolver &e = engine_;
    // SmC <-- S; SmC <-- -1.0 * C + 1.0 * SmC, hence SmC = S^{k+1} - C
    CHECK_CUDA(cudaMemcpy(e.SmC.vals, e.S.vals, sizeof(double) * e.vec_len, D2D));
    axpby_cusparse(e.cusparseH, e.C, e.SmC, -1.0, 1.0);
}

void AdmmState::second_y_step(double sigma)
{
    SDPSolver &e = engine_;
    // rhsy <-- -1.0 * A * SmC + 0.0 * rhsy, hence rhsy = - A(S - C)
    SpMV_cusparse(e.cusparseH, e.A_csr, e.SmC, e.rhsy, -1.0, 0.0, e.SpMV_AX_buffer);
    // rhsy <-- 1/sig * Rp + rhsy, hence rhsy = 1/sigma b - A(X^k /sigma + S^{k+1} - C)
    axpy_cublas(e.cublasH, e.Rp, e.rhsy, 1 / sigma);
    // y <-- linsys(rhsy)
    perform_permutation(e.rhsy_perm, e.rhsy, e.perm_inv);
    e.solve_AAt_perm();
    y_solves_++;
    perform_permutation(e.y, e.y_perm, e.perm);
    // Aty <-- 1.0 * At * y + 0.0 * Aty, hence Aty = A^T y^{k+1}
    SpMV_cusparse(e.cusparseH, e.At_csr, e.y, e.Aty, 1.0, 0.0, e.SpMV_Aty_buffer);
    // Rd1 <-- Aty; Rd1 <-- (-1.0) * C + 1.0 * Rd1, hence Rd1 = A^T y^{k+1} - C
    CHECK_CUDA(cudaMemcpy(e.Rd1.vals, e.Aty.vals, sizeof(double) * e.vec_len, D2D));
    axpby_cusparse(e.cusparseH, e.C, e.Rd1, -1.0, 1.0);
}

void AdmmState::multiplier_step(double sigma, double tau)
{
    SDPSolver &e = engine_;
    // Rd <-- 1.0 * Rd1 + 1.0 * S, hence Rd = A^T y^{k+1} - C + S
    dense_vector_add_dense_vector(e.Rd, e.Rd1, e.S, 1.0, 1.0);
    // X <-- X + (tau * sig) * Rd
    dense_vector_add_dense_vector(e.X, e.Rd, 1.0, tau * sigma);
}

KktResiduals AdmmState::evaluate_kkt()
{
    SDPSolver &e = engine_;
    // Rp <-- -1.0 * A * X + 0.0 * Rp; Rp <-- 1.0 * b + 1.0 * Rp, hence Rp = b - A X
    SpMV_cusparse(e.cusparseH, e.A_csr, e.X, e.Rp, -1.0, 0.0, e.SpMV_AX_buffer);
    axpby_cusparse(e.cusparseH, e.b, e.Rp, 1.0, 1.0);
    // primal error and objective (original units)
    dense_vector_mul_dense_vector_mul_scalar(e.Rporg, e.normA, e.Rp, e.bscale);
    e.errRp = e.Rporg.get_norm(e.cublasH) / e.norm_borg;
    e.pobj = SparseVV_cusparse(e.cusparseH, e.C, e.X, e.SpVV_CtX_buffer) * e.objscale;
    // dual error and objective
    dense_vector_mul_scalar(e.Rdorg, e.Rd, e.Cscale);
    e.errRd = e.Rdorg.get_norm(e.cublasH) / e.norm_Corg;
    e.dobj = SparseVV_cusparse(e.cusparseH, e.b, e.y, e.SpVV_bty_buffer) * e.objscale;
    e.maxfeas = std::max(e.errRp, e.errRd);
    e.relgap = std::abs(e.pobj - e.dobj) / (1 + std::abs(e.pobj) + std::abs(e.dobj));
    KktResiduals r;
    r.eta_p = e.errRp;
    r.eta_d = e.errRd;
    r.eta_g = e.relgap;
    r.pobj = e.pobj;
    r.dobj = e.dobj;
    return r;
}

KktResiduals AdmmState::evaluate_kkt_initial()
{
    SDPSolver &e = engine_;
    // the starting point: SmC = S - C, Rp = b - A X, Aty = A^T y, Rd = Aty + SmC (the rescale path of solve() of main)
    CHECK_CUDA(cudaMemcpy(e.SmC.vals, e.S.vals, sizeof(double) * e.vec_len, D2D));
    axpby_cusparse(e.cusparseH, e.C, e.SmC, -1.0, 1.0);
    SpMV_cusparse(e.cusparseH, e.A_csr, e.X, e.Rp, -1.0, 0.0, e.SpMV_AX_buffer);
    axpby_cusparse(e.cusparseH, e.b, e.Rp, 1.0, 1.0);
    SpMV_cusparse(e.cusparseH, e.At_csr, e.y, e.Aty, 1.0, 0.0, e.SpMV_Aty_buffer);
    dense_vector_add_dense_vector(e.Rd, e.Aty, e.SmC);
    dense_vector_mul_dense_vector_mul_scalar(e.Rporg, e.normA, e.Rp, e.bscale);
    dense_vector_mul_scalar(e.Rdorg, e.Rd, e.Cscale);
    e.errRp = e.Rporg.get_norm(e.cublasH) / e.norm_borg;
    e.errRd = e.Rdorg.get_norm(e.cublasH) / e.norm_Corg;
    e.maxfeas = std::max(e.errRp, e.errRd);
    e.pobj = SparseVV_cusparse(e.cusparseH, e.C, e.X, e.SpVV_CtX_buffer) * e.objscale;
    e.dobj = SparseVV_cusparse(e.cusparseH, e.b, e.y, e.SpVV_bty_buffer) * e.objscale;
    e.relgap = std::abs(e.pobj - e.dobj) / (1 + std::abs(e.pobj) + std::abs(e.dobj));
    KktResiduals r;
    r.eta_p = e.errRp;
    r.eta_d = e.errRd;
    r.eta_g = e.relgap;
    r.pobj = e.pobj;
    r.dobj = e.dobj;
    return r;
}

void AdmmState::scale_iterate()
{
    SDPSolver &e = engine_;
    // X, y, S given in original units: scale them as init() and the rescale path of solve() of main do
    dense_vector_mul_dense_vector(e.y, e.normA);
    dense_vector_div_scalar(e.X, e.bscale);
    dense_vector_div_scalar(e.S, e.Cscale);
    dense_vector_div_scalar(e.y, e.Cscale);
}

void AdmmState::unscale_iterate()
{
    SDPSolver &e = engine_;
    // recover the original solution by unscaling
    dense_vector_mul_scalar(e.X, e.bscale);
    dense_vector_div_dense_vector_mul_scalar(e.y, e.normA, e.Cscale);
    dense_vector_mul_scalar(e.S, e.Cscale);
}

void AdmmState::begin_iteration(int k, double eta, int switch_iter, double switch_tol)
{
    SDPSolver &e = engine_;
    // check if the conditions to switch the projection method are met
    if (!e.switched_proj_method && (k > switch_iter || eta < switch_tol) && k > 1)
    {
        if (e.final_proj_method != e.current_proj_method)
            std::cout << " ---------------- Switching projection method to " << get_projection_method_name(e.final_proj_method) << "------" << std::endl;
        e.current_proj_method = e.final_proj_method;
        e.switched_proj_method = true;
        e.switched_proj_method_iter = k;
    }
}

} // namespace admm
} // namespace cuadmm
