/*

    cone/psd/batched/batch_projectors.cu

    Steps 3.2 and 3.3 of SDPSolver::solve() of main (the medium and small tiers of Milestone 1's
    PsdCone::project_tiers), moved without a change of their calls.

*/

#include "cone/psd/batched/batch_projectors.h"

#include <iostream>
#include <stdexcept>

#include "cone/psd/eig_info.h"
#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

/* small tier */

JacobiBatchedEvd::JacobiBatchedEvd(SDPSolver &engine) : engine_(engine) {}

int JacobiBatchedEvd::decompose(const char *what, bool warn)
{
    SDPSolver &e = engine_;
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
    int total_not_converged = 0;
    info_offset = 0;
    for (int i = 0; i < (int)e.sizes.small_mat_sizes.size(); i++)
    {
        const int not_converged = check_eig_info(e.small_info.vals + info_offset, e.sizes.small_mat_nums[i], what, e.sizes.small_mat_sizes[i]);
        if (warn)
        {
            if (not_converged > 0 && e.jacobi_not_converged == 0)
                std::cout << " WARNING: batched Jacobi EVD did not reach its tolerance for " << not_converged
                          << " matrices of size " << e.sizes.small_mat_sizes[i]
                          << " (results are still accurate to machine precision; reported once per solve)" << std::endl;
            e.jacobi_not_converged += not_converged;
        }
        total_not_converged += not_converged;
        info_offset += e.sizes.small_mat_nums[i];
    }
    return total_not_converged;
}

void JacobiBatchedEvd::project()
{
    SDPSolver &e = engine_;
    decompose("cusolverDnDsyevjBatched (small matrix)", true);
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

/* medium tier */

BatchedEvd::BatchedEvd(SDPSolver &engine) : engine_(engine) {}

bool BatchedEvd::enabled(int group) const
{
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
    return engine_.medium_group_batched[group] != 0;
#else
    (void)group;
    return false;
#endif
}

void BatchedEvd::decompose(int group, int info_offset)
{
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
    SDPSolver &e = engine_;
    const int i = group;
    const int n = e.sizes.medium_mat_sizes[i];
    CHECK_CUSOLVER(cusolverDnXsyevBatched(
        e.cusolverH_eig_medium_batch.cusolver_dn_handle, e.eig_param_single.param,
        e.eig_param_single.jobz, e.eig_param_single.uplo,
        n, CUDA_R_64F, e.medium_mat.vals + e.sizes.medium_mat_offset(i, 0), n,
        CUDA_R_64F, e.medium_W.vals + e.sizes.medium_W_offset(i, 0), CUDA_R_64F,
        e.medium_batch_buffer.vals, e.medium_batch_buffer_size,
        e.medium_batch_cpu_buffer.vals, e.medium_batch_cpu_buffer_size,
        e.medium_info.vals + info_offset, e.sizes.medium_mat_nums[i]));
#else
    (void)group;
    (void)info_offset;
    throw std::logic_error("BatchedEvd: cusolverDnXsyevBatched needs CUDA 12.8");
#endif
}

StreamedEvd::StreamedEvd(SDPSolver &engine) : engine_(engine) {}

void StreamedEvd::decompose(int group, int j, int counter)
{
    SDPSolver &e = engine_;
    const int i = group;
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
}

void StreamedEvd::synchronize()
{
    SDPSolver &e = engine_;
    for (int i = 0; i < e.eig_stream_num_per_gpu; i++)
        CHECK_CUDA(cudaStreamSynchronize(e.eig_medium_stream_arr[i].stream));
}

MediumTier::MediumTier(SDPSolver &engine) : engine_(engine), batched_(engine), streamed_(engine) {}

void MediumTier::decompose(const char *what, bool allow_batched)
{
    SDPSolver &e = engine_;
    int all_counter = 0; // serves as an info offset
    for (int i = 0; i < (int)e.sizes.medium_mat_sizes.size(); i++)
    {
        if (allow_batched && batched_.enabled(i))
        {
            batched_.decompose(i, all_counter);
            all_counter += e.sizes.medium_mat_nums[i];
            continue;
        }
        for (int j = 0; j < e.sizes.medium_mat_nums[i]; j++)
        {
            streamed_.decompose(i, j, all_counter);
            all_counter++;
        }
    }
    // synchronize the streams used for the eigenvalue decomposition of medium matrices
    streamed_.synchronize();
    check_eig_info(e.medium_info.vals, e.sizes.medium_mat_num, what);
}

void MediumTier::project()
{
    SDPSolver &e = engine_;
    decompose("cusolverDnXsyevd (medium matrix)", true);
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
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
