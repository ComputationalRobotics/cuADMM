/*

    cone/psd/large/projectors/full_evd.cu

    SDPSolver::large_full_eig_project of Milestone 1, moved without a change of its calls.

*/

#include "cone/psd/large/projectors/full_evd.h"

#include "cone/psd/eig_info.h"
#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

FullEvd::FullEvd(SDPSolver &engine, RankAnalyzer &analyzer) : engine_(engine), analyzer_(analyzer) {}

void FullEvd::decompose(const LargeBlockJob &job, int info_offset, const char *what)
{
    SDPSolver &e = engine_;
    const int i = job.size_index, j = job.index_in_group;
    single_eig_cusolver(
        e.cusolverH_eig_large, e.eig_param_single,
        e.large_mat, e.large_W,
        e.eig_large_buffer, e.cpu_eig_large_buffer, e.large_info,
        job.n,
        e.eig_large_buffer_size[i], e.cpu_eig_large_buffer_size[i],
        job.mat_offset, job.W_offset,
        e.sizes.large_buffer_offset(i, j, e.eig_large_buffer_size),
        e.sizes.large_cpu_buffer_offset(i, j, e.cpu_eig_large_buffer_size),
        info_offset);
    check_eig_info(e.large_info.vals + info_offset, 1, what);
}

void FullEvd::project(const LargeBlockJob &job, bool analyze_rank, bool multiply)
{
    SDPSolver &e = engine_;
    const int n = job.n;
    const int mat_offset = job.mat_offset;
    const int W_offset = job.W_offset;
    const double one = 1.0;
    const double zero = 0.0;

    // compute the EVD using cuSOLVER: large_mat <- eigenvectors, large_W <- eigenvalues (increasing order)
    decompose(job, job.state.idx, "cusolverDnXsyevd (large matrix)");

    if (analyze_rank)
        analyzer_.analyze(job, e.stop_tol);

    // P = Q max(W, 0) Q^T
    max_dense_vector_zero(e.large_W.vals + W_offset, n);
    dense_matrix_mul_diag_batch(
        e.large_mat_tmp, e.large_mat, e.large_W,
        n, 1, mat_offset, W_offset);
    if (!multiply)
        return; // the caller multiplies the whole size group with one batched GEMM
    CHECK_CUBLAS(cublasDgemm(
        e.cublasH_eig_large.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, n,
        &one,
        e.large_mat_tmp.vals + mat_offset, n,
        e.large_mat.vals + mat_offset, n,
        &zero,
        e.large_mat_P.vals + mat_offset, n));
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
