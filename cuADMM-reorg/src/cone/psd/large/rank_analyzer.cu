/*

    cone/psd/large/rank_analyzer.cu

    The rank-analysis block of SDPSolver::large_full_eig_project of Milestone 1, moved without a change of its
    calls (the macros LOBPCG_RATIO, LOBPCG_RELAXATION, LOBPCG_RANK_RTOL, LOBPCG_RANK_STOP_TOL_FACTOR are the
    configuration now).

*/

#include "cone/psd/large/rank_analyzer.h"

#include <algorithm>
#include <cmath>

#include "cone/psd/kernels/precision.h"
#include "cuadmm_legacy/rank.h"
#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

RankAnalyzer::RankAnalyzer(const RankAnalyzerConfig &config, SDPSolver &engine) : config_(config), engine_(engine) {}

double RankAnalyzer::rank_tolerance(double lambda_min, double lambda_max, double stop_tol) const
{
    // relative to the extreme eigenvalues, but not below a fraction of the stopping tolerance (the negative part
    // of C - A^T y - X/sigma is -X/sigma, tiny when sigma is large)
    return std::max(config_.rank_rtol * std::max(std::abs(lambda_min), std::abs(lambda_max)), config_.rank_stop_factor * stop_tol);
}

void RankAnalyzer::analyze(const LargeBlockJob &job, double stop_tol)
{
    SDPSolver &e = engine_;
    const int n = job.n;
    const int mat_offset = job.mat_offset;
    const int W_offset = job.W_offset;
    const LargeBlockState &st = job.state;

    double lambda_min, lambda_max;
    CHECK_CUDA(cudaMemcpy(&lambda_min, e.large_W.vals + W_offset, sizeof(double), D2H));
    CHECK_CUDA(cudaMemcpy(&lambda_max, e.large_W.vals + W_offset + n - 1, sizeof(double), D2H));
    const double rank_tol = rank_tolerance(lambda_min, lambda_max, stop_tol);
    st.rank_tol = rank_tol;

    // compute the ranks
    compute_ranks(e.large_W.vals + W_offset, n, st.d_positive_rank, st.d_negative_rank, rank_tol);

    // copy ranks to CPU
    CHECK_CUDA(cudaMemcpy(&st.positive_rank, st.d_positive_rank, sizeof(int), D2H));
    CHECK_CUDA(cudaMemcpy(&st.negative_rank, st.d_negative_rank, sizeof(int), D2H));
    const int positive_rank = st.positive_rank;
    const int negative_rank = st.negative_rank;

    // copy the extreme eigenpairs to use as a warmstart for LOBPCG
    if (positive_rank < config_.ratio * n && positive_rank > 0)
    {
        const int k = std::min<int>(std::ceil(config_.relaxation * positive_rank), st.k_alloc);
        st.mode = kModeLowRankPositive;
        st.k = k;

        // largest k eigenpairs, in decreasing order: copy the eigenvalues and reverse them
        reverse_vector(e.large_W.vals + W_offset + n - k, st.D, k);
        // copy the eigenvectors and reverse the columns
        reverse_columns(e.large_mat.vals + mat_offset + (n - k) * n, st.V, n, k);
    }
    else if (negative_rank < config_.ratio * n && negative_rank > 0)
    {
        const int k = std::min<int>(std::ceil(config_.relaxation * negative_rank), st.k_alloc);
        st.mode = kModeLowRankNegative;
        st.k = k;

        // smallest k eigenpairs, in increasing order (i.e. decreasing order for -A): copy without reversing
        CHECK_CUDA(cudaMemcpy(st.D, e.large_W.vals + W_offset, sizeof(double) * k, D2D));
        CHECK_CUDA(cudaMemcpy(st.V, e.large_mat.vals + mat_offset, sizeof(double) * k * n, D2D));
        // LOBPCG runs on -A: store the eigenvalues of -A (the eigenvectors do not change)
        const double minus_one = -1.0;
        CHECK_CUBLAS(cublasDscal(e.cublasH_eig_large.cublas_handle, k, &minus_one, st.D, 1));
    }
    else
    {
        st.mode = kModeFull;
        st.k = 0;
    }
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
