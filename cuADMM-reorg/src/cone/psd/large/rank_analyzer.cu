/*

    cone/psd/large/rank_analyzer.cu

    The rank-analysis block of SDPSolver::large_full_eig_project of Milestone 1, moved without a change of its
    calls (the macros LOBPCG_RATIO, LOBPCG_RELAXATION, LOBPCG_RANK_RTOL, LOBPCG_RANK_STOP_TOL_FACTOR are the
    configuration now; the eigenpairs come from the tier's M and eigenvalue regions).

*/

#include "cone/psd/large/rank_analyzer.h"

#include <algorithm>
#include <cmath>

#include "cone/psd/kernels/precision.h"
#include "cone/psd/large/large_tier.h"
#include "core/check.h"
#include "cuadmm_legacy/rank.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

RankAnalyzer::RankAnalyzer(const RankAnalyzerConfig &config, LargeTier &tier) : config_(config), tier_(tier) {}

double RankAnalyzer::rank_tolerance(double lambda_min, double lambda_max, double stop_tol) const
{
    // relative to the extreme eigenvalues, but not below a fraction of the stopping tolerance (the negative part
    // of C - A^T y - X/sigma is -X/sigma, tiny when sigma is large)
    return std::max(config_.rank_rtol * std::max(std::abs(lambda_min), std::abs(lambda_max)), config_.rank_stop_factor * stop_tol);
}

void RankAnalyzer::analyze(const LargeBlockJob &job, double stop_tol)
{
    const int n = job.n;
    const LargeBlockState &st = job.state;
    const double *eig = tier_.eigenvalues();
    const double *Q = tier_.M();

    double lambda_min, lambda_max;
    CUADMM_CHECK_CUDA(cudaMemcpy(&lambda_min, eig, sizeof(double), cudaMemcpyDeviceToHost));
    CUADMM_CHECK_CUDA(cudaMemcpy(&lambda_max, eig + n - 1, sizeof(double), cudaMemcpyDeviceToHost));
    const double rank_tol = rank_tolerance(lambda_min, lambda_max, stop_tol);
    st.rank_tol = rank_tol;

    // compute the ranks
    compute_ranks(eig, n, st.d_positive_rank, st.d_negative_rank, rank_tol);

    // copy ranks to CPU
    CUADMM_CHECK_CUDA(cudaMemcpy(&st.positive_rank, st.d_positive_rank, sizeof(int), cudaMemcpyDeviceToHost));
    CUADMM_CHECK_CUDA(cudaMemcpy(&st.negative_rank, st.d_negative_rank, sizeof(int), cudaMemcpyDeviceToHost));
    const int positive_rank = st.positive_rank;
    const int negative_rank = st.negative_rank;

    // copy the extreme eigenpairs to use as a warmstart for LOBPCG
    if (positive_rank < config_.ratio * n && positive_rank > 0)
    {
        const int k = std::min<int>(std::ceil(config_.relaxation * positive_rank), st.k_alloc);
        st.mode = kModeLowRankPositive;
        st.k = k;

        // largest k eigenpairs, in decreasing order: copy the eigenvalues and reverse them
        reverse_vector(eig + n - k, st.D, k);
        // copy the eigenvectors and reverse the columns
        reverse_columns(Q + (std::size_t)(n - k) * n, st.V, n, k);
    }
    else if (negative_rank < config_.ratio * n && negative_rank > 0)
    {
        const int k = std::min<int>(std::ceil(config_.relaxation * negative_rank), st.k_alloc);
        st.mode = kModeLowRankNegative;
        st.k = k;

        // smallest k eigenpairs, in increasing order (i.e. decreasing order for -A): copy without reversing
        CUADMM_CHECK_CUDA(cudaMemcpy(st.D, eig, sizeof(double) * k, cudaMemcpyDeviceToDevice));
        CUADMM_CHECK_CUDA(cudaMemcpy(st.V, Q, sizeof(double) * k * n, cudaMemcpyDeviceToDevice));
        // LOBPCG runs on -A: store the eigenvalues of -A (the eigenvectors do not change)
        const double minus_one = -1.0;
        CUADMM_CHECK_CUBLAS(cublasDscal(tier_.ctx().cublas, k, &minus_one, st.D, 1));
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
