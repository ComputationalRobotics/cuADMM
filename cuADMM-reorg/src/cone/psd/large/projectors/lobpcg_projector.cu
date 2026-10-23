/*

    cone/psd/large/projectors/lobpcg_projector.cu

    SDPSolver::large_lobpcg_project of Milestone 1, moved without a change of its calls (the macros
    LOBPCG_WARMSTART, LOBPCG_MAXIT and LOBPCG_TOL are the configuration now; the warm start is always on).

*/

#include "cone/psd/large/projectors/lobpcg_projector.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "cone/psd/kernels/lobpcg.h"
#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

Lobpcg::Lobpcg(const LobpcgConfig &config, SDPSolver &engine) : config_(config), engine_(engine) {}

bool Lobpcg::project(const LargeBlockJob &job)
{
    SDPSolver &e = engine_;
    const int n = job.n;
    const LargeBlockState &st = job.state;
    const int k = st.k;
    const int mat_offset = job.mat_offset;
    const bool negative = st.mode == kModeLowRankNegative;
    const double one = 1.0;
    const double minus_one = -1.0;

    cublasHandle_t cublasH = e.cublasH_eig_large.cublas_handle;
    double *A = e.large_mat.vals + mat_offset;
    double *P = e.large_mat_P.vals + mat_offset;
    double *eigenvectors = st.V;
    double *eigenvalues = st.D;
    double *relu_eigenvalues = e.lobpcg_W_relu.vals;
    double *scaled_eigenvectors = e.large_mat_tmp.vals + mat_offset; // n x k workspace

    if (negative)
    {
        // change the matrix sign to reuse LOBPCG code: A <- -A
        // (the warmstart eigenvalues in D are already those of -A)
        CHECK_CUBLAS(cublasDscal(cublasH, n * n, &minus_one, A, 1));
    }

    // compute the largest eigenpairs (of -A in the negative case), in decreasing order;
    // the pairs that were above the rank tolerance at the last rank analysis and the pairs currently
    // above it take part in the convergence test (the remaining relaxation pairs need not converge)
    const double rank_tol = st.rank_tol;
    const int rank = negative ? st.negative_rank : st.positive_rank;
    LobpcgInfo info;
    lobpcg(
        cublasH, e.cusolverH_eig_large.cusolver_dn_handle,
        A, eigenvectors, eigenvalues,
        n, k, true, config_.max_iter, config_.tol, false,
        &info, rank_tol, std::min(rank, k));
    e.lobpcg_calls++;

    if (negative)
    {
        // restore the sign of A
        CHECK_CUBLAS(cublasDscal(cublasH, n * n, &minus_one, A, 1));
    }

    // accept the result only if LOBPCG converged and the smallest computed Ritz value is below the
    // rank tolerance: otherwise an eigenvalue of the wanted sign may have been left out (e.g. the
    // rank grew since the last rank analysis), and the projection would not be PSD. This is necessary,
    // not sufficient: Ritz values are lower bounds, and an eigenvector (numerically) orthogonal to the
    // LOBPCG search subspace cannot be detected; the full EVD every reevaluation period and the
    // exact PSD-violation check at termination bound this
    std::vector<double> ritz_values(k);
    CHECK_CUDA(cudaMemcpy(ritz_values.data(), eigenvalues, sizeof(double) * k, D2H));
    bool finite = true;
    for (int l = 0; l < k; l++)
        finite = finite && std::isfinite(ritz_values[l]);
    if (!(finite && info.converged && ritz_values[k - 1] <= rank_tol + info.residual_norm))
    {
        e.lobpcg_fallbacks++;
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

} // namespace psd
} // namespace cone
} // namespace cuadmm
