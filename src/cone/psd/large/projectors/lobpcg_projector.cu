/*

    cone/psd/large/projectors/lobpcg_projector.cu

    SDPSolver::large_lobpcg_project of Milestone 1 on the tier's regions: the LOBPCG workspace, the scaled
    eigenvectors and the result P are carved from W (B7); the gemm_compat product keeps the calls of Milestone 1.

*/

#include "cone/psd/large/projectors/lobpcg_projector.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "cone/psd/kernels/lobpcg.h"
#include "cone/psd/large/large_tier.h"
#include "cone/psd/psd_kernels.h"
#include "core/check.h"
#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

Lobpcg::Lobpcg(const LobpcgConfig &config, LargeTier &tier) : config_(config), tier_(tier) {}

bool Lobpcg::project(const LargeBlockJob &job, const double *z, double *s)
{
    SDPSolver &e = tier_.engine();
    const int n = job.n;
    const LargeBlockState &st = job.state;
    const int k = st.k;
    const bool negative = st.mode == kModeLowRankNegative;
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    const double one = 1.0;
    const double minus_one = -1.0;
    const core::ExecContext ctx = tier_.ctx();
    cublasHandle_t cublasH = ctx.cublas;

    double *A = tier_.M();
    double *eigenvectors = st.V;
    double *eigenvalues = st.D;
    // W: P (n^2), the scaled eigenvectors (n x k), the ReLU'd eigenvalues (k), then LOBPCG's workspace
    double *P = tier_.W();
    double *scaled_eigenvectors = P + nn;
    double *relu_eigenvalues = scaled_eigenvectors + static_cast<std::size_t>(n) * k;
    double *workspace = relu_eigenvalues + k;

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
        cublasH, ctx.cusolver,
        A, eigenvectors, eigenvalues,
        n, k, true, config_.max_iter, config_.tol, false,
        &info, rank_tol, std::min(rank, k), workspace);
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
    CHECK_CUDA(cudaMemcpy(ritz_values.data(), eigenvalues, sizeof(double) * k, cudaMemcpyDeviceToHost));
    bool finite = true;
    for (int l = 0; l < k; l++)
        finite = finite && std::isfinite(ritz_values[l]);
    if (!(finite && info.converged && ritz_values[k - 1] <= rank_tol + info.residual_norm))
    {
        e.lobpcg_fallbacks++;
        return false;
    }

    if (tier_.config().gemm_compat)
    {
        // V diag(max(D, 0))
        CHECK_CUDA(cudaMemcpy(relu_eigenvalues, eigenvalues, sizeof(double) * k, cudaMemcpyDeviceToDevice));
        max_dense_vector_zero(relu_eigenvalues, k, ctx.stream);
        CHECK_CUBLAS(cublasDdgmm(
            cublasH, CUBLAS_SIDE_RIGHT, n, k,
            eigenvectors, n, relu_eigenvalues, 1,
            scaled_eigenvectors, n));

        // positive case: P = V max(D, 0) V^T
        // negative case: P = A + V max(-D', 0) V^T = A + proj(-A), where D' are the eigenvalues of -A
        double beta = 0.0;
        if (negative)
        {
            CHECK_CUDA(cudaMemcpy(P, A, sizeof(double) * n * n, cudaMemcpyDeviceToDevice));
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
        tier_.pack(st.idx, P, s, nullptr);
        return true;
    }

    // SYRK: B = V diag(sqrt(max(D, 0))), P = B B^T (the upper triangle); the negative mode adds the input slice while packing
    const double zero = 0.0;
    scale_columns(eigenvectors, scaled_eigenvectors, n, 0, k, eigenvalues, SCALE_SQRT_RELU, ctx.stream);
    CHECK_CUBLAS(cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &one, scaled_eigenvectors, n, &zero, P, n));
    tier_.pack(st.idx, P, s, negative ? z : nullptr);
    return true;
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
