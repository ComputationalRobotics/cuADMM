/*

    cone/psd/large/projectors/full_evd.cu

*/

#include "cone/psd/large/projectors/full_evd.h"

#include <vector>

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

FullEvd::FullEvd(LargeTier &tier, RankAnalyzer &analyzer) : tier_(tier), analyzer_(analyzer) {}

void FullEvd::decompose(const LargeBlockJob &job, const char *what)
{
    tier_.decompose(job.state.idx, what);
}

void FullEvd::project(const LargeBlockJob &job, bool analyze_rank, const double *z, double *s)
{
    const int n = job.n;
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    double *Q = tier_.M();
    double *eig = tier_.eigenvalues();
    const core::ExecContext ctx = tier_.ctx();
    const double one = 1.0, zero = 0.0;

    // compute the EVD using cuSOLVER: M <- eigenvectors, eig <- eigenvalues (increasing order)
    decompose(job, "cusolverDnXsyevd (large matrix)");

    if (analyze_rank)
        analyzer_.analyze(job, tier_.engine_stop_tol());

    if (tier_.config().gemm_compat)
    {
        // today's product: P = (Q max(W, 0)) Q^T with the n^2 temporary
        double *tmp = tier_.W();
        double *P = tmp + nn;
        max_dense_vector_zero(eig, n, ctx.stream);
        scale_columns(Q, tmp, n, 0, n, eig, SCALE_RELU, ctx.stream);
        CUADMM_CHECK_CUBLAS(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, tmp, n, Q, n, &zero, P, n));
        tier_.pack(job.state.idx, P, s, nullptr);
        return;
    }

    // SYRK on the smaller eigen-side (plan Section 10.5): the signs of the eigenvalues on the host
    std::vector<double> lambda(n);
    CUADMM_CHECK_CUDA(cudaMemcpy(lambda.data(), eig, sizeof(double) * n, cudaMemcpyDeviceToHost));
    int positive = 0, negative = 0;
    for (int i = 0; i < n; i++)
    {
        positive += lambda[i] > 0.0;
        negative += lambda[i] < 0.0;
    }
    double *P = tier_.W();
    if (positive <= negative)
    {
        // P = sum_{lambda_i > 0} lambda_i q_i q_i^T = B B^T with B the last `positive` columns scaled by sqrt(lambda_i)
        if (positive == 0)
            CUADMM_CHECK_CUDA(cudaMemsetAsync(P, 0, sizeof(double) * nn, ctx.stream));
        else
        {
            scale_columns(Q, Q, n, n - positive, positive, eig, SCALE_SQRT_RELU, ctx.stream);
            CUADMM_CHECK_CUBLAS(cublasDsyrk(ctx.cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, positive, &one, Q + static_cast<std::size_t>(n - positive) * n, n, &zero, P, n));
        }
        tier_.pack(job.state.idx, P, s, nullptr);
    }
    else
    {
        // P = Z + P' with P' = sum_{lambda_i < 0} (-lambda_i) q_i q_i^T: the first `negative` columns scaled by sqrt(-lambda_i)
        scale_columns(Q, Q, n, 0, negative, eig, SCALE_SQRT_NEG, ctx.stream);
        CUADMM_CHECK_CUBLAS(cublasDsyrk(ctx.cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, negative, &one, Q, n, &zero, P, n));
        tier_.pack(job.state.idx, P, s, z);
    }
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
