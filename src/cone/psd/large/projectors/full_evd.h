/*

    cone/psd/large/projectors/full_evd.h

    FullEvd (plan Table "PSD projection strategies", Section 10.5): the exact projection of a large block with
    cusolverDnXsyevd in place (M <- Q, the eigenvalues increasing), then the final product into the tier's W:

    - syrk (the default since Phase 6b): with r_+ positive and r_- negative eigenvalues, if r_+ <= r_- the last
      r_+ columns of Q are scaled in place by sqrt(lambda_i) and P = B B^T comes from one SYRK (the upper triangle,
      the one the packing reads, as main's map_M1 did);
      otherwise the first r_- columns are scaled by sqrt(-lambda_i), P' = B B^T, and the result is Z + P', the
      input slice added while packing. n^2 min(r_+, r_-) flops, no n^2 temporary, only the triangle svec stores.
    - gemm_compat: today's product, Q max(lambda, 0) into an n^2 temporary and one GEMM (bit for bit the
      reference on the exact path).

    It is the exact phase, the fallback of every large strategy, and the provider of eigenvalues for
    RankAnalyzer (the analysis runs before the in-place scaling).

*/

#ifndef CUADMM_CONE_PSD_LARGE_PROJECTORS_FULL_EVD_H
#define CUADMM_CONE_PSD_LARGE_PROJECTORS_FULL_EVD_H

#include "cone/psd/large/large_block_state.h"
#include "cone/psd/large/rank_analyzer.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

class LargeTier;

class FullEvd
{
public:
    FullEvd(LargeTier &tier, RankAnalyzer &analyzer);
    /// The decomposition of the block in M (M <- Q, the tier's eigenvalues); `what` names the caller in errors.
    void decompose(const LargeBlockJob &job, const char *what);
    /// The projection of the block in M into s (its svec slice); with analyze_rank the RankAnalyzer sets the
    /// block's mode and warm start first. z: the cone-step input (the negative SYRK side adds its slice).
    void project(const LargeBlockJob &job, bool analyze_rank, const double *z, double *s);

private:
    LargeTier &tier_;
    RankAnalyzer &analyzer_;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_PROJECTORS_FULL_EVD_H
