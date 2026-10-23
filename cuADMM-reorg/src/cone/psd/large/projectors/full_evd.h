/*

    cone/psd/large/projectors/full_evd.h

    FullEvd (plan Table "PSD projection strategies"): the exact projection of a large block with
    cusolverDnXsyevd in place (large_mat <- Q, large_W <- lambda increasing), then P = Q max(lambda, 0) Q^T.
    It is the exact phase, the fallback of every large strategy, and the provider of eigenvalues for
    RankAnalyzer. Phase 6a keeps today's product (psd.large.final_product = gemm_compat): Q max(lambda, 0) into
    large_mat_tmp with the diagonal-scaling kernel, then one GEMM into large_mat_P, or, when the caller
    multiplies a whole size group with one strided-batched GEMM, no GEMM here (the batched product does not
    round like per-matrix GEMMs; Phase 6b replaces both with SYRK, within rounding).

*/

#ifndef CUADMM_CONE_PSD_LARGE_PROJECTORS_FULL_EVD_H
#define CUADMM_CONE_PSD_LARGE_PROJECTORS_FULL_EVD_H

#include "cone/psd/large/large_block_state.h"
#include "cone/psd/large/rank_analyzer.h"

class SDPSolver;

namespace cuadmm
{
namespace cone
{
namespace psd
{

class FullEvd
{
public:
    FullEvd(SDPSolver &engine, RankAnalyzer &analyzer);
    /// The decomposition alone: large_mat <- eigenvectors, large_W <- eigenvalues (increasing); throws when
    /// cuSOLVER reports a failure (`what` names the caller in the message; info_offset indexes large_info).
    void decompose(const LargeBlockJob &job, int info_offset, const char *what);
    /// The projection into large_mat_P (multiply) or Q max(lambda, 0) into large_mat_tmp for the caller's batched
    /// product (!multiply); with analyze_rank the RankAnalyzer sets the block's mode and warm start first.
    void project(const LargeBlockJob &job, bool analyze_rank, bool multiply);

private:
    SDPSolver &engine_;
    RankAnalyzer &analyzer_;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_PROJECTORS_FULL_EVD_H
