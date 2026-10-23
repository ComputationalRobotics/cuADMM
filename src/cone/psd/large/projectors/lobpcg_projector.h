/*

    cone/psd/large/projectors/lobpcg_projector.h

    Lobpcg (plan Table "PSD projection strategies", Section 10.5): the warm-started LOBPCG projection of a large
    block in its low-rank mode, its buffers carved from the tier's W (B7). Positive mode: the k largest eigenpairs
    (V, D) of Z give P = V max(D, 0) V^T; negative mode: LOBPCG runs on -Z (the block is negated in place and
    restored) and P = Z + V max(D', 0) V^T with D' the eigenvalues of -Z. The result is accepted only when LOBPCG
    converged, every Ritz value is finite and the smallest computed Ritz value is below the rank tolerance plus
    the residual norm (an eigenvalue of the wanted sign may otherwise have been left out: the rank grew since
    the last analysis); a rejected block falls back to FullEvd with a new rank analysis (the policy's rule).

    Final product: syrk (B = V diag(sqrt(max(D, 0))), P = B B^T, the input slice added while packing in the
    negative mode) or gemm_compat (today's V diag(max(D, 0)) V^T with a GEMM, Z copied into P first in the
    negative mode: bit for bit the reference).

*/

#ifndef CUADMM_CONE_PSD_LARGE_PROJECTORS_LOBPCG_PROJECTOR_H
#define CUADMM_CONE_PSD_LARGE_PROJECTORS_LOBPCG_PROJECTOR_H

#include "cone/psd/large/large_block_state.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

class LargeTier;

struct LobpcgConfig
{
    int max_iter = 20;  // psd.lowrank.max_iter
    double tol = 1e-5;  // psd.lowrank.tol (absolute; DECISIONS 45)
};

class Lobpcg
{
public:
    Lobpcg(const LobpcgConfig &config, LargeTier &tier);
    /// The projection of the block in M into s (its svec slice); false when the result was rejected (M is left as
    /// it was: the exact fallback needs no new unpack). z: the cone-step input (the negative mode adds its slice).
    bool project(const LargeBlockJob &job, const double *z, double *s);
    const LobpcgConfig &config() const { return config_; }

private:
    LobpcgConfig config_;
    LargeTier &tier_;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_PROJECTORS_LOBPCG_PROJECTOR_H
