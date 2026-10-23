/*

    cone/psd/large/projectors/lobpcg_projector.h

    Lobpcg (plan Table "PSD projection strategies"): the warm-started LOBPCG projection of a large block in its
    low-rank mode. Positive mode: the k largest eigenpairs (V, D) of Z give P = V max(D, 0) V^T; negative mode:
    LOBPCG runs on -Z (the block is negated in place and restored) and P = Z + V max(D', 0) V^T with D' the
    eigenvalues of -Z. The result is accepted only when LOBPCG converged, every Ritz value is finite and the
    smallest computed Ritz value is below the rank tolerance plus the residual norm (an eigenvalue of the wanted
    sign may otherwise have been left out: the rank grew since the last analysis); a rejected block falls back
    to FullEvd with a new rank analysis (the policy's rule). The calls are those of
    SDPSolver::large_lobpcg_project of Milestone 1.

*/

#ifndef CUADMM_CONE_PSD_LARGE_PROJECTORS_LOBPCG_PROJECTOR_H
#define CUADMM_CONE_PSD_LARGE_PROJECTORS_LOBPCG_PROJECTOR_H

#include "cone/psd/large/large_block_state.h"

class SDPSolver;

namespace cuadmm
{
namespace cone
{
namespace psd
{

struct LobpcgConfig
{
    int max_iter = 20;  // psd.lowrank.max_iter
    double tol = 1e-5;  // psd.lowrank.tol (absolute; DECISIONS 45)
};

class Lobpcg
{
public:
    Lobpcg(const LobpcgConfig &config, SDPSolver &engine);
    /// The projection into large_mat_P; false when the result was rejected (the block is left as it was).
    bool project(const LargeBlockJob &job);
    const LobpcgConfig &config() const { return config_; }

private:
    LobpcgConfig config_;
    SDPSolver &engine_;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_PROJECTORS_LOBPCG_PROJECTOR_H
