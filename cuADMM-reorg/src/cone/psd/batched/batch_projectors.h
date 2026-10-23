/*

    cone/psd/batched/batch_projectors.h

    The small and medium tiers (plan Section 10.5, Table "PSD projection strategies"), on the engine's resident
    batch buffers of Milestone 1 (Phase 6b changes the memory model):

    - JacobiBatchedEvd: the small blocks (n <= psd.small_max), cusolverDnDsyevjBatched, one call per size group;
      info = n + 1 (tolerance not reached) is counted and reported once per solve, not an error.
    - BatchedEvd: a medium size group with at least psd.medium_batch_min blocks, cusolverDnXsyevBatched in one
      call (CUDA >= 12.8; the engine decided per group at setup).
    - StreamedEvd: the other medium blocks, cusolverDnXsyevd per block on the engine's streams (round robin).
    - MediumTier: the medium tier as a whole: the decompositions of its groups, then max(lambda, 0), the diagonal
      scaling and one strided-batched GEMM per group (today's three-buffer product; Phase 6b scales Q in place
      and uses SYRK, within rounding).

    The decompose() entry points also serve the termination measures (the eigenvalues of X, S and C - A^T y),
    which use the streamed EVD for every medium block as the engine's cone_measures did (bit for bit).

*/

#ifndef CUADMM_CONE_PSD_BATCHED_BATCH_PROJECTORS_H
#define CUADMM_CONE_PSD_BATCHED_BATCH_PROJECTORS_H

class SDPSolver;

namespace cuadmm
{
namespace cone
{
namespace psd
{

class JacobiBatchedEvd
{
public:
    explicit JacobiBatchedEvd(SDPSolver &engine);
    /// Eigenpairs of every small block in place (small_mat <- Q, small_W <- lambda); returns the number of blocks
    /// whose Jacobi sweeps did not reach the tolerance (reported once per solve when warn is set).
    int decompose(const char *what, bool warn);
    /// The projection of every small block into small_mat_P.
    void project();

private:
    SDPSolver &engine_;
};

class BatchedEvd
{
public:
    explicit BatchedEvd(SDPSolver &engine);
    /// True when the build and the engine's setup use the batched EVD for this size group.
    bool enabled(int group) const;
    /// The decomposition of one medium size group in one batched call (info written from info_offset).
    void decompose(int group, int info_offset);

private:
    SDPSolver &engine_;
};

class StreamedEvd
{
public:
    explicit StreamedEvd(SDPSolver &engine);
    /// The decomposition of block j of medium size group `group` on stream counter % streams (info at counter).
    void decompose(int group, int j, int counter);
    /// Waits for every stream.
    void synchronize();

private:
    SDPSolver &engine_;
};

class MediumTier
{
public:
    explicit MediumTier(SDPSolver &engine);
    /// Eigenpairs of every medium block in place (medium_mat <- Q, medium_W <- lambda), batched where the engine
    /// enabled it and allow_batched is set, streamed otherwise; synchronized and checked on return.
    void decompose(const char *what, bool allow_batched);
    /// The projection of every medium block into medium_mat_P.
    void project();

private:
    SDPSolver &engine_;
    BatchedEvd batched_;
    StreamedEvd streamed_;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_BATCHED_BATCH_PROJECTORS_H
