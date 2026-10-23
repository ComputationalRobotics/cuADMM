/*

    cone/psd/batched/batched_tier.h

    BatchedTier (plan Section 10.5 "Small and medium tiers", Phase 6b): the small tier (n <= psd.small_max,
    cusolverDnDsyevjBatched, one call per size group: JacobiBatchedEvd) or the medium tier (psd.small_max < n <=
    psd.medium_max: cusolverDnXsyevBatched in chunks for the size groups with at least psd.medium_batch_min
    blocks, BatchedEvd; cusolverDnXsyevd per block on the pool's medium streams otherwise, StreamedEvd), on
    resident batch buffers of the WorkspaceArena:

        mats: sum n^2 (the blocks, overwritten by their eigenvectors)   W: sum n (eigenvalues)
        P:    sum n^2 (the results)                                     tmp: sum n^2, gemm_compat only
        the EVD workspaces: one per medium stream (sized for the largest medium block), one for the batched
        chunks (sized for the largest chunk), one per small size group (B20: not one per block)

    Final product: scaled_gemm (default since Phase 6b: the columns of Q scaled in place by sqrt(max(lambda, 0)),
    P = B B^T with one strided-batched GEMM per group; the third buffer dropped, B19) or gemm_compat (today's
    Q max(Lambda, 0) into tmp and the GEMM: bit for bit the reference). The termination measures use
    decompose() with the streamed EVD for every medium block, as the engine's cone_measures did.

*/

#ifndef CUADMM_CONE_PSD_BATCHED_BATCHED_TIER_H
#define CUADMM_CONE_PSD_BATCHED_BATCHED_TIER_H

#include <cstddef>
#include <string>
#include <vector>

#include "cone/psd/psd_kernels.h"
#include "core/stream_pool.h"
#include "core/workspace_arena.h"
#include "cuadmm_legacy/cusolver.h"

class SDPSolver;

namespace cuadmm
{
namespace cone
{
namespace psd
{

enum class TierKind
{
    Small,
    Medium
};

struct BatchedTierConfig
{
    bool gemm_compat = false;    // psd.batched.final_product = gemm_compat
    int medium_batch_min = 2;    // psd.medium_batch_min (0: never batched)
    int medium_max_batch = 0;    // psd.medium.max_batch: blocks per batched call (0: the whole group)
    double jacobi_tol = 1e-16;   // psd.jacobi_tol
    int jacobi_max_sweeps = 15;  // psd.jacobi_max_sweeps
};

class BatchedTier
{
public:
    BatchedTier(SDPSolver &engine, TierKind kind, const BatchedTierConfig &config, core::WorkspaceArena &arena, core::StreamPool &pool);

    TierKind kind() const { return kind_; }
    int count() const { return static_cast<int>(refs_.size()); }
    const std::vector<BlockRef> &refs() const { return refs_; } // flat order (size group by size group)
    int group_count() const { return static_cast<int>(group_size_.size()); }
    bool group_batched(int group) const { return batched_[group] != 0; }
    const BatchedTierConfig &config() const { return config_; }

    /// Resolves the regions after the arena's commit (idempotent).
    void bind();
    /// mats <- the dense blocks of z.
    void unpack(const double *z);
    /// Eigenpairs of every block in place (mats <- Q, W <- lambda increasing); allow_batched: the batched EVD for
    /// the groups that use it (the projections) or the streamed EVD for every medium block (the measures); warn:
    /// count and report once the small blocks whose Jacobi sweeps did not reach the tolerance.
    void decompose(const char *what, bool allow_batched, bool warn);
    /// The projection of every block: unpack, decompose, the final product, pack into s.
    void project(const double *z, double *s);
    /// The eigenvalues (device, after decompose) of block idx start at w_offset(idx).
    const double *eigenvalues() const { return W_; }
    long long w_offset(int idx) const { return refs_[idx].w_offset; }
    std::size_t sum_n() const { return sum_n_; }
    std::string memory_text() const;

private:
    void decompose_small(const char *what, bool warn);
    void decompose_medium(const char *what, bool allow_batched);
    SDPSolver &engine_;
    TierKind kind_;
    BatchedTierConfig config_;
    core::WorkspaceArena &arena_;
    core::StreamPool &pool_;
    core::ExecContext ctx_; // the tier's role (main in Milestone 2)
    std::vector<BlockRef> refs_;
    std::vector<int> group_size_, group_count_, group_first_; // per size group: n, blocks, first flat index
    std::vector<char> batched_;
    std::size_t total_nn_ = 0, sum_n_ = 0;
    BlockTable table_;
    SingleEigParameter eig_param_;
    BatchEigParameter jacobi_param_;
    // workspaces
    std::vector<std::size_t> group_ws_bytes_, group_ws_offset_; // small: per size group (Jacobi)
    std::size_t stream_ws_bytes_ = 0, stream_host_bytes_ = 0;    // medium: per stream
    std::size_t chunk_ws_bytes_ = 0, chunk_host_bytes_ = 0;      // medium: the batched chunks
    std::vector<std::vector<unsigned char>> host_ws_;
    std::vector<unsigned char> chunk_host_ws_;
    int h_mats_ = -1, h_W_ = -1, h_P_ = -1, h_tmp_ = -1, h_info_ = -1, h_ws_ = -1, h_chunk_ws_ = -1;
    double *mats_ = nullptr, *W_ = nullptr, *P_ = nullptr, *tmp_ = nullptr;
    int *info_ = nullptr;
    unsigned char *ws_ = nullptr, *chunk_ws_ = nullptr;
    bool bound_ = false;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_BATCHED_BATCHED_TIER_H
