/*

    cone/psd/large/large_tier.h

    LargeTier (plan Sections 10.4-10.5, Phase 6b): the large PSD blocks, one at a time through one region of the
    WorkspaceArena sized for the largest block n_max:

        M:   n_max^2 doubles, the dense input block (unpacked from its svec slice); the EVD overwrites it with Q
        W:   the largest workspace of any strategy at n_max: the cusolverDnXsyevd workspace (about 4 n^2), or the
             final product P (n^2; plus the n^2 temporary of gemm_compat), or P with the LOBPCG buffers, or the
             FP32/FP16 buffers of the composite filter with the Lanczos vectors (B7: nothing is allocated per call)
        eig: n_max eigenvalues; info: one cuSOLVER status
        per block: the LOBPCG warm start V_b (n_b x k_alloc_b) and D_b (k_alloc_b), k_alloc = ceil(1.2 rho n_b)

    Nothing else persists: the result is packed into the cone-step output s from the upper triangle of P (or
    added to the input slice on the negative side). The per-block scalars of the rank analysis (mode, k, rank
    tolerance, rank counters) stay in the engine in Milestone 2 (DECISIONS 46); LargeBlockState views them.
    The tier runs on the pool's large role (the legacy default stream in Milestone 2) with its handles.

*/

#ifndef CUADMM_CONE_PSD_LARGE_LARGE_TIER_H
#define CUADMM_CONE_PSD_LARGE_LARGE_TIER_H

#include <cstddef>
#include <string>
#include <vector>

#include "cone/psd/large/large_block_state.h"
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

struct LargeTierConfig
{
    bool gemm_compat = false;         // psd.large.final_product = gemm_compat (today's product with an n^2 temporary)
    bool lowrank = true;              // the warm starts V_b, D_b exist (psd.lowrank.enabled and the engine's use_lobpcg)
    double lowrank_ratio = 0.03;      // k_alloc = ceil(n * ratio * relaxation)
    double lowrank_relaxation = 1.2;
    bool composite = false;           // the FP32/FP16 buffers are part of W (some block gets the warm-up)
    int lanczos_steps = 20;           // psd.composite.lanczos_steps (the Lanczos vectors in W)
};

class LargeTier
{
public:
    /// Requests the regions from the arena (the owner commits it); the engine provides the block structure.
    LargeTier(SDPSolver &engine, const LargeTierConfig &config, core::WorkspaceArena &arena, core::StreamPool &pool);

    int count() const { return static_cast<int>(refs_.size()); }
    int size(int idx) const { return refs_[idx].n; }
    int n_max() const { return n_max_; }
    int k_max() const { return k_max_; }
    const BlockRef &ref(int idx) const { return refs_[idx]; }
    int size_index(int idx) const { return size_index_[idx]; }
    int index_in_group(int idx) const { return index_in_group_[idx]; }
    const LargeTierConfig &config() const { return config_; }
    core::ExecContext ctx() const { return ctx_; }
    SingleEigParameter &eig_param() { return eig_param_; }
    SDPSolver &engine() { return engine_; }
    /// The stopping tolerance of the current solve (the rank tolerance floor).
    double engine_stop_tol() const;

    /// Resolves the regions after the arena's commit (idempotent; throws if the arena is not committed).
    void bind();
    double *M() { return M_; }
    double *W() { return W_; }
    std::size_t W_bytes() const { return W_bytes_; }
    double *eigenvalues() { return eig_; }
    int *info() { return info_; }
    double *V(int idx) { return V_[idx]; }
    double *D(int idx) { return D_[idx]; }
    int k_alloc(int idx) const { return k_alloc_[idx]; }
    std::size_t evd_workspace_bytes() const { return evd_bytes_; }

    /// M <- the dense block idx of z; s[slice of idx] <- svec of the lower triangle of P (+ the slice of z_add).
    void unpack(int idx, const double *z);
    void pack(int idx, const double *P, double *s, const double *z_add);
    /// cusolverDnXsyevd of block idx in M (M <- Q, eigenvalues() <- lambda increasing); throws on failure.
    void decompose(int idx, const char *what);
    /// The state view of block idx (the engine's scalars, the tier's warm start).
    LargeBlockState state(int idx);
    /// The memory of the tier for the setup print.
    std::string memory_text() const;

private:
    SDPSolver &engine_;
    LargeTierConfig config_;
    core::WorkspaceArena &arena_;
    core::ExecContext ctx_;
    std::vector<BlockRef> refs_;
    std::vector<int> size_index_, index_in_group_, k_alloc_;
    int n_max_ = 0, k_max_ = 0;
    SingleEigParameter eig_param_;
    std::size_t evd_bytes_ = 0, evd_host_bytes_ = 0, W_bytes_ = 0;
    std::vector<unsigned char> host_ws_;
    int h_M_ = -1, h_W_ = -1, h_eig_ = -1, h_info_ = -1;
    std::vector<int> h_V_, h_D_;
    double *M_ = nullptr, *W_ = nullptr, *eig_ = nullptr;
    int *info_ = nullptr;
    std::vector<double *> V_, D_;
    bool bound_ = false;
    int dummy_rank_[2] = {0, 0};
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_LARGE_TIER_H
