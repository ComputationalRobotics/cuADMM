/*

    cone/psd/psd_cone.h

    PsdCone (plan Section 10): all PSD blocks of the problem, projected by tier: the small and medium blocks on
    the resident batches of BatchedTier (JacobiBatchedEvd; BatchedEvd or StreamedEvd on the pool's medium
    streams), the large blocks one at a time through LargeTier's arena region with the projector that
    LargeBlockPolicy selects for the iteration (FullEvd, FullEvd + RankAnalyzer, Lobpcg, CompositeFilter during
    the warm-up). Since Phase 6b the cone owns the tiers' memory (WorkspaceArena regions requested at
    construction, bound after the owner's commit) and runs on the StreamPool; the engine keeps the block
    structure, the per-block scalars of the rank analysis and the counters (DECISIONS 46).

    The large-block schedule depends on the iteration: AdmmSolver calls begin_iteration(k, eta) through the
    product before the cone step of iteration k (the warm-up switch k_sw, the periodic rank analysis).

*/

#ifndef CUADMM_CONE_PSD_PSD_CONE_H
#define CUADMM_CONE_PSD_PSD_CONE_H

#include <string>
#include <vector>

#include "cone/cone.h"
#include "cone/psd/batched/batched_tier.h"
#include "cone/psd/large/large_block_policy.h"
#include "cone/psd/large/large_block_state.h"
#include "cone/psd/large/large_tier.h"
#include "cone/psd/large/projectors/composite_filter.h"
#include "cone/psd/large/projectors/full_evd.h"
#include "cone/psd/large/projectors/lobpcg_projector.h"
#include "cone/psd/large/rank_analyzer.h"
#include "core/stream_pool.h"
#include "core/workspace_arena.h"
#include "cuadmm/options.h"

class SDPSolver;

namespace cuadmm
{
namespace cone
{

/// The configuration of the PSD cone: the policy, the rank analysis, LOBPCG and the composite filter (from the
/// psd.* options and the seed), the final products and the batched-tier settings.
struct PsdConeConfig
{
    psd::LargeBlockPolicyConfig policy;
    psd::RankAnalyzerConfig rank;
    psd::LobpcgConfig lobpcg;
    psd::CompositeFilterConfig composite;
    bool gemm_compat = false;         // psd.large.final_product = gemm_compat (syrk otherwise)
    bool batched_gemm_compat = false; // psd.batched.final_product = gemm_compat (scaled_gemm otherwise)
    int medium_batch_min = 2;         // psd.medium_batch_min
    int medium_max_batch = 0;         // psd.medium.max_batch
    double jacobi_tol = 1e-16;        // psd.jacobi_tol
    int jacobi_max_sweeps = 15;       // psd.jacobi_max_sweeps
    static PsdConeConfig from_options(const PsdOptions &o, unsigned long long seed);
};

class PsdCone final : public Cone
{
public:
    PsdCone(std::vector<BlockSlice> blocks, SDPSolver &engine, const PsdConeConfig &config, core::WorkspaceArena &arena, core::StreamPool &pool);
    ConeKind kind() const override { return ConeKind::Psd; }
    std::string_view name() const override { return "psd"; }
    std::int64_t dimension() const override { return dimension_; }
    bool is_self_dual() const override { return true; }
    const std::vector<BlockSlice> &blocks() const override { return blocks_; }

    /// The beginning of iteration k (1-based) with the KKT residual eta of the current iterate: the policy
    /// decides the warm-up switch; returns true at the iteration that switches.
    bool begin_iteration(int k, double eta, bool verbose);
    /// The iteration of the next project_dual (1-based); begin_iteration sets it too.
    void set_iteration(int k) { iteration_ = k; }
    void project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, core::ExecContext &ctx) override;
    // project_primal: Moreau (the cone is self-dual: Pi_K = Pi_K*; the default of Cone applies)
    ConeMeasures measure(core::DeviceSpan<const double> v, Side side, core::ExecContext &ctx) override;
    ScalingCompatibility scaling_compatibility() const override { return ScalingCompatibility::Congruence; }
    const HostReferenceMeasure &host_reference() const override;

    const PsdConeConfig &config() const { return config_; }
    const psd::LargeBlockPolicy &policy() const { return policy_; }
    /// The strategy each large block got in the last cone step (empty before the first).
    const std::vector<psd::LargeStrategy> &last_strategies() const { return strategies_; }
    /// The job of large block idx (its size-group position and its state view).
    psd::LargeBlockJob job(int idx);
    psd::LargeTier &large_tier() { return large_; }
    psd::BatchedTier &medium_tier() { return medium_; }
    psd::BatchedTier &small_tier() { return small_; }
    psd::FullEvd &full_evd() { return full_; }
    psd::Lobpcg &lobpcg() { return lobpcg_; }
    psd::CompositeFilter &composite_filter() { return composite_; }
    psd::RankAnalyzer &rank_analyzer() { return analyzer_; }
    /// Binds the tiers to the committed arena (project_dual and measure do it on first use).
    void bind();
    /// The memory of the tiers, one line each, for the setup print.
    std::string memory_text() const;

private:
    void project_tiers(const double *z, double *s);
    std::vector<BlockSlice> blocks_;
    SDPSolver &engine_;
    PsdConeConfig config_;
    psd::LargeBlockPolicy policy_;
    psd::LargeTier large_;
    psd::RankAnalyzer analyzer_;
    psd::FullEvd full_;
    psd::Lobpcg lobpcg_;
    psd::CompositeFilter composite_;
    psd::BatchedTier medium_;
    psd::BatchedTier small_;
    std::vector<psd::LargeStrategy> strategies_;
    std::int64_t dimension_ = 0;
    int iteration_ = 1;
    bool verbose_ = true;
};

} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_PSD_CONE_H
