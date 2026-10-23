/*

    cone/psd/psd_cone.h

    PsdCone (plan Section 10): all PSD blocks of the problem. In Phase 5 it wraps the three tiers of the ported
    engine unchanged (small: batched Jacobi EVD; medium: batched or streamed EVD; large: exact EVD, LOBPCG, or the
    FP16 composite filter during the warm-up), on the engine's buffers and maps; the strategy classes of Phase 6a
    and the memory model of Phase 6b replace the internals behind the same interface.

    The large-block schedule depends on the iteration (the warm-up switch, the periodic rank analysis): the product
    sets it with set_iteration(k) before the cone step.

*/

#ifndef CUADMM_CONE_PSD_PSD_CONE_H
#define CUADMM_CONE_PSD_PSD_CONE_H

#include "cone/cone.h"

class SDPSolver;

namespace cuadmm
{
namespace cone
{

class PsdCone final : public Cone
{
public:
    PsdCone(std::vector<BlockSlice> blocks, SDPSolver &engine);
    ConeKind kind() const override { return ConeKind::Psd; }
    std::string_view name() const override { return "psd"; }
    std::int64_t dimension() const override { return dimension_; }
    bool is_self_dual() const override { return true; }
    const std::vector<BlockSlice> &blocks() const override { return blocks_; }
    /// The iteration of the next project_dual (1-based; the large-block schedule of the engine).
    void set_iteration(int k) { iteration_ = k; }
    void project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, core::ExecContext &ctx) override;
    // project_primal: Moreau (the cone is self-dual: Pi_K = Pi_K*; the default of Cone applies)
    ConeMeasures measure(core::DeviceSpan<const double> v, Side side, core::ExecContext &ctx) override;
    ScalingCompatibility scaling_compatibility() const override { return ScalingCompatibility::Congruence; }
    const HostReferenceMeasure &host_reference() const override;

private:
    void project_tiers(int iter);
    std::vector<BlockSlice> blocks_;
    SDPSolver &engine_;
    std::int64_t dimension_ = 0;
    int iteration_ = 1;
};

} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_PSD_CONE_H
