/*

    cone/nonnegative_cone.h

    NonnegativeCone (plan Section 8.2): K = K* = R^n_+. project_dual: s = max(z, 0) on its blocks; measure: block_min
    = min_i v_i, bound_coef = sum_i min(0, v_i), violation = max(0, -min_i v_i).

*/

#ifndef CUADMM_CONE_NONNEGATIVE_CONE_H
#define CUADMM_CONE_NONNEGATIVE_CONE_H

#include "cone/cone.h"
#include "cone/slice_table.h"

namespace cuadmm
{
namespace cone
{

class NonnegativeCone final : public Cone
{
public:
    explicit NonnegativeCone(std::vector<BlockSlice> blocks);
    ConeKind kind() const override { return ConeKind::Nonnegative; }
    std::string_view name() const override { return "nonnegative"; }
    std::int64_t dimension() const override { return table_.total(); }
    bool is_self_dual() const override { return true; }
    const std::vector<BlockSlice> &blocks() const override { return blocks_; }
    void project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, core::ExecContext &ctx) override;
    void project_primal(core::DeviceSpan<const double> z, core::DeviceSpan<double> x, core::ExecContext &ctx) override;
    ConeMeasures measure(core::DeviceSpan<const double> v, Side side, core::ExecContext &ctx) override;
    ScalingCompatibility scaling_compatibility() const override { return ScalingCompatibility::AnyPositiveDiagonal; }
    const HostReferenceMeasure &host_reference() const override;

private:
    std::vector<BlockSlice> blocks_;
    SliceTable table_;
};

} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_NONNEGATIVE_CONE_H
