/*

    cone/free_cone.h

    FreeCone (plan Section 9): K = R^n, K* = {0}. project_dual writes 0 on its blocks, project_primal the identity;
    measure: block_min = max_i |v_i|, bound_coef = -sum_i |v_i| (the certificate term of a free block with the
    entrywise bound |X_u| <= R_u), violation of the dual side max_i |v_i| (S_u must be 0), of the primal side 0.

*/

#ifndef CUADMM_CONE_FREE_CONE_H
#define CUADMM_CONE_FREE_CONE_H

#include "cone/cone.h"
#include "cone/slice_table.h"

namespace cuadmm
{
namespace cone
{

class FreeCone final : public Cone
{
public:
    explicit FreeCone(std::vector<BlockSlice> blocks);
    ConeKind kind() const override { return ConeKind::Free; }
    std::string_view name() const override { return "free"; }
    std::int64_t dimension() const override { return table_.total(); }
    bool is_self_dual() const override { return false; }
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

#endif // CUADMM_CONE_FREE_CONE_H
