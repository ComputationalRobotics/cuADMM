/*

    cone/free_cone.cu

*/

#include "cone/free_cone.h"

#include <cmath>

namespace cuadmm
{
namespace cone
{

namespace
{
double free_block_min(const double *v, int n, std::vector<double> &, std::vector<int> &)
{
    double abs_max = 0.0;
    for (int k = 0; k < n; k++)
        abs_max = std::max(abs_max, std::abs(v[k]));
    return abs_max;
}
const HostReferenceMeasure kFreeReference{&free_block_min};
} // namespace

FreeCone::FreeCone(std::vector<BlockSlice> blocks) : blocks_(std::move(blocks)), table_(blocks_) {}

void FreeCone::project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, core::ExecContext &ctx)
{
    slice_apply(table_, z.data, s.data, SLICE_ZERO, ctx.stream); // S_u = 0
}

void FreeCone::project_primal(core::DeviceSpan<const double> z, core::DeviceSpan<double> x, core::ExecContext &ctx)
{
    if (z.data != x.data)
        slice_apply(table_, z.data, x.data, SLICE_COPY, ctx.stream); // identity
}

ConeMeasures FreeCone::measure(core::DeviceSpan<const double> v, Side side, core::ExecContext &ctx)
{
    // the loops of SDPSolver::cone_measures of main, on host copies of the blocks (same arithmetic, same order)
    const std::vector<double> h = table_.gather_host(v.data, ctx.stream);
    ConeMeasures m;
    m.block_min.assign(blocks_.size(), 0.0);
    m.bound_coef.assign(blocks_.size(), 0.0);
    m.violation.assign(blocks_.size(), 0.0);
    std::int64_t pos = 0;
    for (std::size_t b = 0; b < blocks_.size(); b++)
    {
        double abs_max = 0.0, abs_sum = 0.0;
        for (std::int64_t k = 0; k < blocks_[b].length; k++)
        {
            abs_max = std::max(abs_max, std::abs(h[pos + k]));
            abs_sum += std::abs(h[pos + k]);
        }
        m.block_min[b] = abs_max;
        m.bound_coef[b] = -abs_sum;
        m.violation[b] = side == Side::Dual ? abs_max : 0.0; // S_u must be 0; X_u is free
        pos += blocks_[b].length;
    }
    return m;
}

const HostReferenceMeasure &FreeCone::host_reference() const { return kFreeReference; }

} // namespace cone
} // namespace cuadmm
