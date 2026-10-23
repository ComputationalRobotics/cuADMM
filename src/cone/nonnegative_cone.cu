/*

    cone/nonnegative_cone.cu

*/

#include "cone/nonnegative_cone.h"

#include <algorithm>

namespace cuadmm
{
namespace cone
{

namespace
{
double nonneg_block_min(const double *v, int n, std::vector<double> &, std::vector<int> &)
{
    double m = v[0];
    for (int k = 1; k < n; k++)
        m = std::min(m, v[k]);
    return m;
}
const HostReferenceMeasure kNonnegReference{&nonneg_block_min};
} // namespace

NonnegativeCone::NonnegativeCone(std::vector<BlockSlice> blocks) : blocks_(std::move(blocks)), table_(blocks_) {}

void NonnegativeCone::project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, core::ExecContext &ctx)
{
    slice_apply(table_, z.data, s.data, SLICE_RELU, ctx.stream);
}

void NonnegativeCone::project_primal(core::DeviceSpan<const double> z, core::DeviceSpan<double> x, core::ExecContext &ctx)
{
    slice_apply(table_, z.data, x.data, SLICE_RELU, ctx.stream); // self-dual
}

ConeMeasures NonnegativeCone::measure(core::DeviceSpan<const double> v, Side, core::ExecContext &ctx)
{
    const std::vector<double> h = table_.gather_host(v.data, ctx.stream);
    ConeMeasures m;
    m.block_min.assign(blocks_.size(), 0.0);
    m.bound_coef.assign(blocks_.size(), 0.0);
    m.violation.assign(blocks_.size(), 0.0);
    std::int64_t pos = 0;
    for (std::size_t b = 0; b < blocks_.size(); b++)
    {
        double entry_min = h[pos], negative_sum = 0.0;
        for (std::int64_t k = 0; k < blocks_[b].length; k++)
        {
            entry_min = std::min(entry_min, h[pos + k]);
            negative_sum += std::min(0.0, h[pos + k]);
        }
        m.block_min[b] = entry_min;
        m.bound_coef[b] = negative_sum;
        m.violation[b] = std::max(0.0, -entry_min);
        pos += blocks_[b].length;
    }
    return m;
}

const HostReferenceMeasure &NonnegativeCone::host_reference() const { return kNonnegReference; }

} // namespace cone
} // namespace cuadmm
