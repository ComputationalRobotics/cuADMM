/*

    termination/certificate.cu

*/

#include "termination/certificate.h"

#include <cmath>
#include <limits>

namespace cuadmm
{
namespace termination
{

CertificateResult Certificate::measure(cone::ConeProduct &cones, core::DeviceSpan<const double> Z_scaled, double Cscale, core::ExecContext &ctx)
{
    const cone::ProductMeasures m = cones.measure(Z_scaled, cone::Side::Dual, ctx);
    CertificateResult r;
    r.dual_block_min = m.block_min;
    r.dual_cone_coef = m.bound_coef;
    for (std::size_t b = 0; b < m.kind.size(); b++)
    {
        r.dual_block_min[b] *= Cscale;
        r.dual_cone_coef[b] *= Cscale;
        r.coef_sum += r.dual_cone_coef[b];
    }
    r.lower_bound = std::numeric_limits<double>::quiet_NaN();
    return r;
}

void Certificate::evaluate(CertificateResult &r, double dobj, const std::vector<double> &bounds, const std::vector<ConeKind> &kinds)
{
    r.missing_bounds.clear();
    r.conditional_on_free_bounds = false;
    r.lower_bound = std::numeric_limits<double>::quiet_NaN();
    if (bounds.empty())
        return;
    if (bounds.size() != r.dual_cone_coef.size())
        throw std::invalid_argument("certificate: " + std::to_string(bounds.size()) + " bounds for " + std::to_string(r.dual_cone_coef.size()) + " blocks");
    double bound = dobj;
    for (std::size_t b = 0; b < bounds.size(); b++)
    {
        if (!(bounds[b] >= 0.0)) // NaN or negative: no bound for this block
        {
            r.missing_bounds.push_back(static_cast<int>(b));
            continue;
        }
        if (kinds[b] == ConeKind::Free)
            r.conditional_on_free_bounds = true;
        bound += bounds[b] * r.dual_cone_coef[b];
    }
    if (r.missing_bounds.empty())
        r.lower_bound = bound;
    else
        r.note = "unavailable: no bound for " + std::to_string(r.missing_bounds.size()) + " block(s)";
}

std::vector<double> Certificate::bounds_from_options(const CertificateOptions &o, const std::vector<ConeKind> &kinds, const std::vector<int> &sizes,
                                                     bool scaled_variant)
{
    std::vector<double> bounds;
    if (scaled_variant)
    {
        if (o.trace_bound_scale < 0)
            return bounds;
        bounds.resize(kinds.size());
        for (std::size_t b = 0; b < kinds.size(); b++)
            bounds[b] = kinds[b] == ConeKind::Psd ? o.trace_bound_scale * sizes[b] : o.trace_bound_scale;
        return bounds;
    }
    if (!o.block_bounds.empty())
        return o.block_bounds;
    if (o.trace_bound >= 0)
        bounds.assign(kinds.size(), o.trace_bound);
    return bounds;
}

} // namespace termination
} // namespace cuadmm
