/*

    cone/cone_product.cu

*/

#include "cone/cone_product.h"

#include <algorithm>
#include <stdexcept>

#include "cone/psd/psd_cone.h"
#include "core/check.h"

namespace cuadmm
{
namespace cone
{

double ProductMeasures::max_violation(bool psd, bool nonnegative, bool free) const
{
    double m = 0.0;
    for (std::size_t b = 0; b < kind.size(); b++)
    {
        const bool take = (kind[b] == ConeKind::Psd && psd) || (kind[b] == ConeKind::Nonnegative && nonnegative) || (kind[b] == ConeKind::Free && free);
        if (take)
            m = std::max(m, violation[b]);
    }
    return m;
}

double ProductMeasures::min_block_min_psd_nonneg() const
{
    double m = 0.0;
    for (std::size_t b = 0; b < kind.size(); b++)
        if (kind[b] != ConeKind::Free)
            m = std::min(m, block_min[b]);
    return m;
}

ConeProduct::ConeProduct(const std::vector<ConeSpec> &cones, const ConeBuildContext &ctx) { build(cones, ctx); }

ConeProduct::ConeProduct(const std::vector<char> &types, const std::vector<int> &sizes, const ConeBuildContext &ctx)
{
    if (types.size() != sizes.size())
        throw std::invalid_argument("ConeProduct: types and sizes differ in length");
    std::vector<ConeSpec> specs(types.size());
    for (std::size_t b = 0; b < types.size(); b++)
    {
        specs[b].kind = cone_kind_of(types[b]);
        specs[b].size = sizes[b];
    }
    build(specs, ctx);
}

void ConeProduct::build(const std::vector<ConeSpec> &specs, const ConeBuildContext &ctx)
{
    blocks_ = ConeRegistry::slices(specs);
    kinds_.resize(specs.size());
    vec_len_ = 0;
    for (std::size_t b = 0; b < specs.size(); b++)
    {
        kinds_[b] = specs[b].kind;
        vec_len_ += blocks_[b].length;
    }
    if (ctx.vec_len != 0 && ctx.vec_len != vec_len_)
        throw std::invalid_argument("ConeProduct: the blocks give vec_len " + std::to_string(vec_len_) + ", the caller " + std::to_string(ctx.vec_len));
    cones_ = ConeRegistry::instance().build(specs, ctx);
}

bool ConeProduct::has(ConeKind kind) const
{
    for (const auto &c : cones_)
        if (c->kind() == kind)
            return true;
    return false;
}

Cone *ConeProduct::cone(ConeKind kind)
{
    for (auto &c : cones_)
        if (c->kind() == kind)
            return c.get();
    return nullptr;
}

void ConeProduct::project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, int k, core::ExecContext &ctx)
{
    if (z.size != static_cast<std::size_t>(vec_len_) || s.size != static_cast<std::size_t>(vec_len_))
        throw std::invalid_argument("ConeProduct::project_dual: vector length mismatch");
    for (auto &c : cones_)
    {
        if (c->kind() == ConeKind::Psd)
            static_cast<PsdCone *>(c.get())->set_iteration(k);
        c->project_dual(z, s, ctx);
    }
}

void ConeProduct::project_primal(core::DeviceSpan<const double> z, core::DeviceSpan<double> x, core::ExecContext &ctx)
{
    if (z.size != static_cast<std::size_t>(vec_len_) || x.size != static_cast<std::size_t>(vec_len_))
        throw std::invalid_argument("ConeProduct::project_primal: vector length mismatch");
    for (auto &c : cones_)
        c->project_primal(z, x, ctx);
}

ProductMeasures ConeProduct::measure(core::DeviceSpan<const double> v, Side side, core::ExecContext &ctx)
{
    if (v.size != static_cast<std::size_t>(vec_len_))
        throw std::invalid_argument("ConeProduct::measure: vector length mismatch");
    ProductMeasures out;
    const std::size_t nb = blocks_.size();
    out.kind = kinds_;
    out.block_min.assign(nb, 0.0);
    out.bound_coef.assign(nb, 0.0);
    out.violation.assign(nb, 0.0);
    for (auto &c : cones_)
    {
        const ConeMeasures m = c->measure(v, side, ctx);
        const std::vector<BlockSlice> &bl = c->blocks();
        for (std::size_t i = 0; i < bl.size(); i++)
        {
            out.block_min[bl[i].block] = m.block_min[i];
            out.bound_coef[bl[i].block] = m.bound_coef[i];
            out.violation[bl[i].block] = m.violation[i];
        }
    }
    return out;
}

} // namespace cone
} // namespace cuadmm
