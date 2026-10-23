/*

    cone/cone_registry.cc

*/

#include "cone/cone_registry.h"

#include <stdexcept>

#include "cone/free_cone.h"
#include "cone/nonnegative_cone.h"
#include "cone/psd/psd_cone.h"

namespace cuadmm
{
namespace cone
{

ConeRegistry::ConeRegistry()
{
    table_.push_back({'s', ConeKind::Psd, "psd", true, "",
                      [](const std::vector<BlockSlice> &blocks, const ConeBuildContext &ctx) -> std::unique_ptr<Cone>
                      {
                          if (ctx.engine == nullptr)
                              throw std::logic_error("ConeRegistry: the PSD cone of Phase 5 needs the engine (ConeBuildContext::engine)");
                          return std::make_unique<PsdCone>(blocks, *ctx.engine, ctx.psd != nullptr ? *ctx.psd : PsdConeConfig{});
                      }});
    table_.push_back({'l', ConeKind::Nonnegative, "nonnegative", true, "",
                      [](const std::vector<BlockSlice> &blocks, const ConeBuildContext &) -> std::unique_ptr<Cone>
                      { return std::make_unique<NonnegativeCone>(blocks); }});
    table_.push_back({'u', ConeKind::Free, "free", true, "",
                      [](const std::vector<BlockSlice> &blocks, const ConeBuildContext &) -> std::unique_ptr<Cone>
                      { return std::make_unique<FreeCone>(blocks); }});
    table_.push_back({'q', ConeKind::SecondOrder, "second_order", false, "second-order cones come with Phase 9 (plan Section 8.4)", nullptr});
    table_.push_back({'r', ConeKind::RotatedSecondOrder, "rotated_second_order", false, "rotated second-order cones come with Phase 9", nullptr});
    table_.push_back({'e', ConeKind::Exponential, "exponential", false, "exponential cones are not implemented (plan Section 8.4)", nullptr});
    table_.push_back({'p', ConeKind::Power, "power", false, "power cones are not implemented (plan Section 8.4)", nullptr});
}

const ConeRegistry &ConeRegistry::instance()
{
    static const ConeRegistry registry;
    return registry;
}

const ConeRegistration &ConeRegistry::by_code(char code) const
{
    for (const ConeRegistration &r : table_)
        if (r.code == code)
            return r;
    throw std::invalid_argument(std::string("unknown cone code '") + code + "'");
}

const ConeRegistration &ConeRegistry::by_kind(ConeKind kind) const
{
    for (const ConeRegistration &r : table_)
        if (r.kind == kind)
            return r;
    throw std::invalid_argument("unknown cone kind");
}

std::vector<BlockSlice> ConeRegistry::slices(const std::vector<ConeSpec> &cones)
{
    std::vector<BlockSlice> out;
    std::int64_t offset = 0;
    for (std::size_t b = 0; b < cones.size(); b++)
    {
        BlockSlice s;
        s.block = static_cast<int>(b);
        s.size = static_cast<int>(cones[b].size);
        s.offset = offset;
        s.length = cones[b].kind == ConeKind::Psd ? cones[b].size * (cones[b].size + 1) / 2 : cones[b].size;
        out.push_back(s);
        offset += s.length;
    }
    return out;
}

std::vector<std::unique_ptr<Cone>> ConeRegistry::build(const std::vector<ConeSpec> &cones, const ConeBuildContext &ctx) const
{
    const std::vector<BlockSlice> all = slices(cones);
    for (const ConeSpec &c : cones)
    {
        const ConeRegistration &r = by_kind(c.kind);
        if (!r.supported)
            throw std::invalid_argument(r.message);
    }
    std::vector<std::unique_ptr<Cone>> out;
    for (ConeKind kind : {ConeKind::Psd, ConeKind::Nonnegative, ConeKind::Free, ConeKind::SecondOrder, ConeKind::RotatedSecondOrder,
                          ConeKind::Exponential, ConeKind::Power})
    {
        std::vector<BlockSlice> mine;
        for (std::size_t b = 0; b < cones.size(); b++)
            if (cones[b].kind == kind)
                mine.push_back(all[b]);
        if (mine.empty())
            continue;
        const ConeRegistration &r = by_kind(kind);
        if (!r.supported)
            throw std::invalid_argument(r.message);
        out.push_back(r.factory(mine, ctx));
    }
    return out;
}

} // namespace cone
} // namespace cuadmm
