/*

    cone/cone_registry.h

    ConeRegistry (plan Section 8.1): maps the input codes of the formats ('s', 'l', 'u'; 'q' and 'r' with Phase 9)
    to cone kinds and to the factories that build the Cone objects of a problem, so that the readers, the MATLAB
    gateway, the validator and the solver never switch on cone letters themselves.

*/

#ifndef CUADMM_CONE_CONE_REGISTRY_H
#define CUADMM_CONE_CONE_REGISTRY_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cone/cone.h"

class SDPSolver; // the ported engine that still owns the PSD tiers (Phase 5; replaced in Phase 6)

namespace cuadmm
{
namespace cone
{

/// What a factory receives: the blocks of one kind, the length of the svec vector, and (Phase 5 only) the engine
/// whose tiers the PSD cone wraps.
struct ConeBuildContext
{
    std::int64_t vec_len = 0;
    SDPSolver *engine = nullptr;
};

using ConeFactory = std::function<std::unique_ptr<Cone>(const std::vector<BlockSlice> &, const ConeBuildContext &)>;

struct ConeRegistration
{
    char code = 0;
    ConeKind kind = ConeKind::Psd;
    std::string name;
    bool supported = false; // false: rejected with the phase that adds it (plan Phase 9)
    std::string message;    // the rejection message
    ConeFactory factory;
};

class ConeRegistry
{
public:
    static const ConeRegistry &instance();
    /// The registration of an input code; throws std::invalid_argument for an unknown code.
    const ConeRegistration &by_code(char code) const;
    const ConeRegistration &by_kind(ConeKind kind) const;
    /// Builds the cone objects of a problem: the blocks grouped by kind, in the order Psd, Nonnegative, Free (the
    /// order the product runs them in); throws for an unsupported kind with its message.
    std::vector<std::unique_ptr<Cone>> build(const std::vector<ConeSpec> &cones, const ConeBuildContext &ctx) const;
    /// The block slices of every block of a problem (input order), with the svec offsets.
    static std::vector<BlockSlice> slices(const std::vector<ConeSpec> &cones);

private:
    ConeRegistry();
    std::vector<ConeRegistration> table_;
};

} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_CONE_REGISTRY_H
