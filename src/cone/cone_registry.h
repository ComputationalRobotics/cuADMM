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

class SDPSolver; // the ported engine (the block structure and the per-block scalars of the PSD tiers, Milestone 2)

namespace cuadmm
{
namespace core
{
class WorkspaceArena;
class StreamPool;
} // namespace core
} // namespace cuadmm

namespace cuadmm
{
namespace cone
{

struct PsdConeConfig;

/// What a factory receives: the blocks of one kind, the length of the svec vector, the engine that holds the block
/// structure and the per-block scalars (Milestone 2), the PSD cone's configuration (null: the defaults of the
/// options, with the low-rank path following the engine's use_lobpcg), and the arena and stream pool the PSD cone's
/// tiers run on (Phase 6b; the owner commits the arena after building the product).
struct ConeBuildContext
{
    std::int64_t vec_len = 0;
    SDPSolver *engine = nullptr;
    const PsdConeConfig *psd = nullptr;
    core::WorkspaceArena *arena = nullptr;
    core::StreamPool *pool = nullptr;
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
