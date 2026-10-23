/*

    cone/cone.h

    The cone layer (plan Section 8): one Cone object per cone kind, holding every block of that kind of the problem,
    with the dual-cone projection of the cone step (3), the optional primal projection (Moreau), the violation and
    certificate measures of a vector, and the host reference measure of the independent validator.

    Milestone 1 stores the svec vector in input order, so the blocks of one kind are a set of ranges of the vector
    (plan Phase 8, deferred, makes them one contiguous slice). Until then every cone receives the full vectors and
    its block table; it reads and writes only its own entries.

*/

#ifndef CUADMM_CONE_CONE_H
#define CUADMM_CONE_CONE_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/device_buffer.h"
#include "core/gpu_context.h"
#include "cuadmm/problem.h" // ConeKind

namespace cuadmm
{
namespace cone
{

/// Which iterate a measure describes.
enum class Side { Primal, Dual };

/// One block of a cone in the svec vector.
struct BlockSlice
{
    int block = 0;        // index of the block in the problem (input order)
    int size = 0;         // n: matrix side ('s'), number of entries ('l', 'u')
    std::int64_t offset = 0; // first entry of the block in the svec vector
    std::int64_t length = 0; // n (n + 1) / 2 for 's', n otherwise
};

/// Per-block measures of a vector v (plan Section 8.1, the certificate of Section 7.4 / B24): block_min and the
/// coefficient c_b such that <v_b, X_b> >= R_b c_b for every X_b in the primal cone bounded by R_b (trace bound for
/// PSD blocks, entrywise bound for 'l' and 'u' blocks): min(0, lambda_min) ('s'), sum_i min(0, v_i) ('l'),
/// -sum_i |v_i| ('u'). violation is the cone violation of v on the block: max(0, -lambda_min) ('s'),
/// max(0, -min_i v_i) ('l'); for the dual cone {0} of a free block max_i |v_i|.
struct ConeMeasures
{
    std::vector<double> block_min;  // per block of this cone, in the cone's block order
    std::vector<double> bound_coef; // per block
    std::vector<double> violation;  // per block
};

enum class ScalingCompatibility { AnyPositiveDiagonal, Congruence, UniformFactor, None };

/// The host reference measure of the validator: on an svec block given on the host, the smallest eigenvalue
/// ('s'), the smallest entry ('l'), or the largest absolute entry ('u').
struct HostReferenceMeasure
{
    double (*block_min)(const double *v, int n, std::vector<double> &work, std::vector<int> &iwork) = nullptr;
};

class Cone
{
public:
    virtual ~Cone() = default;
    virtual ConeKind kind() const = 0;
    virtual std::string_view name() const = 0;
    /// Length of this cone's slice: the sum of the lengths of its blocks.
    virtual std::int64_t dimension() const = 0;
    virtual bool is_self_dual() const = 0;
    virtual const std::vector<BlockSlice> &blocks() const = 0;
    /// Cone step (3): s = Pi_{K*}(z) on this cone's blocks (the other entries of s are left untouched).
    virtual void project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, core::ExecContext &ctx) = 0;
    /// Pi_K(z) on this cone's blocks; the default uses Moreau: Pi_K(z) = z + Pi_{K*}(-z).
    virtual void project_primal(core::DeviceSpan<const double> z, core::DeviceSpan<double> x, core::ExecContext &ctx);
    /// The measures of v on this cone's blocks (Side::Primal: v is an X; Side::Dual: v is an S or C - A^T y).
    virtual ConeMeasures measure(core::DeviceSpan<const double> v, Side side, core::ExecContext &ctx) = 0;
    virtual ScalingCompatibility scaling_compatibility() const = 0;
    virtual const HostReferenceMeasure &host_reference() const = 0;
};

} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_CONE_H
