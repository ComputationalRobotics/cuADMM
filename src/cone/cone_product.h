/*

    cone/cone_product.h

    ConeProduct (plan Section 8.1): the product cone of the problem, one Cone object per kind present (PSD,
    nonnegative, free; Phase 9 adds the rest). The cone step (3) is project_dual(); measure() gives the per-block
    measures of a vector in the problem's block order (the cone violations of X and S at termination, and the
    certificate coefficients of C - A^T y).

*/

#ifndef CUADMM_CONE_CONE_PRODUCT_H
#define CUADMM_CONE_CONE_PRODUCT_H

#include <memory>
#include <vector>

#include "cone/cone.h"
#include "cone/cone_registry.h"

namespace cuadmm
{
namespace cone
{

/// Per-block measures of a vector in the problem's block order (input order).
struct ProductMeasures
{
    std::vector<ConeKind> kind;
    std::vector<double> block_min;
    std::vector<double> bound_coef;
    std::vector<double> violation;
    /// The largest violation over the blocks of the given kinds (0 when there is none).
    double max_violation(bool psd, bool nonnegative, bool free) const;
    /// min over the 's' and 'l' blocks of block_min (today's errPSD measure); 0 when there is none.
    double min_block_min_psd_nonneg() const;
};

class ConeProduct
{
public:
    /// From the cones of a problem (the engine is the Phase 5 provider of the PSD tiers).
    ConeProduct(const std::vector<ConeSpec> &cones, const ConeBuildContext &ctx);
    /// The same from the engine's block table (types 's', 'l', 'u' and sizes).
    ConeProduct(const std::vector<char> &types, const std::vector<int> &sizes, const ConeBuildContext &ctx);

    std::int64_t vec_len() const { return vec_len_; }
    const std::vector<BlockSlice> &blocks() const { return blocks_; }
    const std::vector<ConeKind> &kinds() const { return kinds_; }
    bool has(ConeKind kind) const;
    Cone *cone(ConeKind kind);
    const std::vector<std::unique_ptr<Cone>> &cones() const { return cones_; }

    /// The beginning of iteration k (1-based) with the KKT residual eta of the current iterate: the PSD cone's
    /// large-block policy decides its warm-up switch; returns true at the iteration that switches.
    bool begin_iteration(int k, double eta, bool verbose);
    /// Cone step (3): s = Pi_{K*}(z) on every block; k is the iteration (the PSD cone's large-block schedule).
    void project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, int k, core::ExecContext &ctx);
    /// Pi_K(z) on every block (Moreau where a cone has no primal projection of its own).
    void project_primal(core::DeviceSpan<const double> z, core::DeviceSpan<double> x, core::ExecContext &ctx);
    /// The measures of v on every block.
    ProductMeasures measure(core::DeviceSpan<const double> v, Side side, core::ExecContext &ctx);

private:
    void build(const std::vector<ConeSpec> &specs, const ConeBuildContext &ctx);
    std::vector<std::unique_ptr<Cone>> cones_;
    std::vector<BlockSlice> blocks_;
    std::vector<ConeKind> kinds_;
    std::int64_t vec_len_ = 0;
};

} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_CONE_PRODUCT_H
