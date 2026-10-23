/*

    termination/certificate.h

    Certificate (plan Section 7.4, B24): the certified lower bound on the optimal value, <b, y> + sum_b R_b c_b,
    from the per-block coefficients c_b of C - A^T y (Cone::measure, Side::Dual) and explicit per-block bounds R_b:
    the trace of a PSD block, the entrywise bound of a nonnegative or free block (plan Section 9.2). A block
    without a bound (NaN) makes the bound unavailable; the result lists the blocks.

*/

#ifndef CUADMM_TERMINATION_CERTIFICATE_H
#define CUADMM_TERMINATION_CERTIFICATE_H

#include <string>
#include <vector>

#include "cone/cone_product.h"
#include "cuadmm/options.h"

namespace cuadmm
{
namespace termination
{

struct CertificateResult
{
    std::vector<double> dual_block_min; // per block, original units: lambda_min ('s'), min entry ('l'), max |entry| ('u')
    std::vector<double> dual_cone_coef; // per block, original units: c_b with <(C - A^T y)_b, X_b> >= R_b c_b
    double coef_sum = 0.0;              // sum of the coefficients (the "R * coef_sum" line of the console)
    double lower_bound = 0.0;           // NaN when unavailable
    std::vector<int> missing_bounds;    // blocks without a bound (R_b is NaN): the bound is unavailable
    bool conditional_on_free_bounds = false; // a free block's bound is in force (B24): valid only if |X_u| <= R_u
    std::string note;
};

class Certificate
{
public:
    /// The measures of Z = C - A^T y (scaled units; Cscale converts to original units) through the product cone.
    static CertificateResult measure(cone::ConeProduct &cones, core::DeviceSpan<const double> Z_scaled, double Cscale, core::ExecContext &ctx);
    /// The lower bound <b, y> + sum_b R_b c_b with explicit bounds (NaN: missing), filled into `r`.
    static void evaluate(CertificateResult &r, double dobj, const std::vector<double> &bounds, const std::vector<ConeKind> &kinds);
    /// Per-block bounds from the options: block_bounds as given (a negative or NaN entry means missing), else
    /// trace_bound for every block, else trace_bound_scale (times n_b for PSD blocks); an empty vector when the
    /// options ask for no certificate.
    static std::vector<double> bounds_from_options(const CertificateOptions &o, const std::vector<ConeKind> &kinds, const std::vector<int> &sizes,
                                                   bool scaled_variant);
};

} // namespace termination
} // namespace cuadmm

#endif // CUADMM_TERMINATION_CERTIFICATE_H
