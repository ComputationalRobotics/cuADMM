/*

    svec.h

    The svec layout of cuADMM (plan Section 12, Appendix C): a PSD block of side n is stored as its upper triangle,
    column by column, n(n+1)/2 entries, the off-diagonal entries multiplied by sqrt(2) so that
    svec(A) . svec(X) = <A, X>. Entry (k, l), 0-based, k <= l, sits at offset + l(l+1)/2 + k. Free and nonnegative
    blocks are stored as they are. Shared by every reader and writer of src/io/.

*/

#ifndef CUADMM_IO_SVEC_H
#define CUADMM_IO_SVEC_H

#include <cmath>
#include <cstdint>
#include <vector>

#include "cuadmm/problem.h"

namespace cuadmm
{
namespace io
{
namespace detail
{

constexpr double kSqrt2 = 1.4142135623730951;      // sqrt(2) rounded to double
constexpr double kHalfSqrt2 = 0.7071067811865476;  // sqrt(2)/2: the two mirrored halves of an off-diagonal entry sum
                                                   // to exactly value * kSqrt2 (scaling by 2 is exact)

/// @brief Entries of a cone in svec form.
inline std::int64_t cone_length(const ConeSpec &c)
{
    return c.kind == ConeKind::Psd ? c.size * (c.size + 1) / 2 : c.size;
}

/// @brief svec position of entry (k, l) of a PSD block, 0-based, any order of k and l.
inline std::int64_t svec_index(std::int64_t offset, std::int64_t k, std::int64_t l)
{
    if (k > l)
    {
        const std::int64_t t = k;
        k = l;
        l = t;
    }
    return offset + l * (l + 1) / 2 + k;
}

/// @brief Whether svec position `r` (relative to the block start) of a block of side n lies on the diagonal.
inline bool is_diagonal_position(std::int64_t r, std::int64_t n)
{
    // the diagonal (l, l) sits at l(l+1)/2 + l = l(l+3)/2; invert with the column of r
    (void)n;
    // column l of position r: the largest l with l(l+1)/2 <= r
    std::int64_t l = static_cast<std::int64_t>((std::sqrt(8.0 * static_cast<double>(r) + 1.0) - 1.0) / 2.0);
    while (l * (l + 1) / 2 > r)
        l--;
    while ((l + 1) * (l + 2) / 2 <= r)
        l++;
    return r - l * (l + 1) / 2 == l;
}

/// @brief Row and column (k <= l) of svec position r relative to the block start.
inline void svec_position(std::int64_t r, std::int64_t &k, std::int64_t &l)
{
    l = static_cast<std::int64_t>((std::sqrt(8.0 * static_cast<double>(r) + 1.0) - 1.0) / 2.0);
    while (l * (l + 1) / 2 > r)
        l--;
    while ((l + 1) * (l + 2) / 2 <= r)
        l++;
    k = r - l * (l + 1) / 2;
}

/// @brief Start offsets of the cones in svec form (one more entry than cones: the total length).
inline std::vector<std::int64_t> cone_offsets(const std::vector<ConeSpec> &cones)
{
    std::vector<std::int64_t> off(cones.size() + 1, 0);
    for (std::size_t b = 0; b < cones.size(); b++)
        off[b + 1] = off[b] + cone_length(cones[b]);
    return off;
}

} // namespace detail
} // namespace io
} // namespace cuadmm

#endif // CUADMM_IO_SVEC_H
