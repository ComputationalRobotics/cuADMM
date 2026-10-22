/*

    sedumi.cc

    SeDuMi .mat files (plan Section 12): A (m x N or N x m) or At (N x m), b, c (or C), K with fields f, l, q, r, s.
    K.f -> 'u', K.l -> 'l', K.s -> 's' blocks in that order (K.q and K.r come with Phase 9; complex data is
    rejected). When N = m the variable name decides the orientation. PSD blocks are full column-major n^2 vectors
    and need not be symmetric: the svec entry of (k, l), k < l, is (A_kl + A_lk)/sqrt2 (each stored entry
    contributes value * sqrt2/2 and the two halves are merged by summation, which gives exactly value * sqrt2 for
    a symmetric matrix). PLATO's .mat files are in this format (with At).

*/

#include "io/readers.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "io/csc_builder.h"
#include "io/svec.h"

#ifdef CUADMM_HAVE_MATIO
#include "io/mat_common.h"
#endif

namespace cuadmm
{
namespace io
{
namespace detail
{

#ifdef CUADMM_HAVE_MATIO

namespace
{
struct SedumiLayout
{
    std::int64_t f = 0, l = 0;
    std::vector<std::int64_t> s;
    std::vector<std::int64_t> full_offset; // start of each PSD block in the N layout (n^2 entries)
    std::vector<std::int64_t> svec_offset; // start of each PSD block in svec form
    std::int64_t N = 0, vec_len = 0;

    // variable index r of the file -> svec index and the factor on the value
    std::int64_t to_svec(std::int64_t r, double &scale) const
    {
        scale = 1.0;
        if (r < f + l)
            return r;
        const std::int64_t r2 = r - f - l;
        const std::size_t j = static_cast<std::size_t>(std::upper_bound(full_offset.begin(), full_offset.end(), r2) - full_offset.begin()) - 1;
        const std::int64_t n = s[j];
        const std::int64_t k = r2 - full_offset[j];
        const std::int64_t row = k % n, col = k / n;
        if (row != col)
            scale = kHalfSqrt2;
        return svec_index(svec_offset[j], row, col);
    }
};
} // namespace

ProblemData read_sedumi(const std::string &path, const IoOptions &options)
{
    MatFile file(path, MAT_ACC_RDONLY);
    auto fail = [&](const std::string &why) { throw std::invalid_argument(path + ": " + why); };
    MatVar A = read_variable(file, "A"), At = read_variable(file, "At"), b = read_variable(file, "b"), c = read_variable(file, "c");
    MatVar K = read_variable(file, "K");
    if (!c)
        c = read_variable(file, "C");
    if (!K)
        fail("the variable K (cone description) is missing");
    if (!b)
        fail("the variable b is missing");
    if (!c)
        fail("the variable c is missing");
    if (!A && !At)
        fail("neither A nor At is present");

    // K
    SedumiLayout L;
    auto k_values = [&](const char *name)
    {
        const matvar_t *fld = struct_field(K.get(), name, "K");
        if (!fld || is_empty(fld))
            return std::vector<std::int64_t>();
        return integer_values(fld, std::string("K.") + name);
    };
    auto k_scalar = [&](const char *name)
    {
        const std::vector<std::int64_t> v = k_values(name);
        if (v.empty())
            return std::int64_t(0);
        if (v.size() != 1)
            fail(std::string("K.") + name + " must be a scalar");
        if (v[0] < 0)
            fail(std::string("K.") + name + " must be non-negative");
        return v[0];
    };
    L.f = k_scalar("f");
    L.l = k_scalar("l");
    for (const char *name : {"q", "r"})
        for (std::int64_t v : k_values(name))
            if (v != 0)
                fail(std::string("K.") + name + " (second-order cones) comes with Phase 9");
    for (const char *name : {"scomplex", "xcomplex", "ycomplex"})
    {
        const matvar_t *fld = struct_field(K.get(), name, "K");
        if (fld && !is_empty(fld))
            for (double v : numeric_values(fld, std::string("K.") + name))
                if (v != 0.0)
                    fail(std::string("K.") + name + ": complex data is not supported");
    }
    for (std::int64_t n : k_values("s"))
    {
        if (n < 0)
            fail("K.s has a negative size");
        if (n > 0)
            L.s.push_back(n);
    }
    L.N = L.f + L.l;
    L.vec_len = L.f + L.l;
    for (std::int64_t n : L.s)
    {
        L.full_offset.push_back(L.N - L.f - L.l);
        L.svec_offset.push_back(L.vec_len);
        L.N += n * n;
        L.vec_len += n * (n + 1) / 2;
    }
    if (L.N == 0)
        fail("K describes no variables");

    // b, orientation of A / At
    const std::vector<double> b_values = numeric_values(b.get(), "b");
    const std::int64_t m = static_cast<std::int64_t>(b_values.size());
    if (m == 0)
        fail("b is empty");
    const matvar_t *mat = At ? At.get() : A.get();
    const bool named_At = static_cast<bool>(At);
    if (mat->rank != 2)
        fail(std::string(named_At ? "At" : "A") + " must be two-dimensional");
    const std::int64_t d0 = static_cast<std::int64_t>(mat->dims[0]), d1 = static_cast<std::int64_t>(mat->dims[1]);
    bool transposed; // true: the stored matrix is N x m (At)
    if (d0 == L.N && d1 == m && (L.N != m || named_At))
        transposed = true;
    else if (d0 == m && d1 == L.N)
        transposed = false;
    else
        fail(std::string(named_At ? "At" : "A") + " is " + dims_string(mat) + ", but K and b give N = " + std::to_string(L.N) + ", m = " + std::to_string(m));
    std::string orientation = std::string(named_At ? "At" : "A") + " (" + dims_string(mat) + ") read as " + (transposed ? "N x m" : "m x N");
    if (named_At != transposed)
        orientation += " (the name says otherwise; the shape decides when N != m)";

    // At in svec form
    ProblemData p;
    CscBuilder builder(L.vec_len, m);
    builder.reserve(nonzero_capacity(mat));
    std::int64_t off_diagonal = 0;
    if (transposed)
        for_each_nonzero(mat, "At", [&](std::int64_t r, std::int64_t j, double v)
        {
            double scale;
            const std::int64_t idx = L.to_svec(r, scale);
            off_diagonal += scale != 1.0;
            builder.add(j, idx, v * scale);
        });
    else
        for_each_nonzero(mat, "A", [&](std::int64_t i, std::int64_t r, double v)
        {
            double scale;
            const std::int64_t idx = L.to_svec(r, scale);
            off_diagonal += scale != 1.0;
            builder.add(i, idx, v * scale);
        });
    BuildStats at_stats, c_stats;
    p.At = builder.finish(DuplicatePolicy::Sum, true, at_stats);
    // c
    SparseVectorBuilder cb(L.vec_len);
    std::int64_t c_off_diagonal = 0;
    if (numel(c.get()) != static_cast<std::size_t>(L.N))
        fail("c has " + std::to_string(numel(c.get())) + " entries, N = " + std::to_string(L.N));
    for_each_nonzero(c.get(), "c", [&](std::int64_t i, std::int64_t j, double v)
    {
        double scale;
        const std::int64_t idx = L.to_svec(i + j, scale); // a column or a row vector
        c_off_diagonal += scale != 1.0;
        cb.add(idx, v * scale);
    });
    p.C = cb.finish(DuplicatePolicy::Sum, true, c_stats);
    p.b.size = m;
    for (std::int64_t i = 0; i < m; i++)
        if (b_values[static_cast<std::size_t>(i)] != 0.0)
        {
            p.b.index.push_back(i);
            p.b.val.push_back(b_values[static_cast<std::size_t>(i)]);
        }
    if (L.f > 0)
        p.cones.push_back({ConeKind::Free, L.f, 0.0});
    if (L.l > 0)
        p.cones.push_back({ConeKind::Nonnegative, L.l, 0.0});
    for (std::int64_t n : L.s)
        p.cones.push_back({ConeKind::Psd, n, 0.0});
    p.sense = Sense::Minimize;
    p.objective_offset = 0.0;
    check_sense_option(options, p.sense, path, "SeDuMi");
    p.source.format = "sedumi";
    p.source.path = path;
    p.source.mat_version = file.version();
    set_identity_original_blocks(p);
    set_dimacs_kinds_from_original(p);
    std::string sizes;
    for (std::size_t j = 0; j < L.s.size() && j < 12; j++)
        sizes += (j ? ", " : "") + std::to_string(L.s[j]);
    if (L.s.size() > 12)
        sizes += ", ...";
    p.source.report.push_back("SeDuMi (MATLAB v" + p.source.mat_version + "): " + orientation + "; m = " + std::to_string(m) + ", N = " +
                              std::to_string(L.N) + "; K.f = " + std::to_string(L.f) + " -> u, K.l = " + std::to_string(L.l) + " -> l, K.s = [" + sizes +
                              "] -> s");
    p.source.report.push_back("PSD blocks: full n^2 storage symmetrized, svec(k,l) = (A_kl + A_lk)/sqrt2: " + std::to_string(off_diagonal) +
                              " off-diagonal entries of A folded into " + std::to_string(off_diagonal - at_stats.duplicates) + " svec entries, " +
                              std::to_string(c_off_diagonal) + " of c into " + std::to_string(c_off_diagonal - c_stats.duplicates) +
                              "; zeros dropped: " + std::to_string(at_stats.zeros_dropped + c_stats.zeros_dropped));
    apply_simplifications(p, options.simplify);
    finish_problem(p, path);
    return p;
}

#else

ProblemData read_sedumi(const std::string &path, const IoOptions &)
{
    throw std::invalid_argument(path + ": the SeDuMi .mat reader needs matio, which is not compiled into this build (CUADMM_WITH_MATIO)");
}

#endif

} // namespace detail
} // namespace io
} // namespace cuadmm
