/*

    mosek.cc

    MOSEK MATLAB toolbox `prob` structs saved in .mat files (plan Section 12; SPOT's data):

        min/max  c'x + sum_j <barc_j, Xbar_j> + cfix
        s.t.     blc <= a x + sum_j <bara_ij, Xbar_j> <= buc,   blx <= x <= bux,   Xbar_j PSD (bardim)

    Mapping: bar variables -> 's' blocks in order; bara and barc list the lower triangle (subk >= subl) of symmetric
    matrices, so the svec value is val on the diagonal and sqrt2 val off it. Scalar variables (columns of a): free
    (missing or infinite bounds) -> one 'u' block first; [0, inf) -> 'l'; a finite lower bound by a shift
    x = blx + x'; a finite upper bound only by x = bux - x'; two finite bounds by a shift and a slack row
    x' + s = bux - blx; blx = bux by elimination. Rows with blc = buc are equalities; a one-sided row gets a slack
    in the 'l' block (a x - s = blc or a x + s = buc); a ranged row gets a slack and a second row s + t = buc - blc;
    a row without bounds is dropped. cfix (and the shifts) go to the objective offset. The objective sense is not
    stored by MOSEK: it comes from a variable `sense` saved next to `prob` or from io.sense, and the reader stops
    without it (SPOT's SOS files are maximizations). For a maximization C = -c. `cones`, `accs`, `f`/`g` come with
    Phase 9; integer variables are rejected. Duplicate (i, j, k, l) in bara / barc are summed (as MOSEK does).
    Block layout: ['u' free scalars] ['l' nonnegative scalars, bound slacks, row slacks] ['s' bar variables...].

*/

#include "io/readers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
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
constexpr double kInf = std::numeric_limits<double>::infinity();

enum class VarKind { Free, Nonneg, ShiftedLower, NegatedUpper, Ranged, Fixed };
enum class RowKind { Equality, LowerSlack, UpperSlack, Ranged, Dropped };

struct VarInfo
{
    VarKind kind = VarKind::Free;
    std::int64_t position = -1;  // svec index (not for Fixed)
    std::int64_t slack = -1;     // svec index of the bound slack (Ranged)
    std::int64_t slack_row = -1; // constraint index of x' + s = bux - blx (Ranged)
    double shift = 0.0, sign = 1.0, range = 0.0;
};

struct RowInfo
{
    RowKind kind = RowKind::Equality;
    std::int64_t position = -1;   // constraint index of the row (its first row when ranged)
    std::int64_t position2 = -1;  // the second row of a ranged row (s + t = buc - blc)
    std::int64_t slack = -1, slack2 = -1; // svec indices of the slacks
    double rhs = 0.0, rhs2 = 0.0;
};

std::optional<Sense> parse_sense_string(std::string s)
{
    s = lower(s);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\0'))
        s.pop_back();
    if (s == "min" || s == "minimize" || s == "minimise")
        return Sense::Minimize;
    if (s == "max" || s == "maximize" || s == "maximise")
        return Sense::Maximize;
    return std::nullopt;
}
} // namespace

ProblemData read_mosek(const std::string &path, const IoOptions &options)
{
    MatFile file(path, MAT_ACC_RDONLY);
    auto fail = [&](const std::string &why) { throw std::invalid_argument(path + ": " + why); };
    MatVar prob = read_variable(file, "prob");
    if (!prob)
        fail("the variable prob is missing");
    if (prob.get()->class_type != MAT_C_STRUCT)
        fail("prob must be a struct (it is " + class_name(prob.get()) + ")");
    auto field = [&](const char *name) -> const matvar_t *
    {
        const matvar_t *f = struct_field(prob.get(), name, "prob");
        return f && !is_empty(f) ? f : nullptr;
    };
    for (const char *name : {"cones", "accs", "f", "g"})
        if (field(name))
            fail(std::string("prob.") + name + " (conic constraints other than PSD) comes with Phase 9");
    if (field("ints"))
        fail("prob.ints: integer variables are not supported");

    // objective sense
    std::optional<Sense> file_sense;
    MatVar sense_var = read_variable(file, "sense");
    std::string sense_origin;
    if (sense_var)
    {
        file_sense = parse_sense_string(string_value(sense_var.get(), "sense"));
        if (!file_sense)
            fail("the variable sense must be 'minimize' or 'maximize'");
        sense_origin = "the variable sense of the file";
    }
    if (file_sense && options.sense && *file_sense != *options.sense)
        fail("io.sense = " + std::string(name_of(*options.sense)) + " contradicts the variable sense of the file (" + name_of(*file_sense) + ")");
    if (!file_sense && options.sense)
    {
        file_sense = options.sense;
        sense_origin = "io.sense";
    }
    if (!file_sense)
        fail("the objective sense is not known: MOSEK's prob struct does not store it; save a variable `sense` ('minimize' or 'maximize') "
             "next to prob, or give io.sense (--sense min|max)");
    const Sense sense = *file_sense;

    // sizes
    const matvar_t *a = field("a"), *c = field("c"), *blc = field("blc"), *buc = field("buc"), *blx = field("blx"), *bux = field("bux");
    const matvar_t *bardim_v = field("bardim"), *barc = field("barc"), *bara = field("bara"), *cfix = field("cfix");
    std::int64_t n = 0, m = 0;
    if (c)
        n = static_cast<std::int64_t>(numel(c));
    if (a)
    {
        if (a->rank != 2)
            fail("prob.a must be two-dimensional");
        m = static_cast<std::int64_t>(a->dims[0]);
        if (c && static_cast<std::int64_t>(a->dims[1]) != n)
            fail("prob.a has " + std::to_string(a->dims[1]) + " columns, prob.c " + std::to_string(n) + " entries");
        n = static_cast<std::int64_t>(a->dims[1]);
    }
    else if (blc)
        m = static_cast<std::int64_t>(numel(blc));
    else if (buc)
        m = static_cast<std::int64_t>(numel(buc));
    if (m == 0)
        fail("no constraints (prob.a, prob.blc and prob.buc are absent or empty)");
    auto bounds = [&](const matvar_t *v, const char *name, std::int64_t count, double missing)
    {
        if (!v)
            return std::vector<double>(static_cast<std::size_t>(count), missing);
        std::vector<double> out = numeric_values(v, std::string("prob.") + name);
        if (static_cast<std::int64_t>(out.size()) != count)
            fail(std::string("prob.") + name + " has " + std::to_string(out.size()) + " entries, expected " + std::to_string(count));
        for (std::size_t i = 0; i < out.size(); i++)
            if (std::isnan(out[i]))
                fail(std::string("prob.") + name + "(" + std::to_string(i + 1) + ") is NaN");
        return out;
    };
    const std::vector<double> lc = bounds(blc, "blc", m, -kInf), uc = bounds(buc, "buc", m, kInf);
    const std::vector<double> lx = bounds(blx, "blx", n, -kInf), ux = bounds(bux, "bux", n, kInf);
    const std::vector<double> c_values = c ? numeric_values(c, "prob.c") : std::vector<double>(static_cast<std::size_t>(n), 0.0);
    std::vector<std::int64_t> bardim;
    if (bardim_v)
        bardim = integer_values(bardim_v, "prob.bardim");
    for (std::int64_t d : bardim)
        if (d <= 0)
            fail("prob.bardim has a non-positive dimension");
    const std::int64_t nbar = static_cast<std::int64_t>(bardim.size());
    const double cfix_value = cfix ? scalar_value(cfix, "prob.cfix") : 0.0;

    // scalar variables
    std::vector<VarInfo> vars(static_cast<std::size_t>(n));
    std::int64_t n_free = 0, n_nonneg = 0, n_shifted = 0, n_negated = 0, n_ranged = 0, n_fixed = 0;
    for (std::int64_t j = 0; j < n; j++)
    {
        VarInfo &v = vars[static_cast<std::size_t>(j)];
        const double l = lx[static_cast<std::size_t>(j)], u = ux[static_cast<std::size_t>(j)];
        if (l > u)
            fail("variable " + std::to_string(j + 1) + " has blx > bux (" + format_double(l) + " > " + format_double(u) + ")");
        if (l == kInf || u == -kInf)
            fail("variable " + std::to_string(j + 1) + " has an infinite bound of the wrong sign");
        if (l == -kInf && u == kInf)
        {
            v.kind = VarKind::Free;
            n_free++;
        }
        else if (l == 0.0 && u == kInf)
        {
            v.kind = VarKind::Nonneg;
            n_nonneg++;
        }
        else if (u == kInf)
        {
            v.kind = VarKind::ShiftedLower;
            v.shift = l;
            n_shifted++;
        }
        else if (l == -kInf)
        {
            v.kind = VarKind::NegatedUpper;
            v.shift = u;
            v.sign = -1.0;
            n_negated++;
        }
        else if (l == u)
        {
            v.kind = VarKind::Fixed;
            v.shift = l;
            n_fixed++;
        }
        else
        {
            v.kind = VarKind::Ranged;
            v.shift = l;
            v.range = u - l;
            n_ranged++;
        }
    }
    // rows
    std::vector<RowInfo> rows(static_cast<std::size_t>(m));
    std::int64_t n_eq = 0, n_lower = 0, n_upper = 0, n_rranged = 0, n_dropped = 0;
    for (std::int64_t i = 0; i < m; i++)
    {
        RowInfo &r = rows[static_cast<std::size_t>(i)];
        const double l = lc[static_cast<std::size_t>(i)], u = uc[static_cast<std::size_t>(i)];
        if (l > u)
            fail("row " + std::to_string(i + 1) + " has blc > buc (" + format_double(l) + " > " + format_double(u) + ")");
        if (l == kInf || u == -kInf)
            fail("row " + std::to_string(i + 1) + " has an infinite bound of the wrong sign");
        if (l == -kInf && u == kInf)
        {
            r.kind = RowKind::Dropped;
            n_dropped++;
        }
        else if (l == u)
        {
            r.kind = RowKind::Equality;
            r.rhs = l;
            n_eq++;
        }
        else if (u == kInf)
        {
            r.kind = RowKind::LowerSlack;
            r.rhs = l;
            n_lower++;
        }
        else if (l == -kInf)
        {
            r.kind = RowKind::UpperSlack;
            r.rhs = u;
            n_upper++;
        }
        else
        {
            r.kind = RowKind::Ranged;
            r.rhs = l;
            r.rhs2 = u - l;
            n_rranged++;
        }
    }
    // layout: u block, l block (nonnegative scalars, bound slacks, row slacks), bar blocks
    const std::int64_t n_l = n_nonneg + n_shifted + n_negated + 2 * n_ranged + n_lower + n_upper + 2 * n_rranged;
    const std::int64_t off_u = 0, off_l = n_free;
    std::vector<std::int64_t> off_bar(static_cast<std::size_t>(nbar) + 1, off_l + n_l);
    for (std::int64_t j = 0; j < nbar; j++)
        off_bar[static_cast<std::size_t>(j) + 1] = off_bar[static_cast<std::size_t>(j)] + bardim[static_cast<std::size_t>(j)] * (bardim[static_cast<std::size_t>(j)] + 1) / 2;
    const std::int64_t vec_len = off_bar.back();
    std::int64_t next_u = off_u, next_l = off_l;
    for (VarInfo &v : vars)
    {
        if (v.kind == VarKind::Free)
            v.position = next_u++;
        else if (v.kind != VarKind::Fixed)
            v.position = next_l++;
    }
    for (VarInfo &v : vars)
        if (v.kind == VarKind::Ranged)
            v.slack = next_l++;
    for (RowInfo &r : rows)
    {
        if (r.kind == RowKind::LowerSlack || r.kind == RowKind::UpperSlack)
            r.slack = next_l++;
        else if (r.kind == RowKind::Ranged)
        {
            r.slack = next_l++;
            r.slack2 = next_l++;
        }
    }
    // constraint indices: the file's rows first (in order, dropped ones skipped), then the second rows of ranged
    // rows, then the bound rows of ranged variables
    std::int64_t m_new = 0;
    for (RowInfo &r : rows)
        if (r.kind != RowKind::Dropped)
            r.position = m_new++;
    for (RowInfo &r : rows)
        if (r.kind == RowKind::Ranged)
            r.position2 = m_new++;
    for (VarInfo &v : vars)
        if (v.kind == VarKind::Ranged)
            v.slack_row = m_new++;
    if (m_new == 0)
        fail("every row is unbounded (blc = -inf and buc = inf): nothing to solve");

    // b (right-hand sides, with the contributions of the shifts), C, At
    std::vector<double> rhs(static_cast<std::size_t>(m_new), 0.0);
    for (const RowInfo &r : rows)
        if (r.kind != RowKind::Dropped)
        {
            rhs[static_cast<std::size_t>(r.position)] = r.rhs;
            if (r.kind == RowKind::Ranged)
                rhs[static_cast<std::size_t>(r.position2)] = r.rhs2;
        }
    for (const VarInfo &v : vars)
        if (v.kind == VarKind::Ranged)
            rhs[static_cast<std::size_t>(v.slack_row)] = v.range;
    double offset = cfix_value;
    SparseVectorBuilder cb(vec_len);
    for (std::int64_t j = 0; j < n; j++)
    {
        const VarInfo &v = vars[static_cast<std::size_t>(j)];
        const double cj = c_values[static_cast<std::size_t>(j)];
        if (v.shift != 0.0)
            offset += cj * v.shift;
        if (v.kind != VarKind::Fixed && cj != 0.0)
            cb.add(v.position, cj * v.sign);
    }
    CscBuilder builder(vec_len, m_new);
    builder.reserve(nonzero_capacity(a) + nonzero_capacity(bara ? struct_field(bara, "val", "prob.bara") : nullptr) + static_cast<std::size_t>(n_l));
    std::int64_t a_entries = 0, a_dropped = 0;
    if (a)
        for_each_nonzero(a, "prob.a", [&](std::int64_t i, std::int64_t j, double value)
        {
            const RowInfo &r = rows[static_cast<std::size_t>(i)];
            const VarInfo &v = vars[static_cast<std::size_t>(j)];
            a_entries++;
            if (r.kind == RowKind::Dropped)
            {
                a_dropped++;
                return;
            }
            if (v.shift != 0.0)
                rhs[static_cast<std::size_t>(r.position)] -= value * v.shift;
            if (v.kind != VarKind::Fixed)
                builder.add(r.position, v.position, value * v.sign);
        });
    for (const VarInfo &v : vars)
        if (v.kind == VarKind::Ranged)
        {
            builder.add(v.slack_row, v.position, 1.0);
            builder.add(v.slack_row, v.slack, 1.0);
        }
    for (const RowInfo &r : rows)
    {
        if (r.kind == RowKind::LowerSlack)
            builder.add(r.position, r.slack, -1.0);
        else if (r.kind == RowKind::UpperSlack)
            builder.add(r.position, r.slack, 1.0);
        else if (r.kind == RowKind::Ranged)
        {
            builder.add(r.position, r.slack, -1.0);
            builder.add(r.position2, r.slack, 1.0);
            builder.add(r.position2, r.slack2, 1.0);
        }
    }
    // bara, barc (lower triangles)
    auto read_triplets = [&](const matvar_t *s, const char *what, bool with_rows, std::vector<std::int64_t> &subi, std::vector<std::int64_t> &subj,
                             std::vector<std::int64_t> &subk, std::vector<std::int64_t> &subl, std::vector<double> &val)
    {
        if (s->class_type != MAT_C_STRUCT)
            fail(std::string(what) + " must be a struct");
        auto f = [&](const char *name)
        {
            const matvar_t *v = struct_field(s, name, what);
            if (!v)
                fail(std::string(what) + "." + name + " is missing");
            return v;
        };
        if (with_rows)
            subi = integer_values(f("subi"), std::string(what) + ".subi");
        subj = integer_values(f("subj"), std::string(what) + ".subj");
        subk = integer_values(f("subk"), std::string(what) + ".subk");
        subl = integer_values(f("subl"), std::string(what) + ".subl");
        val = numeric_values(f("val"), std::string(what) + ".val");
        const std::size_t t = val.size();
        if (subj.size() != t || subk.size() != t || subl.size() != t || (with_rows && subi.size() != t))
            fail(std::string(what) + ": subi, subj, subk, subl and val differ in length");
        for (std::size_t e = 0; e < t; e++)
        {
            if (with_rows && (subi[e] < 1 || subi[e] > m))
                fail(std::string(what) + ".subi(" + std::to_string(e + 1) + ") = " + std::to_string(subi[e]) + " outside [1, " + std::to_string(m) + "]");
            if (subj[e] < 1 || subj[e] > nbar)
                fail(std::string(what) + ".subj(" + std::to_string(e + 1) + ") = " + std::to_string(subj[e]) + " outside [1, " + std::to_string(nbar) + "]");
            const std::int64_t d = bardim[static_cast<std::size_t>(subj[e] - 1)];
            if (subk[e] < 1 || subk[e] > d || subl[e] < 1 || subl[e] > d)
                fail(std::string(what) + ": entry " + std::to_string(e + 1) + " has (subk, subl) = (" + std::to_string(subk[e]) + ", " + std::to_string(subl[e]) +
                     ") outside bar variable " + std::to_string(subj[e]) + " of dimension " + std::to_string(d));
            if (subk[e] < subl[e])
                fail(std::string(what) + ": entry " + std::to_string(e + 1) + " has subk < subl (" + std::to_string(subk[e]) + " < " + std::to_string(subl[e]) +
                     "); MOSEK lists the lower triangle only");
        }
    };
    std::int64_t bara_entries = 0, bara_dropped = 0, barc_entries = 0;
    if (bara)
    {
        std::vector<std::int64_t> subi, subj, subk, subl;
        std::vector<double> val;
        read_triplets(bara, "prob.bara", true, subi, subj, subk, subl, val);
        bara_entries = static_cast<std::int64_t>(val.size());
        for (std::size_t e = 0; e < val.size(); e++)
        {
            const RowInfo &r = rows[static_cast<std::size_t>(subi[e] - 1)];
            if (r.kind == RowKind::Dropped)
            {
                bara_dropped++;
                continue;
            }
            const std::int64_t k = subk[e] - 1, l = subl[e] - 1;
            builder.add(r.position, svec_index(off_bar[static_cast<std::size_t>(subj[e] - 1)], l, k), k == l ? val[e] : val[e] * kSqrt2);
        }
    }
    if (barc)
    {
        std::vector<std::int64_t> subi, subj, subk, subl;
        std::vector<double> val;
        read_triplets(barc, "prob.barc", false, subi, subj, subk, subl, val);
        barc_entries = static_cast<std::int64_t>(val.size());
        for (std::size_t e = 0; e < val.size(); e++)
        {
            const std::int64_t k = subk[e] - 1, l = subl[e] - 1;
            cb.add(svec_index(off_bar[static_cast<std::size_t>(subj[e] - 1)], l, k), k == l ? val[e] : val[e] * kSqrt2);
        }
    }
    ProblemData p;
    BuildStats at_stats, c_stats;
    p.At = builder.finish(DuplicatePolicy::Sum, true, at_stats);
    p.C = cb.finish(DuplicatePolicy::Sum, true, c_stats);
    if (sense == Sense::Maximize)
        for (double &v : p.C.val)
            v = -v;
    p.b.size = m_new;
    for (std::int64_t i = 0; i < m_new; i++)
        if (rhs[static_cast<std::size_t>(i)] != 0.0)
        {
            p.b.index.push_back(i);
            p.b.val.push_back(rhs[static_cast<std::size_t>(i)]);
        }
    if (n_free > 0)
        p.cones.push_back({ConeKind::Free, n_free, 0.0});
    if (n_l > 0)
        p.cones.push_back({ConeKind::Nonnegative, n_l, 0.0});
    for (std::int64_t d : bardim)
        p.cones.push_back({ConeKind::Psd, d, 0.0});
    p.sense = sense;
    p.objective_offset = offset;
    p.source.format = "mosek";
    p.source.path = path;
    p.source.mat_version = file.version();
    // original blocks: the bar variables; the scalar blocks are described by the mapping
    for (std::int64_t j = 0; j < nbar; j++)
    {
        p.source.original_block_kinds.push_back('s');
        p.source.original_block_sizes.push_back(bardim[static_cast<std::size_t>(j)]);
    }
    if (n_free > 0)
        p.source.original_block_of.push_back(-1);
    if (n_l > 0)
        p.source.original_block_of.push_back(-1);
    for (std::int64_t j = 0; j < nbar; j++)
        p.source.original_block_of.push_back(static_cast<int>(j));
    for (const VarInfo &v : vars)
    {
        p.source.mosek_var_position.push_back(v.kind == VarKind::Fixed ? -1 : v.position);
        p.source.mosek_var_shift.push_back(v.shift);
        p.source.mosek_var_sign.push_back(v.sign);
    }
    for (const RowInfo &r : rows)
        p.source.mosek_row_position.push_back(r.kind == RowKind::Dropped ? -1 : r.position);
    set_dimacs_kinds_from_original(p);
    p.source.report.push_back("MOSEK prob (MATLAB v" + p.source.mat_version + "): " + std::to_string(n) + " scalar variables (free " + std::to_string(n_free) +
                              " -> u; nonnegative " + std::to_string(n_nonneg) + ", shifted " + std::to_string(n_shifted) + ", negated " +
                              std::to_string(n_negated) + ", ranged " + std::to_string(n_ranged) + " -> l; fixed " + std::to_string(n_fixed) +
                              " eliminated), " + std::to_string(nbar) + " bar variables -> s; " + std::to_string(m) + " rows (equalities " +
                              std::to_string(n_eq) + ", lower " + std::to_string(n_lower) + ", upper " + std::to_string(n_upper) + ", ranged " +
                              std::to_string(n_rranged) + " with slacks, dropped " + std::to_string(n_dropped) + ") -> " + std::to_string(m_new) +
                              " constraints; objective sense " + name_of(sense) + " from " + sense_origin + (sense == Sense::Maximize ? " (C = -c)" : "") +
                              "; cfix = " + format_double(cfix_value) + ", offset = " + format_double(offset));
    p.source.report.push_back("a: " + std::to_string(a_entries) + " entries (" + std::to_string(a_dropped) + " in dropped rows); bara: " +
                              std::to_string(bara_entries) + " lower-triangle entries (sqrt2 off the diagonal, " + std::to_string(bara_dropped) +
                              " in dropped rows), barc: " + std::to_string(barc_entries) + "; duplicates summed: " +
                              std::to_string(at_stats.duplicates + c_stats.duplicates) + "; zeros dropped: " + std::to_string(at_stats.zeros_dropped + c_stats.zeros_dropped));
    apply_simplifications(p, options.simplify);
    finish_problem(p, path);
    return p;
}

#else

ProblemData read_mosek(const std::string &path, const IoOptions &)
{
    throw std::invalid_argument(path + ": the MOSEK .mat reader needs matio, which is not compiled into this build (CUADMM_WITH_MATIO)");
}

#endif

} // namespace detail
} // namespace io
} // namespace cuadmm
