/*

    sdpt3.cc

    SDPT3 .mat files (plan Section 12): blk (cell p x 2: type in {'s', 'l', 'u'}, sizes), At (cell: At{p} in svec
    form with cuADMM's convention, n(n+1)/2 x m for 's', n x m otherwise), C (cell: n x n matrix for 's', vector
    otherwise), b (m x 1). An 's' entry that lists several sizes is split into separate blocks (the svec of SDPT3
    stacks the sub-blocks, C{p} is block diagonal). 'q' comes with Phase 9. At is copied straight into the CSC
    arrays (one column per constraint, blocks in order); C is converted to svec with symmetrization.

*/

#include "io/readers.h"

#include <algorithm>
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
struct BlockEntry
{
    char type;
    std::vector<std::int64_t> sizes;   // several for an 's' group
    std::int64_t rows = 0;             // rows of At{p}: sum n_i(n_i+1)/2 for 's', n for 'l'/'u'
    std::vector<std::int64_t> cum_svec; // cumulative svec sizes of the sub-blocks (size k+1)
    std::vector<std::int64_t> cum_side; // cumulative sides of the sub-blocks (size k+1)
    std::int64_t first_cone = 0;       // index of its first cone
};
} // namespace

ProblemData read_sdpt3(const std::string &path, const IoOptions &options)
{
    MatFile file(path, MAT_ACC_RDONLY);
    auto fail = [&](const std::string &why) { throw std::invalid_argument(path + ": " + why); };
    MatVar blk = read_variable(file, "blk"), At = read_variable(file, "At"), C = read_variable(file, "C"), b = read_variable(file, "b");
    for (const auto &[var, name] : {std::pair<const MatVar *, const char *>{&blk, "blk"}, {&At, "At"}, {&C, "C"}, {&b, "b"}})
        if (!*var)
            fail(std::string("the variable ") + name + " is missing");
    if (blk.get()->class_type != MAT_C_CELL || blk.get()->rank != 2 || blk.get()->dims[1] < 2)
        fail("blk must be a cell array with two columns (it is " + class_name(blk.get()) + " " + dims_string(blk.get()) + ")");
    const std::size_t P = blk.get()->dims[0];
    if (numel(At.get()) != P || numel(C.get()) != P)
        fail("At and C must have one cell per row of blk (" + std::to_string(P) + ")");
    const std::vector<double> b_values = numeric_values(b.get(), "b");
    const std::int64_t m = static_cast<std::int64_t>(b_values.size());
    if (m == 0)
        fail("b is empty");

    ProblemData p;
    std::vector<BlockEntry> entries(P);
    std::int64_t vec_len = 0, split_groups = 0;
    for (std::size_t q = 0; q < P; q++)
    {
        BlockEntry &e = entries[q];
        const std::string type = string_value(cell_at(blk.get(), q, "blk"), "blk{" + std::to_string(q + 1) + ",1}");
        if (type.size() != 1)
            fail("blk{" + std::to_string(q + 1) + ",1} = '" + type + "' is not a block type");
        e.type = type[0];
        e.sizes = integer_values(cell_at(blk.get(), q + P, "blk"), "blk{" + std::to_string(q + 1) + ",2}");
        for (std::int64_t n : e.sizes)
            if (n <= 0)
                fail("blk{" + std::to_string(q + 1) + ",2} has a non-positive size");
        if (e.sizes.empty())
            fail("blk{" + std::to_string(q + 1) + ",2} is empty");
        e.first_cone = static_cast<std::int64_t>(p.cones.size());
        if (e.type == 's')
        {
            e.cum_svec.push_back(0);
            e.cum_side.push_back(0);
            for (std::int64_t n : e.sizes)
            {
                p.cones.push_back({ConeKind::Psd, n, 0.0});
                e.cum_svec.push_back(e.cum_svec.back() + n * (n + 1) / 2);
                e.cum_side.push_back(e.cum_side.back() + n);
            }
            e.rows = e.cum_svec.back();
            if (e.sizes.size() > 1)
                split_groups++;
        }
        else if (e.type == 'l' || e.type == 'u')
        {
            std::int64_t n = 0;
            for (std::int64_t s : e.sizes)
                n += s;
            p.cones.push_back({e.type == 'l' ? ConeKind::Nonnegative : ConeKind::Free, n, 0.0});
            e.rows = n;
        }
        else if (e.type == 'q' || e.type == 'r')
            fail("blk{" + std::to_string(q + 1) + ",1} = '" + type + "' (second-order cones) comes with Phase 9");
        else
            fail("blk{" + std::to_string(q + 1) + ",1} = '" + type + "' is not a supported block type ('s', 'l', 'u')");
        vec_len += e.rows;
        // original block: one per blk row
        p.source.original_block_kinds.push_back(e.type);
        p.source.original_block_sizes.push_back(e.type == 's' ? e.cum_side.back() : e.rows);
        for (std::int64_t k = e.first_cone; k < static_cast<std::int64_t>(p.cones.size()); k++)
            p.source.original_block_of.push_back(static_cast<int>(q));
    }
    const std::vector<std::int64_t> off = cone_offsets(p.cones);

    // At: count per column, then copy block by block (rows sorted within each column because the blocks are in order)
    p.At.rows = vec_len;
    p.At.cols = m;
    p.At.colptr.assign(static_cast<std::size_t>(m) + 1, 0);
    std::size_t total = 0;
    std::vector<const matvar_t *> at_cells(P);
    for (std::size_t q = 0; q < P; q++)
    {
        at_cells[q] = cell_at(At.get(), q, "At");
        const std::string what = "At{" + std::to_string(q + 1) + "}";
        if (is_empty(at_cells[q]))
        {
            p.source.report.push_back(what + " is empty: no constraint touches block " + std::to_string(q + 1));
            continue;
        }
        if (at_cells[q]->rank != 2 || static_cast<std::int64_t>(at_cells[q]->dims[0]) != entries[q].rows || static_cast<std::int64_t>(at_cells[q]->dims[1]) != m)
            fail(what + " is " + dims_string(at_cells[q]) + ", expected " + std::to_string(entries[q].rows) + " x " + std::to_string(m));
        for_each_nonzero(at_cells[q], what, [&](std::int64_t, std::int64_t j, double) { p.At.colptr[j + 1]++; });
        total += nonzero_capacity(at_cells[q]);
    }
    for (std::int64_t j = 0; j < m; j++)
        p.At.colptr[j + 1] += p.At.colptr[j];
    p.At.rowind.resize(static_cast<std::size_t>(p.At.colptr[m]));
    p.At.val.resize(static_cast<std::size_t>(p.At.colptr[m]));
    std::vector<std::int64_t> cursor(p.At.colptr.begin(), p.At.colptr.end() - 1);
    for (std::size_t q = 0; q < P; q++)
    {
        if (is_empty(at_cells[q]))
            continue;
        const BlockEntry &e = entries[q];
        const std::int64_t base = off[static_cast<std::size_t>(e.first_cone)];
        for_each_nonzero(at_cells[q], "At{" + std::to_string(q + 1) + "}", [&](std::int64_t r, std::int64_t j, double v)
        {
            const std::int64_t k = cursor[j]++;
            p.At.rowind[k] = base + r; // sub-blocks of an 's' group are consecutive cones: the stacked svec maps directly
            p.At.val[k] = v;
        });
    }
    // a malformed file may list rows out of order or twice within a column
    std::int64_t resorted = 0;
    std::vector<std::pair<std::int64_t, double>> tmp;
    for (std::int64_t j = 0; j < m; j++)
    {
        const std::int64_t lo = p.At.colptr[j], hi = p.At.colptr[j + 1];
        if (std::is_sorted(p.At.rowind.begin() + lo, p.At.rowind.begin() + hi))
            continue;
        resorted++;
        tmp.assign(static_cast<std::size_t>(hi - lo), {0, 0.0});
        for (std::int64_t k = lo; k < hi; k++)
            tmp[k - lo] = {p.At.rowind[k], p.At.val[k]};
        std::stable_sort(tmp.begin(), tmp.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (std::int64_t k = lo; k < hi; k++)
        {
            p.At.rowind[k] = tmp[k - lo].first;
            p.At.val[k] = tmp[k - lo].second;
        }
    }
    for (std::int64_t j = 0; j < m; j++)
        for (std::int64_t k = p.At.colptr[j] + 1; k < p.At.colptr[j + 1]; k++)
            if (p.At.rowind[k] == p.At.rowind[k - 1])
                fail("At lists svec entry " + std::to_string(p.At.rowind[k]) + " twice in constraint " + std::to_string(j + 1));

    // C
    SparseVectorBuilder cb(vec_len);
    std::int64_t c_off_diagonal = 0;
    for (std::size_t q = 0; q < P; q++)
    {
        const BlockEntry &e = entries[q];
        const matvar_t *cq = cell_at(C.get(), q, "C");
        const std::string what = "C{" + std::to_string(q + 1) + "}";
        if (is_empty(cq))
            continue;
        if (e.type == 's')
        {
            const std::int64_t side = e.cum_side.back();
            if (cq->rank != 2 || static_cast<std::int64_t>(cq->dims[0]) != side || static_cast<std::int64_t>(cq->dims[1]) != side)
                fail(what + " is " + dims_string(cq) + ", expected " + std::to_string(side) + " x " + std::to_string(side));
            for_each_nonzero(cq, what, [&](std::int64_t r, std::int64_t c, double v)
            {
                const std::size_t br = static_cast<std::size_t>(std::upper_bound(e.cum_side.begin(), e.cum_side.end(), r) - e.cum_side.begin()) - 1;
                const std::size_t bc = static_cast<std::size_t>(std::upper_bound(e.cum_side.begin(), e.cum_side.end(), c) - e.cum_side.begin()) - 1;
                if (br != bc)
                    fail(what + " has the nonzero (" + std::to_string(r + 1) + ", " + std::to_string(c + 1) + ") outside the diagonal blocks of the group");
                const std::int64_t k = r - e.cum_side[br], l = c - e.cum_side[br];
                const std::int64_t idx = svec_index(off[static_cast<std::size_t>(e.first_cone) + br], k, l);
                if (k != l)
                    c_off_diagonal++;
                cb.add(idx, k == l ? v : v * kHalfSqrt2);
            });
        }
        else
        {
            if (numel(cq) != static_cast<std::size_t>(e.rows) || cq->rank != 2 || (cq->dims[0] != 1 && cq->dims[1] != 1))
                fail(what + " is " + dims_string(cq) + ", expected a vector of " + std::to_string(e.rows) + " entries");
            for_each_nonzero(cq, what, [&](std::int64_t r, std::int64_t c, double v) { cb.add(off[static_cast<std::size_t>(e.first_cone)] + r + c, v); });
        }
    }
    BuildStats c_stats;
    p.C = cb.finish(DuplicatePolicy::Sum, true, c_stats);
    p.b.size = m;
    for (std::int64_t i = 0; i < m; i++)
        if (b_values[static_cast<std::size_t>(i)] != 0.0)
        {
            p.b.index.push_back(i);
            p.b.val.push_back(b_values[static_cast<std::size_t>(i)]);
        }
    p.sense = Sense::Minimize;
    p.objective_offset = 0.0;
    check_sense_option(options, p.sense, path, "SDPT3");
    p.source.format = "sdpt3";
    p.source.path = path;
    p.source.mat_version = file.version();
    set_dimacs_kinds_from_original(p);
    p.source.report.insert(p.source.report.begin(), "SDPT3 (MATLAB v" + p.source.mat_version + "): " + std::to_string(P) + " blk entries -> " +
                                                        std::to_string(p.cones.size()) + " blocks (" + std::to_string(split_groups) +
                                                        " 's' entries with several sizes split); m = " + std::to_string(m) + ", vec_len = " +
                                                        std::to_string(vec_len) + "; At copied in svec form, C converted (symmetrized, " +
                                                        std::to_string(c_off_diagonal) + " off-diagonal entries)");
    if (resorted > 0)
        p.source.report.push_back("At: " + std::to_string(resorted) + " columns with unsorted rows were sorted");
    apply_simplifications(p, options.simplify);
    finish_problem(p, path);
    return p;
}

#else

ProblemData read_sdpt3(const std::string &path, const IoOptions &)
{
    throw std::invalid_argument(path + ": the SDPT3 .mat reader needs matio, which is not compiled into this build (CUADMM_WITH_MATIO)");
}

#endif

} // namespace detail
} // namespace io
} // namespace cuadmm
