/*

    simplify.cc

    The exact simplifications at read time (plan Section 12, option io.simplify, on by default, each one reported):
    a 1x1 PSD block becomes an `l` block, and a PSD block whose data (A and C) touch only its diagonal becomes an `l`
    block of the same size (X >= 0 can be replaced by diag(X) >= 0 without changing the optimal value, and the dual
    slack is diagonal). Also the common finalization of the readers and the mapping of solutions back to the
    original block structure.

*/

#include "io/readers.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>

#include "io/svec.h"

namespace cuadmm
{
namespace io
{
namespace detail
{

std::string format_double(double v)
{
    char buf[64];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, r.ptr);
}

std::string lower(std::string s)
{
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool ends_with(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void set_identity_original_blocks(ProblemData &problem)
{
    problem.source.original_block_of.clear();
    problem.source.original_block_sizes.clear();
    problem.source.original_block_kinds.clear();
    for (std::size_t b = 0; b < problem.cones.size(); b++)
    {
        problem.source.original_block_of.push_back(static_cast<int>(b));
        problem.source.original_block_sizes.push_back(problem.cones[b].size);
        problem.source.original_block_kinds.push_back(cone_code(problem.cones[b].kind));
    }
}

void set_dimacs_kinds_from_original(ProblemData &problem)
{
    problem.source.dimacs_kinds.clear();
    for (std::size_t b = 0; b < problem.cones.size(); b++)
    {
        const int o = b < problem.source.original_block_of.size() ? problem.source.original_block_of[b] : -1;
        const char kind = o >= 0 && static_cast<std::size_t>(o) < problem.source.original_block_kinds.size() ? problem.source.original_block_kinds[o]
                                                                                                             : cone_code(problem.cones[b].kind);
        problem.source.dimacs_kinds.push_back(kind == 's' ? 's' : 'l');
    }
}

void apply_simplifications(ProblemData &problem, bool simplify)
{
    if (!simplify)
        return;
    const std::size_t B = problem.cones.size();
    const std::vector<std::int64_t> off = cone_offsets(problem.cones);
    // which PSD blocks have data off the diagonal: the scan stops as soon as every PSD block of side > 1 is known
    // to have some (the common case: a few entries per block), so it costs nothing on large problems
    std::vector<char> off_diagonal(B, 0), touched(B, 0);
    std::size_t undecided = 0;
    for (std::size_t b = 0; b < B; b++)
        if (problem.cones[b].kind == ConeKind::Psd && problem.cones[b].size > 1)
            undecided++;
    // block of an svec index: a coarse table (one entry per 1024 indices) followed by a short forward scan
    constexpr int kBucketShift = 10;
    std::vector<std::uint32_t> bucket_block(static_cast<std::size_t>((off.back() >> kBucketShift) + 1), 0);
    for (std::size_t b = 0, k = 0; k < bucket_block.size(); k++)
    {
        const std::int64_t start = static_cast<std::int64_t>(k) << kBucketShift;
        while (b + 1 < B && off[b + 1] <= start)
            b++;
        bucket_block[k] = static_cast<std::uint32_t>(b);
    }
    auto block_of = [&](std::int64_t idx)
    {
        std::size_t b = bucket_block[static_cast<std::size_t>(idx >> kBucketShift)];
        while (b + 1 < B && off[b + 1] <= idx)
            b++;
        return b;
    };
    auto mark = [&](std::int64_t idx)
    {
        const std::size_t b = block_of(idx);
        touched[b] = 1;
        if (problem.cones[b].kind == ConeKind::Psd && !off_diagonal[b] && !is_diagonal_position(idx - off[b], problem.cones[b].size))
        {
            off_diagonal[b] = 1;
            if (problem.cones[b].size > 1)
                undecided--;
        }
    };
    for (std::int64_t idx : problem.C.index)
        mark(idx);
    for (std::size_t k = 0; k < problem.At.rowind.size() && undecided > 0; k++)
        mark(problem.At.rowind[k]);
    if (undecided == 0)
    {
        // every PSD block (side > 1) keeps its kind; only the 1x1 blocks and the touched flags matter below, and
        // a 1x1 block is simplified whether or not it is touched
        for (std::size_t b = 0; b < B; b++)
            if (problem.cones[b].kind == ConeKind::Psd && problem.cones[b].size > 1)
                touched[b] = 1;
    }

    std::vector<ConeSpec> cones = problem.cones;
    std::vector<std::int64_t> new_off(B + 1, 0);
    bool changed = false;
    for (std::size_t b = 0; b < B; b++)
    {
        ConeSpec &c = cones[b];
        if (c.kind == ConeKind::Psd && c.size == 1)
        {
            c.kind = ConeKind::Nonnegative;
            problem.source.report.push_back("block " + std::to_string(b) + ": 1x1 PSD block read as 'l 1' (simplification)");
            changed = true;
        }
        else if (c.kind == ConeKind::Psd && !off_diagonal[b])
        {
            c.kind = ConeKind::Nonnegative;
            problem.source.report.push_back("block " + std::to_string(b) + ": PSD block of side " + std::to_string(c.size) +
                                            (touched[b] ? " with data on the diagonal only" : " untouched by A and C") + " read as 'l " +
                                            std::to_string(c.size) + "' (exact simplification)");
            changed = true;
        }
        new_off[b + 1] = new_off[b] + cone_length(c);
    }
    if (!changed)
        return;
    // remap the svec indices (monotone: sorted columns stay sorted)
    auto remap = [&](std::int64_t idx) -> std::int64_t
    {
        const std::size_t b = block_of(idx);
        const std::int64_t r = idx - off[b];
        if (problem.cones[b].kind == ConeKind::Psd && cones[b].kind == ConeKind::Nonnegative)
        {
            std::int64_t k, l;
            svec_position(r, k, l);
            return new_off[b] + l; // diagonal entry (l, l)
        }
        return new_off[b] + r;
    };
    for (std::int64_t &idx : problem.At.rowind)
        idx = remap(idx);
    for (std::int64_t &idx : problem.C.index)
        idx = remap(idx);
    problem.At.rows = new_off[B];
    problem.C.size = new_off[B];
    problem.cones = cones;
}

void check_sense_option(const IoOptions &options, Sense file_sense, const std::string &path, const char *format_name)
{
    if (options.sense && *options.sense != file_sense)
        throw std::invalid_argument(path + ": io.sense = " + name_of(*options.sense) + " contradicts the " + format_name +
                                    " convention (the file is a " + (file_sense == Sense::Minimize ? "minimization" : "maximization") +
                                    "); io.sense is for MOSEK files without a sense variable");
}

void finish_problem(ProblemData &problem, const std::string &path)
{
    for (std::size_t k = 0; k < problem.At.val.size(); k++)
        if (!std::isfinite(problem.At.val[k]))
            throw std::invalid_argument(path + ": non-finite value in A (svec index " + std::to_string(problem.At.rowind[k]) + ")");
    for (std::size_t k = 0; k < problem.b.val.size(); k++)
        if (!std::isfinite(problem.b.val[k]))
            throw std::invalid_argument(path + ": non-finite value in b (constraint " + std::to_string(problem.b.index[k]) + ")");
    for (std::size_t k = 0; k < problem.C.val.size(); k++)
        if (!std::isfinite(problem.C.val[k]))
            throw std::invalid_argument(path + ": non-finite value in C (svec index " + std::to_string(problem.C.index[k]) + ")");
    if (!std::isfinite(problem.objective_offset))
        throw std::invalid_argument(path + ": non-finite objective offset");
    try
    {
        problem.validate();
    }
    catch (const std::invalid_argument &e)
    {
        throw std::invalid_argument(path + ": " + e.what());
    }
}

std::vector<OriginalBlockValues> expand_to_original_blocks(const ProblemData &problem, const std::vector<double> &v)
{
    const SourceInfo &s = problem.source;
    if (static_cast<std::int64_t>(v.size()) != problem.vec_len())
        throw std::invalid_argument("expand_to_original_blocks: the vector has " + std::to_string(v.size()) + " entries, the problem " +
                                    std::to_string(problem.vec_len()));
    const std::vector<std::int64_t> off = cone_offsets(problem.cones);
    std::vector<OriginalBlockValues> out(s.original_block_sizes.size());
    for (std::size_t o = 0; o < out.size(); o++)
    {
        out[o].kind = s.original_block_kinds[o];
        out[o].size = s.original_block_sizes[o];
        out[o].values.assign(static_cast<std::size_t>(out[o].kind == 's' ? out[o].size * out[o].size : out[o].size), 0.0);
    }
    std::vector<std::int64_t> filled(out.size(), 0); // entries of the original block already placed (split cones)
    for (std::size_t b = 0; b < problem.cones.size(); b++)
    {
        const int o = b < s.original_block_of.size() ? s.original_block_of[b] : -1;
        if (o < 0)
            continue; // a block the reader created (slacks): not part of the file's structure
        OriginalBlockValues &dst = out[static_cast<std::size_t>(o)];
        const ConeSpec &c = problem.cones[b];
        const std::int64_t base = filled[static_cast<std::size_t>(o)]; // position of this cone inside the original block
        if (dst.kind == 's')
        {
            const std::int64_t n = dst.size;
            if (c.kind == ConeKind::Psd)
            {
                for (std::int64_t l = 0; l < c.size; l++)
                    for (std::int64_t k = 0; k <= l; k++)
                    {
                        const double x = v[off[b] + l * (l + 1) / 2 + k];
                        const double full = k == l ? x : x * kHalfSqrt2;
                        dst.values[(base + k) + (base + l) * n] = full;
                        dst.values[(base + l) + (base + k) * n] = full;
                    }
            }
            else // simplified to a diagonal block
                for (std::int64_t k = 0; k < c.size; k++)
                    dst.values[(base + k) + (base + k) * n] = v[off[b] + k];
        }
        else
            for (std::int64_t k = 0; k < c.size; k++)
                dst.values[base + k] = v[off[b] + k];
        filled[static_cast<std::size_t>(o)] += c.size;
    }
    for (std::size_t o = 0; o < out.size(); o++)
        if (filled[o] != out[o].size)
            throw std::invalid_argument("expand_to_original_blocks: original block " + std::to_string(o) + " of size " +
                                        std::to_string(out[o].size) + " is covered by cones of total size " + std::to_string(filled[o]));
    return out;
}

std::string solution_file_path(const std::string &path, const std::vector<std::string> &extensions, const std::string &default_name)
{
    const std::string l = lower(path);
    for (const std::string &ext : extensions)
        if (ends_with(l, ext))
            return path;
    return path + "/" + default_name;
}

} // namespace detail
} // namespace io
} // namespace cuadmm
