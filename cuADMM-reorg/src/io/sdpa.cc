/*

    sdpa.cc

    SDPA sparse format (.dat-s, .dat-s.gz), plan Section 12, with the checks of the reference converter
    experiments/2026-09-29_plato_cold_plain_admm_sigma/scripts/sdpa_to_cuadmm.py:

        SDPA pair:    (P) min c^T x  s.t.  X = sum_i F_i x_i - F_0 >= 0;   (D) max <F_0, Y>  s.t.  <F_i, Y> = c_i, Y >= 0
        cuADMM:       min <C, X>  s.t.  <A_i, X> = b_i, X in K
        mapping:      C = -F_0,  A_i = F_i,  b_i = c_i,  X_cuADMM = Y_SDPA,  y_cuADMM = -x_SDPA,  S_cuADMM = X_SDPA;
                      the file's problem is a maximization of <F_0, Y> = -<C, X> (sense = max, offset 0), so the
                      reported objective is the published SDPA value.

    Blocks: a positive size n is an n x n PSD block ('s n'), a negative size -n a diagonal block ('l n'); a 1 x 1
    PSD block becomes 'l 1' through io.simplify (as the converter wrote it). The DIMACS kind of every block ('s' for
    PSD in the file, 'l' for diagonal) is kept for the error measures.
    Checks (the file is rejected): header and counts consistent; matrix number in [0, m], block in [1, nBLOCK],
    indices in range (1-based), i == j in diagonal blocks, finite values, duplicate entries (same matrix, block and
    unordered (i, j)), empty constraint matrices F_i (A would be rank deficient).
    Reported but accepted: explicit zeros (dropped), lower-triangle entries (mirrored), blocks that no F_i touches.

*/

#include "io/readers.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>

#include "io/csc_builder.h"
#include "io/svec.h"
#include "io/text_source.h"

namespace cuadmm
{
namespace io
{
namespace detail
{

namespace
{

inline bool is_separator(char c) { return c == ' ' || c == '\t' || c == ',' || c == '{' || c == '}' || c == '(' || c == ')' || c == '\r' || c == '\f' || c == '\v'; }

// splits a line at whitespace, commas, braces and parentheses (as the converter's tokens())
void tokenize(std::string_view line, std::vector<std::string_view> &tokens)
{
    tokens.clear();
    std::size_t i = 0;
    while (i < line.size())
    {
        while (i < line.size() && is_separator(line[i]))
            i++;
        const std::size_t start = i;
        while (i < line.size() && !is_separator(line[i]))
            i++;
        if (i > start)
            tokens.push_back(line.substr(start, i - start));
    }
}

// an integer token, also accepted with a trailing ".0" (as the converter's int(float(t)) for `[-+]?\d+(\.0*)?`)
bool parse_int_token(std::string_view t, std::int64_t &value)
{
    if (t.empty())
        return false;
    std::size_t start = (t[0] == '+') ? 1 : 0;
    const char *b = t.data() + start, *e = t.data() + t.size();
    std::int64_t v = 0;
    const std::from_chars_result r = std::from_chars(b, e, v);
    if (r.ec != std::errc() || r.ptr == b)
        return false;
    if (r.ptr != e)
    {
        if (*r.ptr != '.')
            return false;
        for (const char *p = r.ptr + 1; p < e; p++)
            if (*p != '0')
                return false;
    }
    value = v;
    return true;
}

bool parse_double_token(std::string_view t, double &value)
{
    if (t.empty())
        return false;
    std::size_t start = (t[0] == '+') ? 1 : 0;
    const char *b = t.data() + start, *e = t.data() + t.size();
    const std::from_chars_result r = std::from_chars(b, e, value);
    return r.ec == std::errc() && r.ptr == e;
}

struct BlockInfo
{
    std::int64_t signed_size; // as in the file
    ConeKind kind;
    std::int64_t size;
    std::int64_t offset; // svec offset
};

} // namespace

ProblemData read_sdpa(const std::string &path, const IoOptions &options)
{
    TextSource src(path);
    auto fail = [&](const std::string &why) { throw std::invalid_argument(path + ", line " + std::to_string(src.line_number()) + ": " + why); };
    std::string_view line;
    std::vector<std::string_view> tokens;

    // leading comment lines (`"` or `*`) and blank lines
    bool have_line = false;
    std::int64_t comment_lines = 0;
    while (src.next_line(line))
    {
        std::size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
            i++;
        if (i == line.size() || line[i] == '"' || line[i] == '*')
        {
            comment_lines++;
            continue;
        }
        have_line = true;
        break;
    }
    if (!have_line)
        fail("unexpected end of file while reading m");

    // header integers: `count` values; a non-numeric token ends the current line, the values may continue on the next
    bool line_pending = true; // `line` holds an unread line
    auto next_ints = [&](std::int64_t count, const char *what)
    {
        std::vector<std::int64_t> vals;
        while (static_cast<std::int64_t>(vals.size()) < count)
        {
            if (!line_pending && !src.next_line(line))
                fail(std::string("unexpected end of file while reading ") + what);
            line_pending = false;
            tokenize(line, tokens);
            for (std::string_view t : tokens)
            {
                if (static_cast<std::int64_t>(vals.size()) == count)
                    break; // trailing text (e.g. '= mDIM') is ignored
                std::int64_t v;
                if (!parse_int_token(t, v))
                    break; // a non-numeric token ends the header line
                vals.push_back(v);
            }
        }
        return vals;
    };
    const std::int64_t m = next_ints(1, "m")[0];
    const std::int64_t nblock = next_ints(1, "nBLOCK")[0];
    if (m <= 0 || nblock <= 0)
        fail("bad header: m = " + std::to_string(m) + ", nBLOCK = " + std::to_string(nblock));
    const std::vector<std::int64_t> sizes = next_ints(nblock, "bLOCKsTRUCT");
    for (std::int64_t s : sizes)
        if (s == 0)
            fail("bad header: a block of size 0");

    // c (m values, possibly over several lines; extra tokens on the last line are ignored)
    std::vector<double> c;
    c.reserve(static_cast<std::size_t>(m));
    while (static_cast<std::int64_t>(c.size()) < m)
    {
        if (!src.next_line(line))
            fail("unexpected end of file while reading c");
        tokenize(line, tokens);
        for (std::string_view t : tokens)
        {
            if (static_cast<std::int64_t>(c.size()) == m)
                break;
            double v;
            if (!parse_double_token(t, v))
                fail("not a number in c: '" + std::string(t) + "'");
            c.push_back(v);
        }
    }

    // blocks
    std::vector<BlockInfo> blocks(static_cast<std::size_t>(nblock));
    std::int64_t vec_len = 0;
    for (std::int64_t b = 0; b < nblock; b++)
    {
        BlockInfo &bi = blocks[static_cast<std::size_t>(b)];
        bi.signed_size = sizes[static_cast<std::size_t>(b)];
        bi.kind = bi.signed_size > 0 ? ConeKind::Psd : ConeKind::Nonnegative;
        bi.size = std::abs(bi.signed_size);
        bi.offset = vec_len;
        vec_len += bi.kind == ConeKind::Psd ? bi.size * (bi.size + 1) / 2 : bi.size;
    }

    // entries
    CscBuilder At(vec_len, m);
    SparseVectorBuilder C(vec_len);
    std::int64_t entry_lines = 0, lower_mirrored = 0;
    std::vector<char> touched(static_cast<std::size_t>(nblock), 0);
    while (src.next_line(line))
    {
        tokenize(line, tokens);
        if (tokens.empty())
            continue;
        if (tokens.size() != 5)
            fail("expected 5 fields, got " + std::to_string(tokens.size()) + ": '" + std::string(line) + "'");
        std::int64_t matno, blk, i, j;
        if (!parse_int_token(tokens[0], matno) || !parse_int_token(tokens[1], blk) || !parse_int_token(tokens[2], i) || !parse_int_token(tokens[3], j))
            fail("the first four fields must be integers: '" + std::string(line) + "'");
        double v;
        if (!parse_double_token(tokens[4], v))
            fail("not a number: '" + std::string(tokens[4]) + "'");
        entry_lines++;
        if (matno < 0 || matno > m)
            fail("matrix number " + std::to_string(matno) + " outside [0, " + std::to_string(m) + "]");
        if (blk < 1 || blk > nblock)
            fail("block " + std::to_string(blk) + " outside [1, " + std::to_string(nblock) + "]");
        const BlockInfo &bi = blocks[static_cast<std::size_t>(blk - 1)];
        if (i < 1 || i > bi.size || j < 1 || j > bi.size)
            fail("index (" + std::to_string(i) + ", " + std::to_string(j) + ") outside a block of size " + std::to_string(bi.size));
        if (bi.kind == ConeKind::Nonnegative && i != j)
            fail("off-diagonal entry (" + std::to_string(i) + ", " + std::to_string(j) + ") in the diagonal block " + std::to_string(blk));
        if (!std::isfinite(v))
            fail("non-finite value " + std::string(tokens[4]));
        if (i > j)
        {
            std::swap(i, j);
            lower_mirrored++;
        }
        const std::int64_t idx = bi.kind == ConeKind::Psd ? svec_index(bi.offset, i - 1, j - 1) : bi.offset + i - 1;
        const double val = (bi.kind == ConeKind::Psd && i != j) ? v * kSqrt2 : v;
        if (matno == 0)
            C.add(idx, -val); // C = -F_0
        else
        {
            At.add(matno - 1, idx, val);
            touched[static_cast<std::size_t>(blk - 1)] = 1;
        }
    }

    // duplicates are rejected (same matrix, block and unordered (i, j)); explicit zeros dropped afterwards
    auto name_item = [&](std::int64_t matno, std::int64_t idx)
    {
        std::int64_t b = 0;
        while (b + 1 < nblock && blocks[static_cast<std::size_t>(b + 1)].offset <= idx)
            b++;
        const BlockInfo &bi = blocks[static_cast<std::size_t>(b)];
        std::int64_t k = idx - bi.offset, l = k;
        if (bi.kind == ConeKind::Psd)
            svec_position(idx - bi.offset, k, l);
        return "matrix " + std::to_string(matno) + ", block " + std::to_string(b + 1) + ", entry (" + std::to_string(k + 1) + ", " +
               std::to_string(l + 1) + ")";
    };
    ProblemData p;
    BuildStats at_stats, c_stats;
    p.At = At.finish(DuplicatePolicy::Reject, true, at_stats);
    if (at_stats.duplicates > 0)
        throw std::invalid_argument(path + ": " + std::to_string(at_stats.duplicates) + " duplicate entries (first: " +
                                    name_item(at_stats.first_duplicate_col + 1, at_stats.first_duplicate_row) + ")");
    p.C = C.finish(DuplicatePolicy::Reject, true, c_stats);
    if (c_stats.duplicates > 0)
        throw std::invalid_argument(path + ": " + std::to_string(c_stats.duplicates) + " duplicate entries in F_0 (first: " +
                                    name_item(0, c_stats.first_duplicate_row) + ")");
    std::vector<std::int64_t> empty;
    for (std::int64_t k = 0; k < m; k++)
        if (p.At.colptr[k] == p.At.colptr[k + 1])
            empty.push_back(k + 1);
    if (!empty.empty())
    {
        std::string first;
        for (std::size_t k = 0; k < empty.size() && k < 5; k++)
            first += (k ? ", " : "") + std::to_string(empty[k]);
        throw std::invalid_argument(path + ": " + std::to_string(empty.size()) + " empty constraint matrices F_i (first: " + first + ")");
    }

    p.b.size = m;
    for (std::int64_t k = 0; k < m; k++)
        if (c[static_cast<std::size_t>(k)] != 0.0)
        {
            if (!std::isfinite(c[static_cast<std::size_t>(k)]))
                throw std::invalid_argument(path + ": non-finite value in c (entry " + std::to_string(k + 1) + ")");
            p.b.index.push_back(k);
            p.b.val.push_back(c[static_cast<std::size_t>(k)]);
        }
    for (const BlockInfo &bi : blocks)
    {
        ConeSpec cs;
        cs.kind = bi.kind;
        cs.size = bi.size;
        p.cones.push_back(cs);
    }
    p.sense = Sense::Maximize; // the file's value is max <F_0, Y> = -<C, X>
    p.objective_offset = 0.0;
    check_sense_option(options, p.sense, path, "SDPA");
    p.source.format = "sdpa";
    p.source.path = path;
    set_identity_original_blocks(p);
    set_dimacs_kinds_from_original(p);
    std::string untouched;
    std::int64_t n_untouched = 0;
    for (std::int64_t b = 0; b < nblock; b++)
        if (!touched[static_cast<std::size_t>(b)])
        {
            if (n_untouched < 10)
                untouched += (n_untouched ? ", " : "") + std::to_string(b + 1);
            n_untouched++;
        }
    p.source.report.push_back(std::string("SDPA sparse") + (src.compressed() ? " (gzip)" : "") + ": m = " + std::to_string(m) + ", " +
                              std::to_string(nblock) + " blocks, " + std::to_string(entry_lines) + " entries, " + std::to_string(comment_lines) +
                              " comment lines; mapped as C = -F_0, A_i = F_i, b = c (the file's objective is max <F_0, Y>)");
    p.source.report.push_back("explicit zeros dropped: " + std::to_string(at_stats.zeros_dropped + c_stats.zeros_dropped) +
                              "; lower-triangle entries mirrored: " + std::to_string(lower_mirrored));
    if (n_untouched > 0)
        p.source.report.push_back("blocks not touched by any F_i: " + std::to_string(n_untouched) + " (" + untouched + (n_untouched > 10 ? ", ..." : "") + ")");
    apply_simplifications(p, options.simplify);
    finish_problem(p, path);
    return p;
}

} // namespace detail
} // namespace io
} // namespace cuadmm
