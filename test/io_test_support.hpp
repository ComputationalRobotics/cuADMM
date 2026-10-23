/*

    io_test_support.hpp  (CPU only)

    Shared helpers of the format tests (plan Section 16, "Formats"): temporary directories, file writing, the
    comparison of two ProblemData (bitwise, or with a normalized tolerance), a small hand-made problem, a fake
    SolveResult, and the locations of the external data (skipped when absent).

*/

#ifndef CUADMM_IO_TEST_SUPPORT_HPP
#define CUADMM_IO_TEST_SUPPORT_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "cuadmm/io.h"
#include "cuadmm/problem.h"
#include "cuadmm/result.h"

#ifndef CUADMM_EXTERNAL_DATA_DIR
#define CUADMM_EXTERNAL_DATA_DIR ""
#endif
#ifndef CUADMM_SPOT_DATA_DIR
#define CUADMM_SPOT_DATA_DIR ""
#endif

namespace io_test
{

using namespace cuadmm;

inline std::string temp_dir(const std::string &name)
{
    const std::string dir = "io_test_tmp/" + name;
    std::filesystem::create_directories(dir);
    return dir;
}

inline std::string write_file(const std::string &path, const std::string &content)
{
    std::ofstream f(path);
    f << content;
    return path;
}

inline std::string external(const std::string &relative)
{
    const char *env = std::getenv("CUADMM_EXTERNAL_DATA_DIR");
    const std::string base = env && *env ? env : CUADMM_EXTERNAL_DATA_DIR;
    return base.empty() ? "" : base + "/" + relative;
}

inline std::string spot(const std::string &relative)
{
    const char *env = std::getenv("CUADMM_SPOT_DATA_DIR");
    const std::string base = env && *env ? env : CUADMM_SPOT_DATA_DIR;
    return base.empty() ? "" : base + "/" + relative;
}

inline bool exists(const std::string &path) { return !path.empty() && std::filesystem::exists(path); }

#define IO_TEST_REQUIRE_FILE(path)                                                                                                        \
    if (!io_test::exists(path))                                                                                                            \
    GTEST_SKIP() << "external data not available: " << (path)

// the entries of At as a map (svec index, constraint) -> value, and of a sparse vector as index -> value
inline std::map<std::pair<std::int64_t, std::int64_t>, double> entries(const CscMatrix &M)
{
    std::map<std::pair<std::int64_t, std::int64_t>, double> out;
    for (std::int64_t j = 0; j < M.cols; j++)
        for (std::int64_t k = M.colptr[j]; k < M.colptr[j + 1]; k++)
            out[{M.rowind[k], j}] += M.val[k];
    return out;
}

inline std::map<std::int64_t, double> entries(const SparseVector &v)
{
    std::map<std::int64_t, double> out;
    for (std::size_t k = 0; k < v.val.size(); k++)
        out[v.index[k]] += v.val[k];
    return out;
}

struct Difference
{
    double max_abs = 0.0;        // max |a - b| over the union of the entries
    double max_ref = 0.0;        // max |b|
    double normalized() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
    std::size_t only_on_one_side = 0;
};

template <typename K>
Difference compare_maps(const std::map<K, double> &a, const std::map<K, double> &b)
{
    Difference d;
    for (const auto &kv : b)
        d.max_ref = std::max(d.max_ref, std::fabs(kv.second));
    for (const auto &kv : a)
    {
        auto it = b.find(kv.first);
        if (it == b.end())
        {
            d.only_on_one_side++;
            d.max_abs = std::max(d.max_abs, std::fabs(kv.second));
        }
        else
            d.max_abs = std::max(d.max_abs, std::fabs(kv.second - it->second));
    }
    for (const auto &kv : b)
        if (a.find(kv.first) == a.end())
        {
            d.only_on_one_side++;
            d.max_abs = std::max(d.max_abs, std::fabs(kv.second));
        }
    return d;
}

inline std::string cones_string(const std::vector<ConeSpec> &cones)
{
    std::string s;
    for (const ConeSpec &c : cones)
        s += std::string(1, cone_code(c.kind)) + std::to_string(c.size) + " ";
    return s;
}

// the same problem up to `tol` on max|diff| / max|ref| (0: bitwise); blocks, sense and offset must be equal
inline void expect_same_problem(const ProblemData &a, const ProblemData &b, double tol, const std::string &what)
{
    SCOPED_TRACE(what);
    ASSERT_EQ(a.cones.size(), b.cones.size()) << cones_string(a.cones) << " vs " << cones_string(b.cones);
    for (std::size_t k = 0; k < a.cones.size(); k++)
    {
        EXPECT_EQ(a.cones[k].kind, b.cones[k].kind) << "block " << k;
        EXPECT_EQ(a.cones[k].size, b.cones[k].size) << "block " << k;
    }
    EXPECT_EQ(a.vec_len(), b.vec_len());
    EXPECT_EQ(a.num_constraints(), b.num_constraints());
    EXPECT_EQ(a.sense, b.sense);
    EXPECT_EQ(a.objective_offset, b.objective_offset);
    const Difference dA = compare_maps(entries(a.At), entries(b.At)), db = compare_maps(entries(a.b), entries(b.b)), dC = compare_maps(entries(a.C), entries(b.C));
    EXPECT_EQ(dA.only_on_one_side, 0u) << "At entries present on one side only";
    EXPECT_EQ(db.only_on_one_side, 0u);
    EXPECT_EQ(dC.only_on_one_side, 0u);
    EXPECT_LE(dA.normalized(), tol) << "At: max|diff| " << dA.max_abs << ", max|ref| " << dA.max_ref;
    EXPECT_LE(db.normalized(), tol) << "b: max|diff| " << db.max_abs;
    EXPECT_LE(dC.normalized(), tol) << "C: max|diff| " << dC.max_abs;
    std::printf("  %s: At %zu entries, max|diff|/max|ref| = %.3e; b %.3e; C %.3e\n", what.c_str(), a.At.val.size(), dA.normalized(), db.normalized(), dC.normalized());
}

inline void expect_bitwise(const ProblemData &a, const ProblemData &b, const std::string &what)
{
    expect_same_problem(a, b, 0.0, what);
    EXPECT_EQ(a.At.colptr, b.At.colptr);
    EXPECT_EQ(a.At.rowind, b.At.rowind);
    EXPECT_EQ(a.At.val, b.At.val);
    EXPECT_EQ(a.b.index, b.b.index);
    EXPECT_EQ(a.b.val, b.b.val);
    EXPECT_EQ(a.C.index, b.C.index);
    EXPECT_EQ(a.C.val, b.C.val);
}

// a small problem: blocks s 2 (svec 0..2), l 2 (3..4), u 1 (5); 3 constraints
inline ProblemData small_problem()
{
    ProblemData p;
    p.cones = {{ConeKind::Psd, 2, 0.0}, {ConeKind::Nonnegative, 2, 0.0}, {ConeKind::Free, 1, 0.0}};
    p.At.rows = 6;
    p.At.cols = 3;
    p.At.colptr = {0, 3, 5, 7};
    p.At.rowind = {0, 1, 2, 3, 5, 1, 4};
    p.At.val = {1.0, 1.4142135623730951, 1.0, 2.0, -1.0, 0.5, 3.0};
    p.b.size = 3;
    p.b.index = {0, 2};
    p.b.val = {1.0, -2.5};
    p.C.size = 6;
    p.C.index = {0, 2, 3, 5};
    p.C.val = {1.0, 1.0, 0.25, -1.0};
    p.sense = Sense::Minimize;
    p.objective_offset = 0.0;
    p.source.format = "test";
    p.source.path = "";
    p.source.report = {"hand-made problem"};
    p.source.dimacs_kinds = {'s', 'l', 'l'};
    p.source.original_block_of = {0, 1, 2};
    p.source.original_block_sizes = {2, 2, 1};
    p.source.original_block_kinds = {'s', 'l', 'u'};
    p.validate();
    return p;
}

inline SolveResult fake_result(const ProblemData &p)
{
    SolveResult r;
    r.status = Status::Converged;
    r.stop_reason = "internal_kkt";
    r.converged = true;
    r.iterations = 7;
    r.best.iteration = 7;
    for (std::int64_t i = 0; i < p.vec_len(); i++)
    {
        r.best.X.push_back(0.5 * static_cast<double>(i) + 0.25);
        r.best.S.push_back(-0.125 * static_cast<double>(i) + 1.0);
    }
    for (std::int64_t i = 0; i < p.num_constraints(); i++)
        r.best.y.push_back(static_cast<double>(i) - 0.75);
    r.final = r.best;
    r.pobj = 3.5;
    r.dobj = 3.25;
    r.eta_p = 1e-5;
    r.eta_d = 2e-5;
    r.eta_g = 3e-5;
    r.sigma_final_original = 0.7;
    return r;
}

} // namespace io_test

#endif // CUADMM_IO_TEST_SUPPORT_HPP
