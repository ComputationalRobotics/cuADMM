/*

    presolve_test.hpp

    CPU tests of the Phase 7 presolve (problem/presolve.h): dependent rows (duplicates and a linear combination)
    detected and removed with a consistent b, the inconsistent b relaxed, the row map round trip, a full-rank
    problem untouched, and eps = 0 (CHOLMOD stops at the exactly zero pivot).

*/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "problem/presolve.h"

namespace presolve_test
{

using cuadmm::problem::presolve;
using cuadmm::problem::PresolveOptions;
using cuadmm::problem::PresolveResult;
using cuadmm::problem::RowMap;

// the constraint rows of A as dense n-vectors; A^T in CSC is built from them
struct Rows
{
    int n = 0;
    std::vector<std::vector<double>> rows;
    int m() const { return int(rows.size()); }
};

inline Rows random_rows(int m, int n, double density, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    Rows R;
    R.n = n;
    for (int i = 0; i < m; i++)
    {
        std::vector<double> row(std::size_t(n), 0.0);
        for (int j = 0; j < n; j++)
            if (ud(gen) < density || j == i % n)
                row[std::size_t(j)] = nd(gen);
        R.rows.push_back(row);
    }
    return R;
}

struct AtCsc
{
    int m = 0, n = 0;
    std::vector<int> p, i;
    std::vector<double> x;
};

inline AtCsc to_at_csc(const Rows &R)
{
    AtCsc At;
    At.m = R.m();
    At.n = R.n;
    At.p.push_back(0);
    for (const auto &row : R.rows)
    {
        for (int j = 0; j < R.n; j++)
            if (row[std::size_t(j)] != 0.0)
            {
                At.i.push_back(j);
                At.x.push_back(row[std::size_t(j)]);
            }
        At.p.push_back(int(At.i.size()));
    }
    return At;
}

inline std::vector<double> times(const Rows &R, const std::vector<double> &x)
{
    std::vector<double> b(std::size_t(R.m()), 0.0);
    for (int i = 0; i < R.m(); i++)
        for (int j = 0; j < R.n; j++)
            b[std::size_t(i)] += R.rows[std::size_t(i)][std::size_t(j)] * x[std::size_t(j)];
    return b;
}

inline std::vector<double> random_vector(int n, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd;
    std::vector<double> v(static_cast<std::size_t>(n));
    for (double &e : v)
        e = nd(gen);
    return v;
}

// 30 independent random rows (n = 100), then rows 30..34 duplicate rows 0..4 and row 35 = 2 row_5 - 3 row_6
inline Rows dependent_rows_problem()
{
    Rows R = random_rows(30, 100, 0.3, 301);
    for (int i = 0; i < 5; i++)
        R.rows.push_back(R.rows[std::size_t(i)]);
    std::vector<double> combo(std::size_t(R.n));
    for (int j = 0; j < R.n; j++)
        combo[std::size_t(j)] = 2.0 * R.rows[5][std::size_t(j)] - 3.0 * R.rows[6][std::size_t(j)];
    R.rows.push_back(combo);
    return R;
}

inline PresolveResult run(const Rows &R, const std::vector<double> &b, const PresolveOptions &options = PresolveOptions())
{
    const AtCsc At = to_at_csc(R);
    return presolve(At.m, At.n, At.p.data(), At.i.data(), At.x.data(), b.data(), options);
}

} // namespace presolve_test

TEST(Presolve, DependentRowsRemovedWithConsistentB)
{
    using namespace presolve_test;
    const Rows R = dependent_rows_problem();
    const std::vector<double> x0 = random_vector(R.n, 302);
    const std::vector<double> b = times(R, x0); // consistent by construction
    const PresolveResult res = run(R, b);
    std::printf("[ PRESOLVE ] %s\n", res.message.c_str());
    std::printf("[ PRESOLVE ] pivots in [%.2e, %.2e], %d tiny, %d nonpositive, span residual %.2e, consistency residual %.2e, %d rounds\n",
                res.min_pivot, res.max_pivot, res.tiny_pivots, res.nonpositive_pivots, res.span_residual, res.consistency_residual, res.rounds);
    EXPECT_FALSE(res.relaxed);
    EXPECT_TRUE(res.consistent);
    EXPECT_EQ(res.removed_rows, 6);
    EXPECT_EQ(res.dependent_rows, 6);
    EXPECT_EQ(res.m, 30);
    EXPECT_EQ(res.n, R.n);
    EXPECT_EQ(res.map.m_full, 36);
    EXPECT_EQ(res.map.m_reduced, 30);
    EXPECT_EQ(int(res.map.removed.size()), 6);
    EXPECT_LE(res.span_residual, 1e-12);
    EXPECT_LE(res.consistency_residual, 1e-12);
    EXPECT_GE(res.tiny_pivots, 5); // the first factorization sees the dependent rows (one of each pair, and the combination)
    // one row of each duplicate pair is kept, and at least two of {5, 6, 35}
    for (int i = 0; i < 5; i++)
        EXPECT_NE(res.map.to_reduced[std::size_t(i)] >= 0, res.map.to_reduced[std::size_t(30 + i)] >= 0) << "pair " << i;
    const int kept_of_combo = int(res.map.to_reduced[5] >= 0) + int(res.map.to_reduced[6] >= 0) + int(res.map.to_reduced[35] >= 0);
    EXPECT_EQ(kept_of_combo, 2);
    // the reduced problem is the kept columns of A^T and the kept entries of b
    const AtCsc At = to_at_csc(R);
    ASSERT_EQ(int(res.at_colptr.size()), res.m + 1);
    for (int k = 0; k < res.m; k++)
    {
        const int j = res.map.kept[std::size_t(k)];
        ASSERT_EQ(res.at_colptr[std::size_t(k) + 1] - res.at_colptr[std::size_t(k)], At.p[std::size_t(j) + 1] - At.p[std::size_t(j)]);
        for (int q = 0; q < At.p[std::size_t(j) + 1] - At.p[std::size_t(j)]; q++)
        {
            EXPECT_EQ(res.at_rowind[std::size_t(res.at_colptr[std::size_t(k)] + q)], At.i[std::size_t(At.p[std::size_t(j)] + q)]);
            EXPECT_EQ(res.at_val[std::size_t(res.at_colptr[std::size_t(k)] + q)], At.x[std::size_t(At.p[std::size_t(j)] + q)]);
        }
        EXPECT_EQ(res.b[std::size_t(k)], b[std::size_t(j)]);
    }
    // the reduced problem has full row rank: a second presolve leaves it untouched
    const PresolveResult again = presolve(res.m, res.n, res.at_colptr.data(), res.at_rowind.data(), res.at_val.data(), res.b.data());
    std::printf("[ PRESOLVE ] reduced problem again: %s\n", again.message.c_str());
    EXPECT_EQ(again.removed_rows, 0);
    EXPECT_EQ(again.dependent_rows, 0);
    EXPECT_TRUE(again.map.is_identity());
}

TEST(Presolve, InconsistentBRelaxes)
{
    using namespace presolve_test;
    const Rows R = dependent_rows_problem();
    std::vector<double> b = times(R, random_vector(R.n, 303));
    b[32] += 1.0;  // the duplicate of row 2 now asks for a different value
    b[35] -= 0.5;  // and the combination row
    const PresolveResult res = run(R, b);
    std::printf("[ PRESOLVE ] %s\n", res.message.c_str());
    EXPECT_TRUE(res.relaxed);
    EXPECT_FALSE(res.consistent);
    EXPECT_EQ(res.removed_rows, 0);
    EXPECT_EQ(res.dependent_rows, 6);
    // the residual is relative to max(1, ||b||_inf); whichever row of the pair {2, 32} was removed, its implied value
    // misses by the full unit perturbation
    double b_inf = 0.0;
    for (double e : b)
        b_inf = std::max(b_inf, std::abs(e));
    EXPECT_GE(res.consistency_residual, 0.999 / std::max(1.0, b_inf));
    EXPECT_LE(res.consistency_residual, 1.001 / std::max(1.0, b_inf));
    EXPECT_NE(res.message.find("inconsistent"), std::string::npos);
    EXPECT_TRUE(res.map.is_identity());
    EXPECT_EQ(res.m, 36);
    EXPECT_EQ(res.b, b);
    const AtCsc At = to_at_csc(R);
    EXPECT_EQ(res.at_colptr, At.p);
    EXPECT_EQ(res.at_rowind, At.i);
    EXPECT_EQ(res.at_val, At.x);
    // relax_on_failure = false: the inconsistency is an error
    PresolveOptions strict;
    strict.relax_on_failure = false;
    EXPECT_THROW(run(R, b, strict), std::runtime_error);
}

TEST(Presolve, FullRankUntouched)
{
    using namespace presolve_test;
    const Rows R = random_rows(40, 100, 0.3, 304);
    const std::vector<double> b = random_vector(R.m(), 305);
    const PresolveResult res = run(R, b);
    std::printf("[ PRESOLVE ] %s\n", res.message.c_str());
    EXPECT_FALSE(res.relaxed);
    EXPECT_TRUE(res.consistent);
    EXPECT_EQ(res.removed_rows, 0);
    EXPECT_EQ(res.dependent_rows, 0);
    EXPECT_EQ(res.tiny_pivots, 0);
    EXPECT_EQ(res.rounds, 1);
    EXPECT_GT(res.min_pivot, 0.0);
    EXPECT_TRUE(res.map.is_identity());
    const AtCsc At = to_at_csc(R);
    EXPECT_EQ(res.m, At.m);
    EXPECT_EQ(res.at_colptr, At.p);
    EXPECT_EQ(res.at_rowind, At.i);
    EXPECT_EQ(res.at_val, At.x);
    EXPECT_EQ(res.b, b);
    EXPECT_NE(res.message.find("independent"), std::string::npos);
}

TEST(Presolve, RowMapRoundTrip)
{
    using namespace presolve_test;
    const RowMap map = RowMap::from_removed(10, {7, 2});
    EXPECT_EQ(map.m_full, 10);
    EXPECT_EQ(map.m_reduced, 8);
    EXPECT_EQ(map.removed, (std::vector<int>{2, 7}));
    EXPECT_EQ(map.kept, (std::vector<int>{0, 1, 3, 4, 5, 6, 8, 9}));
    EXPECT_EQ(map.to_reduced, (std::vector<int>{0, 1, -1, 2, 3, 4, 5, -1, 6, 7}));
    const std::vector<double> v = random_vector(10, 306);
    const std::vector<double> reduced = map.restrict(v);
    ASSERT_EQ(int(reduced.size()), 8);
    for (int k = 0; k < 8; k++)
        EXPECT_EQ(reduced[std::size_t(k)], v[std::size_t(map.kept[std::size_t(k)])]);
    const std::vector<double> full = map.expand(reduced);
    ASSERT_EQ(int(full.size()), 10);
    for (int i = 0; i < 10; i++)
        EXPECT_EQ(full[std::size_t(i)], map.to_reduced[std::size_t(i)] < 0 ? 0.0 : v[std::size_t(i)]);
    EXPECT_THROW(map.expand(v), std::invalid_argument);
    EXPECT_THROW(map.restrict(reduced), std::invalid_argument);
    EXPECT_THROW(RowMap::from_removed(10, {10}), std::invalid_argument);
    const RowMap id = RowMap::identity(4);
    EXPECT_TRUE(id.is_identity());
    EXPECT_EQ(id.expand(std::vector<double>{1, 2, 3, 4}), (std::vector<double>{1, 2, 3, 4}));
}

TEST(Presolve, ZeroEpsExactDuplicateStopsCholmodAndIsRemoved)
{
    using namespace presolve_test;
    Rows R = random_rows(20, 60, 0.3, 307);
    R.rows.push_back(R.rows[3]);
    const std::vector<double> b = times(R, random_vector(R.n, 308));
    PresolveOptions options;
    options.eps = 0.0;
    const PresolveResult res = run(R, b, options);
    std::printf("[ PRESOLVE ] eps = 0: %s\n", res.message.c_str());
    EXPECT_FALSE(res.relaxed);
    EXPECT_EQ(res.removed_rows, 1);
    EXPECT_NE(res.map.to_reduced[3] >= 0, res.map.to_reduced[20] >= 0);
    EXPECT_LE(res.span_residual, 1e-12);
    EXPECT_LE(res.consistency_residual, 1e-12);
}

TEST(Presolve, RejectsMalformedInput)
{
    using namespace presolve_test;
    const Rows R = random_rows(3, 5, 0.5, 309);
    AtCsc At = to_at_csc(R);
    const std::vector<double> b(3, 0.0);
    EXPECT_THROW(presolve(At.m, At.n, nullptr, At.i.data(), At.x.data(), b.data()), std::invalid_argument);
    At.i[0] = 99;
    EXPECT_THROW(presolve(At.m, At.n, At.p.data(), At.i.data(), At.x.data(), b.data()), std::invalid_argument);
    // no constraints at all
    const int p0[1] = {0};
    const PresolveResult empty = presolve(0, 5, p0, nullptr, nullptr, b.data());
    EXPECT_EQ(empty.m, 0);
    EXPECT_FALSE(empty.changed());
}
