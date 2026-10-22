/*

    io_mat_test.hpp  (CPU only; skipped without matio)

    SeDuMi, SDPT3 and MOSEK prob .mat files, MATLAB v5 and v7.3: hand-made edge cases written through matio
    (A vs At, N = m, asymmetric PSD storage, K.q rejected, complex rejected; an 's' entry with several sizes,
    l / u blocks; MOSEK bounds and ranges, missing sense, maximization, cones rejected), the detection from the
    variables, the Python-written SDPT3 files of test/data, the PLATO buck3 SeDuMi file against the SDPA file,
    the SPOT MOSEK files against their TXT conversions, and the solution writers read back.

*/

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "io/readers.h"
#include "io_test_support.hpp"

#ifdef CUADMM_HAVE_MATIO
#include "io/mat_common.h"
#endif

namespace
{
using namespace io_test;
#ifdef CUADMM_HAVE_MATIO
using namespace cuadmm::io::detail;

std::vector<double> dense(std::int64_t rows, std::int64_t cols, std::initializer_list<double> column_major)
{
    std::vector<double> v(column_major);
    if (static_cast<std::int64_t>(v.size()) != rows * cols)
        throw std::logic_error("dense: wrong size");
    return v;
}

// SeDuMi: K.f = 1, K.l = 2, K.s = [2]: N = 1 + 2 + 4 = 7, m = 2 (or N = m = 7 with 7 constraints)
// A (m x N) row 0: [1 2 0 | 3 0.5 0.5 4]  (symmetric PSD part: (A_01 + A_10)/sqrt2 = 1/sqrt2)
//           row 1: [0 0 1 | 0 1.0 0.0 0]  (asymmetric: only A_10 = 1 -> svec = 1/sqrt2)
void write_sedumi(const std::string &path, bool v73, bool store_At, std::int64_t m_override = 2)
{
    const std::int64_t N = 7, m = m_override;
    std::vector<double> A(static_cast<std::size_t>(m * N), 0.0);
    auto set = [&](std::int64_t i, std::int64_t j, double v) { A[static_cast<std::size_t>(i + j * m)] = v; };
    set(0, 0, 1.0); set(0, 1, 2.0); set(0, 3, 3.0); set(0, 4, 0.5); set(0, 5, 0.5); set(0, 6, 4.0);
    set(1, 2, 1.0); set(1, 4, 1.0);
    for (std::int64_t i = 2; i < m; i++)
        set(i, 3, static_cast<double>(i)); // extra rows for the N = m case
    MatFile f = MatFile::create(path, v73);
    if (store_At)
    {
        std::vector<double> At(static_cast<std::size_t>(N * m));
        for (std::int64_t i = 0; i < m; i++)
            for (std::int64_t j = 0; j < N; j++)
                At[static_cast<std::size_t>(j + i * N)] = A[static_cast<std::size_t>(i + j * m)];
        write_variable(f, make_double_matrix("At", N, m, At.data()));
    }
    else
        write_variable(f, make_double_matrix("A", m, N, A.data()));
    std::vector<double> b(static_cast<std::size_t>(m), 1.0);
    b[1] = -2.0;
    write_variable(f, make_double_matrix("b", m, 1, b.data()));
    const std::vector<double> c = dense(N, 1, {1.0, 0.0, 0.0, 5.0, 1.0, 1.0, 6.0});
    write_variable(f, make_double_matrix("c", N, 1, c.data()));
    std::vector<std::pair<std::string, matvar_t *>> K;
    const double kf = 1.0, kl = 2.0, ks = 2.0;
    K.emplace_back("f", make_double_matrix("f", 1, 1, &kf));
    K.emplace_back("l", make_double_matrix("l", 1, 1, &kl));
    K.emplace_back("s", make_double_matrix("s", 1, 1, &ks));
    write_variable(f, make_struct("K", K));
}

void check_sedumi(const ProblemData &p, const std::string &what)
{
    SCOPED_TRACE(what);
    ASSERT_EQ(p.cones.size(), 3u) << cones_string(p.cones);
    EXPECT_EQ(p.cones[0].kind, ConeKind::Free);
    EXPECT_EQ(p.cones[0].size, 1);
    EXPECT_EQ(p.cones[1].kind, ConeKind::Nonnegative);
    EXPECT_EQ(p.cones[1].size, 2);
    EXPECT_EQ(p.cones[2].kind, ConeKind::Psd);
    EXPECT_EQ(p.cones[2].size, 2);
    EXPECT_EQ(p.vec_len(), 6);
    const auto e = entries(p.At);
    EXPECT_EQ(e.at({0, 0}), 1.0);
    EXPECT_EQ(e.at({1, 0}), 2.0);
    EXPECT_EQ(e.at({3, 0}), 3.0);
    EXPECT_EQ(e.at({4, 0}), 0.5 * kSqrt2); // symmetric pair -> exactly 0.5 * sqrt2
    EXPECT_EQ(e.at({5, 0}), 4.0);
    EXPECT_EQ(e.at({2, 1}), 1.0);
    EXPECT_EQ(e.at({4, 1}), 1.0 * kHalfSqrt2); // only A_10 given -> (0 + 1)/sqrt2
    EXPECT_EQ(e.size(), 7u);
    EXPECT_EQ(entries(p.C).at(4), 1.0 * kSqrt2);
    EXPECT_EQ(entries(p.C).at(5), 6.0);
    EXPECT_EQ(p.sense, Sense::Minimize);
    EXPECT_EQ(p.source.format, "sedumi");
    EXPECT_EQ(p.source.dimacs_kinds, (std::vector<char>{'l', 'l', 's'}));
}

// SDPT3: blk = {'s', [2 1]; 'l', 2; 'u', 1}: cones s2, s1, l2, u1 (svec 3 + 1 + 2 + 1 = 7), m = 2
void write_sdpt3(const std::string &path, bool v73)
{
    MatFile f = MatFile::create(path, v73);
    std::vector<matvar_t *> blk_cells;
    const std::vector<double> sizes_s = {2.0, 1.0};
    const double two = 2.0, one = 1.0;
    blk_cells.push_back(make_string("", "s"));
    blk_cells.push_back(make_string("", "l"));
    blk_cells.push_back(make_string("", "u"));
    blk_cells.push_back(make_double_matrix("", 1, 2, sizes_s.data()));
    blk_cells.push_back(make_double_matrix("", 1, 1, &two));
    blk_cells.push_back(make_double_matrix("", 1, 1, &one));
    write_variable(f, make_cell("blk", 3, 2, blk_cells));
    // At{1}: 4 x 2 (svec of the 2x2 block then the 1x1), At{2}: 2 x 2, At{3}: 1 x 2
    std::vector<matvar_t *> at_cells;
    at_cells.push_back(make_double_matrix("", 4, 2, dense(4, 2, {1.0, kSqrt2, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0}).data()));
    SparseMatrix sp; // sparse 2 x 2: (0,0) = 3, (1,1) = -1
    sp.rows = 2;
    sp.cols = 2;
    sp.colptr = {0, 1, 2};
    sp.rowind = {0, 1};
    sp.val = {3.0, -1.0};
    at_cells.push_back(make_sparse("", sp));
    at_cells.push_back(make_double_matrix("", 1, 2, dense(1, 2, {0.0, 7.0}).data()));
    write_variable(f, make_cell("At", 3, 1, at_cells));
    // C{1}: 3 x 3 block diagonal [1 2 0; 2 3 0; 0 0 4], C{2}: [1; 0], C{3}: [-1]
    std::vector<matvar_t *> c_cells;
    c_cells.push_back(make_double_matrix("", 3, 3, dense(3, 3, {1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0}).data()));
    c_cells.push_back(make_double_matrix("", 2, 1, dense(2, 1, {1.0, 0.0}).data()));
    const double minus_one = -1.0;
    c_cells.push_back(make_double_matrix("", 1, 1, &minus_one));
    write_variable(f, make_cell("C", 3, 1, c_cells));
    write_variable(f, make_double_matrix("b", 2, 1, dense(2, 1, {1.0, 2.0}).data()));
}

void check_sdpt3(const ProblemData &p, const std::string &what)
{
    SCOPED_TRACE(what);
    ASSERT_EQ(p.cones.size(), 4u) << cones_string(p.cones);
    EXPECT_EQ(p.cones[0].kind, ConeKind::Psd);
    EXPECT_EQ(p.cones[0].size, 2);
    EXPECT_EQ(p.cones[1].kind, ConeKind::Nonnegative); // the 1x1 of the group, simplified
    EXPECT_EQ(p.cones[1].size, 1);
    EXPECT_EQ(p.cones[2].kind, ConeKind::Nonnegative);
    EXPECT_EQ(p.cones[3].kind, ConeKind::Free);
    EXPECT_EQ(p.vec_len(), 7);
    EXPECT_EQ(p.source.original_block_of, (std::vector<int>{0, 0, 1, 2}));
    EXPECT_EQ(p.source.original_block_sizes, (std::vector<std::int64_t>{3, 2, 1}));
    EXPECT_EQ(p.source.original_block_kinds, (std::vector<char>{'s', 'l', 'u'}));
    const auto e = entries(p.At);
    EXPECT_EQ(e.at({0, 0}), 1.0);
    EXPECT_EQ(e.at({1, 0}), kSqrt2);
    EXPECT_EQ(e.at({3, 0}), 5.0);
    EXPECT_EQ(e.at({4, 0}), 3.0);
    EXPECT_EQ(e.at({2, 1}), 2.0);
    EXPECT_EQ(e.at({5, 1}), -1.0);
    EXPECT_EQ(e.at({6, 1}), 7.0);
    EXPECT_EQ(e.size(), 7u);
    const auto c = entries(p.C);
    EXPECT_EQ(c.at(0), 1.0);
    EXPECT_EQ(c.at(1), 2.0 * kSqrt2);
    EXPECT_EQ(c.at(2), 3.0);
    EXPECT_EQ(c.at(3), 4.0);
    EXPECT_EQ(c.at(4), 1.0);
    EXPECT_EQ(c.at(6), -1.0);
    EXPECT_EQ(p.b.val, (std::vector<double>{1.0, 2.0}));
    EXPECT_EQ(p.source.format, "sdpt3");
}

// MOSEK: 6 scalar variables (free, nonneg, shifted [2,inf), negated (-inf,5], ranged [1,4], fixed 7), 5 rows
// (equality 1, lower >= 0, upper <= 3, ranged [-1, 2], free), bardim [2 1]
struct MosekCase
{
    std::vector<double> c = {1.0, 2.0, 0.0, 3.0, 0.0, -1.0};
    std::vector<double> blx = {-INFINITY, 0.0, 2.0, -INFINITY, 1.0, 7.0};
    std::vector<double> bux = {INFINITY, INFINITY, INFINITY, 5.0, 4.0, 7.0};
    std::vector<double> blc = {1.0, 0.0, -INFINITY, -1.0, -INFINITY};
    std::vector<double> buc = {1.0, INFINITY, 3.0, 2.0, INFINITY};
    // a (5 x 6) dense entries (i, j, v)
    std::vector<std::tuple<int, int, double>> a = {{0, 0, 1.0}, {0, 2, 1.0}, {0, 5, 2.0}, {1, 1, 1.0}, {1, 3, -1.0}, {2, 4, 1.0}, {2, 2, 0.5},
                                                   {3, 0, 1.0}, {3, 4, 1.0}, {4, 1, 9.0}};
    std::vector<double> bardim = {2.0, 1.0};
    // bara: (subi, subj, subk, subl, val), lower triangle
    std::vector<std::tuple<int, int, int, int, double>> bara = {{1, 1, 1, 1, 1.0}, {1, 1, 2, 1, 0.5}, {2, 1, 2, 2, 1.0}, {4, 2, 1, 1, 1.0}, {5, 1, 1, 1, 3.0},
                                                                {1, 1, 2, 1, 0.25}}; // a duplicate: summed
    std::vector<std::tuple<int, int, int, double>> barc = {{1, 1, 1, 1.0}, {1, 2, 1, 2.0}, {2, 1, 1, -1.0}};
    double cfix = 0.5;
};

void write_mosek(const std::string &path, bool v73, const MosekCase &k, const char *sense, bool with_barc = true, bool with_cones = false,
                 bool with_quadratic = false)
{
    MatFile f = MatFile::create(path, v73);
    std::vector<std::pair<std::string, matvar_t *>> prob;
    const std::int64_t n = static_cast<std::int64_t>(k.c.size()), m = static_cast<std::int64_t>(k.blc.size());
    prob.emplace_back("c", make_double_matrix("c", n, 1, k.c.data()));
    SparseMatrix a; // build CSC from the triplets
    a.rows = m;
    a.cols = n;
    a.colptr.assign(static_cast<std::size_t>(n) + 1, 0);
    std::vector<std::tuple<int, int, double>> sorted = k.a;
    std::sort(sorted.begin(), sorted.end(), [](const auto &x, const auto &y) { return std::make_pair(std::get<1>(x), std::get<0>(x)) < std::make_pair(std::get<1>(y), std::get<0>(y)); });
    for (const auto &t : sorted)
    {
        a.colptr[static_cast<std::size_t>(std::get<1>(t)) + 1]++;
        a.rowind.push_back(std::get<0>(t));
        a.val.push_back(std::get<2>(t));
    }
    for (std::int64_t j = 0; j < n; j++)
        a.colptr[static_cast<std::size_t>(j) + 1] += a.colptr[static_cast<std::size_t>(j)];
    prob.emplace_back("a", make_sparse("a", a));
    prob.emplace_back("blc", make_double_matrix("blc", m, 1, k.blc.data()));
    prob.emplace_back("buc", make_double_matrix("buc", m, 1, k.buc.data()));
    prob.emplace_back("blx", make_double_matrix("blx", n, 1, k.blx.data()));
    prob.emplace_back("bux", make_double_matrix("bux", n, 1, k.bux.data()));
    prob.emplace_back("bardim", make_double_matrix("bardim", static_cast<std::int64_t>(k.bardim.size()), 1, k.bardim.data()));
    {
        std::vector<double> subi, subj, subk, subl, val;
        for (const auto &t : k.bara)
        {
            subi.push_back(std::get<0>(t));
            subj.push_back(std::get<1>(t));
            subk.push_back(std::get<2>(t));
            subl.push_back(std::get<3>(t));
            val.push_back(std::get<4>(t));
        }
        std::vector<std::pair<std::string, matvar_t *>> s;
        const std::int64_t t = static_cast<std::int64_t>(val.size());
        s.emplace_back("subi", make_double_matrix("subi", t, 1, subi.data()));
        s.emplace_back("subj", make_double_matrix("subj", t, 1, subj.data()));
        s.emplace_back("subk", make_double_matrix("subk", t, 1, subk.data()));
        s.emplace_back("subl", make_double_matrix("subl", t, 1, subl.data()));
        s.emplace_back("val", make_double_matrix("val", t, 1, val.data()));
        prob.emplace_back("bara", make_struct("bara", s));
    }
    if (with_barc)
    {
        std::vector<double> subj, subk, subl, val;
        for (const auto &t : k.barc)
        {
            subj.push_back(std::get<0>(t));
            subk.push_back(std::get<1>(t));
            subl.push_back(std::get<2>(t));
            val.push_back(std::get<3>(t));
        }
        std::vector<std::pair<std::string, matvar_t *>> s;
        const std::int64_t t = static_cast<std::int64_t>(val.size());
        s.emplace_back("subj", make_double_matrix("subj", t, 1, subj.data()));
        s.emplace_back("subk", make_double_matrix("subk", t, 1, subk.data()));
        s.emplace_back("subl", make_double_matrix("subl", t, 1, subl.data()));
        s.emplace_back("val", make_double_matrix("val", t, 1, val.data()));
        prob.emplace_back("barc", make_struct("barc", s));
    }
    prob.emplace_back("cfix", make_double_matrix("cfix", 1, 1, &k.cfix));
    if (with_cones)
    {
        std::vector<std::pair<std::string, matvar_t *>> cones;
        const double one = 1.0;
        cones.emplace_back("type", make_double_matrix("type", 1, 1, &one));
        prob.emplace_back("cones", make_struct("cones", cones));
    }
    if (with_quadratic)
    {
        const double one = 1.0;
        prob.emplace_back("qosubi", make_double_matrix("qosubi", 1, 1, &one));
        prob.emplace_back("qosubj", make_double_matrix("qosubj", 1, 1, &one));
        prob.emplace_back("qoval", make_double_matrix("qoval", 1, 1, &one));
    }
    write_variable(f, make_struct("prob", prob));
    if (sense)
        write_variable(f, make_string("sense", sense));
}

// evaluates the original MOSEK problem at a point and checks that the mapped point satisfies the ProblemData
void check_mosek_mapping(const ProblemData &p, const MosekCase &k, Sense sense)
{
    const cuadmm::io::SourceInfo &s = p.source;
    ASSERT_EQ(s.mosek_var_position.size(), 6u);
    ASSERT_EQ(s.mosek_row_position.size(), 5u);
    // original point: x within the bounds, Xbar_1 = [2 0.5; 0.5 1], Xbar_2 = [3]; x0 (free) chosen so that the
    // equality row 0 holds (x0 + x2 + 2 x5 + <bara_01, Xbar_1> = 1 with <bara_01, Xbar_1> = 2 + 2 * 0.75 * 0.5);
    // the inequality rows hold through their slacks whatever the point
    const std::vector<double> x = {-18.25, 0.5, 2.5, 4.0, 1.5, 7.0};
    const double X1[3] = {2.0, 0.5, 1.0}; // (0,0), (0,1), (1,1)
    const double X2 = 3.0;
    // row activities
    std::vector<double> act(5, 0.0);
    for (const auto &t : k.a)
        act[static_cast<std::size_t>(std::get<0>(t))] += std::get<2>(t) * x[static_cast<std::size_t>(std::get<1>(t))];
    for (const auto &t : k.bara)
    {
        const int i = std::get<0>(t) - 1, j = std::get<1>(t), kk = std::get<2>(t), ll = std::get<3>(t);
        const double v = std::get<4>(t);
        const double xe = j == 1 ? (kk == ll ? X1[kk == 1 ? 0 : 2] : X1[1]) : X2;
        act[static_cast<std::size_t>(i)] += (kk == ll ? 1.0 : 2.0) * v * xe; // <bara, Xbar> with symmetric matrices
    }
    // the mapped vector: X[position] = sign * (x - shift) for the scalars; the slacks from the activities
    std::vector<double> X(static_cast<std::size_t>(p.vec_len()), 0.0);
    for (std::size_t j = 0; j < 6; j++)
        if (s.mosek_var_position[j] >= 0)
            X[static_cast<std::size_t>(s.mosek_var_position[j])] = s.mosek_var_sign[j] * (x[j] - s.mosek_var_shift[j]);
        else
            EXPECT_EQ(s.mosek_var_shift[j], 7.0); // the fixed variable
    // layout of the l block: [nonneg/shifted/negated/ranged scalars (4)] [bound slack of var 4] [row slacks: row1 (lower), row2 (upper), row3 (ranged: 2)]
    const std::int64_t off_l = 1; // after the u block of one free variable
    const std::int64_t bound_slack = off_l + 4, slack_row1 = off_l + 5, slack_row2 = off_l + 6, slack_row3 = off_l + 7, slack_row3b = off_l + 8;
    X[static_cast<std::size_t>(bound_slack)] = (4.0 - 1.0) - (x[4] - 1.0);
    X[static_cast<std::size_t>(slack_row1)] = act[1] - 0.0;
    X[static_cast<std::size_t>(slack_row2)] = 3.0 - act[2];
    X[static_cast<std::size_t>(slack_row3)] = act[3] - (-1.0);
    X[static_cast<std::size_t>(slack_row3b)] = 2.0 - act[3];
    // bar blocks: s 2 then s 1 (simplified to l 1)
    const std::int64_t off_bar = off_l + 9;
    X[static_cast<std::size_t>(off_bar)] = X1[0];
    X[static_cast<std::size_t>(off_bar + 1)] = X1[1] * kSqrt2;
    X[static_cast<std::size_t>(off_bar + 2)] = X1[2];
    X[static_cast<std::size_t>(off_bar + 3)] = X2;
    ASSERT_EQ(p.vec_len(), off_bar + 4);
    // A X = b
    std::vector<double> AX(static_cast<std::size_t>(p.num_constraints()), 0.0);
    for (std::int64_t j = 0; j < p.At.cols; j++)
        for (std::int64_t q = p.At.colptr[j]; q < p.At.colptr[j + 1]; q++)
            AX[static_cast<std::size_t>(j)] += p.At.val[q] * X[static_cast<std::size_t>(p.At.rowind[q])];
    std::vector<double> b(static_cast<std::size_t>(p.num_constraints()), 0.0);
    for (std::size_t q = 0; q < p.b.val.size(); q++)
        b[static_cast<std::size_t>(p.b.index[q])] = p.b.val[q];
    for (std::size_t i = 0; i < AX.size(); i++)
        EXPECT_NEAR(AX[i], b[i], 1e-12) << "constraint " << i;
    EXPECT_EQ(p.num_constraints(), 4 + 1 + 1); // 4 kept rows + the second row of the ranged row + the bound row
    // objective: c'x + <barc, Xbar> + cfix == sign * <C, X> + offset
    double f = k.cfix;
    for (std::size_t j = 0; j < 6; j++)
        f += k.c[j] * x[j];
    for (const auto &t : k.barc)
    {
        const int j = std::get<0>(t), kk = std::get<1>(t), ll = std::get<2>(t);
        const double v = std::get<3>(t);
        const double xe = j == 1 ? (kk == ll ? X1[kk == 1 ? 0 : 2] : X1[1]) : X2;
        f += (kk == ll ? 1.0 : 2.0) * v * xe;
    }
    double CX = 0.0;
    for (std::size_t q = 0; q < p.C.val.size(); q++)
        CX += p.C.val[q] * X[static_cast<std::size_t>(p.C.index[q])];
    const double sign = sense == Sense::Maximize ? -1.0 : 1.0;
    EXPECT_NEAR(sign * CX + p.objective_offset, f, 1e-12);
    EXPECT_EQ(p.sense, sense);
}
#endif
} // namespace

TEST(IoMat, SedumiHandMadeV5AndV73)
{
#ifndef CUADMM_HAVE_MATIO
    EXPECT_FALSE(io::format_available("sedumi"));
    GTEST_SKIP() << "matio not compiled in";
#else
    EXPECT_TRUE(io::format_available("sedumi"));
    const std::string dir = temp_dir("mat");
    for (bool v73 : {false, true})
        for (bool store_At : {false, true})
        {
            const std::string path = dir + "/sedumi_" + (v73 ? "73" : "5") + (store_At ? "_At" : "_A") + ".mat";
            write_sedumi(path, v73, store_At);
            const ProblemData p = io::read_problem(path, IoOptions());
            check_sedumi(p, path);
            EXPECT_EQ(p.source.mat_version, v73 ? "7.3" : "5");
            EXPECT_EQ(cuadmm::io::detail::detect_mat_format(path), IoFormat::Sedumi);
        }
    // N = m: the name decides the orientation
    write_sedumi(dir + "/sedumi_square_A.mat", false, false, 7);
    write_sedumi(dir + "/sedumi_square_At.mat", false, true, 7);
    const ProblemData a = io::read_problem(dir + "/sedumi_square_A.mat", IoOptions());
    const ProblemData t = io::read_problem(dir + "/sedumi_square_At.mat", IoOptions());
    EXPECT_EQ(a.num_constraints(), 7);
    expect_bitwise(a, t, "SeDuMi N = m, A vs At");
    EXPECT_EQ(entries(a.At).at({3, 5}), 5.0);
    // K.q is rejected (Phase 9), complex data is rejected
    {
        MatFile f = MatFile::create(dir + "/sedumi_q.mat", false);
        const double one = 1.0, three = 3.0;
        write_variable(f, make_double_matrix("A", 1, 3, dense(1, 3, {1.0, 1.0, 1.0}).data()));
        write_variable(f, make_double_matrix("b", 1, 1, &one));
        write_variable(f, make_double_matrix("c", 3, 1, dense(3, 1, {1.0, 0.0, 0.0}).data()));
        std::vector<std::pair<std::string, matvar_t *>> K;
        K.emplace_back("q", make_double_matrix("q", 1, 1, &three));
        write_variable(f, make_struct("K", K));
    }
    EXPECT_THROW(io::read_problem(dir + "/sedumi_q.mat", IoOptions()), std::invalid_argument);
    IoOptions forced;
    forced.format = IoFormat::Mosek;
    EXPECT_THROW(io::read_problem(dir + "/sedumi_5_A.mat", forced), std::invalid_argument); // no prob
#endif
}

TEST(IoMat, Sdpt3HandMadeV5AndV73AndPythonFiles)
{
#ifndef CUADMM_HAVE_MATIO
    GTEST_SKIP() << "matio not compiled in";
#else
    const std::string dir = temp_dir("mat");
    for (bool v73 : {false, true})
    {
        const std::string path = dir + "/sdpt3_" + (v73 ? "73" : "5") + ".mat";
        write_sdpt3(path, v73);
        EXPECT_EQ(cuadmm::io::detail::detect_mat_format(path), IoFormat::Sdpt3);
        check_sdpt3(io::read_problem(path, IoOptions()), path);
    }
    // the same problem written by Python (scipy.io.savemat v5 and h5py v7.3), see tools/io/make_sdpt3_testfiles.py
    for (const char *name : {"sdpt3_small_v5.mat", "sdpt3_small_v73.mat"})
    {
        const std::string path = std::string(CUADMM_TEST_DATA_DIR) + "/" + name;
        if (!exists(path))
        {
            std::printf("  (missing %s)\n", path.c_str());
            continue;
        }
        check_sdpt3(io::read_problem(path, IoOptions()), path);
    }
    IoOptions raw;
    raw.simplify = false;
    const ProblemData q = io::read_problem(dir + "/sdpt3_5.mat", raw);
    EXPECT_EQ(q.cones[1].kind, ConeKind::Psd);
    const ProblemData a = io::read_problem(dir + "/sdpt3_5.mat", IoOptions()), b = io::read_problem(dir + "/sdpt3_73.mat", IoOptions());
    expect_bitwise(a, b, "SDPT3 v5 vs v7.3");
#endif
}

TEST(IoMat, MosekBoundsRangesSenseAndRejections)
{
#ifndef CUADMM_HAVE_MATIO
    GTEST_SKIP() << "matio not compiled in";
#else
    const std::string dir = temp_dir("mat");
    const MosekCase k;
    for (bool v73 : {false, true})
    {
        const std::string path = dir + "/mosek_" + (v73 ? "73" : "5") + ".mat";
        write_mosek(path, v73, k, "maximize");
        EXPECT_EQ(cuadmm::io::detail::detect_mat_format(path), IoFormat::Mosek);
        const ProblemData p = io::read_problem(path, IoOptions());
        check_mosek_mapping(p, k, Sense::Maximize);
        EXPECT_EQ(cones_string(p.cones), "u1 l9 s2 l1 ");
        EXPECT_EQ(p.source.original_block_of, (std::vector<int>{-1, -1, 0, 1}));
        EXPECT_EQ(p.source.dimacs_kinds, (std::vector<char>{'l', 'l', 's', 's'}));
        IoOptions agree;
        agree.sense = Sense::Maximize;
        EXPECT_NO_THROW(io::read_problem(path, agree));
        agree.sense = Sense::Minimize;
        EXPECT_THROW(io::read_problem(path, agree), std::invalid_argument);
    }
    // without a sense: io.sense decides, nothing is guessed
    write_mosek(dir + "/mosek_nosense.mat", false, k, nullptr);
    EXPECT_THROW(io::read_problem(dir + "/mosek_nosense.mat", IoOptions()), std::invalid_argument);
    IoOptions min;
    min.sense = Sense::Minimize;
    check_mosek_mapping(io::read_problem(dir + "/mosek_nosense.mat", min), k, Sense::Minimize);
    // no barc (SOS style), cones rejected, subk < subl rejected
    write_mosek(dir + "/mosek_nobarc.mat", false, k, "minimize", false);
    EXPECT_NO_THROW(io::read_problem(dir + "/mosek_nobarc.mat", IoOptions()));
    write_mosek(dir + "/mosek_cones.mat", false, k, "minimize", true, true);
    EXPECT_THROW(io::read_problem(dir + "/mosek_cones.mat", IoOptions()), std::invalid_argument);
    MosekCase upper = k;
    upper.bara.push_back({1, 1, 1, 2, 1.0});
    write_mosek(dir + "/mosek_upper.mat", false, upper, "minimize");
    EXPECT_THROW(io::read_problem(dir + "/mosek_upper.mat", IoOptions()), std::invalid_argument);
    MosekCase infeasible = k;
    infeasible.blc[0] = 2.0; // blc > buc
    write_mosek(dir + "/mosek_inf.mat", false, infeasible, "minimize");
    EXPECT_THROW(io::read_problem(dir + "/mosek_inf.mat", IoOptions()), std::invalid_argument);
    write_mosek(dir + "/mosek_quad.mat", false, k, "minimize", true, false, true);
    try
    {
        io::read_problem(dir + "/mosek_quad.mat", IoOptions());
        ADD_FAILURE() << "quadratic terms were accepted";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find("quadratic"), std::string::npos) << e.what();
    }
#endif
}

// the written MOSEK duals satisfy MOSEK's dual constraint c - a' y = slx - sux for every scalar variable kind (free,
// nonnegative, shifted, negated, ranged, fixed), and slc - suc = y, for a minimization and a maximization
TEST(IoMat, MosekWrittenDualsSatisfyTheDualConstraint)
{
#ifndef CUADMM_HAVE_MATIO
    GTEST_SKIP() << "matio not compiled in";
#else
    const std::string dir = temp_dir("mat");
    const MosekCase k;
    for (const char *sense : {"minimize", "maximize"})
    {
        SCOPED_TRACE(sense);
        const std::string in = dir + "/mosek_dual_" + sense + ".mat", out = dir + "/mosek_dual_" + sense + "_sol.mat";
        write_mosek(in, false, k, sense);
        const ProblemData p = io::read_problem(in, IoOptions());
        // a dual point: any y with S = C - A' y (dual feasibility is not needed for the identity)
        SolveResult r = fake_result(p);
        for (std::size_t i = 0; i < r.best.y.size(); i++)
            r.best.y[i] = 0.1 * static_cast<double>(i + 1) - 0.35;
        std::vector<double> S(static_cast<std::size_t>(p.vec_len()), 0.0);
        for (std::size_t q = 0; q < p.C.val.size(); q++)
            S[static_cast<std::size_t>(p.C.index[q])] += p.C.val[q];
        for (std::int64_t j = 0; j < p.At.cols; j++)
            for (std::int64_t q = p.At.colptr[j]; q < p.At.colptr[j + 1]; q++)
                S[static_cast<std::size_t>(p.At.rowind[q])] -= p.At.val[q] * r.best.y[static_cast<std::size_t>(j)];
        r.best.S = S;
        io::write_solution(out, p, r);
        MatFile f(out, MAT_ACC_RDONLY);
        MatVar res = read_variable(f, "res");
        const matvar_t *itr = struct_field(struct_field(res.get(), "sol", "res"), "itr", "res.sol");
        const std::vector<double> y = numeric_values(struct_field(itr, "y", "itr"), "y");
        const std::vector<double> slx = numeric_values(struct_field(itr, "slx", "itr"), "slx");
        const std::vector<double> sux = numeric_values(struct_field(itr, "sux", "itr"), "sux");
        const std::vector<double> slc = numeric_values(struct_field(itr, "slc", "itr"), "slc");
        const std::vector<double> suc = numeric_values(struct_field(itr, "suc", "itr"), "suc");
        ASSERT_EQ(y.size(), 5u);
        ASSERT_EQ(slx.size(), 6u);
        const double sgn = std::string(sense) == "maximize" ? -1.0 : 1.0;
        for (std::size_t i = 0; i < 5; i++)
        {
            EXPECT_NEAR(slc[i] - suc[i], y[i], 1e-12) << "row " << i;
            if (p.source.mosek_row_position[i] >= 0)
                EXPECT_EQ(y[i], sgn * r.best.y[static_cast<std::size_t>(p.source.mosek_row_position[i])]);
            else
                EXPECT_EQ(y[i], 0.0);
        }
        for (std::size_t j = 0; j < 6; j++)
        {
            double lhs = k.c[j];
            for (const auto &t : k.a)
                if (static_cast<std::size_t>(std::get<1>(t)) == j)
                    lhs -= std::get<2>(t) * y[static_cast<std::size_t>(std::get<0>(t))];
            EXPECT_NEAR(lhs, slx[j] - sux[j], 1e-12) << "variable " << j + 1 << " (c - a'y vs slx - sux)";
            if (sgn > 0)
            {
                EXPECT_GE(slx[j], 0.0);
                EXPECT_GE(sux[j], 0.0);
            }
            else
            {
                EXPECT_LE(slx[j], 0.0);
                EXPECT_LE(sux[j], 0.0);
            }
        }
    }
    // the reviewer's minimal case: min -x, 0 <= x <= 1, x <= 10: the optimum x = 1 has y = 0, slx = 0, sux = 1
    MosekCase minimal;
    minimal.c = {-1.0};
    minimal.blx = {0.0};
    minimal.bux = {1.0};
    minimal.blc = {-INFINITY};
    minimal.buc = {10.0};
    minimal.a = {{0, 0, 1.0}};
    minimal.bardim = {};
    minimal.bara = {};
    minimal.barc = {};
    minimal.cfix = 0.0;
    write_mosek(dir + "/mosek_minimal.mat", false, minimal, "minimize", false);
    const ProblemData p = io::read_problem(dir + "/mosek_minimal.mat", IoOptions());
    EXPECT_EQ(cones_string(p.cones), "l3 ");  // x', the bound slack s, the row slack t
    ASSERT_EQ(p.num_constraints(), 2);        // x' + t = 10, x' + s = 1
    SolveResult r = fake_result(p);
    r.best.X = {1.0, 0.0, 9.0};
    r.best.y = {0.0, -1.0}; // the bound row is active: its multiplier gives S_x' = -1 - (-1) = 0, S_s = 1
    r.best.S = {0.0, 1.0, 0.0};
    io::write_solution(dir + "/mosek_minimal_sol.mat", p, r);
    MatFile f(dir + "/mosek_minimal_sol.mat", MAT_ACC_RDONLY);
    MatVar res = read_variable(f, "res");
    const matvar_t *itr = struct_field(struct_field(res.get(), "sol", "res"), "itr", "res.sol");
    EXPECT_EQ(numeric_values(struct_field(itr, "xx", "itr"), "xx"), (std::vector<double>{1.0}));
    EXPECT_EQ(numeric_values(struct_field(itr, "y", "itr"), "y"), (std::vector<double>{0.0}));
    EXPECT_EQ(numeric_values(struct_field(itr, "slx", "itr"), "slx"), (std::vector<double>{0.0}));
    EXPECT_EQ(numeric_values(struct_field(itr, "sux", "itr"), "sux"), (std::vector<double>{1.0}));
#endif
}

TEST(IoMat, MosekSolutionWriterReadBack)
{
#ifndef CUADMM_HAVE_MATIO
    GTEST_SKIP() << "matio not compiled in";
#else
    const std::string dir = temp_dir("mat");
    const MosekCase k;
    write_mosek(dir + "/mosek_sol_in.mat", false, k, "maximize");
    const ProblemData p = io::read_problem(dir + "/mosek_sol_in.mat", IoOptions());
    const SolveResult r = fake_result(p);
    const std::string out = dir + "/mosek_sol_out.mat";
    io::write_solution(out, p, r);
    MatFile f(out, MAT_ACC_RDONLY);
    MatVar res = read_variable(f, "res");
    ASSERT_TRUE(res);
    const matvar_t *itr = struct_field(struct_field(res.get(), "sol", "res"), "itr", "res.sol");
    ASSERT_NE(itr, nullptr);
    const std::vector<double> xx = numeric_values(struct_field(itr, "xx", "itr"), "xx");
    ASSERT_EQ(xx.size(), 6u);
    for (std::size_t j = 0; j < 6; j++)
    {
        const std::int64_t pos = p.source.mosek_var_position[j];
        const double expected = pos < 0 ? p.source.mosek_var_shift[j] : p.source.mosek_var_shift[j] + p.source.mosek_var_sign[j] * r.best.X[static_cast<std::size_t>(pos)];
        EXPECT_EQ(xx[j], expected) << "xx(" << j + 1 << ")";
    }
    const std::vector<double> y = numeric_values(struct_field(itr, "y", "itr"), "y");
    ASSERT_EQ(y.size(), 5u);
    EXPECT_EQ(y[4], 0.0); // the dropped row
    EXPECT_EQ(y[0], -r.best.y[0]); // maximization: MOSEK's y = -y_cuADMM
    const std::vector<double> barx = numeric_values(struct_field(itr, "barx", "itr"), "barx");
    ASSERT_EQ(barx.size(), 3u + 1u); // lower triangles of the 2x2 and the 1x1
    const std::int64_t off_bar = p.vec_len() - 4;
    EXPECT_EQ(barx[0], r.best.X[static_cast<std::size_t>(off_bar)]);
    EXPECT_EQ(barx[1], r.best.X[static_cast<std::size_t>(off_bar + 1)] * kHalfSqrt2); // (2,1)
    EXPECT_EQ(barx[2], r.best.X[static_cast<std::size_t>(off_bar + 2)]);
    EXPECT_EQ(barx[3], r.best.X[static_cast<std::size_t>(off_bar + 3)]); // the simplified 1x1
    const std::vector<double> bars = numeric_values(struct_field(itr, "bars", "itr"), "bars");
    EXPECT_EQ(bars[0], -r.best.S[static_cast<std::size_t>(off_bar)]);
    EXPECT_EQ(scalar_value(struct_field(itr, "pobjval", "itr"), "pobjval"), r.pobj);
    // SeDuMi and SDPT3 conventions of the same solution, read back
    io::write_solution(dir + "/sol_sedumi.mat", p, r, "sedumi");
    {
        MatFile g(dir + "/sol_sedumi.mat", MAT_ACC_RDONLY);
        const std::vector<double> x = numeric_values(read_variable(g, "x").get(), "x");
        EXPECT_EQ(x.size(), 1u + 9u + 4u + 1u); // u, l, full 2x2, 1x1
        EXPECT_EQ(x[0], r.best.X[0]);
        EXPECT_EQ(x[10 + 1], r.best.X[static_cast<std::size_t>(off_bar + 1)] * kHalfSqrt2);
        EXPECT_EQ(numeric_values(read_variable(g, "y").get(), "y"), r.best.y);
    }
    io::write_solution(dir + "/sol_sdpt3.mat", p, r, "sdpt3");
    {
        MatFile g(dir + "/sol_sdpt3.mat", MAT_ACC_RDONLY);
        MatVar X = read_variable(g, "X");
        ASSERT_TRUE(X);
        EXPECT_EQ(numel(X.get()), 4u);
        const std::vector<double> X3 = numeric_values(cell_at(X.get(), 2, "X"), "X{3}");
        EXPECT_EQ(X3.size(), 4u);
        EXPECT_EQ(X3[0], r.best.X[static_cast<std::size_t>(off_bar)]);
        const std::vector<double> obj = numeric_values(read_variable(g, "obj").get(), "obj");
        EXPECT_EQ(obj, (std::vector<double>{r.pobj, r.dobj}));
    }
#endif
}

TEST(IoMat, PlatoBuck3SedumiMatchesSdpa)
{
#ifndef CUADMM_HAVE_MATIO
    GTEST_SKIP() << "matio not compiled in";
#else
    const std::string mat = external("plato_sedumi/buck3.mat"), dat = external("plato_sdpa/buck3.dat-s");
    IO_TEST_REQUIRE_FILE(mat);
    IO_TEST_REQUIRE_FILE(dat);
    const ProblemData s = io::read_problem(mat, IoOptions()), d = io::read_problem(dat, IoOptions());
    EXPECT_EQ(cones_string(s.cones), "s320 s321 l544 ");
    EXPECT_EQ(s.source.format, "sedumi");
    ProblemData d_min = d;
    d_min.sense = Sense::Minimize; // SDPA's file is a maximization of <F_0, Y>; SeDuMi's a minimization of the negated objective
    expect_same_problem(s, d_min, 2e-16, "buck3 SeDuMi vs SDPA");
    bool simplified = false;
    for (const std::string &line : s.source.report)
        simplified = simplified || line.find("block 2: PSD block of side 544 with data on the diagonal only") != std::string::npos;
    EXPECT_TRUE(simplified);
    IoOptions raw;
    raw.simplify = false;
    EXPECT_EQ(cones_string(io::read_problem(mat, raw).cones), "s320 s321 s544 ");
#endif
}

TEST(IoMat, SpotMosekFilesMatchTheirTxtConversions)
{
#ifndef CUADMM_HAVE_MATIO
    GTEST_SKIP() << "matio not compiled in";
#else
    const std::string mosek_dir = spot("MOSEK"), txt_dir = spot("TXT");
    IO_TEST_REQUIRE_FILE(mosek_dir);
    IO_TEST_REQUIRE_FILE(txt_dir);
    struct Case
    {
        const char *name;
        Sense sense;
    };
    for (const Case &c : {Case{"PushBox_N=30_MOMENT", Sense::Minimize}, Case{"PushBox_N=30_SOS", Sense::Maximize}, Case{"PushT_N=10_MOMENT", Sense::Minimize}})
    {
        const std::string mat = mosek_dir + "/" + c.name + ".mat", txt = txt_dir + "/" + c.name;
        if (!exists(mat) || !exists(txt))
        {
            std::printf("  (skipping %s)\n", c.name);
            continue;
        }
        IoOptions o;
        o.sense = c.sense;   // the .mat files do not record it
        o.simplify = false;  // the TXT conversions keep every block
        const ProblemData m = io::read_problem(mat, o);
        ProblemData t = io::read_txt(txt);
        t.sense = c.sense;
        expect_same_problem(m, t, 6e-17, std::string(c.name) + " MOSEK vs TXT");
        EXPECT_EQ(m.objective_offset, 0.0);
        // the solution writer produces barx with one lower triangle per bar variable (recover_mosek_sol_blk layout)
        if (std::string(c.name) == "PushT_N=10_MOMENT")
        {
            const ProblemData ms = io::read_problem(mat, IoOptions{IoFormat::Auto, c.sense, true, "", false, "", ""});
            EXPECT_LT(ms.vec_len(), m.vec_len()); // the simplifications shrink the problem
            const SolveResult r = fake_result(ms);
            const std::string out = temp_dir("mat") + "/pusht_solution.mat";
            io::write_solution(out, ms, r);
            MatFile f(out, MAT_ACC_RDONLY);
            MatVar res = read_variable(f, "res");
            const matvar_t *itr = struct_field(struct_field(res.get(), "sol", "res"), "itr", "res.sol");
            std::int64_t expected = 0;
            for (std::int64_t n : ms.source.original_block_sizes)
                expected += n * (n + 1) / 2;
            EXPECT_EQ(static_cast<std::int64_t>(numel(struct_field(itr, "barx", "itr"))), expected);
            EXPECT_EQ(numel(struct_field(itr, "xx", "itr")), 0u);
        }
    }
#endif
}
