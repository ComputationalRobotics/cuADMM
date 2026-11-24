/*

    validation_device_test.hpp

    Eigenvalue backends of ExternalValidator (branch siopt-rebuttal): lambda_min of the PSD blocks computed on the
    device (cusolverDnXsyevd, double precision, eigenvalues only) against LAPACK dsyevd on the CPU.

      - random blocks of several sizes with linear blocks in between: the two backends agree, the host-side metrics
        are identical, the counters say where each eigenvalue was computed, the device result is reproducible;
      - blocks with a known spectrum (the svec scaling of the off-diagonal entries, the layout of the unpacking);
      - a solve with validation.gpu_eigenvalues on and off stops validated at the same iteration.

*/

#ifndef CUADMM_VALIDATION_DEVICE_TEST_HPP
#define CUADMM_VALIDATION_DEVICE_TEST_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "cuadmm_legacy/external_validation.h"
#include "cuadmm_legacy/solver.h"

#include "regression_problems.hpp"
#include "solver_setup.hpp"

namespace validation_device_test
{

struct SmallProblem
{
    int vec_len = 0, con_num = 0;
    std::vector<int> cp, ri, bi, ci;
    std::vector<double> av, bv, cv, X, y, S;
    std::vector<char> types;
    std::vector<int> sizes;

    ExternalValidator validator() const
    {
        return ExternalValidator(vec_len, con_num, cp.data(), ri.data(), av.data(), (int)bi.size(), bi.data(), bv.data(),
                                 (int)ci.size(), ci.data(), cv.data(), types, sizes);
    }
};

inline int svec_length(const std::vector<char> &types, const std::vector<int> &sizes)
{
    int len = 0;
    for (size_t k = 0; k < types.size(); k++)
        len += types[k] == 's' ? sizes[k] * (sizes[k] + 1) / 2 : sizes[k];
    return len;
}

// random data, iterate included: symmetric blocks with N(0, 1) svec entries are indefinite
inline SmallProblem random_problem(const std::vector<char> &types, const std::vector<int> &sizes, unsigned seed)
{
    SmallProblem p;
    p.types = types;
    p.sizes = sizes;
    p.vec_len = svec_length(types, sizes);
    p.con_num = 4;
    std::mt19937 gen(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    const int step = std::max(1, p.vec_len / 12);
    p.cp.push_back(0);
    for (int j = 0; j < p.con_num; j++)
    {
        int count = 0;
        for (int r = j % step; r < p.vec_len && count < 12; r += step, count++)
        {
            p.ri.push_back(r);
            p.av.push_back(g(gen));
        }
        p.cp.push_back((int)p.ri.size());
    }
    for (int j = 0; j < p.con_num; j++)
    {
        p.bi.push_back(j);
        p.bv.push_back(g(gen));
    }
    for (int i = 0; i < p.vec_len; i++)
    {
        p.ci.push_back(i);
        p.cv.push_back(g(gen));
    }
    p.X.resize(p.vec_len);
    p.S.resize(p.vec_len);
    p.y.resize(p.con_num);
    for (double &x : p.X)
        x = g(gen);
    for (double &s : p.S)
        s = g(gen);
    for (double &v : p.y)
        v = g(gen);
    return p;
}

// svec of the n x n symmetric matrix with entries f(r, c), r <= c (upper triangle column by column, sqrt(2) off the diagonal)
template <typename F>
inline void append_svec(std::vector<double> &v, int n, F f)
{
    for (int c = 0; c < n; c++)
        for (int r = 0; r <= c; r++)
            v.push_back(r == c ? f(r, c) : std::sqrt(2.0) * f(r, c));
}

} // namespace validation_device_test

// 1. The device and LAPACK give the same cone metrics; everything computed on the host is identical.
TEST(ValidationDevice, DeviceEigenvaluesMatchLapack)
{
    using namespace validation_device_test;
    const SmallProblem p = random_problem({'s', 'l', 's', 's', 'l', 's'}, {1, 5, 7, 96, 3, 400}, 20261124u);
    ExternalValidator v = p.validator();
    EXPECT_EQ(v.eig_device(), ExternalValidator::EigDevice::Gpu); // the default
    const ValidationTolerances tol = ValidationTolerances::all(1e-4);
    const ValidationResult g = v.evaluate(p.X.data(), p.y.data(), p.S.data(), tol);
    v.set_eig_device(ExternalValidator::EigDevice::Cpu);
    const ValidationResult c = v.evaluate(p.X.data(), p.y.data(), p.S.data(), tol, 2);

    // four PSD blocks, three matrices each (X, Z = C - A^T y, S)
    EXPECT_EQ(g.gpu_eigs, 12);
    EXPECT_EQ(g.cpu_eigs, 0);
    EXPECT_EQ(c.gpu_eigs, 0);
    EXPECT_EQ(c.cpu_eigs, 12);

    // host-side metrics do not depend on the eigenvalue backend
    EXPECT_TRUE(g.finite);
    EXPECT_EQ(g.primal_res, c.primal_res);
    EXPECT_EQ(g.dual_res, c.dual_res);
    EXPECT_EQ(g.relgap, c.relgap);
    EXPECT_EQ(g.pobj, c.pobj);
    EXPECT_EQ(g.dobj, c.dobj);
    EXPECT_EQ(g.dimacs[0], c.dimacs[0]);
    EXPECT_EQ(g.dimacs[2], c.dimacs[2]);
    EXPECT_EQ(g.dimacs[4], c.dimacs[4]);
    EXPECT_EQ(g.dimacs[5], c.dimacs[5]);
    EXPECT_EQ(g.err6_z, c.err6_z);

    // eigenvalues: entries are O(1) and n <= 400, both solvers are backward stable
    const double eps = 1e-10;
    EXPECT_LT(c.min_eig_X, -1.0);
    EXPECT_LT(c.min_eig_S, -1.0);
    EXPECT_NEAR(g.min_eig_X, c.min_eig_X, eps);
    EXPECT_NEAR(g.min_eig_Z, c.min_eig_Z, eps);
    EXPECT_NEAR(g.min_eig_S, c.min_eig_S, eps);
    EXPECT_NEAR(g.X_cone_violation, c.X_cone_violation, eps);
    EXPECT_NEAR(g.X_cone_violation_normb, c.X_cone_violation_normb, eps);
    EXPECT_NEAR(g.dual_cone_violation, c.dual_cone_violation, eps);
    EXPECT_NEAR(g.S_cone_violation, c.S_cone_violation, eps);
    EXPECT_NEAR(g.dimacs[1], c.dimacs[1], eps);
    EXPECT_NEAR(g.dimacs[3], c.dimacs[3], eps);
    EXPECT_NEAR(g.err4_z, c.err4_z, eps);
    EXPECT_NEAR(g.certificate_lower_bound_conditional, c.certificate_lower_bound_conditional, 1e-7);
    EXPECT_EQ(g.X_worst_block, c.X_worst_block);
    EXPECT_EQ(g.X_negative_blocks, c.X_negative_blocks);
    EXPECT_EQ(g.validated, c.validated);
    EXPECT_EQ(g.failed, c.failed);

    // the device result is reproducible, also from a copy of the validator
    v.set_eig_device(ExternalValidator::EigDevice::Gpu);
    const ExternalValidator w = v;
    const ValidationResult g2 = w.evaluate(p.X.data(), p.y.data(), p.S.data(), tol);
    EXPECT_EQ(g2.gpu_eigs, 12);
    EXPECT_EQ(g2.min_eig_X, g.min_eig_X);
    EXPECT_EQ(g2.min_eig_Z, g.min_eig_Z);
    EXPECT_EQ(g2.min_eig_S, g.min_eig_S);
    printf("[ VALDEV   ] lambda_min X %.12f / Z %.12f / S %.12f, |gpu - cpu| = %.1e / %.1e / %.1e\n", g.min_eig_X, g.min_eig_Z,
           g.min_eig_S, std::fabs(g.min_eig_X - c.min_eig_X), std::fabs(g.min_eig_Z - c.min_eig_Z), std::fabs(g.min_eig_S - c.min_eig_S));
}

// 2. Known spectra: X carries [[2, 1], [1, -1]] (lambda_min = (1 - sqrt 13) / 2), S carries J - 3 I of order 64
//    (lambda_min = -3: every off-diagonal entry is used with its 1 / sqrt(2)), Z = C carries tridiag(1, -3, 1) of
//    order 129 (lambda_min = -3 - 2 cos(pi / 130)). The other blocks are identities.
TEST(ValidationDevice, KnownSpectraOnTheDevice)
{
    using namespace validation_device_test;
    SmallProblem p;
    p.types = {'s', 's', 's'};
    p.sizes = {2, 64, 129};
    p.vec_len = svec_length(p.types, p.sizes);
    p.con_num = 1;
    p.cp = {0, 1};
    p.ri = {0};
    p.av = {1.0};
    p.bi = {0};
    p.bv = {1.0};
    p.y = {0.0}; // Z = C
    auto identity = [](int r, int c) { return r == c ? 1.0 : 0.0; };
    append_svec(p.X, 2, [](int r, int c) { return r == c ? (r == 0 ? 2.0 : -1.0) : 1.0; });
    append_svec(p.X, 64, identity);
    append_svec(p.X, 129, identity);
    append_svec(p.S, 2, identity);
    append_svec(p.S, 64, [](int r, int c) { return r == c ? -2.0 : 1.0; }); // J - 3 I
    append_svec(p.S, 129, identity);
    std::vector<double> C;
    append_svec(C, 2, identity);
    append_svec(C, 64, identity);
    append_svec(C, 129, [](int r, int c) { return r == c ? -3.0 : (c - r == 1 ? 1.0 : 0.0); });
    ASSERT_EQ((int)C.size(), p.vec_len);
    for (int i = 0; i < p.vec_len; i++)
        if (C[i] != 0.0)
        {
            p.ci.push_back(i);
            p.cv.push_back(C[i]);
        }
    const double pi = std::acos(-1.0);
    for (const ExternalValidator::EigDevice device : {ExternalValidator::EigDevice::Gpu, ExternalValidator::EigDevice::Cpu})
    {
        SCOPED_TRACE(device == ExternalValidator::EigDevice::Gpu ? "gpu" : "cpu");
        ExternalValidator v = p.validator();
        v.set_eig_device(device);
        const ValidationResult r = v.evaluate(p.X.data(), p.y.data(), p.S.data(), ValidationTolerances::all(1e-4));
        EXPECT_EQ(device == ExternalValidator::EigDevice::Gpu ? r.gpu_eigs : r.cpu_eigs, 9);
        EXPECT_EQ(device == ExternalValidator::EigDevice::Gpu ? r.cpu_eigs : r.gpu_eigs, 0);
        EXPECT_NEAR(r.min_eig_X, (1.0 - std::sqrt(13.0)) / 2.0, 1e-12);
        EXPECT_NEAR(r.min_eig_S, -3.0, 1e-11);
        EXPECT_NEAR(r.min_eig_Z, -3.0 - 2.0 * std::cos(pi / 130.0), 1e-12);
        EXPECT_EQ(r.X_negative_blocks, 1);
        EXPECT_EQ(r.X_worst_block, 0);
    }
}

// 3. In a solve, the device and the CPU validations stop at the same validated iteration.
TEST(ValidationDevice, SolveStopsAtTheSameIterationOnBothDevices)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    int iterations[2] = {-1, -2};
    for (int d = 0; d < 2; d++)
    {
        SCOPED_TRACE(d == 0 ? "gpu" : "cpu");
        SDPSolver s;
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
        s.validation.interval = 100;
        s.validation.tol = ValidationTolerances::all(1e-4);
        s.validation.gpu_eigenvalues = d == 0;
        ASSERT_NO_THROW(s.solve(1000000, 1e-4, 0, 1e-2));
        ASSERT_EQ(s.stop_reason, "externally_validated");
        ASSERT_FALSE(s.validation_history.empty());
        for (const ValidationRecord &rec : s.validation_history)
        {
            EXPECT_GT(d == 0 ? rec.result.gpu_eigs : rec.result.cpu_eigs, 0);
            EXPECT_EQ(d == 0 ? rec.result.cpu_eigs : rec.result.gpu_eigs, 0);
        }
        iterations[d] = s.info_iter_num;
    }
    EXPECT_EQ(iterations[0], iterations[1]);
}

#endif // CUADMM_VALIDATION_DEVICE_TEST_HPP
