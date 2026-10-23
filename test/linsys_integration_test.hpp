/*

    linsys_integration_test.hpp

    Phase 7, the normal-equation backends in the solver: the default (CHOLMOD of main, bit for bit on the
    53 cases: GPU-evidence list) against the dense cuSOLVER backend and the cuDSS backend (when compiled in) end to
    end on the generator's problems (the same optimum, the iteration counts within a few percent: plan Phase 7
    acceptance), the backend's name in the result, and the presolve on a problem with dependent constraint rows
    (duplicated rows: the rows are removed, the solve converges, y comes back on the full problem).
    Uses the facade helpers of cone_test.hpp.

*/

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "cuadmm/solver.h"
#include "regression_problems.hpp"

namespace linsys_integration
{
cuadmm::SolverOptions options_for(const std::string &name, cuadmm::LinsysBackend backend)
{
    cuadmm::SolverOptions o = quiet_options();
    o.linsys.backend = backend;
    if (name.rfind("Large_", 0) == 0)
    {
        o.penalty.sigma0 = 1e-4;
        o.algorithm.max_iter = 5000;
    }
    return o;
}

/// The generated problem with `copies` duplicates of its first rows appended (dependent constraints, consistent b).
cuadmm::ProblemData with_duplicate_rows(const regression::RegressionProblem &p, int copies)
{
    cuadmm::ProblemData d = to_problem_data(p);
    const int m = p.con_num;
    // A^T in CSC: duplicate the first `copies` columns (rows of A) at the end, and the entries of b
    std::vector<std::int64_t> colptr = d.At.colptr, rowind = d.At.rowind;
    std::vector<double> val = d.At.val;
    for (int c = 0; c < copies; c++)
    {
        const std::int64_t begin = d.At.colptr[c], end = d.At.colptr[c + 1];
        rowind.insert(rowind.end(), d.At.rowind.begin() + begin, d.At.rowind.begin() + end);
        val.insert(val.end(), d.At.val.begin() + begin, d.At.val.begin() + end);
        colptr.push_back(static_cast<std::int64_t>(rowind.size()));
    }
    d.At.colptr = colptr;
    d.At.rowind = rowind;
    d.At.val = val;
    d.At.cols = m + copies;
    for (int c = 0; c < copies; c++)
    {
        // b of the duplicated row: the same value (consistent)
        double bc = 0.0;
        for (std::size_t i = 0; i < d.b.index.size(); i++)
            if (d.b.index[i] == c)
                bc = d.b.val[i];
        if (bc != 0.0)
        {
            d.b.index.push_back(m + c);
            d.b.val.push_back(bc);
        }
    }
    d.b.size = m + copies;
    return d;
}
} // namespace linsys_integration

TEST(LinsysIntegration, DefaultBackendIsCholmodAndReportsIt)
{
    const cuadmm::SolveResult r = solve_generated("PSD_Free", linsys_integration::options_for("PSD_Free", cuadmm::LinsysBackend::Auto));
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_EQ(r.linsys_backend, "cholmod");
    EXPECT_GT(r.aat_nnz_L, 0.0);
    EXPECT_EQ(r.presolve_removed_rows, 0);
}

TEST(LinsysIntegration, DenseBackendMatchesCholmodEndToEnd)
{
    for (const std::string name : {"PSD_Free", "Nonneg_Free_PSD", "Large_LowRankS"})
    {
        SCOPED_TRACE(name);
        const regression::RegressionProblem p = regression::generate(name);
        const cuadmm::SolveResult a = solve_generated(name, linsys_integration::options_for(name, cuadmm::LinsysBackend::Cholmod));
        const cuadmm::SolveResult b = solve_generated(name, linsys_integration::options_for(name, cuadmm::LinsysBackend::Dense));
        EXPECT_TRUE(a.converged) << a.stop_reason;
        EXPECT_TRUE(b.converged) << b.stop_reason;
        EXPECT_EQ(b.linsys_backend.rfind("dense", 0), 0u) << b.linsys_backend;
        EXPECT_NEAR(b.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
        EXPECT_NEAR(a.pobj, b.pobj, 1e-6 * (1.0 + std::abs(a.pobj)));
        // the iteration counts within a few percent (plan Phase 7 acceptance): the solves differ by rounding only
        EXPECT_LE(std::abs(a.iterations - b.iterations), std::max(2, a.iterations / 20)) << a.iterations << " vs " << b.iterations;
        printf("[ LINSYS   ] %s: cholmod %d iterations, dense %d iterations, |pobj diff| = %.2e\n", name.c_str(), a.iterations, b.iterations,
               std::abs(a.pobj - b.pobj));
    }
}

TEST(LinsysIntegration, CudssBackendMatchesCholmodEndToEnd)
{
#ifndef CUADMM_HAVE_CUDSS
    GTEST_SKIP() << "built without cuDSS (CUADMM_WITH_CUDSS)";
#else
    for (const std::string name : {"PSD_Free", "Large_LowRankS"})
    {
        SCOPED_TRACE(name);
        const regression::RegressionProblem p = regression::generate(name);
        const cuadmm::SolveResult a = solve_generated(name, linsys_integration::options_for(name, cuadmm::LinsysBackend::Cholmod));
        const cuadmm::SolveResult b = solve_generated(name, linsys_integration::options_for(name, cuadmm::LinsysBackend::Cudss));
        EXPECT_TRUE(a.converged) << a.stop_reason;
        EXPECT_TRUE(b.converged) << b.stop_reason;
        EXPECT_EQ(b.linsys_backend.rfind("cudss", 0), 0u) << b.linsys_backend;
        EXPECT_NEAR(b.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
        EXPECT_NEAR(a.pobj, b.pobj, 1e-6 * (1.0 + std::abs(a.pobj)));
        EXPECT_LE(std::abs(a.iterations - b.iterations), std::max(2, a.iterations / 20)) << a.iterations << " vs " << b.iterations;
        printf("[ LINSYS   ] %s: cholmod %d iterations, cudss %d iterations, |pobj diff| = %.2e\n", name.c_str(), a.iterations, b.iterations,
               std::abs(a.pobj - b.pobj));
    }
#endif
}

TEST(LinsysIntegration, PresolveRemovesDuplicatedRowsAndExpandsY)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    const cuadmm::ProblemData d = linsys_integration::with_duplicate_rows(p, 3);
    cuadmm::SolverOptions o = quiet_options();
    o.linsys.presolve = true;
    cuadmm::Solver solver(o);
    solver.setup(d);
    const cuadmm::SolveResult r = solver.solve();
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_EQ(r.presolve_removed_rows, 3) << r.presolve_message;
    EXPECT_NEAR(r.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    ASSERT_EQ(r.best.y.size(), static_cast<std::size_t>(p.con_num + 3)) << "y of the full problem";
    for (int c = 0; c < 3; c++)
        EXPECT_EQ(r.best.y[p.con_num + c], 0.0) << "a removed row's multiplier is zero";
    // the same problem without the presolve: the factorization sees the dependent rows (tiny pivots) and still solves
    cuadmm::SolverOptions off = quiet_options();
    cuadmm::Solver plain(off);
    plain.setup(d);
    const cuadmm::SolveResult q = plain.solve();
    EXPECT_GE(q.aat_tiny_pivots + q.aat_nonpositive_pivots, 3) << "the duplicated rows make A A^T singular";
    EXPECT_EQ(q.presolve_removed_rows, 0);
    EXPECT_NEAR(q.pobj, p.p_star, 1e-3 * (1.0 + std::abs(p.p_star)));
}
