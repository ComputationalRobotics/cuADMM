/*

    warm_restart_test.hpp

    Tests of a second solve() on the same solver (if_first = false), as used for warm restarts (e.g. MPC).

*/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cuadmm/solver.h"
#include "regression_problems.hpp"
#include "solver_setup.hpp"

// A second solve() with if_first = false starts from the X, y, S stored in the solver (unscaled): its stopping
// test must use the residuals of that point, not the stale ones of the previous solve.
TEST(WarmRestart, SecondSolveRecomputesResiduals)
{
    using solver_setup::init_solver;
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver solver;
    ASSERT_NO_THROW(init_solver(p, solver));
    ASSERT_NO_THROW(solver.solve(20000, 1e-6, 500, 50, 100, 0, 0));
    ASSERT_TRUE(solver.converged);
    const double pobj = solver.pobj, dobj = solver.dobj;

    // save the returned (unscaled) solution
    std::vector<double> X(p.vec_len), y(p.con_num), S(p.vec_len);
    ASSERT_EQ(cudaMemcpy(X.data(), solver.X.vals, sizeof(double) * p.vec_len, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(y.data(), solver.y.vals, sizeof(double) * p.con_num, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(S.data(), solver.S.vals, sizeof(double) * p.vec_len, cudaMemcpyDeviceToHost), cudaSuccess);

    // restart from zero with no iteration allowed: this point is far from optimal and must not be reported as converged
    ASSERT_EQ(cudaMemset(solver.X.vals, 0, sizeof(double) * p.vec_len), cudaSuccess);
    ASSERT_EQ(cudaMemset(solver.y.vals, 0, sizeof(double) * p.con_num), cudaSuccess);
    ASSERT_EQ(cudaMemset(solver.S.vals, 0, sizeof(double) * p.vec_len), cudaSuccess);
    ASSERT_NO_THROW(solver.solve(0, 1e-6, 500, 50, 100, 0, 0, 1e-2, 2, false));
    EXPECT_FALSE(solver.converged);
    EXPECT_GT(std::max(solver.errRp, std::max(solver.errRd, solver.relgap)), 1e-2);

    // restart from the saved solution: it satisfies the tolerance again, with the same objectives
    ASSERT_EQ(cudaMemcpy(solver.X.vals, X.data(), sizeof(double) * p.vec_len, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(solver.y.vals, y.data(), sizeof(double) * p.con_num, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(solver.S.vals, S.data(), sizeof(double) * p.vec_len, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_NO_THROW(solver.solve(0, 1e-6, 500, 50, 100, 0, 0, 1e-2, 2, false));
    EXPECT_TRUE(solver.converged);
    EXPECT_EQ(solver.info_iter_num, 0);
    EXPECT_NEAR(solver.pobj, pobj, 1e-9 * (1 + std::abs(pobj)));
    EXPECT_NEAR(solver.dobj, dobj, 1e-9 * (1 + std::abs(dobj)));
}
