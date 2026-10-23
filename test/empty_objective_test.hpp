/*

    empty_objective_test.hpp

    A problem whose objective vector is empty (C = 0, a feasibility problem such as the PLATO instance cnhil10 of
    examples/plato/TXT) sets up and iterates through the facade: the sparse-vector kernels of the engine skip the
    zero-size launch that 1.x issued silently and the launch checks of 2.0 reject (src/legacy/kernels/sparse_scalar.cu,
    sparse_dense.cu). Uses to_problem_data() and quiet_options() of cone_test.hpp.

*/

#pragma once

#include <gtest/gtest.h>

#include <cmath>

#include "cuadmm/cuadmm.h"
#include "regression_problems.hpp"

TEST(EmptyObjective, SolverSetsUpAndIteratesWithoutC)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    cuadmm::ProblemData d = to_problem_data(p);
    d.C.index.clear();
    d.C.val.clear();
    ASSERT_EQ(d.C.nnz(), 0);
    cuadmm::SolverOptions o = quiet_options();
    o.algorithm.max_iter = 50;
    o.algorithm.tol = 1e-300;
    cuadmm::Solver solver(o);
    ASSERT_NO_THROW(solver.setup(d));
    cuadmm::SolveResult r;
    ASSERT_NO_THROW(r = solver.solve());
    EXPECT_EQ(r.iterations, 50);
    EXPECT_EQ(r.pobj, 0.0);
    EXPECT_TRUE(std::isfinite(r.eta_p));
    EXPECT_TRUE(std::isfinite(r.eta_d));
    EXPECT_TRUE(std::isfinite(r.eta_g));
}
