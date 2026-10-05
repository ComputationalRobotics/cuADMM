/*

    dense_ystep_test.hpp

    The y-step solves with the dense factor on the GPU (SDPSolver::AAt_dense_gpu, used when the factor of
    eps*I + AA^T is dense) give the same iterates as the CHOLMOD solves on the CPU, up to rounding, in plain ADMM
    and in sGS-ADMM. Same for the batched EVD of the medium blocks (SDPSolver::medium_group_batched).

*/

#ifdef CUADMM_PLAIN_ADMM_ONLY
#define CUADMM_PLAIN_AND_SGS_SWITCHES {0}
#else
#define CUADMM_PLAIN_AND_SGS_SWITCHES {0, 100000}
#endif
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cuadmm/solver.h"
#include "regression_problems.hpp"
#include "solver_setup.hpp"

TEST(DenseYStep, MatchesCpuSolves)
{
    // one PSD block of size 80 (vec_len 3240) and 2200 constraints of 40 entries: the factor of AA^T is dense
    const regression::ProblemSpec spec = {{{'s', 80, 40, 0, false}}, 2200, 40, 3001};
    const regression::RegressionProblem p = regression::generate(spec);
    for (int switch_admm : CUADMM_PLAIN_AND_SGS_SWITCHES)
    {
        SCOPED_TRACE(switch_admm > 0 ? "sGS-ADMM" : "ADMM");
        SDPSolver dense, cpu;
        ASSERT_NO_THROW(solver_setup::init_solver(p, dense));
        ASSERT_NO_THROW(solver_setup::init_solver(p, cpu));
        ASSERT_TRUE(dense.AAt_dense_gpu) << "nnz(L) = " << dense.cpu_AAt_solver.cc.lnz;
        cpu.AAt_dense_gpu = false; // same factor, CHOLMOD solves on the CPU
        ASSERT_NO_THROW(dense.solve(300, 1e-12, 500, 50, 100, switch_admm, 0));
        ASSERT_NO_THROW(cpu.solve(300, 1e-12, 500, 50, 100, switch_admm, 0));
        ASSERT_EQ(dense.info_iter_num, cpu.info_iter_num);
        double max_rel = 0.0;
        for (int k = 0; k < dense.info_iter_num; k++)
        {
            const double a = dense.info_errRp_arr[k], b = cpu.info_errRp_arr[k];
            const double c = dense.info_pobj_arr[k], d = cpu.info_pobj_arr[k];
            max_rel = std::max(max_rel, std::abs(a - b) / std::max(std::abs(b), 1e-8)); // 1e-8: well above the rounding floor
            max_rel = std::max(max_rel, std::abs(c - d) / (1.0 + std::abs(d)));
        }
        printf("[ DENSE    ] switch_admm=%d: max relative difference of errRp / pobj over %d iterations = %.2e, "
               "final errRp = %.2e, pobj - p* = %.2e\n",
               switch_admm, dense.info_iter_num, max_rel, dense.errRp, dense.pobj - p.p_star);
        EXPECT_LT(max_rel, 1e-6);
        EXPECT_NEAR(dense.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    }
}

// The batched EVD of the medium size groups with several matrices (cusolverDnXsyevBatched) gives the iterates of
// one cusolverDnXsyevd per matrix on the streams, up to rounding, in plain ADMM and in sGS-ADMM.
TEST(BatchedMediumEVD, MatchesStreamedEVD)
{
    // six 40 x 40 blocks and three 50 x 50 blocks (medium), a 12 x 12 block (small) and a free block
    const regression::ProblemSpec spec = {
        {{'s', 40, 20, 0, false}, {'s', 50, 25, 0, false}, {'s', 40, 20, 0, false}, {'u', 30, 0, 0, false},
         {'s', 40, 20, 0, false}, {'s', 50, 25, 0, false}, {'s', 40, 20, 0, false}, {'s', 12, 6, 0, false},
         {'s', 40, 20, 0, false}, {'s', 50, 25, 0, false}, {'s', 40, 20, 0, false}},
        0, 12, 3002};
    const regression::RegressionProblem p = regression::generate(spec);
    for (int switch_admm : CUADMM_PLAIN_AND_SGS_SWITCHES)
    {
        SCOPED_TRACE(switch_admm > 0 ? "sGS-ADMM" : "ADMM");
        SDPSolver batched, streamed;
        streamed.medium_batch_min_count = 0; // one EVD per matrix on the streams
        ASSERT_NO_THROW(solver_setup::init_solver(p, batched));
        ASSERT_NO_THROW(solver_setup::init_solver(p, streamed));
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
        ASSERT_EQ(std::count(batched.medium_group_batched.begin(), batched.medium_group_batched.end(), 1), 2);
#endif
        ASSERT_EQ(std::count(streamed.medium_group_batched.begin(), streamed.medium_group_batched.end(), 1), 0);
        ASSERT_NO_THROW(batched.solve(300, 1e-12, 500, 50, 100, switch_admm, 0));
        ASSERT_NO_THROW(streamed.solve(300, 1e-12, 500, 50, 100, switch_admm, 0));
        ASSERT_EQ(batched.info_iter_num, streamed.info_iter_num);
        double max_rel = 0.0;
        for (int k = 0; k < batched.info_iter_num; k++)
        {
            const double a = batched.info_errRp_arr[k], b = streamed.info_errRp_arr[k];
            const double c = batched.info_pobj_arr[k], d = streamed.info_pobj_arr[k];
            max_rel = std::max(max_rel, std::abs(a - b) / std::max(std::abs(b), 1e-8)); // 1e-8: well above the rounding floor
            max_rel = std::max(max_rel, std::abs(c - d) / (1.0 + std::abs(d)));
        }
        printf("[ BATCHED  ] switch_admm=%d: max relative difference of errRp / pobj over %d iterations = %.2e, "
               "pobj - p* = %.2e\n", switch_admm, batched.info_iter_num, max_rel, batched.pobj - p.p_star);
        EXPECT_LT(max_rel, 1e-6);
        EXPECT_NEAR(batched.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    }
}
