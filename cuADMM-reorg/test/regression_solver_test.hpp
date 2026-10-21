/*

    regression_solver_test.hpp

    End-to-end regression tests of SDPSolver on the deterministic problems of regression_problems.hpp,
    whose optimal value p* is known analytically. Each problem is written as a TXT directory under
    $CUADMM_REGRESSION_DATA_DIR (default: "regression_data", relative to the working directory),
    loaded with Problem::from_txt and solved like src/main.cu does (standard ADMM, LOBPCG phase from
    iteration 2), with a tighter tolerance. The written directories can also be solved with the CLI:
        ./cuadmm_exe regression_data/<name>

*/

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "cuadmm/problem.h"
#include "cuadmm/solver.h"
#include "regression_problems.hpp"

namespace
{

// directory of the generated TXT problems
std::string regression_data_dir()
{
    const char *dir = std::getenv("CUADMM_REGRESSION_DATA_DIR");
    return (dir != nullptr && dir[0] != '\0') ? std::string(dir) : std::string("regression_data");
}

// Generates the named problem into gen, writes it as a TXT directory, loads it back with Problem::from_txt
// and solves it like src/main.cu does: standard ADMM, switch to the final projection method (LOBPCG phase)
// at iteration 2. sig: initial sigma; switch_admm: sGS-ADMM for iter < switch_admm (0: plain ADMM)
void solve_regression(
    const std::string &name, bool use_lobpcg, double sig, int max_iter, double stop_tol,
    regression::RegressionProblem &gen, SDPSolver &solver, int switch_admm = 0)
{
    gen = regression::generate(name);
    std::string dir = regression_data_dir() + "/" + name;
    regression::write_txt(gen, dir);

    Problem problem;
    problem.from_txt(dir);
    ASSERT_EQ(problem.vec_len, gen.vec_len);
    ASSERT_EQ(problem.con_num, gen.con_num);

    std::vector<char> blk_types;
    std::vector<int> blk_sizes;
    for (const auto &blk : problem.blk_vals)
    {
        blk_types.push_back(std::get<0>(blk));
        blk_sizes.push_back(std::get<1>(blk));
    }

    solver.init(
        15,
        problem.vec_len, problem.con_num,
        problem.At_csc_col_ptrs.data(), problem.At_csc_row_ids.data(), problem.At_csc_vals.data(), problem.At_nnz,
        problem.b_indices.data(), problem.b_vals.data(), problem.b_nnz,
        problem.C_indices.data(), problem.C_vals.data(), problem.C_nnz,
        blk_types.data(), blk_sizes.data(), problem.mat_num,
        ProjectionMethod::EIG_FP64,
        ProjectionMethod::EIG_FP64,
        nullptr, nullptr, nullptr,
        sig, use_lobpcg);
    solver.solve(max_iter, stop_tol, 500, 50, 100, switch_admm, 0);
}

// residuals and objectives of the returned iterate, recomputed on the host
struct HostKKT
{
    double errRp; // ||b - A(X)|| / (1 + ||b||)
    double errRd; // ||A^T(y) + S - C|| / (1 + ||C||)
    double pobj;  // <C, X>
    double dobj;  // <b, y>
};

// Independent host check of the RETURNED iterate (unscaled at the end of solve()), with the original data.
HostKKT host_kkt(const regression::RegressionProblem &gen, const SDPSolver &solver)
{
    std::vector<double> X(gen.vec_len), y(gen.con_num), S(gen.vec_len);
    CHECK_CUDA(cudaMemcpy(X.data(), solver.X.vals, sizeof(double) * gen.vec_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y.data(), solver.y.vals, sizeof(double) * gen.con_num, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(S.data(), solver.S.vals, sizeof(double) * gen.vec_len, cudaMemcpyDeviceToHost));
    std::vector<double> Rp = gen.b;      // b - A(X)
    std::vector<double> Rd(gen.vec_len); // A^T(y) + S - C
    for (int i = 0; i < gen.vec_len; i++)
        Rd[i] = S[i] - gen.C[i];
    for (size_t e = 0; e < gen.At_vals.size(); e++)
    {
        Rp[gen.At_cols[e]] -= gen.At_vals[e] * X[gen.At_rows[e]];
        Rd[gen.At_rows[e]] += gen.At_vals[e] * y[gen.At_cols[e]];
    }
    auto norm = [](const std::vector<double> &v)
    { return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0)); };
    HostKKT host;
    host.errRp = norm(Rp) / (1.0 + norm(gen.b));
    host.errRd = norm(Rd) / (1.0 + norm(gen.C));
    host.pobj = std::inner_product(gen.C.begin(), gen.C.end(), X.begin(), 0.0);
    host.dobj = std::inner_product(gen.b.begin(), gen.b.end(), y.begin(), 0.0);
    return host;
}

// max KKT residual max(errRp, errRd, relgap) of every iteration of the last solve()
std::vector<double> kkt_history(const SDPSolver &solver)
{
    const size_t first = solver.info_errRp_arr.size() - solver.info_iter_num;
    std::vector<double> kkt(solver.info_iter_num);
    for (size_t it = 0; it < kkt.size(); it++)
        kkt[it] = std::max({solver.info_errRp_arr[first + it], solver.info_errRd_arr[first + it], solver.info_relgap_arr[first + it]});
    return kkt;
}

// Bug 2.2: the returned iterate must be the best iterate of the run, and the reported residuals and
// objectives must describe it.
void expect_best_iterate_returned(const SDPSolver &solver, const HostKKT &host)
{
    EXPECT_NEAR(host.errRp, solver.errRp, 1e-3 * solver.errRp + 1e-10);
    EXPECT_NEAR(host.errRd, solver.errRd, 1e-3 * solver.errRd + 1e-10);
    EXPECT_NEAR(host.pobj, solver.pobj, 1e-8 * (1.0 + std::abs(solver.pobj)));
    EXPECT_NEAR(host.dobj, solver.dobj, 1e-8 * (1.0 + std::abs(solver.dobj)));

    const std::vector<double> kkt = kkt_history(solver);
    ASSERT_FALSE(kkt.empty());
    EXPECT_DOUBLE_EQ(std::max({solver.errRp, solver.errRd, solver.relgap}), *std::min_element(kkt.begin(), kkt.end()))
        << "the returned iterate is not the best iterate of the run";
}

// Generates, writes, loads and solves the named problem, and checks the returned solution.
// sig: initial sigma; expected_large_mode: projection mode of the large matrices after the last rank
// analysis (-1: no check); switch_admm: sGS-ADMM for iter < switch_admm (0: plain ADMM)
void run_regression(const std::string &name, bool use_lobpcg, double sig, int expected_large_mode, int switch_admm = 0)
{
    // with exact projections every problem converges in a few hundred iterations; max_iter only bounds
    // the runs that stall above stop_tol, which then return their best iterate
    const double stop_tol = 1e-6;
    const int max_iter = 5000;

    regression::RegressionProblem gen;
    SDPSolver solver;
    solve_regression(name, use_lobpcg, sig, max_iter, stop_tol, gen, solver, switch_admm);
    if (testing::Test::HasFatalFailure())
        return;
    const HostKKT host = host_kkt(gen, solver);

    const double p_star = gen.p_star;
    const double obj_tol = 1e-4 * (1.0 + std::abs(p_star));
    // no iterate of a run stopped by max_iter is below stop_tol, not even the returned best one
    const bool converged = std::max({solver.errRp, solver.errRd, solver.relgap}) < stop_tol;
    printf("[ RESULT   ] %s lobpcg=%d switch_admm=%d: %s, iter = %d, time = %.2fs, p* = %.10e, pobj - p* = %.2e, dobj - p* = %.2e, "
           "errRp = %.2e, errRd = %.2e, relgap = %.2e, errPSD_X = %.2e, errPSD_S = %.2e, host errRp = %.2e, "
           "host errRd = %.2e, LOBPCG calls = %d, fallbacks = %d\n",
           name.c_str(), int(use_lobpcg), switch_admm, converged ? "converged" : "max_iter reached",
           solver.info_iter_num, solver.total_time, p_star, solver.pobj - p_star, solver.dobj - p_star,
           solver.errRp, solver.errRd, solver.relgap, solver.errPSD_X, solver.errPSD_S,
           host.errRp, host.errRd, solver.lobpcg_calls, solver.lobpcg_fallbacks);

    EXPECT_NEAR(solver.pobj, p_star, obj_tol);
    EXPECT_NEAR(solver.dobj, p_star, obj_tol);
    EXPECT_LE(solver.errRp, 1e-5);
    EXPECT_LE(solver.errRd, 1e-5);
    EXPECT_LE(solver.relgap, 1e-5);
    EXPECT_LE(solver.errPSD_S, 1e-6);
    // X is not a projection (tau != 1, and sGS Step 3): its cone violation is measured over all blocks
    EXPECT_LE(solver.errPSD_X, 1e-5);
    EXPECT_LE(host.errRp, 1e-4);

    expect_best_iterate_returned(solver, host);

    if (expected_large_mode >= 0)
    {
        for (int idx = 0; idx < solver.sizes.large_mat_num; idx++)
            EXPECT_EQ(solver.large_mode[idx], expected_large_mode) << "unexpected projection mode of large matrix " << idx;
        if (expected_large_mode != LargeProjectionMode::FULL)
            EXPECT_GT(solver.lobpcg_calls, 0) << "LOBPCG was never used";
    }
}

} // namespace

// Each problem is solved with LOBPCG (as src/main.cu does) and without it (full EVD of the large matrices).
// Large_LowRankS starts from a small sigma so that it is not solved before the rank analysis of iteration
// 102, the first one to see the final ranks (+3 -1097): it then takes the LOBPCG positive branch.
#define REGRESSION_SOLVER_TEST(name, sig, expected_large_mode)        \
    TEST(RegressionSolver, name)                                      \
    {                                                                 \
        run_regression(#name, true, sig, expected_large_mode);        \
    }                                                                 \
    TEST(RegressionSolver, name##_NoLOBPCG)                           \
    {                                                                 \
        run_regression(#name, false, sig, LargeProjectionMode::FULL); \
    }

REGRESSION_SOLVER_TEST(PSD, 1e2, -1)
REGRESSION_SOLVER_TEST(Free, 1e2, -1)
REGRESSION_SOLVER_TEST(PSD_Free, 1e2, -1)
REGRESSION_SOLVER_TEST(Free_PSD, 1e2, -1)
REGRESSION_SOLVER_TEST(PSD_Free_PSD, 1e2, -1)
REGRESSION_SOLVER_TEST(Nonneg_Free_PSD, 1e2, -1)
REGRESSION_SOLVER_TEST(Free_Free_PSD, 1e2, -1)
REGRESSION_SOLVER_TEST(PSD_Nonneg, 1e2, -1)
REGRESSION_SOLVER_TEST(SmallBatch_Free, 1e2, -1)
REGRESSION_SOLVER_TEST(Medium_Free, 1e2, -1)
REGRESSION_SOLVER_TEST(Large_LowRankX, 1e2, LargeProjectionMode::LOBPCG_NEGATIVE)
REGRESSION_SOLVER_TEST(Large_LowRankS, 1e-4, LargeProjectionMode::LOBPCG_POSITIVE)
REGRESSION_SOLVER_TEST(Large_HalfRank, 1e2, LargeProjectionMode::FULL)

#undef REGRESSION_SOLVER_TEST

// The same problems solved with the paper's sGS-ADMM (Algorithm 1: sGS Step 3 at every iteration, tau = 1.95,
// sGS sigma rule), and one hybrid run (sGS-ADMM for 49 iterations, then plain ADMM).
#ifndef CUADMM_PLAIN_ADMM_ONLY // sGS-ADMM tests: not built in the plain-ADMM-only build
#define REGRESSION_SOLVER_SGS_TEST(name)                   \
    TEST(RegressionSolver, name##_sGS)                     \
    {                                                      \
        run_regression(#name, true, 1e2, -1, 100000);      \
    }

REGRESSION_SOLVER_SGS_TEST(PSD)
REGRESSION_SOLVER_SGS_TEST(Free)
REGRESSION_SOLVER_SGS_TEST(PSD_Free)
REGRESSION_SOLVER_SGS_TEST(Free_PSD)
REGRESSION_SOLVER_SGS_TEST(PSD_Free_PSD)
REGRESSION_SOLVER_SGS_TEST(Nonneg_Free_PSD)
REGRESSION_SOLVER_SGS_TEST(Free_Free_PSD)
REGRESSION_SOLVER_SGS_TEST(PSD_Nonneg)
REGRESSION_SOLVER_SGS_TEST(SmallBatch_Free)
REGRESSION_SOLVER_SGS_TEST(Medium_Free)

#undef REGRESSION_SOLVER_SGS_TEST

TEST(RegressionSolver, Nonneg_Free_PSD_Hybrid)
{
    run_regression("Nonneg_Free_PSD", true, 1e2, -1, 50);
}

#endif // CUADMM_PLAIN_ADMM_ONLY

TEST(RegressionSolver, AdmmModeName)
{
    EXPECT_EQ(admm_mode_name(0, 100), "plain ADMM");
    EXPECT_EQ(admm_mode_name(1, 100), "plain ADMM");
    EXPECT_EQ(admm_mode_name(101, 100), "pure sGS-ADMM (Algorithm 1)");
    EXPECT_EQ(admm_mode_name(50, 100), "hybrid: sGS-ADMM for iterations 1-49, then plain ADMM");
}

#ifndef CUADMM_PLAIN_ADMM_ONLY
// In the sGS phase, Step 3 solves AA^T y^{k+1} = b/sigma - A(X^k/sigma + S^{k+1} - C) exactly (up to the eps*I of
// the factorization), so Step 4 gives b - A X^{k+1} = (1 - tau)(b - A X^k): the primal residual contracts by exactly
// |1 - tau| = 0.95 per iteration, whatever sigma and the projection do. A stale Rp or S in Step 3, or a wrong tau,
// breaks this; plain ADMM has no such invariant.
TEST(RegressionSolver, SgsPrimalContraction)
{
    regression::RegressionProblem gen;
    SDPSolver solver;
    // stop_tol 1e-14: the tau rule for errRd < stop_tol never fires, so tau stays 1.95
    solve_regression("PSD_Free", true, 1e2, 40, 1e-14, gen, solver, 1000);
    if (testing::Test::HasFatalFailure())
        return;
    ASSERT_EQ(solver.info_iter_num, 40);
    const size_t first = solver.info_errRp_arr.size() - solver.info_iter_num;
    for (int k = 1; k < solver.info_iter_num; k++)
    {
        const double ratio = solver.info_errRp_arr[first + k] / solver.info_errRp_arr[first + k - 1];
        EXPECT_NEAR(ratio, 0.95, 1e-6) << "iteration " << k + 1;
    }
}
#endif // CUADMM_PLAIN_ADMM_ONLY

// Bug 2.2 on a run stopped by max_iter: the best iterate must be returned with its own residuals and
// objectives. A converged run returns its last iterate (the first one below stop_tol, hence the best),
// so the runs above take this path only when they stall; here PSD_Free is stopped right after the
// iteration whose KKT residual is the furthest above the best one before it.
TEST(RegressionSolver, BestIterate)
{
    const double stop_tol = 1e-6;
    regression::RegressionProblem gen;
    std::vector<double> kkt;
    {
        SDPSolver solver;
        solve_regression("PSD_Free", true, 1e2, 5000, stop_tol, gen, solver);
        if (HasFatalFailure())
            return;
        kkt = kkt_history(solver);
    }
    ASSERT_FALSE(kkt.empty());
    int max_iter = 0;
    double worst_ratio = 1.0;
    double best = kkt[0];
    for (int it = 1; it < int(kkt.size()); it++)
    {
        if (kkt[it] / best > worst_ratio)
        {
            worst_ratio = kkt[it] / best;
            max_iter = it + 1;
        }
        best = std::min(best, kkt[it]);
    }
    ASSERT_GT(max_iter, 0) << "the KKT residual of PSD_Free decreases monotonically";

    SDPSolver solver;
    solve_regression("PSD_Free", true, 1e2, max_iter, stop_tol, gen, solver);
    if (HasFatalFailure())
        return;
    const std::vector<double> history = kkt_history(solver);
    ASSERT_EQ(int(history.size()), max_iter);
    const double best_kkt = *std::min_element(history.begin(), history.end());
    ASSERT_GT(history.back(), best_kkt) << "the last iterate is the best one: the best iterate restore is not exercised";
    const HostKKT host = host_kkt(gen, solver);
    printf("[ RESULT   ] BestIterate: max_iter = %d, last KKT = %.2e, best KKT = %.2e, returned KKT = %.2e, "
           "errRp = %.2e (host %.2e), errRd = %.2e (host %.2e), pobj = %.10e (host %.10e)\n",
           max_iter, history.back(), best_kkt, std::max({solver.errRp, solver.errRd, solver.relgap}),
           solver.errRp, host.errRp, solver.errRd, host.errRd, solver.pobj, host.pobj);

    expect_best_iterate_returned(solver, host);
}
