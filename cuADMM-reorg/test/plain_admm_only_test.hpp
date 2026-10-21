/*

    plain_admm_only_test.hpp

    Tests of the plain-ADMM-only build (CUADMM_PLAIN_ADMM_ONLY, branch experiment/plato-cold-plain-admm-sigma) and of
    the additions for the PLATO campaign:
      - solve() and the command line reject every sGS-ADMM / hybrid configuration; a solve performs exactly one y linear
        solve per iteration and no sGS step;
      - the sigma log records every change of the adaptive rules and reproduces the per-iteration sigma history;
        fixed sigma never changes;
      - the six DIMACS error measures of ExternalValidator on a small problem with hand-computed values, including the
        block kinds of the DIMACS norm;
      - the strict stage (ValidationConfig::strict_dimacs_tol): practical criteria first, then max_abs_DIMACS, and the
        returned iterate at a safety limit;
      - the CLI options --strict-dimacs-tol, --save-final-dir, --sigma-log.
    Uses the helpers of fixed_sigma_validation_test.hpp (included before this file).

*/

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "cuadmm/external_validation.h"
#include "cuadmm/solver.h"
#include "regression_problems.hpp"
#include "solver_setup.hpp"

#ifdef CUADMM_PLAIN_ADMM_ONLY

namespace plain_test
{
using fixed_sigma_test::last_solve;
using fixed_sigma_test::make_validator;
using fixed_sigma_test::returned_iterate;

inline std::string read_file(const std::string &f)
{
    std::ifstream in(f);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
} // namespace plain_test

using namespace plain_test;

// solve() rejects sGS-ADMM (switch_admm > max_iter), the hybrid (1 < switch_admm <= max_iter) and the hybrid sigma
// policy; switch_admm = 0 or 1 is plain ADMM.
TEST(PlainAdmmOnly, SolveRejectsSgsAndHybrid)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    for (int sw : {INT_MAX, 101, 2})
    {
        SDPSolver s;
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
        EXPECT_THROW(s.solve(100, 1e-4, 500, 50, 100, sw, 0, 1e-2, 2.0), std::invalid_argument) << "switch_admm = " << sw;
    }
    SDPSolver h;
    ASSERT_NO_THROW(solver_setup::init_solver(p, h, true, 1.0));
    h.sigma_policy = SigmaPolicy::FIXED_SGS_LEGACY_ADMM;
    EXPECT_THROW(h.solve(100, 1e-4, 500, 50, 100, 0, 0, 1e-2, 2.0), std::invalid_argument);
    for (int sw : {0, 1})
    {
        SDPSolver s;
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
        EXPECT_NO_THROW(s.solve(50, 1e-12, 500, 50, 100, sw, 0, 1e-2, 2.0));
    }
}

// Exactly one y linear solve per iteration, no sGS iteration, tau = 1.618 at every iteration.
TEST(PlainAdmmOnly, OneYSolvePerIterationNoSgsStep)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    for (SigmaPolicy pol : {SigmaPolicy::LEGACY_ADAPTIVE, SigmaPolicy::FIXED})
    {
        SDPSolver s;
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
        s.sigma_policy = pol;
        ASSERT_NO_THROW(s.solve(700, 0.0, 500, 50, 100, 0, 0, 1e-2, 2.0));
        ASSERT_EQ(s.info_iter_num, 700);
        EXPECT_EQ(s.y_solves, 700);
        EXPECT_EQ(s.sgs_phase_iterations, 0);
        EXPECT_EQ(s.admm_phase_iterations, 700);
        for (double t : last_solve(s.info_tau_arr, s.info_iter_num))
            ASSERT_EQ(t, 1.618);
    }
}

// The sigma log reproduces the per-iteration sigma history (6000 iterations reach the Monitor1 region); fixed sigma
// logs nothing and never changes.
TEST(PlainAdmmOnly, SigmaLogReproducesTheHistory)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    ASSERT_NO_THROW(s.solve(6000, 0.0, 500, 50, 100, 0, 0, 1e-2, 2.0));
    ASSERT_GT(s.sigma_changes, 10);
    EXPECT_EQ((int)s.sigma_log.size(), s.sigma_changes);
    const std::vector<double> sig = last_solve(s.info_sig_arr, s.info_iter_num);
    double cur = s.sigma0;
    size_t e = 0;
    for (int k = 1; k <= s.info_iter_num; k++)
    {
        while (e < s.sigma_log.size() && s.sigma_log[e].iter == k)
        {
            ASSERT_EQ(s.sigma_log[e].old_sigma, cur) << "entry " << e;
            cur = s.sigma_log[e].new_sigma;
            EXPECT_TRUE(s.sigma_log[e].rule == "admm_schedule" || s.sigma_log[e].rule == "monitor1");
            e++;
        }
        ASSERT_EQ(sig[k - 1], cur) << "iteration " << k;
    }
    EXPECT_EQ(e, s.sigma_log.size());
    SDPSolver f;
    ASSERT_NO_THROW(solver_setup::init_solver(p, f, true, 1.0));
    f.sigma_policy = SigmaPolicy::FIXED;
    ASSERT_NO_THROW(f.solve(6000, 0.0, 500, 50, 100, 0, 0, 1e-2, 2.0));
    EXPECT_TRUE(f.sigma_log.empty());
    EXPECT_EQ(f.sigma_changes, 0);
    EXPECT_EQ(f.sig, f.sigma0);
}

// DIMACS measures on a hand-made problem: blocks 's' 2 (PSD), 'l' 2 (linear) and 'l' 1 (a 1 x 1 PSD block of the
// original data); two constraints. The expected values are computed here with full matrices (no svec).
TEST(DimacsValidator, HandComputedSmallProblem)
{
    const double r2 = std::sqrt(2.0);
    // At as CSC (vec_len 6 x 2): A1 = [[1, 1], [1, 0]] (+ lp0 = 1), A2 = [[0, 0], [0, 2]] (+ lp1 = -1, 1x1 = 1)
    std::vector<int> cp = {0, 3, 6}, ri = {0, 1, 3, 2, 4, 5};
    std::vector<double> av = {1.0, r2, 1.0, 2.0, -1.0, 1.0};
    std::vector<int> bi = {0, 1}, ci = {0, 1, 2, 3, 4, 5};
    std::vector<double> bv = {1.0, 2.0}, cv = {1.0, 0.5 * r2, 3.0, 2.0, -4.0, 0.25};
    ExternalValidator v(6, 2, cp.data(), ri.data(), av.data(), 2, bi.data(), bv.data(), 6, ci.data(), cv.data(), {'s', 'l', 'l'}, {2, 2, 1});
    const double X[6] = {2.0, 1.0 * r2, -1.0, 3.0, -0.5, 0.1};  // [[2, 1], [1, -1]], [3, -0.5], [0.1]
    const double y[2] = {0.5, -1.0};
    const double S[6] = {1.0, 0.0, -2.0, 0.3, 0.2, -0.05};    // [[1, 0], [0, -2]], [0.3, 0.2], [-0.05]
    // by hand, full matrices
    const double Xm[2][2] = {{2, 1}, {1, -1}}, Sm[2][2] = {{1, 0}, {0, -2}}, Cm[2][2] = {{1, 0.5}, {0.5, 3}};
    const double A1[2][2] = {{1, 1}, {1, 0}}, A2[2][2] = {{0, 0}, {0, 2}};
    auto ip = [](const double a[2][2], const double b[2][2]) { return a[0][0] * b[0][0] + 2 * a[0][1] * b[0][1] + a[1][1] * b[1][1]; };
    const double AX1 = ip(A1, Xm) + 3.0, AX2 = ip(A2, Xm) + 0.5 + 0.1; // lp: 1*3 ; -1*-0.5 ; 1x1: 1*0.1
    const double rp1 = AX1 - 1.0, rp2 = AX2 - 2.0;
    double R[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            R[i][j] = y[0] * A1[i][j] + y[1] * A2[i][j] + Sm[i][j] - Cm[i][j];
    const double rl0 = y[0] * 1.0 + 0.3 - 2.0, rl1 = y[1] * -1.0 + 0.2 + 4.0, r11 = y[1] * 1.0 - 0.05 - 0.25;
    const double RF = std::sqrt(R[0][0] * R[0][0] + 2 * R[0][1] * R[0][1] + R[1][1] * R[1][1]);
    const double binf = 2.0, cinf = 4.0; // max |b_i|; max |C_ij| over matrix entries (0.5 off-diagonal, not 0.5 sqrt 2)
    auto lmin2 = [](const double a[2][2]) { return 0.5 * (a[0][0] + a[1][1]) - std::sqrt(0.25 * (a[0][0] - a[1][1]) * (a[0][0] - a[1][1]) + a[0][1] * a[0][1]); };
    const double lminX = std::min({lmin2(Xm), 3.0, -0.5, 0.1}), lminS = std::min({lmin2(Sm), 0.3, 0.2, -0.05});
    const double pobj = ip(Cm, Xm) + 2.0 * 3.0 - 4.0 * -0.5 + 0.25 * 0.1, dobj = 1.0 * y[0] + 2.0 * y[1];
    const double den = 1 + std::fabs(pobj) + std::fabs(dobj);
    const double xs = ip(Xm, Sm) + 3.0 * 0.3 - 0.5 * 0.2 + 0.1 * -0.05;
    const double e1 = std::sqrt(rp1 * rp1 + rp2 * rp2) / (1 + binf), e2 = std::max(0.0, -lminX) / (1 + binf);
    const double e4 = std::max(0.0, -lminS) / (1 + cinf), e5 = (pobj - dobj) / den, e6 = xs / den;
    // default kinds: the 1 x 1 block counts as linear (one 2-norm over all linear entries)
    const ValidationResult d = v.evaluate(X, y, S, ValidationTolerances::all(1e-4));
    const double e3_lin = (RF + std::sqrt(rl0 * rl0 + rl1 * rl1 + r11 * r11)) / (1 + cinf);
    const double want[6] = {e1, e2, e3_lin, e4, e5, e6};
    for (int k = 0; k < 6; k++)
        EXPECT_NEAR(d.dimacs[k], want[k], 1e-13 * std::max(1.0, std::fabs(want[k]))) << "err" << k + 1;
    double mx = 0;
    for (double w : want)
        mx = std::max(mx, std::fabs(w));
    EXPECT_NEAR(d.max_abs_dimacs, mx, 1e-13);
    EXPECT_FALSE(d.err6_defined); // X and S are outside the cone
    // SDPA kinds: the 1 x 1 block is a PSD block, so its residual enters the norm by itself (|r|)
    ExternalValidator w(6, 2, cp.data(), ri.data(), av.data(), 2, bi.data(), bv.data(), 6, ci.data(), cv.data(), {'s', 'l', 'l'}, {2, 2, 1});
    w.set_dimacs_kinds({'s', 'l', 's'});
    const ValidationResult k = w.evaluate(X, y, S, ValidationTolerances::all(1e-4));
    EXPECT_NEAR(k.dimacs[2], (RF + std::fabs(r11) + std::sqrt(rl0 * rl0 + rl1 * rl1)) / (1 + cinf), 1e-13);
    EXPECT_EQ(k.dimacs[0], d.dimacs[0]);
    EXPECT_THROW(w.set_dimacs_kinds({'l', 'l', 'l'}), std::invalid_argument); // a PSD block cannot be linear
    // the S cone criterion: max(0, -lambda_min(S_b)) / (1 + ||C||)
    double normC = 0;
    for (double c : cv)
        normC += c * c;
    normC = std::sqrt(normC);
    EXPECT_NEAR(d.S_cone_violation, std::max({0.0, -lmin2(Sm), 0.05}) / (1 + normC), 1e-13);
    EXPECT_NE(d.failed.find("S_cone"), std::string::npos);
}

// Strict stage: the practical criteria are recorded first, then the solve continues to max_abs_DIMACS <= 1e-6; with
// an unattainable strict tolerance the solve stops at the safety cap and returns the validated snapshot with the
// smallest max_abs_DIMACS.
TEST(StrictStage, PracticalThenStrictAndTheReturnedIterate)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-4);
    s.validation.strict_dimacs_tol = 1e-6;
    ASSERT_NO_THROW(s.solve(200000, 1e-4, 500, 50, 100, 0, 0, 1e-2, 2.0));
    ASSERT_TRUE(s.practical_validated);
    ASSERT_EQ(s.stop_reason, "strict_validated");
    EXPECT_TRUE(s.strict_validated);
    EXPECT_LE(s.first_practical_iteration, s.first_strict_iteration);
    EXPECT_LE(s.first_practical_time_s, s.first_strict_time_s);
    EXPECT_EQ(s.first_strict_iteration, s.info_iter_num);
    std::vector<double> X, y, S;
    returned_iterate(s, X, y, S);
    const ValidationResult r = make_validator(p).evaluate(X.data(), y.data(), S.data(), s.validation.tol);
    EXPECT_TRUE(r.validated);
    EXPECT_LE(r.max_abs_dimacs, 1e-6);
    printf("[ STRICT   ] practical at %d (%.3f s), strict at %d (%.3f s), max_abs_DIMACS %.2e\n", s.first_practical_iteration,
           s.first_practical_time_s, s.first_strict_iteration, s.first_strict_time_s, r.max_abs_dimacs);
    // unattainable strict tolerance: the cap, and the best validated snapshot is returned
    SDPSolver t;
    ASSERT_NO_THROW(solver_setup::init_solver(p, t, true, 1.0));
    t.validation.interval = 100;
    t.validation.tol = ValidationTolerances::all(1e-4);
    t.validation.strict_dimacs_tol = 1e-300;
    ASSERT_NO_THROW(t.solve(s.first_practical_iteration + 1500, 1e-4, 500, 50, 100, 0, 0, 1e-2, 2.0));
    EXPECT_EQ(t.stop_reason, "max_iter");
    EXPECT_TRUE(t.practical_validated);
    EXPECT_FALSE(t.strict_validated);
    EXPECT_EQ(t.first_practical_iteration, s.first_practical_iteration);
    double best = std::numeric_limits<double>::infinity();
    for (const ValidationRecord &rec : t.validation_history)
        if (rec.result.validated)
            best = std::min(best, rec.result.max_abs_dimacs);
    EXPECT_EQ(t.best_validation.result.max_abs_dimacs, best);
    EXPECT_TRUE(t.best_validation.result.validated);
    returned_iterate(t, X, y, S);
    const ValidationResult u = make_validator(p).evaluate(X.data(), y.data(), S.data(), t.validation.tol);
    EXPECT_TRUE(u.validated);
    EXPECT_EQ(u.max_abs_dimacs, best);
}

// Command line of the plain-ADMM-only build: sGS options are rejected with a clear error; a plain-ADMM run reports
// the build, one y solve per iteration, writes the sigma log and, at a safety limit, the final iterate.
TEST(PlainAdmmOnlyCli, RejectsSgsOptionsAndWritesTheNewOutputs)
{
    const std::string dir = "fixed_sigma_cli/PSD_Free";
    regression::write_txt(regression::generate("PSD_Free"), dir);
    auto run = [&](const std::string &args, const std::string &tag)
    {
        for (const char *ext : {".json", ".log", "_sigma.csv"})
            std::remove(("fixed_sigma_cli/" + tag + ext).c_str());
        const std::string cmd = "./cuadmm_exe " + dir + " " + args + " --summary fixed_sigma_cli/" + tag + ".json > fixed_sigma_cli/" + tag +
                                ".log 2>&1";
        return std::system(cmd.c_str());
    };
    for (const std::string &bad : {std::string("--algorithm sgs"), std::string("--switch-admm 5"), std::string("--sgs-iterations 10"),
                                   std::string("--sigma-policy fixed_sgs_then_legacy_admm")})
    {
        EXPECT_NE(run(bad, "rejected"), 0) << bad;
        EXPECT_NE(read_file("fixed_sigma_cli/rejected.log").find("disabled in this plain-ADMM-only build"), std::string::npos) << bad;
    }
    ASSERT_EQ(run("--algorithm admm --max-iter 600 --tol 1e-12 --sig 1 --sigma-log fixed_sigma_cli/plain_sigma.csv", "plain"), 0);
    const std::string js = read_file("fixed_sigma_cli/plain.json"), log = read_file("fixed_sigma_cli/plain.log");
    EXPECT_NE(log.find("Build: plain ADMM only"), std::string::npos);
    EXPECT_NE(js.find("\"plain_admm_only_build\": true"), std::string::npos);
    // the run may converge before the cap (internal KKT < 1e-12): one y solve per completed iteration
    const size_t ip = js.find("\"iterations\": ");
    ASSERT_NE(ip, std::string::npos);
    const int iters = std::stoi(js.substr(ip + 14));
    EXPECT_GT(iters, 0);
    EXPECT_LE(iters, 600);
    EXPECT_NE(js.find("\"y_solves\": " + std::to_string(iters) + ","), std::string::npos);
    EXPECT_NE(js.find("\"y_solves_per_iteration\": 1,"), std::string::npos);
    const std::string sl = read_file("fixed_sigma_cli/plain_sigma.csv");
    EXPECT_EQ(sl.substr(0, sl.find('\n')), "iter,old_sigma,new_sigma,rule,errRp,errRd,relgap,feasratio,prim_win,dual_win");
    // the final iterate at the safety cap (unattainable tolerance)
    std::system("rm -rf fixed_sigma_cli/final_iterate");
    ASSERT_EQ(run("--algorithm admm --max-iter 300 --tol 1e-4 --validate-interval 100 --validate-tol 1e-14 --strict-dimacs-tol 1e-8 "
                  "--save-final-dir fixed_sigma_cli/final_iterate", "final"), 0);
    struct stat st;
    EXPECT_EQ(stat("fixed_sigma_cli/final_iterate/X.txt", &st), 0);
    EXPECT_EQ(read_file("fixed_sigma_cli/final_iterate/iteration.txt"), "300\n");
    const std::string fj = read_file("fixed_sigma_cli/final.json");
    EXPECT_NE(fj.find("\"best_max_abs_dimacs\": "), std::string::npos);
    EXPECT_NE(fj.find("\"strict_dimacs_tol\": 1e-08"), std::string::npos);
    EXPECT_NE(run("--strict-dimacs-tol 1e-6", "strict_without_validation"), 0);
    EXPECT_NE(run("--save-final-dir fixed_sigma_cli/x", "final_without_validation"), 0);
}

#endif // CUADMM_PLAIN_ADMM_ONLY
