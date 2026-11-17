/*

    fixed_sigma_validation_test.hpp

    Tests of the fixed sigma policy and of the validation-aware stopping of the solve (ValidationConfig,
    ExternalValidator), on the engine's driver SDPSolver::solve() and on the command line:
      - fixed sigma is exactly constant for the whole solve;
      - validation continues beyond iteration 10000, the solve stops on the validated accuracy, not on the internal
        KKT residual, and not when the cone criterion fails;
      - non-finite values fail closed;
      - the returned iterate is the validated one or, without validation success, the best externally evaluated one
        (also after non-finite residuals);
      - checkpoints, the starting point, the solve time and the command-line interface (the default policy, the
        validation flags, the removed options rejected, JSON parseable).

*/

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "cuadmm_legacy/external_validation.h"
#include "cuadmm_legacy/solver.h"
#include "regression_problems.hpp"
#include "solver_setup.hpp"

#ifndef CUADMM_TEST_DATA_DIR
#define CUADMM_TEST_DATA_DIR "test/data"
#endif

namespace fixed_sigma_test
{

// last solve() of the info arrays
inline std::vector<double> last_solve(const std::vector<double> &arr, int n)
{
    return std::vector<double>(arr.end() - n, arr.end());
}

inline ExternalValidator make_validator(const regression::RegressionProblem &p)
{
    std::vector<int> rows = p.At_rows, cols = p.At_cols, col_ptrs;
    std::vector<double> vals = p.At_vals;
    COO_to_CSC(col_ptrs, cols, rows, vals, (int)vals.size(), p.con_num);
    std::vector<int> b_idx, C_idx;
    std::vector<double> b_val, C_val;
    for (int i = 0; i < p.con_num; i++)
        if (p.b[i] != 0.0)
        {
            b_idx.push_back(i);
            b_val.push_back(p.b[i]);
        }
    for (int i = 0; i < p.vec_len; i++)
        if (p.C[i] != 0.0)
        {
            C_idx.push_back(i);
            C_val.push_back(p.C[i]);
        }
    return ExternalValidator(p.vec_len, p.con_num, col_ptrs.data(), rows.data(), vals.data(), (int)b_idx.size(), b_idx.data(),
                             b_val.data(), (int)C_idx.size(), C_idx.data(), C_val.data(), p.blk_types, p.blk_sizes);
}

// returned (unscaled) iterate of the solver, on the host
inline void returned_iterate(const SDPSolver &s, std::vector<double> &X, std::vector<double> &y, std::vector<double> &S)
{
    X.resize(s.X.size);
    y.resize(s.y.size);
    S.resize(s.S.size);
    cudaMemcpy(X.data(), s.X.vals, sizeof(double) * X.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), s.y.vals, sizeof(double) * y.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(S.data(), s.S.vals, sizeof(double) * S.size(), cudaMemcpyDeviceToHost);
}

// copies a device vector to the host, applies f, copies it back (fault injection from a validation callback)
template <typename F>
inline void modify_device(DeviceDenseVector<double> &v, F f)
{
    std::vector<double> h(v.size);
    cudaMemcpy(h.data(), v.vals, sizeof(double) * v.size, cudaMemcpyDeviceToHost);
    f(h);
    cudaMemcpy(v.vals, h.data(), sizeof(double) * v.size, cudaMemcpyHostToDevice);
}

} // namespace fixed_sigma_test

using namespace fixed_sigma_test;

// 1. Fixed sigma stays exactly sigma_0 for the whole solve (6000 iterations, where balance_freeze would change it).
TEST(FixedSigma, ConstantForTheWholeSolve)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 3.0));
    s.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
    ASSERT_NO_THROW(s.solve(6000, 0.0, 0, 1e-2));
    ASSERT_EQ(s.info_iter_num, 6000);
    EXPECT_EQ(s.iterations, 6000);
    for (double v : last_solve(s.info_sig_arr, s.info_iter_num))
        ASSERT_EQ(v, 3.0);
    EXPECT_EQ(s.sigma0, 3.0);
    EXPECT_EQ(s.sig, 3.0);
    EXPECT_EQ(s.sigma_changes, 0);
}

// 4. The validation keeps running beyond iteration 10000 (no hidden 10000-iteration stop) and records the requested
//    checkpoint; with an unattainable tolerance the solve ends at the safety cap.
TEST(ValidationStopping, ChecksBeyondIteration10000)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-30); // unattainable
    s.validation.checkpoint_iterations = {10000};
    ASSERT_NO_THROW(s.solve(10500, 1e-4, 0, 1e-2));
    EXPECT_EQ(s.stop_reason, "max_iter");
    EXPECT_FALSE(s.externally_validated_converged);
    EXPECT_EQ(s.info_iter_num, 10500);
    bool checkpoint_10000 = false;
    int beyond = 0;
    for (const ValidationRecord &r : s.validation_history)
    {
        checkpoint_10000 = checkpoint_10000 || (r.iter == 10000 && r.trigger == "checkpoint" && r.checkpoint);
        beyond += r.iter > 10000;
    }
    EXPECT_TRUE(checkpoint_10000);
    EXPECT_EQ(beyond, 5); // 10100, ..., 10400 and the final iterate 10500
    EXPECT_EQ(s.validation_history.back().iter, 10500);
    EXPECT_EQ(s.validation_history.back().trigger, "max_iter");
}

// 5 and 8a. With a reachable tolerance the solve stops at the first validated snapshot, before the safety cap, and
//    returns exactly that iterate (re-validated here with a fresh validator on the returned X, y, S).
TEST(ValidationStopping, StopsOnValidatedAccuracyAndReturnsIt)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    for (const cuadmm::PenaltyPolicyKind policy : {cuadmm::PenaltyPolicyKind::BalanceFreeze, cuadmm::PenaltyPolicyKind::Fixed})
    {
        SCOPED_TRACE(cuadmm::name_of(policy));
        SDPSolver s;
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
        s.penalty.policy = policy;
        s.validation.interval = 100;
        s.validation.tol = ValidationTolerances::all(1e-4);
        s.validation.checkpoint_iterations = {10000};
        ASSERT_NO_THROW(s.solve(1000000, 1e-4, 0, 1e-2));
        ASSERT_EQ(s.stop_reason, "externally_validated");
        ASSERT_TRUE(s.externally_validated_converged);
        EXPECT_LT(s.info_iter_num, 1000000);
        EXPECT_EQ(s.first_validated_iteration, s.info_iter_num);
        const ValidationRecord &last = s.validation_history.back();
        EXPECT_TRUE(last.result.validated);
        EXPECT_EQ(last.iter, s.info_iter_num);
        std::vector<double> X, y, S;
        returned_iterate(s, X, y, S);
        const ValidationResult r = make_validator(p).evaluate(X.data(), y.data(), S.data(), ValidationTolerances::all(1e-4));
        EXPECT_TRUE(r.validated) << r.failed;
        EXPECT_EQ(r.primal_res, last.result.primal_res);
        EXPECT_EQ(r.dual_res, last.result.dual_res);
        EXPECT_EQ(r.relgap, last.result.relgap);
        EXPECT_EQ(r.X_cone_violation, last.result.X_cone_violation);
        EXPECT_EQ(r.dual_cone_violation, last.result.dual_cone_violation);
        // the solve time includes every validation, the final one too
        EXPECT_GE(s.solve_time, last.time_s + last.validation_s);
        EXPECT_GE(s.solve_time, s.validation_time_s);
        printf("[ VALID    ] %s: validated at iteration %d (internal KKT first below 1e-4 at %d), %zu validations, %.3f s\n",
               cuadmm::name_of(policy), s.first_validated_iteration, s.first_internal_below_tol,
               s.validation_history.size(), s.validation_time_s);
    }
}

// 6 and 8b. The internal KKT residual passes but the cone criterion cannot (tolerance -1): the solve must not stop at
//    the crossing, it ends at the safety cap and returns the best externally evaluated iterate.
TEST(ValidationStopping, NoStopWhenConeFailsAndBestIterateReturned)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-4);
    s.validation.tol.cone = -1.0; // the X cone violation is >= 0: this criterion always fails
    // after the snapshot at 2900, X is scaled by 1e3 on the device: the final iterate (3000) is then clearly worse
    // than an earlier one, so the best externally evaluated iterate must be restored
    s.validation.on_validation = [&s](const ValidationRecord &rec, const std::vector<double> &, const std::vector<double> &,
                                      const std::vector<double> &)
    {
        if (rec.iter == 2900)
            modify_device(s.X, [](std::vector<double> &h)
                          { for (double &x : h) x *= 1e3; });
    };
    ASSERT_NO_THROW(s.solve(3000, 1e-4, 0, 1e-2));
    ASSERT_GT(s.first_internal_below_tol, 0) << "the internal KKT residual must pass for this test";
    EXPECT_LT(s.first_internal_below_tol, 3000);
    EXPECT_EQ(s.stop_reason, "max_iter");
    EXPECT_FALSE(s.externally_validated_converged);
    EXPECT_FALSE(s.converged);
    EXPECT_EQ(s.info_iter_num, 3000);
    bool crossing_failed_on_cone = false;
    // the ranking of B23: the maximum DIMACS error (the practical merit only for snapshots without a finite one)
    double best_dimacs = std::numeric_limits<double>::infinity(), best_merit = std::numeric_limits<double>::infinity();
    int best_iter = -1;
    for (const ValidationRecord &r : s.validation_history)
    {
        EXPECT_FALSE(r.result.validated);
        crossing_failed_on_cone = crossing_failed_on_cone ||
                                  (r.trigger == "internal_crossing" && r.result.failed.find("X_cone") != std::string::npos);
        const double d = std::isfinite(r.result.max_abs_dimacs) ? r.result.max_abs_dimacs : std::numeric_limits<double>::infinity();
        if (d < best_dimacs || (std::isinf(d) && std::isinf(best_dimacs) && r.result.merit() < best_merit))
        {
            best_dimacs = d;
            best_merit = r.result.merit();
            best_iter = r.iter;
        }
    }
    EXPECT_TRUE(crossing_failed_on_cone);
    EXPECT_TRUE(std::isfinite(best_dimacs)) << "the snapshots of this test have finite DIMACS errors";
    EXPECT_EQ(s.best_validation.iter, best_iter);
    EXPECT_LE(best_iter, 2900);
    ASSERT_TRUE(s.returned_best_external);
    std::vector<double> X, y, S;
    returned_iterate(s, X, y, S);
    const ValidationResult r = make_validator(p).evaluate(X.data(), y.data(), S.data(), s.validation.tol);
    EXPECT_EQ(r.max_abs_dimacs, best_dimacs) << "the returned iterate must be the best externally evaluated one (B23: by max |DIMACS|)";
    EXPECT_EQ(r.merit(), best_merit);
    // and the reported residuals are the solver's own ones of that snapshot
    EXPECT_EQ(s.errRp, s.best_validation.internal_errRp);
    EXPECT_EQ(s.errRd, s.best_validation.internal_errRd);
    EXPECT_EQ(std::max(std::max(s.errRp, s.errRd), s.relgap), s.best_validation.internal_kkt);
}

// 7b and 8c. Non-finite residuals during a validating solve (NaN injected into X after the snapshot at 1000): the
//    solve stops with stop_reason "non_finite" and returns the best externally evaluated iterate, which is finite.
TEST(ValidationStopping, NonFiniteIterateStopsWithTheBestIterate)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-30); // never validated
    s.validation.on_validation = [&s](const ValidationRecord &rec, const std::vector<double> &, const std::vector<double> &,
                                      const std::vector<double> &)
    {
        if (rec.iter == 1000)
            modify_device(s.X, [](std::vector<double> &h)
                          { h[0] = std::numeric_limits<double>::quiet_NaN(); });
    };
    ASSERT_NO_THROW(s.solve(3000, 1e-4, 0, 1e-2));
    EXPECT_EQ(s.stop_reason, "non_finite");
    EXPECT_FALSE(s.externally_validated_converged);
    EXPECT_FALSE(s.converged);
    EXPECT_LT(s.info_iter_num, 1010);
    ASSERT_TRUE(s.returned_best_external);
    EXPECT_LE(s.best_validation.iter, 1000);
    std::vector<double> X, y, S;
    returned_iterate(s, X, y, S);
    const ValidationResult r = make_validator(p).evaluate(X.data(), y.data(), S.data(), s.validation.tol);
    EXPECT_TRUE(r.finite);
    EXPECT_EQ(r.merit(), s.best_validation.result.merit());
    EXPECT_TRUE(std::isfinite(s.errRp) && std::isfinite(s.errRd) && std::isfinite(s.relgap));
}

// 5b. Validation-aware stopping is not internal stopping: with an external tolerance of 1e-3 and an internal one of
//    1e-8, the solve stops at a regular validation snapshot although its internal KKT residual never reached 1e-8.
TEST(ValidationStopping, StopsOnExternalAccuracyNotOnTheInternalResidual)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-3);
    ASSERT_NO_THROW(s.solve(100000, 1e-8, 0, 1e-2));
    ASSERT_EQ(s.stop_reason, "externally_validated");
    EXPECT_EQ(s.first_internal_below_tol, -1);
    EXPECT_FALSE(s.internal_converged_at_return);
    EXPECT_EQ(s.validation_history.back().trigger, "interval");
    EXPECT_EQ(s.info_iter_num % 100, 0);
    EXPECT_GT(s.validation_history.back().internal_kkt, 1e-8);
}

// 4b. A requested checkpoint is flagged (and passed to on_validation) even when it is also the safety-cap iteration.
TEST(ValidationStopping, CheckpointFlaggedWhateverTheTrigger)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-30);
    s.validation.checkpoint_iterations = {150, 400};
    std::vector<int> seen;
    s.validation.on_validation = [&seen](const ValidationRecord &rec, const std::vector<double> &X, const std::vector<double> &,
                                         const std::vector<double> &)
    {
        if (rec.checkpoint && !X.empty())
            seen.push_back(rec.iter);
    };
    ASSERT_NO_THROW(s.solve(400, 1e-4, 0, 1e-2));
    ASSERT_EQ(seen, (std::vector<int>{150, 400}));
    EXPECT_EQ(s.validation_history.back().trigger, "max_iter");
    EXPECT_TRUE(s.validation_history.back().checkpoint);
    // a callback that throws is reported and counted; the solve goes on
    SDPSolver t;
    ASSERT_NO_THROW(solver_setup::init_solver(p, t, true, 1.0));
    t.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
    t.validation.interval = 100;
    t.validation.tol = ValidationTolerances::all(1e-30);
    t.validation.on_validation = [](const ValidationRecord &, const std::vector<double> &, const std::vector<double> &,
                                    const std::vector<double> &)
    { throw std::runtime_error("disk full"); };
    ASSERT_NO_THROW(t.solve(300, 1e-4, 0, 1e-2));
    EXPECT_GE(t.validation_history.size(), 3u); // 100, 200, 300 and the internal crossing
    EXPECT_EQ(t.callback_failures, (int)t.validation_history.size());
    EXPECT_EQ(t.info_iter_num, 300);
}

// The starting point is validated when the solve ends at once (max_iter = 0) and when a warm restart starts from an
//    iterate that already satisfies the internal tolerance, which then stops after 0 iterations as in legacy mode.
TEST(ValidationStopping, StartingPointIsValidated)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    {
        SDPSolver z;
        ASSERT_NO_THROW(solver_setup::init_solver(p, z, true, 1.0));
        z.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
        z.validation.interval = 100;
        z.validation.tol = ValidationTolerances::all(1e-4);
        ASSERT_NO_THROW(z.solve(0, 1e-4, 0, 1e-2));
        ASSERT_EQ(z.validation_history.size(), 1u);
        EXPECT_EQ(z.validation_history[0].iter, 0);
        EXPECT_EQ(z.validation_history[0].trigger, "max_iter");
        EXPECT_EQ(z.stop_reason, "max_iter");
    }
    // solve to the validated accuracy, then restart from that point (if_first = false)
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-4);
    ASSERT_NO_THROW(s.solve(100000, 1e-4, 0, 1e-2));
    ASSERT_TRUE(s.externally_validated_converged);
    ASSERT_NO_THROW(s.solve(100000, 1e-4, 0, 1e-2, false));
    EXPECT_TRUE(s.externally_validated_converged);
    EXPECT_EQ(s.info_iter_num, 0);
    ASSERT_EQ(s.validation_history.size(), 1u);
    EXPECT_EQ(s.validation_history[0].iter, 0);
    EXPECT_EQ(s.validation_history[0].trigger, "internal_crossing");
}

// 7. Non-finite values fail closed: a NaN or Inf anywhere in X, y or S is never validated.
TEST(ValidationStopping, NonFiniteValuesFailClosed)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    const ExternalValidator v = make_validator(p);
    const ValidationTolerances tol = ValidationTolerances::all(1e-4);
    const ValidationResult ok = v.evaluate(p.X_star.data(), p.y_star.data(), p.S_star.data(), tol);
    EXPECT_TRUE(ok.validated) << ok.failed; // the known optimal solution passes
    EXPECT_TRUE(ok.finite);
    const double bad[2] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    for (double b : bad)
        for (int which = 0; which < 3; which++)
        {
            std::vector<double> X = p.X_star, y = p.y_star, S = p.S_star;
            (which == 0 ? X : which == 1 ? y : S)[1] = b;
            const ValidationResult r = v.evaluate(X.data(), y.data(), S.data(), tol);
            EXPECT_FALSE(r.validated);
            EXPECT_FALSE(r.finite);
            EXPECT_EQ(r.failed, "non-finite");
            EXPECT_TRUE(std::isinf(r.merit()));
        }
}

// 10. The command line: the default policy (balance_freeze; explicit and implicit runs identical), its parameters in
//     the summary, the fixed policy with validation-aware stopping, checkpoints, and the options of 2.0 that are gone
//     (the sGS scheme, the reference policies, the experiment knobs, the flags of 1.x) rejected. Runs ./cuadmm_exe from
//     the build directory.
TEST(Cli, DefaultPolicyValidationAndRemovedOptions)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    const std::string dir = "fixed_sigma_cli/PSD_Free";
    regression::write_txt(p, dir);
    auto run = [&](const std::string &args, const std::string &tag)
    {
        for (const char *ext : {".json", ".csv", ".log", "_validation.csv"})
            std::remove(("fixed_sigma_cli/" + tag + ext).c_str());
        const std::string cmd = "./cuadmm_exe " + dir + " " + args + " --summary fixed_sigma_cli/" + tag + ".json --history fixed_sigma_cli/" +
                                tag + ".csv > fixed_sigma_cli/" + tag + ".log 2>&1";
        return std::system(cmd.c_str());
    };
    auto read = [](const std::string &f)
    {
        std::ifstream in(f);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };
    ASSERT_EQ(run("--max-iter 400 --tol 1e-9", "default"), 0);
    ASSERT_EQ(run("--max-iter 400 --tol 1e-9 --policy balance_freeze --sigma0 100 --factor 1.35 --sigma-min 1e-3 --sigma-max 1e3 "
                  "--freeze-iteration 10000",
                  "explicit"),
              0);
    // identical histories apart from the time column
    auto strip_time = [](const std::string &csv)
    {
        std::stringstream in(csv), out;
        std::string line;
        while (std::getline(in, line))
        {
            const size_t a = line.find(','), b = line.find(',', a + 1);
            out << line.substr(0, a) << line.substr(b) << "\n";
        }
        return out.str();
    };
    EXPECT_EQ(strip_time(read("fixed_sigma_cli/default.csv")), strip_time(read("fixed_sigma_cli/explicit.csv")));
    const std::string history = read("fixed_sigma_cli/default.csv");
    EXPECT_EQ(history.substr(0, history.find('\n')), "iter,time_s,pobj,dobj,eta_p,eta_d,eta_g,sigma,tau");
    const std::string def = read("fixed_sigma_cli/default.json");
    EXPECT_NE(def.find("\"sigma_policy\": \"balance_freeze\""), std::string::npos);
    EXPECT_NE(def.find("\"sigma_policy_parameters\": {\"factor\": 1.35, \"sigma_min\": 0.001, \"sigma_max\": 1000, \"freeze_iteration\": 10000"),
              std::string::npos)
        << def;
    EXPECT_NE(def.find("\"build_commit\": \""), std::string::npos);
    EXPECT_NE(def.find("\"validation_interval\": 0"), std::string::npos);
    EXPECT_NE(def.find("\"algorithm\": \"plain ADMM\""), std::string::npos);
    ASSERT_EQ(run("--max-iter 400 --tol 1e-9 --factor 1.2", "factor12"), 0);
    EXPECT_NE(read("fixed_sigma_cli/factor12.json").find("\"sigma_policy_parameters\": {\"factor\": 1.2"), std::string::npos);
    EXPECT_NE(run("--factor 1", "bad_factor"), 0);
    EXPECT_NE(run("--sigma-min 10 --sigma-max 1", "bad_bounds"), 0);
    EXPECT_NE(run("--freeze-iteration -1", "bad_freeze"), 0);

    ASSERT_EQ(run("--policy fixed --sigma0 1 --max-iter 1000000 --validation-interval 100 --validation-tol 1e-4 "
                  "--validation-history fixed_sigma_cli/fixed_validation.csv",
                  "fixed"),
              0);
    EXPECT_FALSE(read("fixed_sigma_cli/fixed.csv").empty());
    const std::string fixed = read("fixed_sigma_cli/fixed.json");
    EXPECT_NE(fixed.find("\"sigma_policy\": \"fixed\""), std::string::npos);
    EXPECT_NE(fixed.find("\"sigma_changes\": 0"), std::string::npos);
    EXPECT_NE(fixed.find("\"stop_reason\": \"externally_validated\""), std::string::npos);
    EXPECT_NE(fixed.find("\"externally_validated_converged\": true"), std::string::npos);
    EXPECT_NE(read("fixed_sigma_cli/fixed.log").find("Algorithm: plain ADMM"), std::string::npos);
    EXPECT_FALSE(read("fixed_sigma_cli/fixed_validation.csv").empty());

    // no JSON value may be nan or inf
    for (const std::string &json : {def, fixed})
        for (const char *bad : {": nan", ": -nan", ": inf", ": -inf"})
            EXPECT_EQ(json.find(bad), std::string::npos) << bad;

    // checkpoints are saved even when one is the safety-cap iteration (unattainable tolerance: no early stop)
    std::system("rm -rf fixed_sigma_cli/ck");
    ASSERT_EQ(run("--policy fixed --sigma0 1 --max-iter 300 --validation-interval 100 --validation-tol 1e-12 "
                  "--checkpoint-iterations 150,300 --checkpoint-dir fixed_sigma_cli/ck",
                  "checkpoints"),
              0);
    struct stat st;
    EXPECT_EQ(stat("fixed_sigma_cli/ck/iter_150/X.txt", &st), 0);
    EXPECT_EQ(stat("fixed_sigma_cli/ck/iter_300/y.txt", &st), 0);
    EXPECT_NE(read("fixed_sigma_cli/checkpoints.json").find("\"stop_reason\": \"max_iter\""), std::string::npos);

    EXPECT_NE(run("--policy adaptive2", "bad_policy"), 0);
    EXPECT_NE(run("--checkpoint-iterations 100", "checkpoint_without_validation"), 0);
    EXPECT_NE(run("--validation-tol 1e-4", "tol_without_validation"), 0);
    EXPECT_NE(run("--validation-interval 100 --checkpoint-iterations 1e4", "bad_checkpoint"), 0);
    // the options of 2.0 that are gone, and the flags of 1.x
    for (const char *removed : {"--policy residual_balancing", "--policy legacy_adaptive", "--scheme sgs_admm", "--sgs-iterations 10",
                                "--pingpong-window 8", "--step-schedule constant", "--freeze-value last", "--schedule 200:10",
                                "--psd-final-product syrk", "--determinism strict", "--autotune 0", "--sig 1", "--sigma-policy fixed",
                                "--algorithm sgs", "--switch-admm 5", "--validate-interval 100", "--streams 4"})
        EXPECT_NE(run(removed, "removed"), 0) << removed;
}
