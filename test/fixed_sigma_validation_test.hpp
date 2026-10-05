/*

    fixed_sigma_validation_test.hpp

    Tests of the explicit sigma policy (SigmaPolicy::FIXED / LEGACY_ADAPTIVE) and of the validation-aware stopping of
    solve() (ValidationConfig, ExternalValidator):
      - fixed sigma is exactly constant for the whole solve, in pure sGS and in plain ADMM;
      - the legacy adaptive policy (and plain ADMM) reproduces the histories of the pre-change solver 290660f
        (test/data/legacy_golden.txt, generated on an H200 by
        experiments/2026-09-28_fixed_sigma_to_tolerance/scripts/gen_legacy_golden.cu; regenerate it on another GPU
        architecture, whose rounding may differ);
      - pure sGS never takes a plain ADMM step;
      - validation continues beyond iteration 10000, the solve stops on the validated accuracy, not on the internal
        KKT residual, and not when the cone criterion fails;
      - non-finite values fail closed;
      - the returned iterate is the validated one or, without validation success, the best externally evaluated one
        (also after non-finite residuals);
      - checkpoints, the starting point, the solve time and the command-line interface (previous defaults kept, new
        flags checked, JSON parseable).

*/

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <sys/stat.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "cuadmm/external_validation.h"
#include "cuadmm/solver.h"
#include "regression_problems.hpp"
#include "solver_setup.hpp"

#ifndef CUADMM_TEST_DATA_DIR
#define CUADMM_TEST_DATA_DIR "test/data"
#endif

// In the plain-ADMM-only build (CUADMM_PLAIN_ADMM_ONLY) solve() rejects sGS-ADMM: the validation tests then run on
// plain ADMM, the sGS-only tests and golden cases are compiled out (see plain_admm_only_test.hpp for the rejection).
#ifdef CUADMM_PLAIN_ADMM_ONLY
#define CUADMM_SGS_OR_PLAIN 0
#define CUADMM_ALGORITHM_FLAG "admm"
static const std::vector<int> CUADMM_SGS_AND_PLAIN = {0};
#else
#define CUADMM_SGS_OR_PLAIN INT_MAX
#define CUADMM_ALGORITHM_FLAG "sgs"
static const std::vector<int> CUADMM_SGS_AND_PLAIN = {INT_MAX, 0};
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

struct GoldenCase
{
    std::string name, problem;
    int switch_admm = 0, max_iter = 0, iterations = 0;
    double stop_tol = 0, sig0 = 0, sigscale = 0;
    std::vector<std::pair<int, double>> sigma;                 // (iteration, sigma) where sigma changes
    std::vector<std::vector<double>> samples;                   // iter, pobj, dobj, errRp, errRd, relgap
    std::vector<double> final_state; // converged, returned_best_iterate, errRp, errRd, relgap, pobj, dobj, sums of X, y, S
};

// The golden file was generated on an H200: there the comparison is exact (bitwise); another GPU model may round
// differently, and the histories are then compared to 1e-9 (relative).
inline bool same_gpu_as_golden()
{
    cudaDeviceProp prop;
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess && std::string(prop.name).find("H200") != std::string::npos;
}

// sum and sum of squares of a device vector (as in gen_legacy_golden.cu)
inline void device_sums(const DeviceDenseVector<double> &v, double &sum, double &sq)
{
    std::vector<double> h(v.size);
    cudaMemcpy(h.data(), v.vals, sizeof(double) * v.size, cudaMemcpyDeviceToHost);
    long double s = 0, q = 0;
    for (double x : h)
    {
        s += x;
        q += (long double)x * x;
    }
    sum = (double)s;
    sq = (double)q;
}

inline std::vector<GoldenCase> read_golden(const std::string &file)
{
    std::ifstream in(file);
    std::vector<GoldenCase> cases;
    std::string tok;
    while (in >> tok)
    {
        if (tok[0] == '#')
        {
            std::getline(in, tok);
            continue;
        }
        if (tok != "case")
            continue;
        GoldenCase c;
        in >> c.name >> c.problem >> c.switch_admm >> c.max_iter >> c.stop_tol >> c.sig0 >> c.sigscale;
        size_t n;
        in >> tok >> c.iterations >> tok >> n;
        for (size_t k = 0; k < n; k++)
        {
            int it;
            double s;
            in >> it >> s;
            c.sigma.push_back({it, s});
        }
        in >> tok >> n;
        for (size_t k = 0; k < n; k++)
        {
            std::vector<double> row(6);
            for (double &v : row)
                in >> v;
            c.samples.push_back(row);
        }
        in >> tok; // final or end
        if (tok == "final")
        {
            c.final_state.resize(13);
            for (double &v : c.final_state)
                in >> v;
            in >> tok; // end
        }
        cases.push_back(c);
    }
    return cases;
}

// Solves a golden case with the current code and compares with the pre-change solver: the sigma history exactly,
// the sampled objectives and residuals and the returned iterate (after the internal stop or the legacy best-iterate
// return) exactly on the golden GPU model, else to 1e-9 (relative).
inline void check_golden(const GoldenCase &c)
{
    const bool exact = same_gpu_as_golden();
    const double tol = exact ? 0.0 : 1e-9;
    SCOPED_TRACE(c.name);
    const regression::RegressionProblem p = regression::generate(c.problem);
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, c.sig0));
    ASSERT_EQ(s.sigma_policy, SigmaPolicy::LEGACY_ADAPTIVE); // the default
    ASSERT_NO_THROW(s.solve(c.max_iter, c.stop_tol, 500, 50, 100, c.switch_admm, 0, 1e-2, c.sigscale));
    ASSERT_EQ(s.info_iter_num, c.iterations);
    const std::vector<double> sig = last_solve(s.info_sig_arr, s.info_iter_num);
    std::vector<std::pair<int, double>> changes;
    for (int k = 0; k < s.info_iter_num; k++)
        if (k == 0 || sig[k] != sig[k - 1])
            changes.push_back({k + 1, sig[k]});
    ASSERT_EQ(changes.size(), c.sigma.size());
    for (size_t k = 0; k < changes.size(); k++)
    {
        EXPECT_EQ(changes[k].first, c.sigma[k].first) << "sigma change " << k;
        EXPECT_EQ(changes[k].second, c.sigma[k].second) << "sigma change " << k;
    }
    const std::vector<double> pobj = last_solve(s.info_pobj_arr, s.info_iter_num), dobj = last_solve(s.info_dobj_arr, s.info_iter_num),
                              rp = last_solve(s.info_errRp_arr, s.info_iter_num), rd = last_solve(s.info_errRd_arr, s.info_iter_num),
                              gap = last_solve(s.info_relgap_arr, s.info_iter_num);
    double worst = 0.0;
    for (const std::vector<double> &row : c.samples)
    {
        const int k = (int)row[0] - 1;
        const double now[5] = {pobj[k], dobj[k], rp[k], rd[k], gap[k]};
        for (int j = 0; j < 5; j++)
            worst = std::max(worst, std::abs(now[j] - row[j + 1]) / std::max(std::abs(row[j + 1]), 1e-300));
    }
    double worst_final = 0.0;
    if (!c.final_state.empty())
    {
        EXPECT_EQ(s.converged ? 1.0 : 0.0, c.final_state[0]);
        EXPECT_EQ(s.returned_best_iterate ? 1.0 : 0.0, c.final_state[1]);
        double now[11] = {s.errRp, s.errRd, s.relgap, s.pobj, s.dobj};
        device_sums(s.X, now[5], now[6]);
        device_sums(s.y, now[7], now[8]);
        device_sums(s.S, now[9], now[10]);
        for (int j = 0; j < 11; j++)
            worst_final = std::max(worst_final, std::abs(now[j] - c.final_state[j + 2]) / std::max(std::abs(c.final_state[j + 2]), 1e-300));
    }
    printf("[ GOLDEN   ] %s: %d iterations, %zu sigma values identical, max relative difference: samples %.2e, returned "
           "iterate %.2e (%s; converged %d, best-iterate return %d)\n",
           c.name.c_str(), s.info_iter_num, changes.size(), worst, worst_final, exact ? "exact comparison on the golden GPU" : "tolerance 1e-9",
           s.converged ? 1 : 0, s.returned_best_iterate ? 1 : 0);
    EXPECT_LE(worst, tol);
    EXPECT_LE(worst_final, tol);
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

// 1. Fixed sigma stays exactly sigma_0 for the whole solve, in pure sGS and in plain ADMM (whose adaptive schedule
//    and Monitor1 correction would otherwise change it; 6000 iterations reach the Monitor1 region).
TEST(FixedSigma, ConstantForTheWholeSolve)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    for (int switch_admm : CUADMM_SGS_AND_PLAIN)
    {
        SCOPED_TRACE(switch_admm == 0 ? "plain ADMM" : "pure sGS");
        SDPSolver s;
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 3.0));
        s.sigma_policy = SigmaPolicy::FIXED;
        ASSERT_NO_THROW(s.solve(6000, 0.0, 500, 50, 100, switch_admm, 0, 1e-2, 2.0));
        ASSERT_EQ(s.info_iter_num, 6000);
        for (double v : last_solve(s.info_sig_arr, s.info_iter_num))
            ASSERT_EQ(v, 3.0);
        EXPECT_EQ(s.sigma0, 3.0);
        EXPECT_EQ(s.sig, 3.0);
        EXPECT_EQ(s.sigma_changes, 0);
    }
}

// 2 and 9. The legacy adaptive policy reproduces the sigma histories of the pre-change solver, and plain ADMM (and
//    pure sGS and the hybrid) give the same objectives and residuals.
TEST(FixedSigma, LegacyAdaptiveAndPlainAdmmUnchanged)
{
    const std::vector<GoldenCase> cases = read_golden(std::string(CUADMM_TEST_DATA_DIR) + "/legacy_golden.txt");
    ASSERT_EQ(cases.size(), 6u);
    for (const GoldenCase &c : cases)
    {
#ifdef CUADMM_PLAIN_ADMM_ONLY
        if (c.switch_admm > 1)
        {
            printf("[ GOLDEN   ] %s: skipped (sGS-ADMM is disabled in the plain-ADMM-only build)\n", c.name.c_str());
            continue;
        }
#endif
        check_golden(c);
    }
}

#ifndef CUADMM_PLAIN_ADMM_ONLY
// 3. Pure sGS (switch_admm = INT_MAX, as --algorithm sgs) never takes a plain ADMM step: the counter stays 0 and the
//    primal residual contracts by |1 - tau| = 0.95 at every iteration (tau = 1.95 of the sGS branch).
TEST(FixedSigma, PureSgsNeverSwitchesToAdmm)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.sigma_policy = SigmaPolicy::FIXED;
    ASSERT_NO_THROW(s.solve(5000, 0.0, 500, 50, 100, INT_MAX, 0, 1e-2, 2.0));
    EXPECT_EQ(s.admm_phase_iterations, 0);
    EXPECT_EQ(admm_mode_name(INT_MAX, 5000), "pure sGS-ADMM (Algorithm 1)");
    const std::vector<double> rp = last_solve(s.info_errRp_arr, s.info_iter_num);
    int checked = 0;
    // exact while the residual is far above the floor of the eps*y term of the y-solve (b - A X^{k+1} =
    // (1 - tau)(b - A X^k) + tau sigma eps y^{k+1}); below about 1e-9 that term shows up in the ratio
    for (size_t k = 1; k < rp.size() && rp[k - 1] > 1e-9; k++, checked++)
        ASSERT_NEAR(rp[k] / rp[k - 1], 0.95, 1e-6) << "iteration " << k + 1;
    EXPECT_GT(checked, 100);
    // and plain ADMM counts its steps
    SDPSolver a;
    ASSERT_NO_THROW(solver_setup::init_solver(p, a, true, 1.0));
    ASSERT_NO_THROW(a.solve(300, 0.0, 500, 50, 100, 0, 0, 1e-2, 2.0));
    EXPECT_EQ(a.admm_phase_iterations, 300);
}

#endif // CUADMM_PLAIN_ADMM_ONLY

// 4. The validation keeps running beyond iteration 10000 (no hidden 10000-iteration stop) and records the requested
//    checkpoint; with an unattainable tolerance the solve ends at the safety cap.
TEST(ValidationStopping, ChecksBeyondIteration10000)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.sigma_policy = SigmaPolicy::FIXED;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-30); // unattainable
    s.validation.checkpoint_iterations = {10000};
    ASSERT_NO_THROW(s.solve(10500, 1e-4, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0));
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
    for (int switch_admm : CUADMM_SGS_AND_PLAIN)
    {
        SCOPED_TRACE(switch_admm == 0 ? "plain ADMM" : "pure sGS");
        SDPSolver s;
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
        s.sigma_policy = switch_admm == 0 ? SigmaPolicy::LEGACY_ADAPTIVE : SigmaPolicy::FIXED;
        s.validation.interval = 100;
        s.validation.tol = ValidationTolerances::all(1e-4);
        s.validation.checkpoint_iterations = {10000};
        ASSERT_NO_THROW(s.solve(1000000, 1e-4, 500, 50, 100, switch_admm, 0, 1e-2, 2.0));
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
               switch_admm == 0 ? "ADMM" : "sGS", s.first_validated_iteration, s.first_internal_below_tol,
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
    s.sigma_policy = SigmaPolicy::FIXED;
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
    ASSERT_NO_THROW(s.solve(3000, 1e-4, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0));
    ASSERT_GT(s.first_internal_below_tol, 0) << "the internal KKT residual must pass for this test";
    EXPECT_LT(s.first_internal_below_tol, 3000);
    EXPECT_EQ(s.stop_reason, "max_iter");
    EXPECT_FALSE(s.externally_validated_converged);
    EXPECT_FALSE(s.converged);
    EXPECT_EQ(s.info_iter_num, 3000);
    bool crossing_failed_on_cone = false;
    double best_merit = std::numeric_limits<double>::infinity();
    int best_iter = -1;
    for (const ValidationRecord &r : s.validation_history)
    {
        EXPECT_FALSE(r.result.validated);
        crossing_failed_on_cone = crossing_failed_on_cone ||
                                  (r.trigger == "internal_crossing" && r.result.failed.find("X_cone") != std::string::npos);
        if (r.result.merit() < best_merit)
        {
            best_merit = r.result.merit();
            best_iter = r.iter;
        }
    }
    EXPECT_TRUE(crossing_failed_on_cone);
    EXPECT_EQ(s.best_validation.iter, best_iter);
    EXPECT_LE(best_iter, 2900);
    ASSERT_TRUE(s.returned_best_external);
    std::vector<double> X, y, S;
    returned_iterate(s, X, y, S);
    const ValidationResult r = make_validator(p).evaluate(X.data(), y.data(), S.data(), s.validation.tol);
    EXPECT_EQ(r.merit(), best_merit) << "the returned iterate must be the best externally evaluated one";
    // and the reported residuals are the solver's own ones of that snapshot
    EXPECT_EQ(s.errRp, s.best_validation.internal_errRp);
    EXPECT_EQ(s.errRd, s.best_validation.internal_errRd);
    EXPECT_EQ(std::max(std::max(s.errRp, s.errRd), s.relgap), s.best_validation.internal_kkt);
}

// 7b and 8c. Non-finite residuals during a validating solve (NaN injected into X after the snapshot at 1000; the
//    sGS y-step recomputes y from scratch, so X is where a NaN persists): the
//    solve stops with stop_reason "non_finite" and returns the best externally evaluated iterate, which is finite.
TEST(ValidationStopping, NonFiniteIterateStopsWithTheBestIterate)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.sigma_policy = SigmaPolicy::FIXED;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-30); // never validated
    s.validation.on_validation = [&s](const ValidationRecord &rec, const std::vector<double> &, const std::vector<double> &,
                                      const std::vector<double> &)
    {
        if (rec.iter == 1000)
            modify_device(s.X, [](std::vector<double> &h)
                          { h[0] = std::numeric_limits<double>::quiet_NaN(); });
    };
    ASSERT_NO_THROW(s.solve(3000, 1e-4, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0));
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
    s.sigma_policy = SigmaPolicy::FIXED;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-3);
    ASSERT_NO_THROW(s.solve(100000, 1e-8, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0));
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
    s.sigma_policy = SigmaPolicy::FIXED;
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
    ASSERT_NO_THROW(s.solve(400, 1e-4, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0));
    ASSERT_EQ(seen, (std::vector<int>{150, 400}));
    EXPECT_EQ(s.validation_history.back().trigger, "max_iter");
    EXPECT_TRUE(s.validation_history.back().checkpoint);
    // a callback that throws is reported and counted; the solve goes on
    SDPSolver t;
    ASSERT_NO_THROW(solver_setup::init_solver(p, t, true, 1.0));
    t.sigma_policy = SigmaPolicy::FIXED;
    t.validation.interval = 100;
    t.validation.tol = ValidationTolerances::all(1e-30);
    t.validation.on_validation = [](const ValidationRecord &, const std::vector<double> &, const std::vector<double> &,
                                    const std::vector<double> &)
    { throw std::runtime_error("disk full"); };
    ASSERT_NO_THROW(t.solve(300, 1e-4, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0));
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
        z.sigma_policy = SigmaPolicy::FIXED;
        z.validation.interval = 100;
        z.validation.tol = ValidationTolerances::all(1e-4);
        ASSERT_NO_THROW(z.solve(0, 1e-4, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0));
        ASSERT_EQ(z.validation_history.size(), 1u);
        EXPECT_EQ(z.validation_history[0].iter, 0);
        EXPECT_EQ(z.validation_history[0].trigger, "max_iter");
        EXPECT_EQ(z.stop_reason, "max_iter");
    }
    // solve to the validated accuracy, then restart from that point (if_first = false)
    SDPSolver s;
    ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
    s.sigma_policy = SigmaPolicy::FIXED;
    s.validation.interval = 100;
    s.validation.tol = ValidationTolerances::all(1e-4);
    ASSERT_NO_THROW(s.solve(100000, 1e-4, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0));
    ASSERT_TRUE(s.externally_validated_converged);
    ASSERT_NO_THROW(s.solve(100000, 1e-4, 500, 50, 100, CUADMM_SGS_OR_PLAIN, 0, 1e-2, 2.0, false));
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

// 10. Command-line compatibility: the old flags keep their meaning (default legacy_adaptive, internal stopping), the
//     new flags work, and conflicting flags are rejected. Runs ./cuadmm_exe from the build directory.
TEST(FixedSigmaCli, BackwardCompatibleAndNewFlags)
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
    ASSERT_EQ(run("--max-iter 400 --tol 1e-9", "legacy"), 0);
    ASSERT_EQ(run("--max-iter 400 --tol 1e-9 --sigma-policy legacy_adaptive", "explicit"), 0);
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
    EXPECT_EQ(strip_time(read("fixed_sigma_cli/legacy.csv")), strip_time(read("fixed_sigma_cli/explicit.csv")));
    const std::string legacy = read("fixed_sigma_cli/legacy.json");
    EXPECT_NE(legacy.find("\"sigma_policy\": \"legacy_adaptive\""), std::string::npos);
    EXPECT_NE(legacy.find("\"validation_interval\": 0"), std::string::npos);
    EXPECT_NE(legacy.find("\"algorithm\": \"plain ADMM\""), std::string::npos);

    ASSERT_EQ(run("--algorithm " CUADMM_ALGORITHM_FLAG " --sigma-policy fixed --sig 1 --max-iter 1000000 --validate-interval 100 --validate-tol 1e-4 "
                  "--validation-history fixed_sigma_cli/new_validation.csv",
                  "new"),
              0);
    EXPECT_FALSE(read("fixed_sigma_cli/new.csv").empty());
    const std::string fresh = read("fixed_sigma_cli/new.json");
    EXPECT_NE(fresh.find("\"sigma_policy\": \"fixed\""), std::string::npos);
    EXPECT_NE(fresh.find("\"sigma_changes\": 0"), std::string::npos);
#ifdef CUADMM_PLAIN_ADMM_ONLY
    EXPECT_NE(fresh.find("\"sgs_phase_iterations\": 0"), std::string::npos);
#else
    EXPECT_NE(fresh.find("\"admm_phase_iterations\": 0"), std::string::npos);
#endif
    EXPECT_NE(fresh.find("\"stop_reason\": \"externally_validated\""), std::string::npos);
    EXPECT_NE(fresh.find("\"externally_validated_converged\": true"), std::string::npos);
#ifdef CUADMM_PLAIN_ADMM_ONLY
    EXPECT_NE(read("fixed_sigma_cli/new.log").find("Build: plain ADMM only"), std::string::npos);
#else
    EXPECT_NE(read("fixed_sigma_cli/new.log").find("Algorithm: pure sGS-ADMM"), std::string::npos);
#endif
    EXPECT_FALSE(read("fixed_sigma_cli/new_validation.csv").empty());

    // no JSON value may be nan or inf
    for (const std::string &json : {legacy, fresh})
        for (const char *bad : {": nan", ": -nan", ": inf", ": -inf"})
            EXPECT_EQ(json.find(bad), std::string::npos) << bad;

    // checkpoints are saved even when one is the safety-cap iteration (unattainable tolerance: no early stop)
    std::system("rm -rf fixed_sigma_cli/ck");
    ASSERT_EQ(run("--algorithm " CUADMM_ALGORITHM_FLAG " --sigma-policy fixed --sig 1 --max-iter 300 --validate-interval 100 --validate-tol 1e-12 "
                  "--checkpoint-iters 150,300 --checkpoint-dir fixed_sigma_cli/ck",
                  "checkpoints"),
              0);
    struct stat st;
    EXPECT_EQ(stat("fixed_sigma_cli/ck/iter_150/X.txt", &st), 0);
    EXPECT_EQ(stat("fixed_sigma_cli/ck/iter_300/y.txt", &st), 0);
    EXPECT_NE(read("fixed_sigma_cli/checkpoints.json").find("\"stop_reason\": \"max_iter\""), std::string::npos);

    EXPECT_NE(run("--algorithm sgs --switch-admm 5", "conflict"), 0);
    EXPECT_NE(run("--sigma-policy adaptive2", "bad_policy"), 0);
    EXPECT_NE(run("--checkpoint-iters 100", "checkpoint_without_validation"), 0);
    EXPECT_NE(run("--validate-tol 1e-4", "tol_without_validation"), 0);
    EXPECT_NE(run("--validate-interval 100 --checkpoint-iters 1e4", "bad_checkpoint"), 0);
}
