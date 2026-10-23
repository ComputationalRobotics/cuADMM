/*

    hybrid_switch_test.hpp

    Tests of the hybrid mode of the public code (sGS-ADMM for iter < switch_admm, then plain ADMM, in one solve();
    not Algorithm 1 of arXiv 2406.05846, which is pure sGS-ADMM):
      - before the switch the iterates are exactly those of pure sGS-ADMM, and after it they are plain ADMM continued
        from the same (X, y, S): the test fails if the state were reinitialized at the switch or if the wrong update
        ran on either side (negative controls show that the comparison discriminates);
      - the phase bookkeeping (iteration counts, switch iteration, phase times, sigma and tau per phase);
      (the hybrid-only policy fixed_sgs_then_legacy_admm of main is not kept in cuADMM-reorg: docs/CHANGES.md);
      - the --sgs-iterations command-line option.
    Uses the helpers of fixed_sigma_validation_test.hpp (included before this file).

*/

#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cuadmm_legacy/solver.h"
#include "regression_problems.hpp"
#include "solver_setup.hpp"

namespace hybrid_test
{
using fixed_sigma_test::last_solve;

struct History
{
    std::vector<double> pobj, dobj, rp, rd, gap, sig, tau;
};

// the last `n` entries of the info arrays (the last solve() when n = info_iter_num)
inline History history(const SDPSolver &s, int n)
{
    return {last_solve(s.info_pobj_arr, n), last_solve(s.info_dobj_arr, n), last_solve(s.info_errRp_arr, n),
            last_solve(s.info_errRd_arr, n), last_solve(s.info_relgap_arr, n), last_solve(s.info_sig_arr, n),
            last_solve(s.info_tau_arr, n)};
}

// entries [from, from + n) of a history
inline History slice(const History &h, int from, int n)
{
    auto cut = [&](const std::vector<double> &v)
    { return std::vector<double>(v.begin() + from, v.begin() + from + n); };
    return {cut(h.pobj), cut(h.dobj), cut(h.rp), cut(h.rd), cut(h.gap), cut(h.sig), cut(h.tau)};
}

// Discrepancy of two histories of equal length, in units of the tolerance 1e-10: the objectives relative, the residuals
// (errRp, errRd, relgap: already relative quantities, which reach the round-off floor of ~1e-15 and are formed by
// cancellation) absolute. <= 1 means equal to the rounding of a floating-point round trip of the state.
inline double discrepancy(const History &a, const History &b)
{
    const double tol = 1e-10;
    double w = 0.0;
    for (size_t k = 0; k < a.pobj.size(); k++)
    {
        w = std::max(w, std::abs(a.pobj[k] - b.pobj[k]) / (tol * std::max(std::abs(a.pobj[k]), std::abs(b.pobj[k]))));
        w = std::max(w, std::abs(a.dobj[k] - b.dobj[k]) / (tol * std::max(std::abs(a.dobj[k]), std::abs(b.dobj[k]))));
        w = std::max(w, std::abs(a.rp[k] - b.rp[k]) / tol);
        w = std::max(w, std::abs(a.rd[k] - b.rd[k]) / tol);
        w = std::max(w, std::abs(a.gap[k] - b.gap[k]) / tol);
    }
    return w;
}

// per series: the largest relative difference, where it occurs and the values there (diagnostics)
inline std::string diff_report(const History &a, const History &b)
{
    const char *names[5] = {"pobj", "dobj", "errRp", "errRd", "relgap"};
    const std::vector<double> *x[5] = {&a.pobj, &a.dobj, &a.rp, &a.rd, &a.gap}, *y[5] = {&b.pobj, &b.dobj, &b.rp, &b.rd, &b.gap};
    std::string out;
    for (int j = 0; j < 5; j++)
    {
        double w = 0.0, u0 = 0, v0 = 0;
        size_t at = 0;
        for (size_t k = 0; k < x[j]->size(); k++)
        {
            const double u = (*x[j])[k], v = (*y[j])[k];
            const double r = std::abs(u - v) / std::max(std::max(std::abs(u), std::abs(v)), 1e-300);
            if (r > w)
            {
                w = r;
                at = k;
                u0 = u;
                v0 = v;
            }
        }
        char buf[200];
        snprintf(buf, sizeof(buf), " %s %.1e at %zu (%.6e vs %.6e);", names[j], w, at, u0, v0);
        out += buf;
    }
    return out;
}

inline bool bitwise_equal(const History &a, const History &b)
{
    return a.pobj == b.pobj && a.dobj == b.dobj && a.rp == b.rp && a.rd == b.rd && a.gap == b.gap && a.sig == b.sig && a.tau == b.tau;
}

inline void device_copy(const DeviceDenseVector<double> &v, std::vector<double> &h)
{
    h.resize(v.size);
    cudaMemcpy(h.data(), v.vals, sizeof(double) * v.size, cudaMemcpyDeviceToHost);
}

} // namespace hybrid_test

using namespace hybrid_test;

// The switch keeps the state and changes only the update: with N = 100 sGS iterations then M = 200 plain-ADMM ones
// (switch_admm = N + 1, fixed sigma = 1, no stopping), (a) the first N iterations and the scaled state (X, y, S) after
// them are bitwise those of pure sGS-ADMM, (b) the last M iterations are those of a plain-ADMM solve continued from
// that state on the same solver object (to the rounding of its unscale/rescale round trip), (c) the negative controls
// (pure sGS continued, plain ADMM from the cold start) differ by far more, so a reinitialized state or the wrong update
// on either side would fail (b), and (d) the per-iteration signatures agree: tau 1.95 / 1.618 and the exact 0.95
// contraction of the primal residual of sGS-ADMM before the switch only.
TEST(HybridSwitch, StateCarriedOverAndOneAlgorithmPerPhase)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    const int N = 100, M = 200;
    auto fresh = [&](SDPSolver &s)
    {
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
        s.sigma_policy = SigmaPolicy::FIXED;
    };
    // the hybrid; the validation callback at iteration N (read-only snapshot) records the scaled device state
    SDPSolver h;
    fresh(h);
    std::vector<double> hX, hy, hS;
    h.validation.interval = N;
    h.validation.tol = ValidationTolerances::all(1e-300); // never validated: no early stop
    h.validation.on_validation = [&](const ValidationRecord &rec, const std::vector<double> &, const std::vector<double> &,
                                     const std::vector<double> &)
    {
        if (rec.iter == N)
        {
            device_copy(h.X, hX);
            device_copy(h.y, hy);
            device_copy(h.S, hS);
        }
    };
    ASSERT_NO_THROW(h.solve(N + M, 0.0, 500, 50, 100, N + 1, 0, 1e-2, 2.0));
    ASSERT_EQ(h.info_iter_num, N + M);
    EXPECT_EQ(h.sgs_phase_iterations, N);
    EXPECT_EQ(h.admm_phase_iterations, M);
    EXPECT_EQ(h.switch_iteration, N + 1);
    EXPECT_EQ(h.sigma_at_switch, 1.0);
    EXPECT_EQ(h.sigma_changes, 0);
    EXPECT_EQ(h.tau_last_sgs, 1.95);
    EXPECT_GT(h.sgs_phase_time_s, 0.0);
    EXPECT_GT(h.admm_phase_time_s, 0.0);
    EXPECT_NEAR(h.sgs_phase_time_s + h.admm_phase_time_s, h.solve_time, 1e-9);
    const History H = history(h, N + M);
    for (int k = 0; k < N + M; k++)
    {
        ASSERT_EQ(H.sig[k], 1.0) << "iteration " << k + 1;
        ASSERT_EQ(H.tau[k], k < N ? 1.95 : 1.618) << "iteration " << k + 1;
    }

    // (a) pure sGS-ADMM for N iterations: the same iterates and the same state at the switch, bitwise
    SDPSolver g;
    fresh(g);
    std::vector<double> gX, gy, gS;
    g.validation = h.validation;
    g.validation.on_validation = [&](const ValidationRecord &rec, const std::vector<double> &, const std::vector<double> &,
                                     const std::vector<double> &)
    {
        if (rec.iter == N)
        {
            device_copy(g.X, gX);
            device_copy(g.y, gy);
            device_copy(g.S, gS);
        }
    };
    ASSERT_NO_THROW(g.solve(N, 0.0, 500, 50, 100, INT_MAX, 0, 1e-2, 2.0));
    EXPECT_TRUE(bitwise_equal(slice(H, 0, N), history(g, N))) << "the sGS phase differs from pure sGS-ADMM";
    ASSERT_FALSE(hX.empty());
    EXPECT_TRUE(hX == gX && hy == gy && hS == gS) << "the state handed to plain ADMM differs from the pure-sGS state";

    // (b) plain ADMM continued from that state on the same object (if_first = false: X, y, S unscaled then rescaled)
    SDPSolver c;
    fresh(c);
    ASSERT_NO_THROW(c.solve(N, 0.0, 500, 50, 100, INT_MAX, 0, 1e-2, 2.0));
    ASSERT_NO_THROW(c.solve(M, 0.0, 500, 50, 100, 0, 0, 1e-2, 2.0, false));
    ASSERT_EQ(c.info_iter_num, M);
    EXPECT_EQ(c.admm_phase_iterations, M);
    const double cont = discrepancy(slice(H, N, M), history(c, M));

    // (c) negative controls: sGS continued instead of plain ADMM, and plain ADMM from the cold start
    SDPSolver q;
    fresh(q);
    ASSERT_NO_THROW(q.solve(N + M, 0.0, 500, 50, 100, INT_MAX, 0, 1e-2, 2.0));
    const double wrong_update = discrepancy(slice(H, N, M), slice(history(q, N + M), N, M));
    SDPSolver a;
    fresh(a);
    ASSERT_NO_THROW(a.solve(M, 0.0, 500, 50, 100, 0, 0, 1e-2, 2.0));
    const double reinitialized = discrepancy(slice(H, N, M), history(a, M));
    printf("[ HYBRID   ] discrepancy (units of 1e-10) of the post-switch iterates vs a plain-ADMM continuation: %.2e; vs sGS continued: "
           "%.2e; vs plain ADMM from the cold start: %.2e\n", cont, wrong_update, reinitialized);
    printf("[ HYBRID   ] continuation per series:%s\n", diff_report(slice(H, N, M), history(c, M)).c_str());
    EXPECT_LE(cont, 1.0);
    EXPECT_GT(wrong_update, 1e3);
    EXPECT_GT(reinitialized, 1e3);

    // (d) the exact contraction b - A X^{k+1} = (1 - tau)(b - A X^k) of sGS-ADMM holds before the switch only
    int sgs_exact = 0, admm_exact = 0;
    for (int k = 1; k < N + M; k++)
    {
        if (H.rp[k - 1] < 1e-9)
            continue;
        const bool exact = std::abs(H.rp[k] / H.rp[k - 1] - 0.95) < 1e-6;
        (k < N ? sgs_exact : admm_exact) += exact;
    }
    EXPECT_GE(sgs_exact, N - 5);
    EXPECT_LE(admm_exact, 5);
}

// Phase bookkeeping in a validating hybrid solve: the switch state is validated (iteration N), the phase counts and
// times add up, and the validation time splits between the phases; with the switch after the validated iterate no
// plain-ADMM step runs.
TEST(HybridSwitch, PhaseBookkeepingWithValidation)
{
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    for (int N : {100, 5000})
    {
        SCOPED_TRACE("N = " + std::to_string(N));
        SDPSolver s;
        ASSERT_NO_THROW(solver_setup::init_solver(p, s, true, 1.0));
        s.sigma_policy = SigmaPolicy::FIXED;
        s.validation.interval = 100;
        s.validation.tol = ValidationTolerances::all(1e-4);
        ASSERT_NO_THROW(s.solve(1000000, 1e-4, 500, 50, 100, N + 1, 0, 1e-2, 2.0));
        ASSERT_TRUE(s.externally_validated_converged);
        EXPECT_EQ(s.sgs_phase_iterations + s.admm_phase_iterations, s.info_iter_num);
        EXPECT_NEAR(s.sgs_phase_validation_s + s.admm_phase_validation_s, s.validation_time_s, 1e-9);
        EXPECT_NEAR(s.sgs_phase_time_s + s.admm_phase_time_s, s.solve_time, 1e-9);
        if (s.info_iter_num > N)
        {
            EXPECT_EQ(s.sgs_phase_iterations, N);
            EXPECT_EQ(s.switch_iteration, N + 1);
            const ValidationRecord *at = nullptr;
            for (const ValidationRecord &r : s.validation_history)
                if (r.iter == N)
                    at = &r;
            ASSERT_NE(at, nullptr) << "the switch state is validated";
            EXPECT_EQ(s.kkt_at_switch, at->internal_kkt);
            EXPECT_EQ(s.sigma_at_switch, 1.0);
            EXPECT_GT(s.admm_phase_time_s, 0.0);
            printf("[ HYBRID   ] N = %d: validated at %d (%d sGS + %d ADMM), phases %.3f + %.3f s\n", N, s.info_iter_num,
                   s.sgs_phase_iterations, s.admm_phase_iterations, s.sgs_phase_time_s, s.admm_phase_time_s);
        }
        else
        {
            EXPECT_EQ(s.switch_iteration, -1);
            EXPECT_EQ(s.admm_phase_iterations, 0);
            EXPECT_EQ(s.sgs_phase_time_s, s.solve_time);
            EXPECT_EQ(s.admm_phase_time_s, 0.0);
        }
    }
}

// --sgs-iterations N: switch_admm = N + 1, N sGS then plain ADMM; the history marks the phase and tau of every
// iteration; 0 is plain ADMM; invalid combinations and values are rejected.
TEST(HybridSwitchCli, SgsIterationsOption)
{
    const std::string dir = "fixed_sigma_cli/PSD_Free"; // written by FixedSigmaCli.BackwardCompatibleAndNewFlags
    regression::write_txt(regression::generate("PSD_Free"), dir);
    auto run = [&](const std::string &args, const std::string &tag)
    {
        for (const char *ext : {".json", ".csv", ".log"})
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
    ASSERT_EQ(run("--sgs-iterations 100 --sigma-policy fixed --sig 1 --max-iter 150 --tol 1e-12", "hybrid"), 0);
    const std::string js = read("fixed_sigma_cli/hybrid.json");
    for (const char *f : {"\"switch_admm\": 101", "\"sgs_iterations_requested\": 100", "\"sgs_phase_iterations\": 100",
                          "\"admm_phase_iterations\": 50", "\"switch_iteration\": 101", "\"sigma_changes\": 0", "\"iterations\": 150"})
        EXPECT_NE(js.find(f), std::string::npos) << f;
    EXPECT_NE(js.find("\"algorithm\": \"hybrid: sGS-ADMM for iterations 1-100, then plain ADMM\""), std::string::npos);
    const std::string log = read("fixed_sigma_cli/hybrid.log");
    EXPECT_NE(log.find("switching to plain ADMM at iteration 101 (switch_admm = 101) after 100 sGS iterations"), std::string::npos);
    EXPECT_NE(log.find("Hybrid mode of the public code (not Algorithm 1)"), std::string::npos);
    std::stringstream csv(read("fixed_sigma_cli/hybrid.csv"));
    std::string line;
    std::getline(csv, line);
    EXPECT_EQ(line, "iter,time_s,pobj,dobj,eta_p,eta_d,eta_g,sigma,tau,phase"); // the plan's names (Phase 2)
    int n = 0, sgs = 0;
    while (std::getline(csv, line))
    {
        n++;
        const bool is_sgs = line.size() > 4 && line.compare(line.size() - 4, 4, ",sgs") == 0;
        sgs += is_sgs;
        EXPECT_EQ(is_sgs, n <= 100) << line;
        // iter,time_s,pobj,dobj,eta_p,eta_d,eta_g,sigma,tau,phase with 17 significant digits: parse sigma and tau
        std::vector<std::string> cols;
        std::stringstream ls(line);
        for (std::string tok; std::getline(ls, tok, ',');)
            cols.push_back(tok);
        ASSERT_EQ(cols.size(), 10u) << line;
        EXPECT_EQ(std::stod(cols[7]), 1.0) << line;
        EXPECT_EQ(std::stod(cols[8]), n <= 100 ? 1.95 : 1.618) << line;
    }
    EXPECT_EQ(n, 150);
    EXPECT_EQ(sgs, 100);
    // 0 sGS iterations is plain ADMM, identical to --algorithm admm
    ASSERT_EQ(run("--sgs-iterations 0 --sigma-policy fixed --sig 1 --max-iter 100 --tol 1e-12", "zero"), 0);
    ASSERT_EQ(run("--algorithm admm --sigma-policy fixed --sig 1 --max-iter 100 --tol 1e-12", "admm"), 0);
    auto strip_time = [](const std::string &text)
    {
        std::stringstream in(text), out;
        std::string l;
        while (std::getline(in, l))
        {
            const size_t a = l.find(','), b = l.find(',', a + 1);
            out << l.substr(0, a) << l.substr(b) << "\n";
        }
        return out.str();
    };
    EXPECT_EQ(strip_time(read("fixed_sigma_cli/zero.csv")), strip_time(read("fixed_sigma_cli/admm.csv")));
    EXPECT_NE(read("fixed_sigma_cli/zero.json").find("\"switch_admm\": 0"), std::string::npos);
    // rejected
    EXPECT_NE(run("--sgs-iterations 5 --switch-admm 6", "conflict1"), 0);
    EXPECT_NE(run("--sgs-iterations 5 --algorithm sgs", "conflict2"), 0);
    EXPECT_NE(run("--sgs-iterations -1", "negative"), 0);
    EXPECT_NE(run("--sgs-iterations 1e3", "float"), 0);
    EXPECT_NE(run("--switch-admm 1e4", "switch_float"), 0);
    EXPECT_NE(run("--max-iter 1e6", "maxiter_float"), 0);
    EXPECT_NE(run("--max-iter 2147483647", "maxiter_huge"), 0);
    EXPECT_NE(run("--sigma-policy fixed_then_adaptive", "bad_policy"), 0);
    EXPECT_NE(run("--sigma-policy fixed_sgs_then_legacy_admm", "removed_policy"), 0);
}
