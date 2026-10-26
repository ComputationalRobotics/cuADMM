/*

    penalty_policy_test.hpp  (CPU only)

    The penalty policies on scripted residual sequences (plan Section 7.3: "Unit tests drive the policies with
    scripted residual sequences stored at full precision, so they need no GPU"):
      - LegacyPenalty and ResidualBalancingPenalty replay the reference histories of test/data/golden (the 9bb4ba5
        and 2d8a755 reference builds on PSD_Free, 6000 iterations, 17 significant digits) and reproduce every sigma
        of the history and every entry of the reference sigma log exactly;
      - FixedPenalty never changes and refuses a changed sigma;
      - balance_freeze: freeze at freeze_iteration, ping-pong detection with the +,+,-,- pattern (net movement, not a
        reversal count), band-centre value, bias, halve_on_reversal, sigma_0 clamp, no change after the freeze, the
        state round trip;
      - residual_balancing and balance_freeze agree before any freeze with the same parameters;
      - balance_freeze agrees bit for bit with the CPU replica of the Phase-4b screen (cuadmm_emu.py v1.2,
        BalanceFreezeState; c4eb89b:cuADMM-reorg/tools/sigma_screen/cuadmm_emu.py) on the residual sequences of the golden
        histories for a set of parameter combinations (test/data/golden/balance_freeze_cases_*.csv, written by
        replay_policy.py of the same screen): every returned
        sigma and every sigma-log entry.

*/

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "admm/penalty_policy.h"
#include "cuadmm/options.h"

#ifndef CUADMM_TEST_DATA_DIR
#define CUADMM_TEST_DATA_DIR "test/data"
#endif

namespace penalty_test
{
using namespace cuadmm;
using namespace cuadmm::admm;

struct Row
{
    int iter;
    double pobj, dobj, eta_p, eta_d, eta_g, sigma;
};

inline std::vector<Row> read_history(const std::string &file)
{
    std::ifstream in(file);
    std::vector<Row> rows;
    std::string line;
    std::getline(in, line); // header: iter,time_s,pobj,dobj,errRp,errRd,relgap,sig,tau,phase
    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ','))
            cols.push_back(tok);
        if (cols.size() < 8)
            continue;
        rows.push_back({std::stoi(cols[0]), std::stod(cols[2]), std::stod(cols[3]), std::stod(cols[4]), std::stod(cols[5]), std::stod(cols[6]),
                        std::stod(cols[7])});
    }
    return rows;
}

struct LogRow
{
    int iter;
    double old_sigma, new_sigma;
    std::string rule;
};

inline std::vector<LogRow> read_sigma_log(const std::string &file)
{
    std::ifstream in(file);
    std::vector<LogRow> rows;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ','))
            cols.push_back(tok);
        if (cols.size() < 4)
            continue;
        rows.push_back({std::stoi(cols[0]), std::stod(cols[1]), std::stod(cols[2]), cols[3]});
    }
    return rows;
}

// Replays a history through the policy and checks the sigma of every iteration and the log against the reference.
inline void replay(PenaltyPolicy &policy, const std::string &name, double sigma0, std::size_t expected_rows)
{
    const std::vector<Row> rows = read_history(std::string(CUADMM_TEST_DATA_DIR) + "/golden/" + name + "_history.csv");
    const std::vector<LogRow> ref_log = read_sigma_log(std::string(CUADMM_TEST_DATA_DIR) + "/golden/" + name + "_sigma_log.csv");
    ASSERT_EQ(rows.size(), expected_rows) << name;
    policy.reset(sigma0);
    double sigma = sigma0;
    int mismatches = 0;
    for (const Row &row : rows)
    {
        KktResiduals r;
        r.eta_p = row.eta_p;
        r.eta_d = row.eta_d;
        r.eta_g = row.eta_g;
        r.pobj = row.pobj;
        r.dobj = row.dobj;
        sigma = policy.update(row.iter, r, sigma, false);
        if (sigma != row.sigma)
        {
            if (mismatches++ == 0)
                ADD_FAILURE() << name << ": first sigma mismatch at iteration " << row.iter << ": policy " << sigma << ", reference " << row.sigma;
        }
    }
    EXPECT_EQ(mismatches, 0) << name;
    ASSERT_EQ(policy.log().size(), ref_log.size()) << name << ": number of sigma changes";
    for (std::size_t i = 0; i < ref_log.size(); i++)
    {
        EXPECT_EQ(policy.log()[i].iteration, ref_log[i].iter) << name << " entry " << i;
        // the reference sigma log has 12 significant digits: compare to that
        EXPECT_NEAR(policy.log()[i].old_sigma, ref_log[i].old_sigma, 1e-11 * ref_log[i].old_sigma) << name << " entry " << i;
        EXPECT_NEAR(policy.log()[i].new_sigma, ref_log[i].new_sigma, 1e-11 * ref_log[i].new_sigma) << name << " entry " << i;
        EXPECT_EQ(policy.log()[i].rule, ref_log[i].rule) << name << " entry " << i;
    }
    printf("[ REPLAY   ] %s: %zu iterations, %zu sigma changes reproduced, final sigma %.17g\n", name.c_str(), rows.size(), policy.log().size(), sigma);
}

inline KktResiduals residuals(double eta_p, double eta_d)
{
    KktResiduals r;
    r.eta_p = eta_p;
    r.eta_d = eta_d;
    r.eta_g = 1e-3;
    r.pobj = 1.0;
    r.dobj = 1.0;
    return r;
}

// drives the policy with a constant vote (primal wins when primal_wins is true) for iterations [from, to]
inline double drive(PenaltyPolicy &policy, double sigma, int from, int to, bool primal_wins)
{
    for (int k = from; k <= to; k++)
        sigma = policy.update(k, primal_wins ? residuals(1e-3, 1e-1) : residuals(1e-1, 1e-3), sigma, false);
    return sigma;
}
} // namespace penalty_test

using namespace penalty_test;

TEST(PenaltyPolicy, LegacyAdaptiveReplaysTheReference9bb4ba5)
{
    PenaltyOptions o;
    o.policy = PenaltyPolicyKind::LegacyAdaptive;
    LegacyPenalty policy(o);
    EXPECT_EQ(policy.sigma_min(), 1e-6);
    EXPECT_EQ(policy.sigma_max(), 1e6);
    replay(policy, "PSD_Free_legacy_adaptive_s100_i6000", 100.0, 6000);
    // mater-1 to tolerance 1e-4 (8117 iterations): the Monitor1 correction fires twice after iteration 5000
    LegacyPenalty monitored(o);
    replay(monitored, "mater-1_legacy_adaptive_s1_tol1e-4", 1.0, 8117);
    int monitor_entries = 0;
    for (const SigmaChange &c : monitored.log())
        monitor_entries += c.rule == "monitor1";
    EXPECT_EQ(monitor_entries, 2);
}

TEST(PenaltyPolicy, ResidualBalancingReplaysTheReference2d8a755)
{
    PenaltyOptions o;
    o.policy = PenaltyPolicyKind::ResidualBalancing;
    ResidualBalancingPenalty policy(o, false);
    EXPECT_EQ(policy.sigma_min(), 1e-3);
    EXPECT_EQ(policy.sigma_max(), 1e3);
    replay(policy, "PSD_Free_residual_balancing_s100_i6000", 100.0, 6000);
    // the guard: a single win at iteration 1 does not change sigma (B14 is what the unguarded rule does)
    policy.reset(100.0);
    EXPECT_EQ(policy.update(1, residuals(1e-3, 1e-1), 100.0, false), 100.0);
    LegacyPenalty legacy(o);
    legacy.reset(100.0);
    EXPECT_EQ(legacy.update(1, residuals(1e-3, 1e-1), 100.0, false), 200.0);
}

TEST(PenaltyPolicy, FixedNeverChanges)
{
    FixedPenalty policy;
    policy.reset(3.0);
    double sigma = 3.0;
    for (int k = 1; k <= 20000; k++)
        sigma = policy.update(k, residuals(1e-3, 1e-1), sigma, k % 2 == 0);
    EXPECT_EQ(sigma, 3.0);
    EXPECT_TRUE(policy.log().empty());
    EXPECT_THROW(policy.update(20001, residuals(1e-3, 1e-1), 4.0, false), std::logic_error);
}

TEST(PenaltyPolicy, LegacyMonitor1ChasesTheObjectives)
{
    // after iteration 5000, every 25 iterations the monitor sees (pobj, dobj, eta_p, eta_d, eta_g); once 20 samples are in
    // with relgap / max(eta_p, eta_d) > 1.2, pobj and dobj both ascending and pobj > dobj, it sets sigma = max(100, sigma)
    PenaltyOptions o;
    o.policy = PenaltyPolicyKind::LegacyAdaptive;
    LegacyPenalty policy(o);
    policy.reset(1.0);
    double sigma = 1.0;
    for (int k = 1; k <= 5000; k++) // balanced votes: alternate, no schedule change
        sigma = policy.update(k, k % 2 ? residuals(1e-3, 1e-1) : residuals(1e-1, 1e-3), sigma, false);
    const double before = sigma;
    int monitor_changes = 0;
    for (int k = 5001; k <= 5000 + 25 * 21; k++)
    {
        KktResiduals r;
        r.eta_p = 1e-4;
        r.eta_d = 1e-4;
        r.eta_g = 1e-2; // ratio 100 > 1.2
        r.pobj = 10.0 + 0.01 * k;
        r.dobj = 5.0 + 0.01 * k; // both ascend, pobj > dobj: "help dual"
        const double s0 = sigma;
        sigma = policy.update(k, r, sigma, false);
        if (sigma != s0 && !policy.log().empty() && policy.log().back().rule == "monitor1")
            monitor_changes++;
    }
    EXPECT_GE(monitor_changes, 1);
    EXPECT_EQ(sigma, std::max(100.0, before));
    // the state round trip keeps the monitor's queues
    const PolicyState st = policy.state();
    LegacyPenalty copy(o);
    copy.reset(1.0);
    copy.restore(st);
    EXPECT_EQ(copy.state().monitor_pobj, st.monitor_pobj);
    EXPECT_EQ(copy.state().sigma, st.sigma);
}

TEST(PenaltyPolicy, BalanceFreezeFreezesAtTheIteration)
{
    PenaltyOptions o;
    o.policy = PenaltyPolicyKind::BalanceFreeze;
    o.freeze_iteration = 300;
    ResidualBalancingPenalty policy(o, true);
    policy.reset(1.0);
    double sigma = drive(policy, 1.0, 1, 299, true); // primal always wins: sigma doubles at every check
    EXPECT_GT(sigma, 1.0);
    EXPECT_FALSE(policy.frozen());
    sigma = policy.update(300, residuals(1e-3, 1e-1), sigma, false);
    EXPECT_TRUE(policy.frozen());
    EXPECT_EQ(policy.freeze_iteration(), 300);
    EXPECT_EQ(policy.freeze_reason(), "freeze_iteration");
    const double frozen = sigma;
    EXPECT_EQ(policy.log().back().rule, "freeze_freeze_iteration");
    // drifting (+ + + ...) never looks like ping-pong: the frozen value is the last one (no band)
    EXPECT_EQ(drive(policy, sigma, 301, 5000, true), frozen);
    EXPECT_EQ(drive(policy, sigma, 5001, 6000, false), frozen);
    // sigma_0 is clamped into the bounds
    policy.reset(1e5);
    EXPECT_EQ(policy.state().sigma, 1e3);
    policy.reset(1e-5);
    EXPECT_EQ(policy.state().sigma, 1e-3);
    ResidualBalancingPenalty plain(o, false);
    plain.reset(1e5);
    EXPECT_EQ(plain.state().sigma, 1e5); // residual_balancing: not clamped (2d8a755)
}

TEST(PenaltyPolicy, BalanceFreezeDetectsPingPongByNetMovement)
{
    PenaltyOptions o;
    o.policy = PenaltyPolicyKind::BalanceFreeze;
    o.freeze_iteration = 100000;
    o.pingpong_min_iter = 1000;
    ResidualBalancingPenalty policy(o, true);
    policy.reset(8.0);
    double sigma = 8.0;
    // the trto1 pattern + + - - repeated: two checks up, two down; one check per period (10, 25, 50, ...)
    const int checks[] = {1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 226, 251, 276,
                          301, 326, 351, 376, 401, 426, 451, 476, 501, 526, 551, 576, 601, 626, 651, 676, 701, 726, 751, 776, 801, 826,
                          851, 876, 901, 926, 951, 976, 1001, 1051, 1101, 1151, 1201, 1251, 1301, 1351, 1401};
    std::size_t c = 0;
    int phase = 0; // 0,1: primal wins (up); 2,3: dual wins (down)
    for (int k = 1; k <= 1500 && !policy.frozen(); k++)
    {
        const bool primal = (phase % 4) < 2;
        // keep the winner's margin above the guard: feed the same vote until the next check, then switch phase
        sigma = policy.update(k, primal ? residuals(1e-3, 1e-1) : residuals(1e-1, 1e-3), sigma, false);
        if (c < sizeof(checks) / sizeof(checks[0]) && k == checks[c])
        {
            c++;
            phase++;
        }
    }
    ASSERT_TRUE(policy.frozen());
    EXPECT_EQ(policy.freeze_reason(), "ping_pong");
    EXPECT_GE(policy.freeze_iteration(), 1000);
    // the frozen value is the geometric mean of the last W sigmas (band centre), within the bounds
    const PolicyState st = policy.state();
    double log_sum = 0.0;
    for (std::size_t i = st.sigmas.size() - o.pingpong_window; i < st.sigmas.size(); i++)
        log_sum += std::log(st.sigmas[i]);
    EXPECT_NEAR(st.sigma, std::exp(log_sum / o.pingpong_window), 1e-12 * st.sigma);
    // with the bias the frozen value is multiplied
    PenaltyOptions ob = o;
    ob.freeze_bias = 4.0;
    ResidualBalancingPenalty biased(ob, true);
    biased.reset(8.0);
    double sb = 8.0;
    c = 0;
    phase = 0;
    for (int k = 1; k <= 1500 && !biased.frozen(); k++)
    {
        const bool primal = (phase % 4) < 2;
        sb = biased.update(k, primal ? residuals(1e-3, 1e-1) : residuals(1e-1, 1e-3), sb, false);
        if (c < sizeof(checks) / sizeof(checks[0]) && k == checks[c])
        {
            c++;
            phase++;
        }
    }
    ASSERT_TRUE(biased.frozen());
    EXPECT_NEAR(biased.state().sigma, std::min(4.0 * st.sigma, 1e3), 1e-12 * biased.state().sigma);
    // the frozen policy ignores every later vote, and survives a state round trip
    const double frozen = sigma;
    EXPECT_EQ(drive(policy, sigma, policy.freeze_iteration() + 1, policy.freeze_iteration() + 3000, true), frozen);
    ResidualBalancingPenalty copy(o, true);
    copy.reset(8.0);
    copy.restore(st);
    EXPECT_TRUE(copy.frozen());
    EXPECT_EQ(copy.state().sigma, st.sigma);
    EXPECT_EQ(copy.state().dirs, st.dirs);
}

TEST(PenaltyPolicy, BalanceFreezeHalveOnReversalShrinksTheStep)
{
    PenaltyOptions o;
    o.policy = PenaltyPolicyKind::BalanceFreeze;
    o.step_schedule = StepSchedule::HalveOnReversal;
    o.freeze_iteration = 100000;
    o.pingpong_window = 100; // keep the ping-pong test out of the way
    ResidualBalancingPenalty policy(o, true);
    policy.reset(1.0);
    double sigma = policy.update(1, residuals(1e-3, 1e-1), 1.0, false); // no change at k = 1 (guard)
    EXPECT_EQ(sigma, 1.0);
    sigma = drive(policy, sigma, 2, 11, true); // check at 11: up by 2
    EXPECT_EQ(sigma, 2.0);
    sigma = drive(policy, sigma, 12, 21, false); // check at 21: reversal, step sqrt(2)
    EXPECT_NEAR(sigma, 2.0 / std::sqrt(2.0), 1e-15);
    sigma = drive(policy, sigma, 22, 31, false); // same direction: step stays sqrt(2)
    EXPECT_NEAR(sigma, 2.0 / 2.0, 1e-15);
    EXPECT_NEAR(policy.state().step, std::sqrt(2.0), 1e-15);
    // keep reversing: 2 -> 2^(1/2) -> 2^(1/4) -> 2^(1/8) -> 2^(1/16) -> freeze ("step") when log2(step) < 1/16
    int reversals = 0;
    bool up = true;
    int k = 32;
    while (!policy.frozen() && k < 5000)
    {
        const int next_check = k <= 200 ? ((k / 10) * 10 + 11) : ((k / 25) * 25 + 26);
        sigma = drive(policy, sigma, k, next_check, up);
        k = next_check + 1;
        up = !up;
        reversals++;
    }
    EXPECT_TRUE(policy.frozen());
    EXPECT_EQ(policy.freeze_reason(), "step");
    EXPECT_LE(reversals, 8);
}

TEST(PenaltyPolicy, BalanceFreezeEqualsResidualBalancingBeforeTheFreeze)
{
    PenaltyOptions o;
    o.policy = PenaltyPolicyKind::BalanceFreeze;
    o.freeze_iteration = 100000;
    o.pingpong_window = 1000; // no ping-pong freeze within the replay
    o.sigma_min = 1e-3;
    o.sigma_max = 1e3;
    ResidualBalancingPenalty freeze(o, true), plain(o, false);
    const std::vector<Row> rows = read_history(std::string(CUADMM_TEST_DATA_DIR) + "/golden/PSD_Free_residual_balancing_s100_i6000_history.csv");
    ASSERT_FALSE(rows.empty());
    freeze.reset(100.0);
    plain.reset(100.0);
    double sf = 100.0, sp = 100.0;
    for (const Row &row : rows)
    {
        KktResiduals r;
        r.eta_p = row.eta_p;
        r.eta_d = row.eta_d;
        r.eta_g = row.eta_g;
        r.pobj = row.pobj;
        r.dobj = row.dobj;
        sf = freeze.update(row.iter, r, sf, false);
        sp = plain.update(row.iter, r, sp, false);
        ASSERT_EQ(sf, sp) << "iteration " << row.iter;
    }
    EXPECT_EQ(freeze.log().size(), plain.log().size());
    EXPECT_FALSE(freeze.frozen());
}

// The cross-check with the Python replica: case,iter,sigma,log rows per case "<sigma0>@balance_freeze[:key=value,...]".
inline void cross_check_with_replica(const std::string &history_name, const std::string &cases_name, std::size_t expected_rows)
{
    const std::vector<Row> rows = read_history(std::string(CUADMM_TEST_DATA_DIR) + "/golden/" + history_name + "_history.csv");
    ASSERT_EQ(rows.size(), expected_rows) << history_name;
    std::ifstream in(std::string(CUADMM_TEST_DATA_DIR) + "/golden/" + cases_name);
    ASSERT_TRUE(in.good()) << cases_name;
    std::string line;
    std::getline(in, line);
    ASSERT_EQ(line, "case,iter,sigma,log") << cases_name;
    struct Expected
    {
        int iter;
        double sigma;
        std::string log;
    };
    std::vector<std::pair<std::string, std::vector<Expected>>> cases;
    while (std::getline(in, line))
    {
        // case,iter,sigma,log: the case is quoted when its parameter list holds commas; the log field holds
        // ';'-joined names, never commas
        std::string name;
        std::size_t c1;
        if (!line.empty() && line[0] == '"')
        {
            const std::size_t q = line.find('"', 1);
            ASSERT_NE(q, std::string::npos) << line;
            name = line.substr(1, q - 1);
            c1 = q + 1;
            ASSERT_EQ(line[c1], ',') << line;
        }
        else
        {
            c1 = line.find(',');
            name = line.substr(0, c1);
        }
        const std::size_t c2 = line.find(',', c1 + 1), c3 = line.find(',', c2 + 1);
        ASSERT_NE(c3, std::string::npos) << line;
        if (cases.empty() || cases.back().first != name)
            cases.emplace_back(name, std::vector<Expected>());
        cases.back().second.push_back({std::stoi(line.substr(c1 + 1, c2 - c1 - 1)), std::strtod(line.substr(c2 + 1, c3 - c2 - 1).c_str(), nullptr),
                                       line.substr(c3 + 1)});
    }
    ASSERT_FALSE(cases.empty()) << cases_name;
    for (const auto &c : cases)
    {
        const std::string &name = c.first;
        ASSERT_EQ(c.second.size(), rows.size()) << name;
        const std::size_t at = name.find('@');
        ASSERT_NE(at, std::string::npos) << name;
        const double sigma0 = std::strtod(name.substr(0, at).c_str(), nullptr);
        std::string spec = name.substr(at + 1);
        ASSERT_EQ(spec.rfind("balance_freeze", 0), 0u) << name;
        SolverOptions so;
        so.penalty.policy = PenaltyPolicyKind::BalanceFreeze;
        if (spec.size() > 15 && spec[14] == ':')
        {
            std::stringstream ss(spec.substr(15));
            std::string item;
            while (std::getline(ss, item, ','))
            {
                const std::size_t eq = item.find('=');
                ASSERT_NE(eq, std::string::npos) << name;
                set_option(so, "penalty." + item.substr(0, eq), item.substr(eq + 1));
            }
        }
        std::unique_ptr<PenaltyPolicy> policy = make_penalty_policy(so.penalty);
        policy->reset(sigma0);
        double sigma = policy->state().sigma; // sigma_0 clamped into the bounds
        int mismatches = 0;
        std::size_t logged = 0;
        for (std::size_t i = 0; i < rows.size(); i++)
        {
            const Row &row = rows[i];
            const Expected &e = c.second[i];
            ASSERT_EQ(row.iter, e.iter) << name;
            KktResiduals r;
            r.eta_p = row.eta_p;
            r.eta_d = row.eta_d;
            r.eta_g = row.eta_g;
            r.pobj = row.pobj;
            r.dobj = row.dobj;
            sigma = policy->update(row.iter, r, sigma, false);
            std::string names;
            for (; logged < policy->log().size(); logged++)
                names += (names.empty() ? "" : ";") + policy->log()[logged].rule;
            if (sigma != e.sigma || names != e.log)
            {
                if (mismatches++ == 0)
                    ADD_FAILURE() << name << ": first mismatch at iteration " << row.iter << ": policy sigma " << sigma << " log '" << names
                                  << "', replica sigma " << e.sigma << " log '" << e.log << "'";
            }
        }
        EXPECT_EQ(mismatches, 0) << name;
        printf("[ REPLICA  ] %s: %zu iterations, %zu log entries, frozen %s at %d (%s), final sigma %.17g\n", name.c_str(), rows.size(),
               policy->log().size(), policy->frozen() ? "true" : "false", policy->freeze_iteration(), policy->freeze_reason().c_str(), sigma);
    }
}

TEST(PenaltyPolicy, BalanceFreezeAgreesWithTheReplicaOnPsdFree)
{
    cross_check_with_replica("PSD_Free_residual_balancing_s100_i6000", "balance_freeze_cases_PSD_Free.csv", 6000);
}

TEST(PenaltyPolicy, BalanceFreezeAgreesWithTheReplicaOnMater1)
{
    cross_check_with_replica("mater-1_legacy_adaptive_s1_tol1e-4", "balance_freeze_cases_mater-1.csv", 8117);
}

TEST(PenaltyPolicy, FactoryAndParameters)
{
    PenaltyOptions o;
    for (PenaltyPolicyKind kind : {PenaltyPolicyKind::Fixed, PenaltyPolicyKind::LegacyAdaptive, PenaltyPolicyKind::ResidualBalancing,
                                   PenaltyPolicyKind::BalanceFreeze})
    {
        o.policy = kind;
        std::unique_ptr<PenaltyPolicy> p = make_penalty_policy(o);
        EXPECT_EQ(p->name(), name_of(kind));
        p->reset(1.0);
        EXPECT_NE(p->parameters_json().find("\"sigma_min\": "), std::string::npos);
    }
    o.policy = PenaltyPolicyKind::ResidualBalancing;
    o.factor = 1.2;
    std::unique_ptr<PenaltyPolicy> p = make_penalty_policy(o);
    EXPECT_NE(p->parameters_json().find("\"factor\": 1.2"), std::string::npos);
}
