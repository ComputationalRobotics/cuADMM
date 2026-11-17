/*

    penalty_policy_test.hpp  (CPU only)

    The penalty policies on scripted and recorded residual sequences (no GPU):
      - FixedPenalty never changes and refuses a changed sigma;
      - balance_freeze: the defaults (sigma_0 100 in [1e-3, 1e3], factor 1.35 at every update, the freeze at
        iteration 10000), the freeze at freeze_iteration and no change after it, the parameter checks;
      - balance_freeze reproduces, bit for bit, the sigma of every iteration and every sigma-log entry recorded with
        cuADMM 2.0.1 (main 8fd203c) on two recorded residual sequences (PSD_Free, 6000 iterations; mater-1, 8117
        iterations) for the default parameters and for other values of sigma_0, the bounds, the factor and the
        freeze iteration (test/data/golden/balance_freeze_changes_*.csv: the first iteration and every iteration
        with a log entry of each case; residuals_*.csv: the residual sequences, 17 significant digits).

*/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
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
    double pobj, dobj, eta_p, eta_d, eta_g;
};

// iter,pobj,dobj,eta_p,eta_d,eta_g
inline std::vector<Row> read_residuals(const std::string &file)
{
    std::ifstream in(file);
    std::vector<Row> rows;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ','))
            cols.push_back(tok);
        if (cols.size() < 6)
            continue;
        rows.push_back({std::stoi(cols[0]), std::stod(cols[1]), std::stod(cols[2]), std::stod(cols[3]), std::stod(cols[4]), std::stod(cols[5])});
    }
    return rows;
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
        sigma = policy.update(k, primal_wins ? residuals(1e-3, 1e-1) : residuals(1e-1, 1e-3), sigma);
    return sigma;
}

// The recorded trajectories: every case "<sigma0>@balance_freeze[:key=value]" of the change file, replayed on the
// residual sequence; the sigma of every iteration and the rule names logged at every iteration must match exactly.
inline void check_recorded_trajectories(const std::string &name, std::size_t expected_rows, std::size_t expected_cases)
{
    const std::string dir = std::string(CUADMM_TEST_DATA_DIR) + "/golden/";
    const std::vector<Row> rows = read_residuals(dir + "residuals_" + name + ".csv");
    ASSERT_EQ(rows.size(), expected_rows) << name;
    std::ifstream in(dir + "balance_freeze_changes_" + name + ".csv");
    ASSERT_TRUE(in.good()) << name;
    std::string line;
    std::getline(in, line);
    ASSERT_EQ(line, "case,iter,sigma,log");
    struct Entry
    {
        int iter;
        double sigma;
        std::string log;
    };
    std::vector<std::pair<std::string, std::vector<Entry>>> cases;
    while (std::getline(in, line))
    {
        const std::size_t c1 = line.find(','), c2 = line.find(',', c1 + 1), c3 = line.find(',', c2 + 1);
        ASSERT_NE(c3, std::string::npos) << line;
        const std::string case_name = line.substr(0, c1);
        if (cases.empty() || cases.back().first != case_name)
            cases.emplace_back(case_name, std::vector<Entry>());
        cases.back().second.push_back({std::stoi(line.substr(c1 + 1, c2 - c1 - 1)), std::strtod(line.substr(c2 + 1, c3 - c2 - 1).c_str(), nullptr),
                                       line.substr(c3 + 1)});
    }
    ASSERT_EQ(cases.size(), expected_cases) << name;
    for (const auto &c : cases)
    {
        const std::string &case_name = c.first;
        const std::vector<Entry> &entries = c.second;
        ASSERT_FALSE(entries.empty());
        ASSERT_EQ(entries.front().iter, rows.front().iter) << case_name << ": the first iteration is recorded";
        const std::size_t at = case_name.find('@');
        ASSERT_NE(at, std::string::npos) << case_name;
        const double sigma0 = std::strtod(case_name.substr(0, at).c_str(), nullptr);
        const std::string spec = case_name.substr(at + 1);
        ASSERT_EQ(spec.rfind("balance_freeze", 0), 0u) << case_name;
        SolverOptions so;
        if (spec.size() > 14)
        {
            ASSERT_EQ(spec[14], ':') << case_name;
            const std::string kv = spec.substr(15);
            const std::size_t eq = kv.find('=');
            ASSERT_NE(eq, std::string::npos) << case_name;
            set_option(so, "penalty." + kv.substr(0, eq), kv.substr(eq + 1));
        }
        std::unique_ptr<PenaltyPolicy> policy = make_penalty_policy(so.penalty);
        policy->reset(sigma0);
        double sigma = std::min(std::max(sigma0, policy->sigma_min()), policy->sigma_max()); // sigma_0 clamped into the bounds
        std::size_t next = 0, logged = 0;
        double expected = 0.0;
        int mismatches = 0;
        for (const Row &row : rows)
        {
            KktResiduals r;
            r.eta_p = row.eta_p;
            r.eta_d = row.eta_d;
            r.eta_g = row.eta_g;
            r.pobj = row.pobj;
            r.dobj = row.dobj;
            sigma = policy->update(row.iter, r, sigma);
            std::string names;
            for (; logged < policy->log().size(); logged++)
                names += (names.empty() ? "" : ";") + policy->log()[logged].rule;
            std::string expected_log;
            if (next < entries.size() && entries[next].iter == row.iter)
            {
                expected = entries[next].sigma;
                expected_log = entries[next].log;
                next++;
            }
            if (sigma != expected || names != expected_log)
            {
                if (mismatches++ == 0)
                    ADD_FAILURE() << case_name << ": first mismatch at iteration " << row.iter << ": sigma " << sigma << " log '" << names
                                  << "', recorded sigma " << expected << " log '" << expected_log << "'";
            }
        }
        EXPECT_EQ(next, entries.size()) << case_name << ": recorded entries not reached";
        EXPECT_EQ(mismatches, 0) << case_name;
        printf("[ GOLDEN   ] %s %s: %zu iterations, %zu log entries, frozen %s at %d, final sigma %.17g\n", name.c_str(), case_name.c_str(),
               rows.size(), policy->log().size(), policy->frozen() ? "true" : "false", policy->freeze_iteration(), sigma);
    }
}
} // namespace penalty_test

using namespace penalty_test;

TEST(PenaltyPolicy, FixedNeverChanges)
{
    FixedPenalty policy;
    policy.reset(3.0);
    double sigma = 3.0;
    for (int k = 1; k <= 20000; k++)
        sigma = policy.update(k, k % 2 ? residuals(1e-3, 1e-1) : residuals(1e-1, 1e-3), sigma);
    EXPECT_EQ(sigma, 3.0);
    EXPECT_TRUE(policy.log().empty());
    EXPECT_THROW(policy.update(20001, residuals(1e-3, 1e-1), 4.0), std::logic_error);
}

TEST(PenaltyPolicy, BalanceFreezeFreezesAtTheIteration)
{
    PenaltyOptions o;
    o.freeze_iteration = 300;
    BalanceFreezePenalty policy(o);
    policy.reset(1.0);
    // the guard: a single win at iteration 1 does not change sigma
    EXPECT_EQ(policy.update(1, residuals(1e-3, 1e-1), 1.0), 1.0);
    double sigma = drive(policy, 1.0, 2, 299, true); // primal always wins: sigma grows by the factor at every check
    EXPECT_GT(sigma, 1.0);
    EXPECT_FALSE(policy.frozen());
    sigma = policy.update(300, residuals(1e-3, 1e-1), sigma);
    EXPECT_TRUE(policy.frozen());
    EXPECT_EQ(policy.freeze_iteration(), 300);
    EXPECT_EQ(policy.freeze_reason(), "freeze_iteration");
    EXPECT_EQ(policy.log().back().rule, "freeze_freeze_iteration");
    const double frozen = sigma;
    EXPECT_EQ(drive(policy, sigma, 301, 5000, true), frozen);
    EXPECT_EQ(drive(policy, sigma, 5001, 6000, false), frozen);
    // the bounds hold for every update
    o.freeze_iteration = 100000;
    BalanceFreezePenalty up(o);
    up.reset(100.0);
    EXPECT_EQ(drive(up, 100.0, 1, 20000, true), 1e3);
    BalanceFreezePenalty down(o);
    down.reset(100.0);
    EXPECT_EQ(drive(down, 100.0, 1, 20000, false), 1e-3);
}

TEST(PenaltyPolicy, InvalidParametersAreRejected)
{
    PenaltyOptions o;
    o.factor = 1.0;
    EXPECT_THROW(BalanceFreezePenalty p(o), std::invalid_argument);
    o = PenaltyOptions();
    o.sigma_min = 10.0;
    o.sigma_max = 1.0;
    EXPECT_THROW(BalanceFreezePenalty p(o), std::invalid_argument);
    o = PenaltyOptions();
    o.freeze_iteration = -1;
    EXPECT_THROW(BalanceFreezePenalty p(o), std::invalid_argument);
}

TEST(PenaltyPolicy, FactoryAndParameters)
{
    PenaltyOptions o;
    o.policy = PenaltyPolicyKind::Fixed;
    std::unique_ptr<PenaltyPolicy> fixed = make_penalty_policy(o);
    EXPECT_EQ(fixed->name(), "fixed");
    fixed->reset(2.0);
    EXPECT_EQ(fixed->parameters_json(), "{\"sigma\": 2}");
    o.policy = PenaltyPolicyKind::BalanceFreeze;
    o.factor = 1.2;
    std::unique_ptr<PenaltyPolicy> p = make_penalty_policy(o);
    EXPECT_EQ(p->name(), "balance_freeze");
    EXPECT_NE(p->parameters_json().find("\"factor\": 1.2"), std::string::npos) << p->parameters_json();
}

// The defaults: balance_freeze from sigma_0 = 100 in [1e-3, 1e3], factor 1.35 at every update, the freeze after
// iteration 10000 at the current value
TEST(PenaltyPolicy, DefaultsAreBalanceFreezeWithFactor135AndTheFreezeAt10000)
{
    const SolverOptions so;
    const PenaltyOptions &o = so.penalty;
    EXPECT_EQ(o.policy, PenaltyPolicyKind::BalanceFreeze);
    EXPECT_EQ(o.sigma0, 100.0);
    EXPECT_EQ(o.sigma_min, 1e-3);
    EXPECT_EQ(o.sigma_max, 1e3);
    EXPECT_EQ(o.factor, 1.35);
    EXPECT_EQ(o.freeze_iteration, 10000);
    std::unique_ptr<PenaltyPolicy> p = make_penalty_policy(o);
    EXPECT_EQ(p->name(), "balance_freeze");
    EXPECT_EQ(p->sigma_min(), 1e-3);
    EXPECT_EQ(p->sigma_max(), 1e3);
    const std::string json = p->parameters_json();
    for (const char *expected : {"\"factor\": 1.35", "\"freeze_iteration\": 10000", "\"win_ratio\": 1.35",
                                 "\"schedule\": \"200:10,1000:25,5000:50,10000:200,inf:1000\""})
        EXPECT_NE(json.find(expected), std::string::npos) << expected << " in " << json;
}

// Every update multiplies or divides sigma by exactly 1.35, reversals included, and only at the check iterations;
// sigma is not frozen before iteration 10000; after iteration 10000 it keeps its current value
TEST(PenaltyPolicy, DefaultBalanceFreezeUsesAConstantFactorAndFreezesOnlyAt10000)
{
    const SolverOptions so;
    BalanceFreezePenalty policy(so.penalty);
    policy.reset(100.0);
    double sigma = 100.0;
    bool up = true;
    std::size_t seen = 0;
    int reversals = 0;
    for (int k = 1; k <= 9999; k++)
    {
        sigma = policy.update(k, up ? residuals(1e-3, 1e-1) : residuals(1e-1, 1e-3), sigma);
        ASSERT_FALSE(policy.frozen()) << "frozen at iteration " << k;
        while (seen < policy.log().size())
        {
            const SigmaChange &c = policy.log()[seen++];
            ASSERT_EQ(c.rule, "admm_schedule") << "iteration " << c.iteration;
            ASSERT_EQ(c.new_sigma, up ? c.old_sigma * 1.35 : c.old_sigma / 1.35) << "iteration " << c.iteration;
            ASSERT_TRUE(BalanceFreezePenalty::is_check(c.iteration)) << c.iteration;
            up = !up; // reverse the vote after every change
            reversals++;
        }
    }
    EXPECT_GT(reversals, 30);
    EXPECT_GE(policy.log().back().iteration, 9000); // the changes went on until the last checks before the freeze
    const double before = sigma;
    sigma = policy.update(10000, up ? residuals(1e-3, 1e-1) : residuals(1e-1, 1e-3), sigma);
    ASSERT_TRUE(policy.frozen());
    EXPECT_EQ(policy.freeze_iteration(), 10000);
    EXPECT_EQ(policy.freeze_reason(), "freeze_iteration");
    EXPECT_EQ(sigma, before);
    EXPECT_EQ(policy.log().back().rule, "freeze_freeze_iteration");
    EXPECT_EQ(policy.log().back().old_sigma, before);
    EXPECT_EQ(policy.log().back().new_sigma, before);
    const std::size_t changes = policy.log().size();
    EXPECT_EQ(drive(policy, sigma, 10001, 20000, true), before);
    EXPECT_EQ(drive(policy, sigma, 20001, 30000, false), before);
    EXPECT_EQ(policy.log().size(), changes);
}

TEST(PenaltyPolicy, BalanceFreezeReproducesTheRecordedTrajectoriesOnPsdFree)
{
    check_recorded_trajectories("PSD_Free", 6000, 10);
}

TEST(PenaltyPolicy, BalanceFreezeReproducesTheRecordedTrajectoriesOnMater1)
{
    check_recorded_trajectories("mater-1", 8117, 10);
}
