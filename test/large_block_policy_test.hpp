/*

    large_block_policy_test.hpp

    Tests of LargeBlockPolicy (plan Section 10.6) on synthetic schedules: the per-block warm-up threshold, the
    one-way switch k_sw by residual or by iteration, the low-rank start and the reevaluation cadence per block,
    the modes, the legacy (C++ API) start, and the argument checks. Host logic only (cuadmm_cpu_tests).

*/

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "cone/psd/large/large_block_policy.h"

namespace
{
using cuadmm::cone::psd::kModeFull;
using cuadmm::cone::psd::kModeLowRankNegative;
using cuadmm::cone::psd::kModeLowRankPositive;
using cuadmm::cone::psd::LargeBlockPolicy;
using cuadmm::cone::psd::LargeBlockPolicyConfig;
using cuadmm::cone::psd::LargeStrategy;

constexpr double kInf = std::numeric_limits<double>::infinity();
} // namespace

TEST(LargeBlockPolicy, NoWarmupStartsTheAnalysisAtStartIterAndRepeatsIt)
{
    LargeBlockPolicyConfig c; // the defaults of the options: no warm-up, start 2, reevaluate 100
    LargeBlockPolicy p(c, {1100});
    EXPECT_FALSE(p.warm(0));
    EXPECT_EQ(p.start(0), 2);

    p.begin_iteration(1, kInf);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvd);
    EXPECT_EQ(p.decide(0, kModeLowRankPositive), LargeStrategy::FullEvd) << "no low-rank mode before an analysis is due";
    p.begin_iteration(2, kInf);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvdWithRankAnalysis);
    p.begin_iteration(3, kInf);
    EXPECT_EQ(p.decide(0, kModeLowRankPositive), LargeStrategy::Lobpcg);
    EXPECT_EQ(p.decide(0, kModeLowRankNegative), LargeStrategy::Lobpcg);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvd);
    p.begin_iteration(101, kInf);
    EXPECT_EQ(p.decide(0, kModeLowRankPositive), LargeStrategy::Lobpcg);
    p.begin_iteration(102, kInf);
    EXPECT_EQ(p.decide(0, kModeLowRankPositive), LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvdWithRankAnalysis);
    // a residual below the switch tolerance changes nothing without a warm-up or the legacy start
    p.begin_iteration(103, 1e-9);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvd);
    EXPECT_EQ(p.start(0), 2);
}

TEST(LargeBlockPolicy, WarmupAppliesPerBlockAboveMinSize)
{
    LargeBlockPolicyConfig c;
    c.warmup = true;
    c.warmup_min_size = 5000;
    c.switch_tol = 1e-2;
    c.switch_iter = 5000;
    LargeBlockPolicy p(c, {6000, 1200, 5000});
    EXPECT_TRUE(p.warm(0));
    EXPECT_FALSE(p.warm(1));
    EXPECT_TRUE(p.warm(2)) << "min_size is inclusive";
    EXPECT_TRUE(p.any_warm());
    EXPECT_EQ(p.start(0), -1) << "unknown until the switch";
    EXPECT_EQ(p.start(1), 2);

    p.begin_iteration(1, kInf);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::CompositeFilter);
    EXPECT_EQ(p.decide(1, kModeFull), LargeStrategy::FullEvd);
    EXPECT_FALSE(p.begin_iteration(2, 1.0));
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::CompositeFilter);
    EXPECT_EQ(p.decide(1, kModeFull), LargeStrategy::FullEvdWithRankAnalysis) << "the small block starts its analysis at 2";
    EXPECT_FALSE(p.begin_iteration(3, 5e-2));
    EXPECT_EQ(p.decide(1, kModeLowRankPositive), LargeStrategy::Lobpcg);
    // the switch: eta below the tolerance at iteration 7
    EXPECT_TRUE(p.begin_iteration(7, 5e-3));
    EXPECT_TRUE(p.switched());
    EXPECT_EQ(p.switch_iteration(), 7);
    EXPECT_EQ(p.start(0), 7);
    EXPECT_EQ(p.start(2), 7);
    EXPECT_EQ(p.start(1), 2);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvdWithRankAnalysis) << "the warm-up ends with an analysis";
    EXPECT_EQ(p.decide(2, kModeFull), LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(p.decide(1, kModeLowRankPositive), LargeStrategy::Lobpcg);
    // the cadences differ per block afterwards
    EXPECT_FALSE(p.begin_iteration(102, 1e-3));
    EXPECT_EQ(p.decide(0, kModeLowRankPositive), LargeStrategy::Lobpcg);
    EXPECT_EQ(p.decide(1, kModeLowRankPositive), LargeStrategy::FullEvdWithRankAnalysis);
    p.begin_iteration(107, 1e-3);
    EXPECT_EQ(p.decide(0, kModeLowRankPositive), LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(p.decide(1, kModeLowRankPositive), LargeStrategy::Lobpcg);
    p.begin_iteration(108, 1e-3);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvd);
}

TEST(LargeBlockPolicy, SwitchByIterationIsOneWayAndNeverAtIterationOne)
{
    LargeBlockPolicyConfig c;
    c.warmup = true;
    c.warmup_min_size = 1;
    c.switch_tol = 1e-2;
    c.switch_iter = 10;
    LargeBlockPolicy p(c, {2000});
    EXPECT_FALSE(p.begin_iteration(1, 0.0)) << "never at iteration 1, whatever the residual";
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::CompositeFilter);
    for (int k = 2; k <= 10; k++)
        EXPECT_FALSE(p.begin_iteration(k, 1.0)) << k;
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::CompositeFilter);
    EXPECT_TRUE(p.begin_iteration(11, 1.0)) << "k > switch_iter";
    EXPECT_EQ(p.switch_iteration(), 11);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_FALSE(p.begin_iteration(12, 1e3)) << "one-way: a large residual does not restart the warm-up";
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvd);
    EXPECT_EQ(p.switch_iteration(), 11);
}

TEST(LargeBlockPolicy, LowrankDisabledKeepsTheExactProjection)
{
    LargeBlockPolicyConfig c;
    c.lowrank_enabled = false;
    c.warmup = true;
    c.warmup_min_size = 1000;
    LargeBlockPolicy p(c, {1100, 900});
    p.begin_iteration(1, kInf);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::CompositeFilter);
    EXPECT_EQ(p.decide(1, kModeFull), LargeStrategy::FullEvd);
    p.begin_iteration(2, 1e-9); // the switch
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvd);
    EXPECT_EQ(p.decide(0, kModeLowRankPositive), LargeStrategy::FullEvd) << "a stale mode is ignored without the low-rank path";
    p.begin_iteration(102, 1e-9);
    EXPECT_EQ(p.decide(1, kModeFull), LargeStrategy::FullEvd);
}

TEST(LargeBlockPolicy, LegacyGlobalStartFollowsTheSwitchForEveryBlock)
{
    // the C++ API of main: no warm-up (EIG_FP64), switch_proj_max_iter 5000, switch_proj_tol 1e-2
    LargeBlockPolicyConfig c;
    c.warmup = false;
    c.switch_tol = 1e-2;
    c.switch_iter = 5000;
    c.legacy_global_start = true;
    LargeBlockPolicy p(c, {1100});
    p.begin_iteration(2, 1.0);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvd) << "no analysis before the switch";
    EXPECT_EQ(p.start(0), -1);
    EXPECT_TRUE(p.begin_iteration(5, 1e-3));
    EXPECT_EQ(p.start(0), 5);
    EXPECT_EQ(p.decide(0, kModeFull), LargeStrategy::FullEvdWithRankAnalysis);
    p.begin_iteration(105, 1e-3);
    EXPECT_EQ(p.decide(0, kModeLowRankNegative), LargeStrategy::FullEvdWithRankAnalysis);
    p.begin_iteration(106, 1e-3);
    EXPECT_EQ(p.decide(0, kModeLowRankNegative), LargeStrategy::Lobpcg);

    // today's CLI (switch_proj_max_iter 0): the switch at iteration 2, as the per-block default
    LargeBlockPolicyConfig cli = c;
    cli.switch_iter = 0;
    LargeBlockPolicy q(cli, {1100});
    EXPECT_FALSE(q.begin_iteration(1, 1.0));
    EXPECT_TRUE(q.begin_iteration(2, 1.0));
    EXPECT_EQ(q.decide(0, kModeFull), LargeStrategy::FullEvdWithRankAnalysis);
    q.begin_iteration(102, 1.0);
    EXPECT_EQ(q.decide(0, kModeFull), LargeStrategy::FullEvdWithRankAnalysis);
}

TEST(LargeBlockPolicy, ArgumentChecks)
{
    LargeBlockPolicyConfig c;
    c.warmup_min_size = 0;
    EXPECT_THROW(LargeBlockPolicy(c, {10}), std::invalid_argument);
    c = LargeBlockPolicyConfig{};
    c.lowrank_start_iter = 1;
    EXPECT_THROW(LargeBlockPolicy(c, {10}), std::invalid_argument);
    c = LargeBlockPolicyConfig{};
    c.lowrank_reevaluate = 0;
    EXPECT_THROW(LargeBlockPolicy(c, {10}), std::invalid_argument);
    c = LargeBlockPolicyConfig{};
    LargeBlockPolicy p(c, {10, 20});
    EXPECT_THROW(p.decide(0, kModeFull), std::logic_error) << "before begin_iteration";
    EXPECT_THROW(p.begin_iteration(0, 1.0), std::invalid_argument);
    p.begin_iteration(1, 1.0);
    EXPECT_THROW(p.decide(2, kModeFull), std::out_of_range);
    EXPECT_EQ(p.block_count(), 2);
    EXPECT_STREQ(cuadmm::cone::psd::large_strategy_name(LargeStrategy::CompositeFilter), "CompositeFilter");
}
