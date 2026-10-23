/*

    cone/psd/large/large_block_policy.cc

    The selection policy of plan Section 10.6 (see the header). Equivalence with the engine's schedule of
    Milestone 1 (one global flag switched_proj_method): without a warm-up the facade mapped the low-rank start to
    the switch (switch_iter = start_iter - 1, switch_tol = 0), so k_sw = start_iter and the per-block rule gives
    the same iterations; with the C++ API's warm-up on every block, warm(b) holds for every block and start(b) =
    k_sw. The baselines never use the warm-up.

*/

#include "cone/psd/large/large_block_policy.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace cuadmm
{
namespace cone
{
namespace psd
{

const char *large_strategy_name(LargeStrategy s)
{
    switch (s)
    {
    case LargeStrategy::FullEvd:
        return "FullEvd";
    case LargeStrategy::FullEvdWithRankAnalysis:
        return "FullEvd+RankAnalyzer";
    case LargeStrategy::Lobpcg:
        return "Lobpcg";
    case LargeStrategy::CompositeFilter:
        return "CompositeFilter";
    }
    return "?";
}

LargeBlockPolicy::LargeBlockPolicy(const LargeBlockPolicyConfig &config, std::vector<int> block_sizes)
    : config_(config), sizes_(std::move(block_sizes))
{
    if (config_.warmup_min_size < 1)
        throw std::invalid_argument("LargeBlockPolicy: warmup.min_size must be at least 1");
    if (config_.lowrank_start_iter < 2)
        throw std::invalid_argument("LargeBlockPolicy: lowrank.start_iter must be at least 2");
    if (config_.lowrank_reevaluate < 1)
        throw std::invalid_argument("LargeBlockPolicy: lowrank.reevaluate must be at least 1");
    if (config_.switch_iter < 0)
        throw std::invalid_argument("LargeBlockPolicy: warmup.switch_iter must not be negative");
}

bool LargeBlockPolicy::begin_iteration(int k, double eta)
{
    if (k < 1)
        throw std::invalid_argument("LargeBlockPolicy::begin_iteration: iterations are 1-based (got " + std::to_string(k) + ")");
    k_ = k;
    // the engine's rule: switch when k > switch_iter or eta < switch_tol, never at k = 1, once
    if (k_sw_ < 0 && k > 1 && (k > config_.switch_iter || eta < config_.switch_tol))
    {
        k_sw_ = k;
        return true;
    }
    return false;
}

bool LargeBlockPolicy::any_warm() const
{
    for (int b = 0; b < block_count(); b++)
        if (warm(b))
            return true;
    return false;
}

int LargeBlockPolicy::start(int b) const
{
    if (warm(b) || config_.legacy_global_start)
        return k_sw_; // -1 until the switch
    return config_.lowrank_start_iter;
}

LargeStrategy LargeBlockPolicy::decide(int b, int mode) const
{
    if (b < 0 || b >= block_count())
        throw std::out_of_range("LargeBlockPolicy::decide: block " + std::to_string(b));
    if (k_ < 1)
        throw std::logic_error("LargeBlockPolicy::decide before begin_iteration");
    if (warm(b) && !switched())
        return LargeStrategy::CompositeFilter;
    if (!config_.lowrank_enabled)
        return LargeStrategy::FullEvd;
    // the low-rank phase of the block: from start(b) on, the periodic analysis and LOBPCG in a low-rank mode; a mode
    // left by an earlier solve (the engine's state persists) is not used before the phase begins, as the engine's
    // switched_proj_method gate did
    const int s = start(b);
    if (s > 0 && k_ >= s)
    {
        if ((k_ - s) % config_.lowrank_reevaluate == 0)
            return LargeStrategy::FullEvdWithRankAnalysis;
        if (mode == kModeLowRankPositive || mode == kModeLowRankNegative)
            return LargeStrategy::Lobpcg;
    }
    return LargeStrategy::FullEvd;
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
