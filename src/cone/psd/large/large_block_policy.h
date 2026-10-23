/*

    cone/psd/large/large_block_policy.h

    LargeBlockPolicy (plan Section 10.6): which projector each large block gets at each iteration.

        warm(b)  = warmup != none and n_b >= warmup.min_size
        k_sw     = first k > 1 with eta < warmup.switch_tol or k > warmup.switch_iter   (global, one-way)
        start(b) = k_sw if warm(b) else lowrank.start_iter

        if warm(b) and k < k_sw:                                     CompositeFilter  (rejected -> FullEvd)
        elif lowrank.enabled and k >= start(b)
             and (k - start(b)) % lowrank.reevaluate == 0:           FullEvd + RankAnalyzer
        elif k >= start(b) and mode(b) is LowRankPositive or LowRankNegative:
                                                                     Lobpcg           (rejected -> FullEvd + RankAnalyzer)
        else:                                                        FullEvd

    The policy is host logic without CUDA, so it is tested on the CPU (test/large_block_policy_test.hpp). The
    switch k_sw is decided at the beginning of iteration k with the KKT residual eta of the current iterate, as
    the engine's switched_proj_method was. `legacy_global_start` reproduces the C++ API of main, where blocks
    without a warm-up also start the low-rank analysis at k_sw (B3: the single flag of today); the facade uses
    the per-block rule above.

*/

#ifndef CUADMM_CONE_PSD_LARGE_LARGE_BLOCK_POLICY_H
#define CUADMM_CONE_PSD_LARGE_LARGE_BLOCK_POLICY_H

#include <vector>

namespace cuadmm
{
namespace cone
{
namespace psd
{

enum class LargeStrategy
{
    FullEvd,                 // exact projection (cusolverDnXsyevd)
    FullEvdWithRankAnalysis, // exact projection, then the rank analysis that sets the block's mode and warm start
    Lobpcg,                  // warm-started LOBPCG in the block's low-rank mode
    CompositeFilter          // the FP16 composite polynomial filter (warm-up)
};

const char *large_strategy_name(LargeStrategy s);

/// The per-block projection mode set by the rank analysis (the values of the engine's LargeProjectionMode).
enum LargeMode
{
    kModeFull = 0,
    kModeLowRankPositive = 1,
    kModeLowRankNegative = 2
};

struct LargeBlockPolicyConfig
{
    bool warmup = false;           // psd.large.warmup == composite_fp16
    int warmup_min_size = 5000;    // psd.large.warmup.min_size
    double switch_tol = 1e-2;      // psd.large.warmup.switch_tol
    int switch_iter = 5000;        // psd.large.warmup.switch_iter
    bool lowrank_enabled = true;   // psd.lowrank.enabled
    int lowrank_start_iter = 2;    // psd.lowrank.start_iter (blocks without a warm-up)
    int lowrank_reevaluate = 100;  // psd.lowrank.reevaluate
    bool legacy_global_start = false; // C++ API of main: every block starts the low-rank analysis at k_sw
};

class LargeBlockPolicy
{
public:
    LargeBlockPolicy(const LargeBlockPolicyConfig &config, std::vector<int> block_sizes);

    /// The beginning of iteration k (1-based) with the KKT residual eta of the current iterate: decides the
    /// switch k_sw (one-way). Returns true at the iteration that switches.
    bool begin_iteration(int k, double eta);
    /// The strategy of block b at the current iteration, given its mode (LargeMode) from the last rank analysis.
    LargeStrategy decide(int b, int mode) const;

    int iteration() const { return k_; }
    bool warm(int b) const { return config_.warmup && sizes_[b] >= config_.warmup_min_size; }
    bool any_warm() const;
    bool switched() const { return k_sw_ > 0; }
    int switch_iteration() const { return k_sw_; } // -1 before the switch
    /// The first iteration of the low-rank analysis of block b (-1 while it depends on a switch not yet made).
    int start(int b) const;
    int block_count() const { return static_cast<int>(sizes_.size()); }
    const LargeBlockPolicyConfig &config() const { return config_; }

private:
    LargeBlockPolicyConfig config_;
    std::vector<int> sizes_;
    int k_ = 0;
    int k_sw_ = -1;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_LARGE_BLOCK_POLICY_H
