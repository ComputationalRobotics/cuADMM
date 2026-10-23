/*

    cone/psd/large/projectors/composite_filter.h

    CompositeFilter (plan Section 10.7): the FP16 composite polynomial filter as the warm-up projection of a
    large block, its FP32/FP16 buffers and Lanczos vectors carved from the tier's W (B7). The FP64 Lanczos
    estimate of ||Z||_2 (deterministic start vector seeded by (seed, block index), B8) gives the scale
    margin * up; the block is divided by it, filtered in place (22 GEMMs on Tensor Cores with FP32
    accumulation; the coefficient table is data in kernels/composite_fp16.cu), multiplied back and packed.
    Safety (B22): the result is rejected when it is not finite or ||P||_F > (1 + norm_slack) ||Z||_F (the
    projection is nonexpansive with Pi(0) = 0); the block is then unpacked again from the input for FullEvd,
    instead of the n^2 copy of Milestone 1.

*/

#ifndef CUADMM_CONE_PSD_LARGE_PROJECTORS_COMPOSITE_FILTER_H
#define CUADMM_CONE_PSD_LARGE_PROJECTORS_COMPOSITE_FILTER_H

#include "cone/psd/large/large_block_state.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

class LargeTier;

struct CompositeFilterConfig
{
    double margin = 1.05;           // psd.composite.margin
    int lanczos_steps = 20;         // psd.composite.lanczos_steps
    double norm_slack = 1e-2;       // psd.composite.norm_slack
    unsigned long long seed = 0;    // options.seed (the Lanczos start vector, B8)
};

class CompositeFilter
{
public:
    CompositeFilter(const CompositeFilterConfig &config, LargeTier &tier);
    /// The projection of the block in M into s (its svec slice); false when the result was rejected (M then holds
    /// the input again, unpacked from z).
    bool project(const LargeBlockJob &job, const double *z, double *s);
    const CompositeFilterConfig &config() const { return config_; }
    /// The Lanczos seed of a block: the solver seed mixed with the block index (B8).
    static unsigned long long block_seed(unsigned long long seed, int idx);

private:
    CompositeFilterConfig config_;
    LargeTier &tier_;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_PROJECTORS_COMPOSITE_FILTER_H
