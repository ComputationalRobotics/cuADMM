/*

    cone/psd/large/projectors/composite_filter.h

    CompositeFilter (plan Section 10.7): the FP16 composite polynomial filter as the warm-up projection of a
    large block. The FP64 Lanczos estimate of ||Z||_2 (deterministic start vector seeded by (seed, block index),
    B8) gives the scale margin * up; the block is divided by it, filtered in the FP32/FP16 workspaces (22 GEMMs
    on Tensor Cores with FP32 accumulation; the coefficient table is data in kernels/composite_fp16.cu), and
    multiplied back. Safety (B22): the result is rejected, the input restored and the block left to FullEvd when
    the result is not finite or ||P||_F > (1 + norm_slack) ||Z||_F (the projection is nonexpansive with
    Pi(0) = 0). The calls are those of SDPSolver::large_composite_project of Milestone 1.

*/

#ifndef CUADMM_CONE_PSD_LARGE_PROJECTORS_COMPOSITE_FILTER_H
#define CUADMM_CONE_PSD_LARGE_PROJECTORS_COMPOSITE_FILTER_H

#include "cone/psd/large/large_block_state.h"

class SDPSolver;

namespace cuadmm
{
namespace cone
{
namespace psd
{

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
    CompositeFilter(const CompositeFilterConfig &config, SDPSolver &engine);
    /// The projection into large_mat_P; false when the result was rejected (the input is restored in large_mat).
    bool project(const LargeBlockJob &job);
    const CompositeFilterConfig &config() const { return config_; }
    /// The Lanczos seed of a block: the solver seed mixed with the block index (B8).
    static unsigned long long block_seed(unsigned long long seed, int idx);

private:
    CompositeFilterConfig config_;
    SDPSolver &engine_;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_PROJECTORS_COMPOSITE_FILTER_H
