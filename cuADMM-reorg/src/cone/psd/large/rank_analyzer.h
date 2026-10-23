/*

    cone/psd/large/rank_analyzer.h

    RankAnalyzer (plan Section 10.6, "Rank analysis"): with the eigenvalues lambda_1 <= ... <= lambda_n of a large
    block (FullEvd just decomposed it),

        tol_r = max(rank_rtol * max(|lambda_1|, |lambda_n|), rank_stop_factor * tol),
        r_+ = #{lambda_i > tol_r},  r_- = #{lambda_i < -tol_r},

    the block's mode becomes LowRankPositive when 0 < r_+ < ratio * n (k = min(ceil(relaxation * r_+), k_alloc)
    pairs, the largest eigenpairs copied in decreasing order as the LOBPCG warm start), LowRankNegative when
    0 < r_- < ratio * n (the smallest eigenpairs, the eigenvalues negated: LOBPCG runs on -Z), and Dense
    otherwise. The positive test comes first, as in main. The calls are those of SDPSolver::large_full_eig_project
    of Milestone 1 (bit for bit).

*/

#ifndef CUADMM_CONE_PSD_LARGE_RANK_ANALYZER_H
#define CUADMM_CONE_PSD_LARGE_RANK_ANALYZER_H

#include "cone/psd/large/large_block_state.h"

class SDPSolver;

namespace cuadmm
{
namespace cone
{
namespace psd
{

struct RankAnalyzerConfig
{
    double ratio = 0.03;            // psd.lowrank.ratio
    double relaxation = 1.2;        // psd.lowrank.relaxation
    double rank_rtol = 1e-8;        // psd.lowrank.rank_rtol
    double rank_stop_factor = 1e-3; // psd.lowrank.rank_stop_factor
};

class RankAnalyzer
{
public:
    RankAnalyzer(const RankAnalyzerConfig &config, SDPSolver &engine);
    /// The rank tolerance of a block from its extreme eigenvalues and the stopping tolerance.
    double rank_tolerance(double lambda_min, double lambda_max, double stop_tol) const;
    /// After FullEvd::decompose(job): sets the block's ranks, rank tolerance, mode, k and warm start.
    void analyze(const LargeBlockJob &job, double stop_tol);
    const RankAnalyzerConfig &config() const { return config_; }

private:
    RankAnalyzerConfig config_;
    SDPSolver &engine_;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_RANK_ANALYZER_H
