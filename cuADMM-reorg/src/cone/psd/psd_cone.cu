/*

    cone/psd/psd_cone.cu

    Phase 6a: the tiers of main (Step 3 of SDPSolver::solve(), moved through AdmmState::project_psd_tiers of Phase 4
    and PsdCone::project_tiers of Phase 5) as named projectors selected per large block by LargeBlockPolicy, on the
    buffers of the ported engine. Every call sequence is the one of Milestone 1: the exact and LOBPCG paths are bit
    for bit; the composite path differs by its deterministic Lanczos start (CHANGES.md, Phase 6a step 1).
    measure() is the PSD part of SDPSolver::cone_measures() of main on the same projectors' decompositions.
    Phase 6b changes the memory layout (arena, SYRK) within rounding.

*/

#include "cone/psd/psd_cone.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "cone/psd/svec_pack.h"
#include "cuadmm_legacy/solver.h"

#ifndef CUADMM_NO_LAPACK
extern "C" void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a, const int *lda, double *w, double *work, const int *lwork,
                        int *iwork, const int *liwork, int *info);
#endif

namespace cuadmm
{
namespace cone
{

namespace
{
double psd_block_min_host(const double *v, int n, std::vector<double> &work, std::vector<int> &iwork)
{
    std::vector<double> A(static_cast<std::size_t>(n) * n, 0.0), w(n, 0.0);
    const double is2 = 1.0 / std::sqrt(2.0);
    for (int c = 0; c < n; c++)
        for (int r = 0; r <= c; r++)
        {
            double x = v[static_cast<long>(c) * (c + 1) / 2 + r];
            if (r != c)
                x *= is2;
            A[static_cast<std::size_t>(c) * n + r] = x;
            A[static_cast<std::size_t>(r) * n + c] = x;
        }
#ifdef CUADMM_NO_LAPACK
    (void)work;
    (void)iwork;
    throw std::runtime_error("the PSD host reference measure needs LAPACK (built with CUADMM_NO_LAPACK)");
#else
    const char jobz = 'N', uplo = 'U';
    int lwork = -1, liwork = -1, info = 0;
    double wq = 0;
    int iq = 0;
    dsyevd_(&jobz, &uplo, &n, A.data(), &n, w.data(), &wq, &lwork, &iq, &liwork, &info);
    if (info != 0)
        return std::numeric_limits<double>::quiet_NaN();
    lwork = static_cast<int>(wq);
    liwork = iq;
    work.assign(lwork, 0.0);
    iwork.assign(liwork, 0);
    dsyevd_(&jobz, &uplo, &n, A.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0)
        return std::numeric_limits<double>::quiet_NaN();
    return w[0];
#endif
}
const HostReferenceMeasure kPsdReference{&psd_block_min_host};

// the sizes of the large blocks in the engine's flat order (size group by size group)
std::vector<int> large_sizes(const SDPSolver &e)
{
    std::vector<int> sizes;
    for (std::size_t i = 0; i < e.sizes.large_mat_sizes.size(); i++)
        for (int j = 0; j < e.sizes.large_mat_nums[i]; j++)
            sizes.push_back(e.sizes.large_mat_sizes[i]);
    return sizes;
}

bool exact_strategy(psd::LargeStrategy s)
{
    return s == psd::LargeStrategy::FullEvd || s == psd::LargeStrategy::FullEvdWithRankAnalysis;
}
} // namespace

PsdConeConfig PsdConeConfig::from_options(const PsdOptions &o, unsigned long long seed)
{
    PsdConeConfig c;
    c.policy.warmup = o.warmup == WarmupKind::CompositeFp16;
    c.policy.warmup_min_size = o.warmup_min_size;
    c.policy.switch_tol = o.warmup_switch_tol;
    c.policy.switch_iter = o.warmup_switch_iter;
    c.policy.lowrank_enabled = o.lowrank_enabled;
    c.policy.lowrank_start_iter = o.lowrank_start_iter;
    c.policy.lowrank_reevaluate = o.lowrank_reevaluate;
    c.policy.legacy_global_start = false;
    c.rank.ratio = o.lowrank_ratio;
    c.rank.relaxation = o.lowrank_relaxation;
    c.rank.rank_rtol = o.lowrank_rank_rtol;
    c.rank.rank_stop_factor = o.lowrank_rank_stop_factor;
    c.lobpcg.max_iter = o.lowrank_max_iter;
    c.lobpcg.tol = o.lowrank_tol;
    c.composite.margin = o.composite_margin;
    c.composite.lanczos_steps = o.composite_lanczos_steps;
    c.composite.norm_slack = o.composite_norm_slack;
    c.composite.seed = seed;
    c.gemm_compat = o.large_final_product == FinalProductKind::GemmCompat;
    return c;
}

PsdCone::PsdCone(std::vector<BlockSlice> blocks, SDPSolver &engine, const PsdConeConfig &config)
    : blocks_(std::move(blocks)), engine_(engine), config_(config), policy_(config.policy, large_sizes(engine)),
      analyzer_(config.rank, engine), full_(engine, analyzer_), lobpcg_(config.lobpcg, engine), composite_(config.composite, engine),
      medium_(engine), small_(engine)
{
    for (const BlockSlice &b : blocks_)
        dimension_ += b.length;
    if (!config_.gemm_compat)
        throw std::invalid_argument("PsdCone: psd.large.final_product = syrk arrives with Phase 6b");
}

psd::LargeBlockJob PsdCone::job(int idx)
{
    SDPSolver &e = engine_;
    if (idx < 0 || idx >= e.sizes.large_mat_num)
        throw std::out_of_range("PsdCone::job: large block " + std::to_string(idx));
    int i = 0, j = idx;
    while (j >= e.sizes.large_mat_nums[i])
    {
        j -= e.sizes.large_mat_nums[i];
        i++;
    }
    const bool lowrank = e.use_lobpcg;
    psd::LargeBlockState st{
        idx,
        e.large_mode[idx],
        e.large_k[idx],
        e.large_rank_tol[idx],
        lowrank ? e.lobpcg_k_alloc[idx] : 0,
        lowrank ? e.positive_ranks.vals + idx : nullptr,
        lowrank ? e.negative_ranks.vals + idx : nullptr,
        lowrank ? e.cpu_positive_ranks.vals[idx] : dummy_rank_[0],
        lowrank ? e.cpu_negative_ranks.vals[idx] : dummy_rank_[1],
        lowrank ? e.lobpcg_P[idx].vals : nullptr,
        lowrank ? e.lobpcg_W[idx].vals : nullptr};
    return psd::LargeBlockJob{i, j, e.sizes.large_mat_sizes[i], e.sizes.large_mat_offset(i, j), e.sizes.large_W_offset(i, j), st};
}

bool PsdCone::begin_iteration(int k, double eta, bool verbose)
{
    SDPSolver &e = engine_;
    verbose_ = verbose;
    iteration_ = k;
    const bool switched = policy_.begin_iteration(k, eta);
    if (switched)
    {
        // the engine's bookkeeping of the switch (its console line and the warm-up fields it reports)
        if (e.final_proj_method != e.current_proj_method && verbose)
            std::cout << " ---------------- Switching projection method to " << get_projection_method_name(e.final_proj_method) << "------" << std::endl;
        e.current_proj_method = e.final_proj_method;
        e.switched_proj_method = true;
        e.switched_proj_method_iter = k;
    }
    return switched;
}

void PsdCone::project_dual(core::DeviceSpan<const double> z, core::DeviceSpan<double> s, core::ExecContext &)
{
    SDPSolver &e = engine_;
    // the matrices of the PSD blocks (svec -> dense, 1/sqrt 2 off the diagonal), as Step 3 of main
    psd_unpack(z.data, e.large_mat.vals, e.medium_mat.vals, e.small_mat.vals, e.map_B.vals, e.map_M1.vals, e.map_M2.vals, e.vec_len);
    CHECK_CUDA(cudaDeviceSynchronize());
    project_tiers();
    // back to the vector (dense -> svec, sqrt 2 off the diagonal); the other cones write their own entries
    psd_pack(s.data, e.large_mat_P.vals, e.medium_mat_P.vals, e.small_mat_P.vals, e.map_B.vals, e.map_M1.vals, e.map_M2.vals, e.vec_len);
}

void PsdCone::project_tiers()
{
    SDPSolver &e = engine_;
    // a cone step without begin_iteration (the tests, a driver of its own): the policy sees the iteration, no
    // residual-based switch
    if (policy_.iteration() != iteration_)
        policy_.begin_iteration(iteration_, std::numeric_limits<double>::infinity());

    /* Step 3.1. The large blocks, one at a time, with the projector the policy selects */
    const int L = e.sizes.large_mat_num;
    strategies_.resize(L);
    // when every large block takes the exact path in this iteration, the blocks of each size group are multiplied
    // by their eigenvectors with one strided-batched GEMM (bit-identical to the code before the refactor: a
    // strided-batched GEMM with batch > 1 does not round like per-matrix GEMMs); gemm_compat only (Phase 6b: SYRK)
    bool group_gemm = config_.gemm_compat;
    bool any_analysis = false;
    for (int idx = 0; idx < L; idx++)
    {
        strategies_[idx] = policy_.decide(idx, e.large_mode[idx]);
        group_gemm = group_gemm && exact_strategy(strategies_[idx]);
    }

    int idx = 0; // flat index of the large block
    for (int i = 0; i < (int)e.sizes.large_mat_sizes.size(); i++)
    {
        for (int j = 0; j < e.sizes.large_mat_nums[i]; j++)
        {
            const psd::LargeBlockJob jb = job(idx);
            switch (strategies_[idx])
            {
            case psd::LargeStrategy::CompositeFilter:
                if (!composite_.project(jb))
                    full_.project(jb, false, true);
                break;
            case psd::LargeStrategy::FullEvd:
                full_.project(jb, false, !group_gemm);
                break;
            case psd::LargeStrategy::FullEvdWithRankAnalysis:
                any_analysis = true;
                full_.project(jb, true, !group_gemm);
                break;
            case psd::LargeStrategy::Lobpcg:
                if (!lobpcg_.project(jb))
                {
                    any_analysis = true;
                    full_.project(jb, true, true);
                }
                break;
            }
            idx++;
        }
        if (group_gemm)
            dense_matrix_mul_trans_batch(
                e.cublasH,
                e.large_mat_P, e.large_mat_tmp, e.large_mat,
                e.sizes.large_mat_sizes[i], e.sizes.large_mat_nums[i],
                e.sizes.large_mat_offset(i, 0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (any_analysis && L > 0 && verbose_)
    {
        int number_low_rank_matrices = 0;
        for (int b = 0; b < L; b++)
            if (e.large_mode[b] != psd::kModeFull)
                number_low_rank_matrices++;
        std::cout << " ------------------ Number of low rank matrices = " << std::setw(3) << number_low_rank_matrices << " / " << std::setw(3) << L << " ------------------" << std::endl;
        for (int b = 0; b < L; b++)
            std::cout << " Ranks (matrix " << b << "):   +" << e.cpu_positive_ranks.vals[b] << "  -" << e.cpu_negative_ranks.vals[b] << std::endl;
    }

    /* Step 3.2. The medium blocks */
    medium_.project();

    /* Step 3.3. The small blocks */
    small_.project();
}

ConeMeasures PsdCone::measure(core::DeviceSpan<const double> v, Side, core::ExecContext &)
{
    SDPSolver &e = engine_;
    // matrices of the PSD blocks
    psd_unpack(v.data, e.large_mat.vals, e.medium_mat.vals, e.small_mat.vals, e.map_B.vals, e.map_M1.vals, e.map_M2.vals, e.vec_len);
    CHECK_CUDA(cudaDeviceSynchronize());

    // eigenvalues of every PSD block, with the machinery of the projection (the eigenvectors overwrite the matrices)
    for (int idx = 0; idx < e.sizes.large_mat_num; idx++)
        full_.decompose(job(idx), idx, "cusolverDnXsyevd (cone measure)");
    medium_.decompose("cusolverDnXsyevd (cone measure)", false);
    small_.decompose("cusolverDnDsyevjBatched (cone measure)", false);
    CHECK_CUDA(cudaDeviceSynchronize());

    // host copies of the eigenvalues
    std::vector<double> large_W_h(e.sizes.sum_large_mat_size);
    std::vector<double> medium_W_h(e.sizes.sum_medium_mat_size);
    std::vector<double> small_W_h(e.sizes.sum_small_mat_size);
    if (!large_W_h.empty())
        CHECK_CUDA(cudaMemcpy(large_W_h.data(), e.large_W.vals, sizeof(double) * large_W_h.size(), D2H));
    if (!medium_W_h.empty())
        CHECK_CUDA(cudaMemcpy(medium_W_h.data(), e.medium_W.vals, sizeof(double) * medium_W_h.size(), D2H));
    if (!small_W_h.empty())
        CHECK_CUDA(cudaMemcpy(small_W_h.data(), e.small_W.vals, sizeof(double) * small_W_h.size(), D2H));

    ConeMeasures m;
    m.block_min.assign(blocks_.size(), 0.0);
    m.bound_coef.assign(blocks_.size(), 0.0);
    m.violation.assign(blocks_.size(), 0.0);
    for (std::size_t i = 0; i < blocks_.size(); i++)
    {
        const BlockInfo &blk = e.blk_info[blocks_[i].block];
        const double *W = blk.category == MatrixSizeCategory::LARGE ? large_W_h.data() : (blk.category == MatrixSizeCategory::MEDIUM ? medium_W_h.data() : small_W_h.data());
        double lambda_min = W[blk.W_offset];
        for (int k = 1; k < blk.size; k++)
            lambda_min = std::min(lambda_min, W[blk.W_offset + k]);
        m.block_min[i] = lambda_min;
        m.bound_coef[i] = std::min(0.0, lambda_min);
        m.violation[i] = std::max(0.0, -lambda_min);
    }
    return m;
}

const HostReferenceMeasure &PsdCone::host_reference() const { return kPsdReference; }

} // namespace cone
} // namespace cuadmm
