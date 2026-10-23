/*

    cone/psd/psd_cone.cu

    Phase 6b: the tiers on the cone's own memory (the WorkspaceArena) and the StreamPool, the large blocks through
    one arena region, the final products by SYRK / scaled GEMM (psd.*.final_product = gemm_compat keeps the
    products of Milestone 1 bit for bit). measure() is the PSD part of SDPSolver::cone_measures() of main on the
    tiers' decompositions.

*/

#include "cone/psd/psd_cone.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/check.h"
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

psd::LargeBlockPolicyConfig effective_policy(const PsdConeConfig &c, const SDPSolver &e)
{
    psd::LargeBlockPolicyConfig p = c.policy;
    p.lowrank_enabled = p.lowrank_enabled && e.use_lobpcg; // the engine's rank counters exist only with use_lobpcg
    return p;
}

psd::LargeTierConfig large_config(const PsdConeConfig &c, const psd::LargeBlockPolicy &policy)
{
    psd::LargeTierConfig t;
    t.gemm_compat = c.gemm_compat;
    t.lowrank = policy.config().lowrank_enabled;
    t.lowrank_ratio = c.rank.ratio;
    t.lowrank_relaxation = c.rank.relaxation;
    t.composite = policy.any_warm();
    t.lanczos_steps = c.composite.lanczos_steps;
    return t;
}

psd::BatchedTierConfig batched_config(const PsdConeConfig &c)
{
    psd::BatchedTierConfig b;
    b.gemm_compat = c.batched_gemm_compat;
    b.medium_batch_min = c.medium_batch_min;
    b.medium_max_batch = c.medium_max_batch;
    b.jacobi_tol = c.jacobi_tol;
    b.jacobi_max_sweeps = c.jacobi_max_sweeps;
    return b;
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
    c.batched_gemm_compat = o.batched_final_product == BatchedProductKind::GemmCompat;
    c.medium_batch_min = o.medium_batch_min;
    c.medium_max_batch = o.medium_max_batch;
    c.jacobi_tol = o.jacobi_tol;
    c.jacobi_max_sweeps = o.jacobi_max_sweeps;
    return c;
}

PsdCone::PsdCone(std::vector<BlockSlice> blocks, SDPSolver &engine, const PsdConeConfig &config, core::WorkspaceArena &arena, core::StreamPool &pool)
    : blocks_(std::move(blocks)), engine_(engine), config_(config), policy_(effective_policy(config, engine), large_sizes(engine)),
      large_(engine, large_config(config, policy_), arena, pool), analyzer_(config.rank, large_), full_(large_, analyzer_),
      lobpcg_(config.lobpcg, large_), composite_(config.composite, large_),
      medium_(engine, psd::TierKind::Medium, batched_config(config), arena, pool), small_(engine, psd::TierKind::Small, batched_config(config), arena, pool)
{
    for (const BlockSlice &b : blocks_)
        dimension_ += b.length;
}

void PsdCone::bind()
{
    large_.bind();
    medium_.bind();
    small_.bind();
}

std::string PsdCone::memory_text() const
{
    std::ostringstream os;
    os << "  " << large_.memory_text() << "\n  " << medium_.memory_text() << "\n  " << small_.memory_text();
    return os.str();
}

psd::LargeBlockJob PsdCone::job(int idx)
{
    const psd::BlockRef &r = large_.ref(idx);
    return psd::LargeBlockJob{large_.size_index(idx), large_.index_in_group(idx), r.n, 0, 0, large_.state(idx)};
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
    bind();
    project_tiers(z.data, s.data);
}

void PsdCone::project_tiers(const double *z, double *s)
{
    SDPSolver &e = engine_;
    // a cone step without begin_iteration (the tests, a driver of its own): the policy sees the iteration, no
    // residual-based switch
    if (policy_.iteration() != iteration_)
        policy_.begin_iteration(iteration_, std::numeric_limits<double>::infinity());

    /* Step 3.1. The large blocks, one at a time through the arena, with the projector the policy selects */
    const int L = large_.count();
    strategies_.resize(L);
    bool any_analysis = false;
    for (int idx = 0; idx < L; idx++)
        strategies_[idx] = policy_.decide(idx, e.large_mode[idx]);
    for (int idx = 0; idx < L; idx++)
    {
        const psd::LargeBlockJob jb = job(idx);
        large_.unpack(idx, z);
        switch (strategies_[idx])
        {
        case psd::LargeStrategy::CompositeFilter:
            if (!composite_.project(jb, z, s))
                full_.project(jb, false, z, s); // the input is in M again
            break;
        case psd::LargeStrategy::FullEvd:
            full_.project(jb, false, z, s);
            break;
        case psd::LargeStrategy::FullEvdWithRankAnalysis:
            any_analysis = true;
            full_.project(jb, true, z, s);
            break;
        case psd::LargeStrategy::Lobpcg:
            if (!lobpcg_.project(jb, z, s))
            {
                any_analysis = true;
                full_.project(jb, true, z, s); // M still holds the input
            }
            break;
        }
    }
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(large_.ctx().stream));

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
    medium_.project(z, s);

    /* Step 3.3. The small blocks */
    small_.project(z, s);
}

ConeMeasures PsdCone::measure(core::DeviceSpan<const double> v, Side, core::ExecContext &)
{
    SDPSolver &e = engine_;
    bind();

    // eigenvalues of every PSD block, with the machinery of the projection (the eigenvectors overwrite the matrices)
    std::vector<double> large_W_h;
    for (int idx = 0; idx < large_.count(); idx++)
    {
        const int n = large_.size(idx);
        large_.unpack(idx, v.data);
        large_.decompose(idx, "cusolverDnXsyevd (cone measure)");
        std::vector<double> w(n);
        CUADMM_CHECK_CUDA(cudaMemcpy(w.data(), large_.eigenvalues(), sizeof(double) * n, cudaMemcpyDeviceToHost));
        large_W_h.insert(large_W_h.end(), w.begin(), w.end());
    }
    medium_.unpack(v.data);
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    medium_.decompose("cusolverDnXsyevd (cone measure)", false, false);
    small_.unpack(v.data);
    small_.decompose("cusolverDnDsyevjBatched (cone measure)", false, false);
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());

    // host copies of the eigenvalues of the batched tiers
    std::vector<double> medium_W_h(medium_.sum_n()), small_W_h(small_.sum_n());
    if (!medium_W_h.empty())
        CUADMM_CHECK_CUDA(cudaMemcpy(medium_W_h.data(), medium_.eigenvalues(), sizeof(double) * medium_W_h.size(), cudaMemcpyDeviceToHost));
    if (!small_W_h.empty())
        CUADMM_CHECK_CUDA(cudaMemcpy(small_W_h.data(), small_.eigenvalues(), sizeof(double) * small_W_h.size(), cudaMemcpyDeviceToHost));

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
