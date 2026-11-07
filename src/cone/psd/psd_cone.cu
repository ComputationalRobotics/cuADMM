/*

    cone/psd/psd_cone.cu

    Phase 6b: the tiers on the cone's own memory (the WorkspaceArena) and the StreamPool, the large blocks through
    one arena region, the final products by SYRK / scaled GEMM (psd.*.final_product = gemm_compat keeps the
    products of Milestone 1 bit for bit). measure() is the PSD part of SDPSolver::cone_measures() of main on the
    tiers' decompositions.

*/

#include "cone/psd/psd_cone.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/check.h"
#include "cuadmm_legacy/block_structure.h"
#include "cuadmm_legacy/matrix_sizes.h"
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

// psd.trace: numbers as the shortest decimal that round-trips (JSON null when not finite), the modes and tiers by name
std::string trace_num(double v)
{
    if (!std::isfinite(v))
        return "null";
    char buf[64];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, r.ptr);
}

const char *mode_name(int mode)
{
    return mode == psd::kModeLowRankPositive ? "lowrank_pos" : (mode == psd::kModeLowRankNegative ? "lowrank_neg" : "full");
}

const char *tier_name(MatrixSizeCategory c)
{
    return c == MatrixSizeCategory::SMALL ? "small" : (c == MatrixSizeCategory::MEDIUM ? "medium" : "large");
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
    c.trace_path = o.trace;
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

    // psd.trace: the input index of each large block (the engine orders the large blocks by size group)
    SDPSolver &e = engine_;
    trace_.assign(large_.count(), BlockTrace{});
    std::vector<int> large_of(e.blk_info.size(), -1);
    for (std::size_t b = 0; b < e.blk_info.size(); b++)
    {
        const BlockInfo &bi = e.blk_info[b];
        if (bi.type != 's' || bi.category != MatrixSizeCategory::LARGE)
            continue;
        int idx = bi.same_size_index;
        for (int i = 0; i < bi.size_index; i++)
            idx += e.sizes.large_mat_nums[i];
        trace_[idx].input_block = static_cast<int>(b);
        large_of[b] = idx;
    }
    if (config_.trace_path.empty())
        return;
    trace_out_.reset(new std::ofstream(config_.trace_path, std::ios::out | std::ios::trunc));
    if (!trace_out_->good())
        throw std::runtime_error("psd.trace: cannot write " + config_.trace_path);
    // the settings the events refer to (the tier cutoffs of this setup, the warm-up, the rank analysis and LOBPCG)
    const psd::LargeBlockPolicyConfig &pc = policy_.config();
    std::ostringstream os;
    os << "{\"ev\": \"config\", \"small_max\": " << MatrixSizes::small_limit << ", \"medium_max\": " << MatrixSizes::medium_limit
       << ", \"warmup\": " << (pc.warmup ? "true" : "false") << ", \"warmup_min_size\": " << pc.warmup_min_size
       << ", \"warmup_switch_tol\": " << trace_num(pc.switch_tol) << ", \"warmup_switch_iter\": " << pc.switch_iter
       << ", \"lowrank\": " << (pc.lowrank_enabled ? "true" : "false") << ", \"lowrank_start_iter\": " << pc.lowrank_start_iter
       << ", \"lowrank_reevaluate\": " << pc.lowrank_reevaluate << ", \"lowrank_ratio\": " << trace_num(config_.rank.ratio)
       << ", \"lowrank_relaxation\": " << trace_num(config_.rank.relaxation) << ", \"rank_rtol\": " << trace_num(config_.rank.rank_rtol)
       << ", \"rank_stop_factor\": " << trace_num(config_.rank.rank_stop_factor) << ", \"stop_tol\": " << trace_num(e.stop_tol)
       << ", \"lobpcg_tol\": " << trace_num(config_.lobpcg.tol) << ", \"lobpcg_max_iter\": " << config_.lobpcg.max_iter << "}";
    trace_line(os.str());
    // every PSD block with its tier; the large ones with their warm-up, warm-start capacity and first analysis
    for (std::size_t b = 0; b < e.blk_info.size(); b++)
    {
        const BlockInfo &bi = e.blk_info[b];
        if (bi.type != 's')
            continue;
        std::ostringstream ob;
        ob << "{\"ev\": \"block\", \"block\": " << b << ", \"n\": " << bi.size << ", \"tier\": \"" << tier_name(bi.category) << "\"";
        if (large_of[b] >= 0)
        {
            const int idx = large_of[b];
            ob << ", \"large\": " << idx << ", \"warm\": " << (policy_.warm(idx) ? "true" : "false") << ", \"k_alloc\": "
               << (pc.lowrank_enabled ? large_.k_alloc(idx) : 0) << ", \"lowrank_eligible_below\": " << trace_num(config_.rank.ratio * bi.size);
        }
        ob << "}";
        trace_line(ob.str());
    }
}

PsdCone::~PsdCone()
{
    if (!trace_out_)
        return;
    try
    {
        // the totals of each large block; the final mode is the one its last rank analysis set (full before any)
        int last = -1;
        for (int idx = 0; idx < static_cast<int>(trace_.size()); idx++)
        {
            const BlockTrace &t = trace_[idx];
            last = std::max(last, t.last_iteration);
            std::ostringstream o;
            o << "{\"ev\": \"block_totals\", \"block\": " << t.input_block << ", \"large\": " << idx << ", \"n\": " << large_.size(idx)
              << ", \"warm\": " << (policy_.warm(idx) ? "true" : "false") << ", \"warmup_exit\": " << (policy_.warm(idx) ? policy_.switch_iteration() : -1)
              << ", \"composite_calls\": " << t.composite << ", \"composite_rejected\": " << t.composite_rejected
              << ", \"full_evd_calls\": " << t.full_evd << ", \"rank_analyses\": " << t.analyses << ", \"lobpcg_calls\": " << t.lobpcg
              << ", \"lobpcg_accepted\": " << (t.lobpcg - t.lobpcg_rejected) << ", \"lobpcg_rejected\": " << t.lobpcg_rejected
              << ", \"first_composite\": " << t.first_composite << ", \"first_full_evd\": " << t.first_full_evd
              << ", \"first_rank_analysis\": " << t.first_analysis << ", \"first_low_rank_analysis\": " << t.first_low_rank
              << ", \"first_lobpcg\": " << t.first_lobpcg << ", \"low_rank_analyses\": " << t.low_rank_analyses << ", \"mode_changes\": " << t.mode_changes
              << ", \"last_iteration\": " << t.last_iteration << ", \"last_rank_analysis\": " << t.last_analysis << ", \"last_pos\": " << t.last_pos
              << ", \"last_neg\": " << t.last_neg << ", \"last_rank_tol\": " << trace_num(t.last_rank_tol) << ", \"final_mode\": \"" << mode_name(t.mode)
              << "\", \"final_k_pairs\": " << (t.mode == psd::kModeFull ? 0 : t.last_k_pairs) << "}";
            trace_line(o.str());
        }
        trace_line("{\"ev\": \"end\", \"last_iteration\": " + std::to_string(last) + ", \"warmup_switch\": " + std::to_string(policy_.switch_iteration()) + "}");
    }
    catch (...)
    {
        // a destructor never throws; the events written so far stay in the file
    }
}

void PsdCone::trace_line(const std::string &json)
{
    if (trace_out_)
        (*trace_out_) << json << "\n" << std::flush;
}

void PsdCone::trace_switch(int k, double eta)
{
    if (trace_out_)
        trace_line("{\"k\": " + std::to_string(k) + ", \"ev\": \"warmup_switch\", \"eta\": " + trace_num(eta) + ", \"reason\": \"" +
                   (eta < policy_.config().switch_tol ? "eta" : "iteration") + "\"}");
}

void PsdCone::trace_analysis(int idx, int mode_before, const char *cause)
{
    SDPSolver &e = engine_;
    BlockTrace &t = trace_[idx];
    const int k = iteration_;
    const int mode_after = e.large_mode[idx];
    t.analyses++;
    if (t.first_analysis < 0)
        t.first_analysis = k;
    if (mode_after != mode_before)
        t.mode_changes++;
    if (mode_after != psd::kModeFull)
    {
        t.low_rank_analyses++;
        if (t.first_low_rank < 0)
            t.first_low_rank = k;
    }
    t.mode = mode_after;
    t.last_analysis = k;
    t.last_pos = e.cpu_positive_ranks.vals[idx];
    t.last_neg = e.cpu_negative_ranks.vals[idx];
    t.last_rank_tol = e.large_rank_tol[idx];
    t.last_k_pairs = e.large_k[idx];
    if (!trace_out_)
        return;
    std::ostringstream o;
    o << "{\"k\": " << k << ", \"ev\": \"rank\", \"block\": " << t.input_block << ", \"large\": " << idx << ", \"n\": " << large_.size(idx)
      << ", \"pos\": " << t.last_pos << ", \"neg\": " << t.last_neg << ", \"rank_tol\": " << trace_num(t.last_rank_tol) << ", \"mode_before\": \""
      << mode_name(mode_before) << "\", \"mode_after\": \"" << mode_name(mode_after) << "\", \"k_pairs\": " << (mode_after == psd::kModeFull ? 0 : t.last_k_pairs)
      << ", \"cause\": \"" << cause << "\"}";
    trace_line(o.str());
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
        trace_switch(k, eta);
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
    if (policy_.iteration() != iteration_ && policy_.begin_iteration(iteration_, std::numeric_limits<double>::infinity()))
        trace_switch(iteration_, std::numeric_limits<double>::infinity());

    /* Step 3.1. The large blocks, one at a time through the arena, with the projector the policy selects */
    const int L = large_.count();
    strategies_.resize(L);
    bool any_analysis = false;
    for (int idx = 0; idx < L; idx++)
        strategies_[idx] = policy_.decide(idx, e.large_mode[idx]);
    for (int idx = 0; idx < L; idx++)
    {
        const psd::LargeBlockJob jb = job(idx);
        // psd.trace bookkeeping (host counters; the projections are unchanged)
        BlockTrace &t = trace_[idx];
        const int k = iteration_;
        const int mode_before = e.large_mode[idx];
        t.last_iteration = k;
        auto full_evd_counted = [&t, k]()
        {
            t.full_evd++;
            if (t.first_full_evd < 0)
                t.first_full_evd = k;
        };
        large_.unpack(idx, z);
        switch (strategies_[idx])
        {
        case psd::LargeStrategy::CompositeFilter:
            t.composite++;
            if (t.first_composite < 0)
                t.first_composite = k;
            if (!composite_.project(jb, z, s))
            {
                t.composite_rejected++;
                full_evd_counted();
                if (trace_out_)
                    trace_line("{\"k\": " + std::to_string(k) + ", \"ev\": \"composite_rejected\", \"block\": " + std::to_string(t.input_block) +
                               ", \"large\": " + std::to_string(idx) + "}");
                full_.project(jb, false, z, s); // the input is in M again
            }
            break;
        case psd::LargeStrategy::FullEvd:
            full_evd_counted();
            full_.project(jb, false, z, s);
            break;
        case psd::LargeStrategy::FullEvdWithRankAnalysis:
            any_analysis = true;
            full_evd_counted();
            full_.project(jb, true, z, s);
            trace_analysis(idx, mode_before, "scheduled");
            break;
        case psd::LargeStrategy::Lobpcg:
            t.lobpcg++;
            if (t.first_lobpcg < 0)
                t.first_lobpcg = k;
            if (!lobpcg_.project(jb, z, s))
            {
                any_analysis = true;
                t.lobpcg_rejected++;
                full_evd_counted();
                full_.project(jb, true, z, s); // M still holds the input
                trace_analysis(idx, mode_before, "lobpcg_rejected");
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
