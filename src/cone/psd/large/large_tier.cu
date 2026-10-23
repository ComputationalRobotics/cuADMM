/*

    cone/psd/large/large_tier.cu

*/

#include "cone/psd/large/large_tier.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "cone/psd/eig_info.h"
#include "cone/psd/kernels/lanczos.h"
#include "cone/psd/kernels/lobpcg.h"
#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

namespace
{
std::size_t align_up(std::size_t x, std::size_t a) { return (x + a - 1) / a * a; }
} // namespace

LargeTier::LargeTier(SDPSolver &engine, const LargeTierConfig &config, core::WorkspaceArena &arena, core::StreamPool &pool)
    : engine_(engine), config_(config), arena_(arena), ctx_(pool.context(core::StreamRole::Large))
{
    SDPSolver &e = engine_;
    const int L = e.sizes.large_mat_num;
    refs_.assign(L, BlockRef{});
    size_index_.assign(L, 0);
    index_in_group_.assign(L, 0);
    k_alloc_.assign(L, 0);
    // the blocks in the engine's flat order (size group by size group), with their svec slices from the block table
    for (std::size_t b = 0; b < e.blk_info.size(); b++)
    {
        const BlockInfo &bi = e.blk_info[b];
        if (bi.type != 's' || bi.category != MatrixSizeCategory::LARGE)
            continue;
        int idx = bi.same_size_index;
        for (int i = 0; i < bi.size_index; i++)
            idx += e.sizes.large_mat_nums[i];
        refs_[idx] = BlockRef{bi.vec_offset, bi.size, 0, 0};
        size_index_[idx] = bi.size_index;
        index_in_group_[idx] = bi.same_size_index;
        n_max_ = std::max(n_max_, bi.size);
        if (config_.lowrank)
        {
            k_alloc_[idx] = static_cast<int>(std::ceil(bi.size * config_.lowrank_ratio * config_.lowrank_relaxation));
            k_max_ = std::max(k_max_, k_alloc_[idx]);
        }
    }
    if (L == 0)
        return;

    // W: the largest workspace of any strategy at n_max (the EVD workspace queried per distinct size)
    const std::size_t nn = static_cast<std::size_t>(n_max_) * n_max_;
    for (int i = 0; i < static_cast<int>(e.sizes.large_mat_sizes.size()); i++)
    {
        const int n = e.sizes.large_mat_sizes[i];
        std::size_t device_bytes = 0, host_bytes = 0;
        core::DeviceBuffer<double> dummy(16);
        CUADMM_CHECK_CUSOLVER(cusolverDnXsyevd_bufferSize(ctx_.cusolver, eig_param_.param, eig_param_.jobz, eig_param_.uplo, n, CUDA_R_64F,
                                                          dummy.data(), n, CUDA_R_64F, dummy.data(), CUDA_R_64F, &device_bytes, &host_bytes));
        evd_bytes_ = std::max(evd_bytes_, device_bytes);
        evd_host_bytes_ = std::max(evd_host_bytes_, host_bytes);
    }
    host_ws_.assign(std::max<std::size_t>(evd_host_bytes_, 8), 0);
    std::size_t product_bytes = (config_.gemm_compat ? 2 : 1) * nn * sizeof(double);
    std::size_t lobpcg_bytes = 0;
    if (config_.lowrank && k_max_ > 0)
        lobpcg_bytes = (nn + static_cast<std::size_t>(n_max_) * k_max_ + k_max_ + lobpcg_workspace_doubles(ctx_.cusolver, n_max_, k_max_)) * sizeof(double);
    std::size_t composite_bytes = 0;
    if (config_.composite)
    {
        const std::size_t stride = nn % 4 == 0 ? nn : nn + (4 - nn % 4); // the alignment the filter's half4 loads need
        composite_bytes = align_up(3 * stride * sizeof(float) + 3 * stride * sizeof(__half), 256) +
                          lanczos_workspace_doubles(ctx_.cusolver, n_max_, config_.lanczos_steps) * sizeof(double);
    }
    W_bytes_ = align_up(std::max({evd_bytes_, product_bytes, lobpcg_bytes, composite_bytes, static_cast<std::size_t>(8)}), 256);

    h_M_ = arena_.request("psd.large.M", nn * sizeof(double));
    h_W_ = arena_.request("psd.large.W", W_bytes_);
    h_eig_ = arena_.request("psd.large.eigenvalues", static_cast<std::size_t>(n_max_) * sizeof(double));
    h_info_ = arena_.request("psd.large.info", sizeof(int));
    if (config_.lowrank)
    {
        h_V_.assign(L, -1);
        h_D_.assign(L, -1);
        for (int idx = 0; idx < L; idx++)
        {
            h_V_[idx] = arena_.request("psd.large.V[" + std::to_string(idx) + "]", static_cast<std::size_t>(refs_[idx].n) * k_alloc_[idx] * sizeof(double));
            h_D_[idx] = arena_.request("psd.large.D[" + std::to_string(idx) + "]", static_cast<std::size_t>(k_alloc_[idx]) * sizeof(double));
        }
    }
}

void LargeTier::bind()
{
    if (bound_ || count() == 0)
    {
        bound_ = true;
        return;
    }
    if (!arena_.committed())
        throw std::logic_error("LargeTier: the workspace arena is not committed");
    M_ = static_cast<double *>(arena_.data(h_M_));
    W_ = static_cast<double *>(arena_.data(h_W_));
    eig_ = static_cast<double *>(arena_.data(h_eig_));
    info_ = static_cast<int *>(arena_.data(h_info_));
    V_.assign(count(), nullptr);
    D_.assign(count(), nullptr);
    if (config_.lowrank)
        for (int idx = 0; idx < count(); idx++)
        {
            V_[idx] = static_cast<double *>(arena_.data(h_V_[idx]));
            D_[idx] = static_cast<double *>(arena_.data(h_D_[idx]));
        }
    bound_ = true;
}

void LargeTier::unpack(int idx, const double *z)
{
    svec_to_dense(z + refs_[idx].svec_offset, refs_[idx].n, M_, ctx_.stream);
}

void LargeTier::pack(int idx, const double *P, double *s, const double *z_add)
{
    const BlockRef &r = refs_[idx];
    dense_to_svec(P, r.n, s + r.svec_offset, z_add != nullptr ? z_add + r.svec_offset : nullptr, ctx_.stream);
}

void LargeTier::decompose(int idx, const char *what)
{
    const int n = refs_[idx].n;
    CUADMM_CHECK_CUSOLVER(cusolverDnXsyevd(ctx_.cusolver, eig_param_.param, eig_param_.jobz, eig_param_.uplo, n, CUDA_R_64F, M_, n,
                                           CUDA_R_64F, eig_, CUDA_R_64F, W_, evd_bytes_, host_ws_.data(), host_ws_.size(), info_));
    check_eig_info(info_, 1, what);
}

LargeBlockState LargeTier::state(int idx)
{
    SDPSolver &e = engine_;
    const bool lr = config_.lowrank;
    return LargeBlockState{
        idx,
        e.large_mode[idx],
        e.large_k[idx],
        e.large_rank_tol[idx],
        lr ? k_alloc_[idx] : 0,
        lr ? e.positive_ranks.vals + idx : nullptr,
        lr ? e.negative_ranks.vals + idx : nullptr,
        lr ? e.cpu_positive_ranks.vals[idx] : dummy_rank_[0],
        lr ? e.cpu_negative_ranks.vals[idx] : dummy_rank_[1],
        lr ? V_[idx] : nullptr,
        lr ? D_[idx] : nullptr};
}

double LargeTier::engine_stop_tol() const { return engine_.stop_tol; }

std::string LargeTier::memory_text() const
{
    std::ostringstream os;
    const std::size_t nn = static_cast<std::size_t>(n_max_) * n_max_;
    std::size_t warm = 0;
    for (int idx = 0; idx < count(); idx++)
        warm += (static_cast<std::size_t>(refs_[idx].n) + 1) * k_alloc_[idx] * sizeof(double);
    os << "large tier: " << count() << " block(s), n_max " << n_max_ << ": M " << nn * sizeof(double) / (1024.0 * 1024.0) << " MiB, W "
       << W_bytes_ / (1024.0 * 1024.0) << " MiB (EVD workspace " << evd_bytes_ / (1024.0 * 1024.0) << " MiB), warm starts "
       << warm / (1024.0 * 1024.0) << " MiB";
    return os.str();
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
