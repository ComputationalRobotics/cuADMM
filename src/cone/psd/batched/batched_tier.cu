/*

    cone/psd/batched/batched_tier.cu

    Steps 3.2 and 3.3 of SDPSolver::solve() of main (the medium and small tiers), on the arena with the pool's
    streams; the call sequences of Milestone 1 under gemm_compat, the scaled product otherwise.

*/

#include "cone/psd/batched/batched_tier.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "cone/psd/eig_info.h"
#include "core/check.h"
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

BatchedTier::BatchedTier(SDPSolver &engine, TierKind kind, const BatchedTierConfig &config, core::WorkspaceArena &arena, core::StreamPool &pool)
    : engine_(engine), kind_(kind), config_(config), arena_(arena), pool_(pool),
      ctx_(pool.context(kind == TierKind::Small ? core::StreamRole::Small : core::StreamRole::Main)),
      jacobi_param_(0, config.jacobi_tol, config.jacobi_max_sweeps, 1)
{
    SDPSolver &e = engine_;
    const bool small = kind_ == TierKind::Small;
    const std::vector<int> &sizes = small ? e.sizes.small_mat_sizes : e.sizes.medium_mat_sizes;
    const std::vector<int> &nums = small ? e.sizes.small_mat_nums : e.sizes.medium_mat_nums;
    const MatrixSizeCategory category = small ? MatrixSizeCategory::SMALL : MatrixSizeCategory::MEDIUM;
    group_size_ = sizes;
    group_count_ = nums;
    group_first_.assign(sizes.size(), 0);
    int total = 0;
    for (std::size_t i = 0; i < sizes.size(); i++)
    {
        group_first_[i] = total;
        total += nums[i];
    }
    refs_.assign(total, BlockRef{});
    for (std::size_t b = 0; b < e.blk_info.size(); b++)
    {
        const BlockInfo &bi = e.blk_info[b];
        if (bi.type != 's' || bi.category != category)
            continue;
        const int idx = group_first_[bi.size_index] + bi.same_size_index;
        refs_[idx] = BlockRef{bi.vec_offset, bi.size, bi.mat_offset, bi.W_offset};
        total_nn_ += static_cast<std::size_t>(bi.size) * bi.size;
        sum_n_ += bi.size;
    }
    table_ = BlockTable(refs_);
    batched_.assign(sizes.size(), 0);
    if (total == 0)
        return;

    // the regions
    h_mats_ = arena_.request(small ? "psd.small.mats" : "psd.medium.mats", total_nn_ * sizeof(double));
    h_W_ = arena_.request(small ? "psd.small.W" : "psd.medium.W", sum_n_ * sizeof(double));
    h_P_ = arena_.request(small ? "psd.small.P" : "psd.medium.P", total_nn_ * sizeof(double));
    if (config_.gemm_compat)
        h_tmp_ = arena_.request(small ? "psd.small.tmp" : "psd.medium.tmp", total_nn_ * sizeof(double));
    h_info_ = arena_.request(small ? "psd.small.info" : "psd.medium.info", static_cast<std::size_t>(total) * sizeof(int));

    core::DeviceBuffer<double> dummy(16);
    double *d = dummy.data();
    if (small)
    {
        // one Jacobi workspace per size group (the batched call needs the whole group's)
        group_ws_bytes_.assign(sizes.size(), 0);
        group_ws_offset_.assign(sizes.size(), 0);
        std::size_t off = 0;
        for (std::size_t i = 0; i < sizes.size(); i++)
        {
            int len = 0;
            CUADMM_CHECK_CUSOLVER(cusolverDnDsyevjBatched_bufferSize(ctx_.cusolver, jacobi_param_.jobz, jacobi_param_.uplo, sizes[i], d, sizes[i], d,
                                                                     &len, jacobi_param_.syevj_param, nums[i]));
            group_ws_bytes_[i] = static_cast<std::size_t>(len) * sizeof(double);
            group_ws_offset_[i] = off;
            off += align_up(group_ws_bytes_[i], 256);
        }
        h_ws_ = arena_.request("psd.small.jacobi_workspace", std::max<std::size_t>(off, 8));
        return;
    }

    // medium: the groups that use the batched EVD (CUDA >= 12.8), the chunk workspace, one streamed workspace per stream
    for (std::size_t i = 0; i < sizes.size(); i++)
    {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
        if (config_.medium_batch_min > 0 && nums[i] >= config_.medium_batch_min)
            batched_[i] = 1;
#endif
        std::size_t device_bytes = 0, host_bytes = 0;
        CUADMM_CHECK_CUSOLVER(cusolverDnXsyevd_bufferSize(ctx_.cusolver, eig_param_.param, eig_param_.jobz, eig_param_.uplo, sizes[i], CUDA_R_64F, d,
                                                          sizes[i], CUDA_R_64F, d, CUDA_R_64F, &device_bytes, &host_bytes));
        stream_ws_bytes_ = std::max(stream_ws_bytes_, device_bytes);
        stream_host_bytes_ = std::max(stream_host_bytes_, host_bytes);
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
        if (batched_[i])
        {
            const int chunk = config_.medium_max_batch > 0 ? std::min(config_.medium_max_batch, nums[i]) : nums[i];
            CUADMM_CHECK_CUSOLVER(cusolverDnXsyevBatched_bufferSize(ctx_.cusolver, eig_param_.param, eig_param_.jobz, eig_param_.uplo, sizes[i],
                                                                    CUDA_R_64F, d, sizes[i], CUDA_R_64F, d, CUDA_R_64F, &device_bytes, &host_bytes, chunk));
            chunk_ws_bytes_ = std::max(chunk_ws_bytes_, device_bytes);
            chunk_host_bytes_ = std::max(chunk_host_bytes_, host_bytes);
        }
#endif
    }
    const int streams = std::max(1, pool_.medium_count());
    stream_ws_bytes_ = align_up(std::max<std::size_t>(stream_ws_bytes_, 8), 256);
    h_ws_ = arena_.request("psd.medium.streamed_workspace[" + std::to_string(streams) + "]", stream_ws_bytes_ * streams);
    host_ws_.assign(streams, std::vector<unsigned char>(std::max<std::size_t>(stream_host_bytes_, 8), 0));
    if (chunk_ws_bytes_ > 0)
    {
        h_chunk_ws_ = arena_.request("psd.medium.batched_workspace", chunk_ws_bytes_);
        chunk_host_ws_.assign(std::max<std::size_t>(chunk_host_bytes_, 8), 0);
    }
}

void BatchedTier::bind()
{
    if (bound_ || count() == 0)
    {
        bound_ = true;
        return;
    }
    if (!arena_.committed())
        throw std::logic_error("BatchedTier: the workspace arena is not committed");
    mats_ = static_cast<double *>(arena_.data(h_mats_));
    W_ = static_cast<double *>(arena_.data(h_W_));
    P_ = static_cast<double *>(arena_.data(h_P_));
    tmp_ = h_tmp_ >= 0 ? static_cast<double *>(arena_.data(h_tmp_)) : nullptr;
    info_ = static_cast<int *>(arena_.data(h_info_));
    ws_ = static_cast<unsigned char *>(arena_.data(h_ws_));
    chunk_ws_ = h_chunk_ws_ >= 0 ? static_cast<unsigned char *>(arena_.data(h_chunk_ws_)) : nullptr;
    bound_ = true;
}

void BatchedTier::unpack(const double *z)
{
    svec_to_dense_batch(table_, z, mats_, ctx_.stream);
}

void BatchedTier::decompose_small(const char *what, bool warn)
{
    SDPSolver &e = engine_;
    int info_offset = 0;
    for (int i = 0; i < group_count(); i++)
    {
        const int n = group_size_[i];
        const BlockRef &first = refs_[group_first_[i]];
        CUADMM_CHECK_CUSOLVER(cusolverDnDsyevjBatched(ctx_.cusolver, jacobi_param_.jobz, jacobi_param_.uplo, n, mats_ + first.mat_offset, n,
                                                      W_ + first.w_offset, reinterpret_cast<double *>(ws_ + group_ws_offset_[i]),
                                                      static_cast<int>(group_ws_bytes_[i] / sizeof(double)), info_ + info_offset,
                                                      jacobi_param_.syevj_param, group_count_[i]));
        info_offset += group_count_[i];
    }
    info_offset = 0;
    for (int i = 0; i < group_count(); i++)
    {
        const int not_converged = check_eig_info(info_ + info_offset, group_count_[i], what, group_size_[i]);
        if (warn)
        {
            if (not_converged > 0 && e.jacobi_not_converged == 0)
                std::cout << " WARNING: batched Jacobi EVD did not reach its tolerance for " << not_converged
                          << " matrices of size " << group_size_[i]
                          << " (results are still accurate to machine precision; reported once per solve)" << std::endl;
            e.jacobi_not_converged += not_converged;
        }
        info_offset += group_count_[i];
    }
}

void BatchedTier::decompose_medium(const char *what, bool allow_batched)
{
    const int streams = std::max(1, pool_.medium_count());
    // the blocks were unpacked on the main stream: the medium streams wait for it
    for (int st = 0; st < streams; st++)
        pool_.wait(core::StreamRole::Medium, st, core::StreamRole::Main, 0);
    int counter = 0; // serves as an info offset and the stream index
    for (int i = 0; i < group_count(); i++)
    {
        const int n = group_size_[i];
        if (allow_batched && batched_[i])
        {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
            const int chunk = config_.medium_max_batch > 0 ? std::min(config_.medium_max_batch, group_count_[i]) : group_count_[i];
            for (int c = 0; c < group_count_[i]; c += chunk)
            {
                const int this_chunk = std::min(chunk, group_count_[i] - c);
                const BlockRef &first = refs_[group_first_[i] + c];
                CUADMM_CHECK_CUSOLVER(cusolverDnXsyevBatched(ctx_.cusolver, eig_param_.param, eig_param_.jobz, eig_param_.uplo, n, CUDA_R_64F,
                                                             mats_ + first.mat_offset, n, CUDA_R_64F, W_ + first.w_offset, CUDA_R_64F, chunk_ws_,
                                                             chunk_ws_bytes_, chunk_host_ws_.data(), chunk_host_ws_.size(), info_ + counter + c, this_chunk));
            }
#endif
            counter += group_count_[i];
            continue;
        }
        for (int j = 0; j < group_count_[i]; j++)
        {
            const int st = counter % streams;
            const core::ExecContext sc = pool_.context(core::StreamRole::Medium, st);
            const BlockRef &r = refs_[group_first_[i] + j];
            CUADMM_CHECK_CUSOLVER(cusolverDnXsyevd(sc.cusolver, eig_param_.param, eig_param_.jobz, eig_param_.uplo, n, CUDA_R_64F, mats_ + r.mat_offset, n,
                                                   CUDA_R_64F, W_ + r.w_offset, CUDA_R_64F, ws_ + static_cast<std::size_t>(st) * stream_ws_bytes_, stream_ws_bytes_,
                                                   host_ws_[st].data(), host_ws_[st].size(), info_ + counter));
            counter++;
        }
    }
    // synchronize the streams used for the eigenvalue decomposition of medium matrices
    for (int st = 0; st < streams; st++)
        pool_.synchronize(core::StreamRole::Medium, st);
    check_eig_info(info_, count(), what);
}

void BatchedTier::decompose(const char *what, bool allow_batched, bool warn)
{
    if (count() == 0)
        return;
    if (kind_ == TierKind::Small)
        decompose_small(what, warn);
    else
        decompose_medium(what, allow_batched);
}

void BatchedTier::project(const double *z, double *s)
{
    if (count() == 0)
        return;
    const double one = 1.0, zero = 0.0;
    unpack(z);
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(ctx_.stream)); // the unpack completes before the EVDs (the main stream is the legacy default stream)
    decompose(kind_ == TierKind::Small ? "cusolverDnDsyevjBatched (small matrix)" : "cusolverDnXsyevd (medium matrix)", true, true);
    for (int i = 0; i < group_count(); i++)
    {
        const int n = group_size_[i];
        const int cnt = group_count_[i];
        const BlockRef &first = refs_[group_first_[i]];
        const long long nn = static_cast<long long>(n) * n;
        const double *Q = mats_ + first.mat_offset;
        double *B;
        if (config_.gemm_compat)
        {
            // today's product: tmp = Q max(Lambda, 0), P = tmp Q^T
            B = tmp_ + first.mat_offset;
            scale_columns_batch(Q, B, n, cnt, W_ + first.w_offset, SCALE_RELU, ctx_.stream);
        }
        else
        {
            // B = Q diag(sqrt(max(lambda, 0))) in place, P = B B^T
            B = mats_ + first.mat_offset;
            scale_columns_batch(Q, B, n, cnt, W_ + first.w_offset, SCALE_SQRT_RELU, ctx_.stream);
        }
        const double *right = config_.gemm_compat ? Q : B;
        CUADMM_CHECK_CUBLAS(cublasDgemmStridedBatched(ctx_.cublas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, B, n, nn, right, n, nn, &zero,
                                                      P_ + first.mat_offset, n, nn, cnt));
    }
    dense_to_svec_batch(table_, P_, s, ctx_.stream);
}

std::string BatchedTier::memory_text() const
{
    std::ostringstream os;
    const double mib = 1024.0 * 1024.0;
    os << (kind_ == TierKind::Small ? "small" : "medium") << " tier: " << count() << " block(s) in " << group_count() << " size group(s): matrices "
       << total_nn_ * sizeof(double) / mib << " MiB x " << (config_.gemm_compat ? 3 : 2) << ", eigenvalues " << sum_n_ * sizeof(double) / mib << " MiB";
    if (kind_ == TierKind::Medium)
        os << ", streamed workspace " << stream_ws_bytes_ * std::max(1, pool_.medium_count()) / mib << " MiB, batched workspace " << chunk_ws_bytes_ / mib << " MiB";
    else
    {
        std::size_t ws = 0;
        for (std::size_t b : group_ws_bytes_)
            ws += b;
        os << ", Jacobi workspace " << ws / mib << " MiB";
    }
    return os.str();
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
