/*

    cone/slice_table.cu

*/

#include "cone/slice_table.h"

#include <stdexcept>

#include "core/check.h"

namespace cuadmm
{
namespace cone
{

SliceTable::SliceTable(const std::vector<BlockSlice> &blocks) : blocks_(blocks)
{
    count_ = static_cast<int>(blocks.size());
    std::vector<std::int64_t> offset(count_), prefix(count_ + 1, 0);
    for (int b = 0; b < count_; b++)
    {
        offset[b] = blocks[b].offset;
        prefix[b + 1] = prefix[b] + blocks[b].length;
    }
    total_ = prefix[count_];
    if (count_ > 0)
    {
        offset_ = core::DeviceBuffer<std::int64_t>(count_);
        prefix_ = core::DeviceBuffer<std::int64_t>(count_ + 1);
        core::to_device(offset_.span(), offset);
        core::to_device(prefix_.span(), prefix);
    }
}

std::vector<double> SliceTable::gather_host(const double *v, cudaStream_t stream) const
{
    std::vector<double> out(total_);
    std::int64_t pos = 0;
    for (const BlockSlice &b : blocks_)
    {
        CUADMM_CHECK_CUDA(cudaMemcpyAsync(out.data() + pos, v + b.offset, sizeof(double) * b.length, cudaMemcpyDeviceToHost, stream));
        pos += b.length;
    }
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(stream));
    return out;
}

namespace
{
__global__ void slice_apply_kernel(const std::int64_t *offset, const std::int64_t *prefix, int count, std::int64_t total,
                                   const double *z, double *s, int op)
{
    const std::int64_t i = blockIdx.x * (std::int64_t)blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    // the block of entry i: prefix[b] <= i < prefix[b + 1]
    int lo = 0, hi = count - 1;
    while (lo < hi)
    {
        const int mid = (lo + hi + 1) / 2;
        if (prefix[mid] <= i)
            lo = mid;
        else
            hi = mid - 1;
    }
    const std::int64_t idx = offset[lo] + (i - prefix[lo]);
    if (op == SLICE_ZERO)
        s[idx] = 0.0;
    else if (op == SLICE_RELU)
    {
        const double v = z[idx];
        s[idx] = (v > 0.0) ? v : 0.0; // as today's conversion kernel
    }
    else
        s[idx] = z[idx];
}
} // namespace

void slice_apply(const SliceTable &table, const double *z, double *s, int op, cudaStream_t stream)
{
    if (table.total() == 0)
        return;
    const int block = 1024;
    const std::int64_t grid = (table.total() + block - 1) / block;
    if (grid > 2147483647LL)
        throw std::runtime_error("slice_apply: too many entries for one launch");
    slice_apply_kernel<<<static_cast<unsigned>(grid), block, 0, stream>>>(table.d_offset(), table.d_prefix(), table.count(), table.total(),
                                                                          z, s, op);
    CUADMM_CHECK_LAUNCH();
}

} // namespace cone
} // namespace cuadmm
