/*

    cone/slice_table.h

    SliceTable: the blocks of one cone as device arrays (offsets and cumulative lengths), so that one kernel launch
    over the cone's entries finds its vector index with a binary search over the blocks (Phase 8 makes the slice
    contiguous and the table trivial). Host copies of the block entries for the measures.

*/

#ifndef CUADMM_CONE_SLICE_TABLE_H
#define CUADMM_CONE_SLICE_TABLE_H

#include <cstdint>
#include <vector>

#include "cone/cone.h"
#include "core/device_buffer.h"

namespace cuadmm
{
namespace cone
{

class SliceTable
{
public:
    SliceTable() = default;
    explicit SliceTable(const std::vector<BlockSlice> &blocks);
    std::int64_t total() const { return total_; }
    int count() const { return count_; }
    const std::int64_t *d_offset() const { return offset_.data(); }
    const std::int64_t *d_prefix() const { return prefix_.data(); }
    /// Copies the entries of every block of v (device) into one host vector in block order (length total()).
    std::vector<double> gather_host(const double *v, cudaStream_t stream) const;
    const std::vector<BlockSlice> &blocks() const { return blocks_; }

private:
    std::vector<BlockSlice> blocks_;
    std::int64_t total_ = 0;
    int count_ = 0;
    core::DeviceBuffer<std::int64_t> offset_; // first entry of each block in the vector
    core::DeviceBuffer<std::int64_t> prefix_; // cumulative lengths, count + 1 entries
};

/// s[idx] = f(z[idx]) on every entry of the table; f is a device functor.
void slice_apply(const SliceTable &table, const double *z, double *s, int op, cudaStream_t stream);
enum SliceOp { SLICE_ZERO = 0, SLICE_RELU = 1, SLICE_COPY = 2 };

} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_SLICE_TABLE_H
