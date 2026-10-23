/*

    cone/psd/psd_kernels.h

    The device kernels of the PSD tiers (Phase 6b): the svec <-> dense conversions of one block (the large tier,
    one block at a time through the arena) and of a table of blocks (the medium and small tiers, resident
    batches), and the column scalings of the final products.

    svec convention (the vectorization of main): the upper triangle by columns, entry (r, c) with r <= c at
    c (c + 1) / 2 + r, off-diagonal entries times sqrt 2. The dense matrices are column-major with both
    triangles filled on unpacking; packing reads the upper triangle (row <= column), as main's map_M1 did, so
    the SYRK products write the upper triangle (a GEMM result is symmetric only up to rounding). The arithmetic is main's: z * (1 / sqrt 2) off the diagonal on the way in, P * sqrt 2 on
    the way out, so the conversions are bit for bit the engine's.

*/

#ifndef CUADMM_CONE_PSD_PSD_KERNELS_H
#define CUADMM_CONE_PSD_PSD_KERNELS_H

#include <vector>

#include <cuda_runtime_api.h>

#include "core/device_buffer.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

/// One block of a tier: its svec slice, its size, and its offsets in the tier's dense and eigenvalue buffers.
struct BlockRef
{
    long long svec_offset = 0;
    int n = 0;
    long long mat_offset = 0;
    long long w_offset = 0;
};

/// The blocks of a tier as device arrays, with the prefix of their svec lengths (binary search in the kernels).
class BlockTable
{
public:
    BlockTable() = default;
    explicit BlockTable(const std::vector<BlockRef> &refs);
    int count() const { return count_; }
    long long total() const { return total_; } // svec entries of every block
    const std::vector<BlockRef> &refs() const { return refs_; }
    const long long *d_svec_offset() const { return svec_offset_.data(); }
    const long long *d_mat_offset() const { return mat_offset_.data(); }
    const long long *d_prefix() const { return prefix_.data(); }
    const int *d_size() const { return size_.data(); }

private:
    std::vector<BlockRef> refs_;
    int count_ = 0;
    long long total_ = 0;
    core::DeviceBuffer<long long> svec_offset_, mat_offset_, prefix_;
    core::DeviceBuffer<int> size_;
};

/// The dense n x n matrix (both triangles) of one block from its svec slice.
void svec_to_dense(const double *z_block, int n, double *M, cudaStream_t stream);
/// The svec slice of one block from the upper triangle of a dense n x n matrix; with z_add (the block's svec
/// slice of the input), s = z_add + svec(P).
void dense_to_svec(const double *P, int n, double *s_block, const double *z_add, cudaStream_t stream);
/// The same for every block of a table (z and s are the full svec vectors, mats the tier's dense buffer).
void svec_to_dense_batch(const BlockTable &table, const double *z, double *mats, cudaStream_t stream);
void dense_to_svec_batch(const BlockTable &table, const double *mats, double *s, cudaStream_t stream);

enum ScaleMode
{
    SCALE_RELU = 0,      // B[:, j] = Q[:, j] * max(w_j, 0)      (today's product Q max(Lambda, 0))
    SCALE_SQRT_RELU = 1, // B[:, j] = Q[:, j] * sqrt(max(w_j, 0)) (the SYRK side of the positive eigenvalues)
    SCALE_SQRT_NEG = 2   // B[:, j] = Q[:, j] * sqrt(max(-w_j, 0)) (the SYRK side of the negative eigenvalues)
};
/// Columns [first, first + count) of Q (n rows, column-major) scaled into B (B may be Q: in place).
void scale_columns(const double *Q, double *B, int n, int first, int count, const double *w, int mode, cudaStream_t stream);
/// All columns of `count` matrices of size n (stride n * n; w: stride n).
void scale_columns_batch(const double *Q, double *B, int n, int count, const double *w, int mode, cudaStream_t stream);

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_PSD_KERNELS_H
