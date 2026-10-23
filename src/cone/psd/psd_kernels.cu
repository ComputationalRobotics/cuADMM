/*

    cone/psd/psd_kernels.cu

*/

#include "cone/psd/psd_kernels.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "core/check.h"
#include "cuadmm_legacy/kernels.h" // SQRT2, SQRT2INV

namespace cuadmm
{
namespace cone
{
namespace psd
{

namespace
{
constexpr double kSqrt2 = SQRT2;       // the constants of main (kernels.h): SQRT2 from its Newton iteration and
constexpr double kSqrt2Inv = SQRT2INV; // SQRT2INV = 1.0 / SQRT2 (one ulp below the nearest double of 1 / sqrt 2)
constexpr int kBlock = 256;

// (r, c), r <= c, of the svec index l within a block: c (c + 1) / 2 + r = l
__device__ inline void svec_rc(long long l, int &r, int &c)
{
    long long cc = (long long)floor((sqrt(8.0 * (double)l + 1.0) - 1.0) * 0.5);
    while (cc * (cc + 1) / 2 > l)
        cc--;
    while ((cc + 1) * (cc + 2) / 2 <= l)
        cc++;
    c = (int)cc;
    r = (int)(l - cc * (cc + 1) / 2);
}

__global__ void svec_to_dense_kernel(const double *z, int n, double *M, long long len)
{
    const long long l = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= len)
        return;
    int r, c;
    svec_rc(l, r, c);
    const double v = z[l];
    if (r == c)
        M[(long long)r + (long long)c * n] = v;
    else
    {
        const double x = v * kSqrt2Inv;
        M[(long long)r + (long long)c * n] = x;
        M[(long long)c + (long long)r * n] = x;
    }
}

__global__ void dense_to_svec_kernel(const double *P, int n, double *s, const double *z_add, long long len)
{
    const long long l = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= len)
        return;
    int r, c;
    svec_rc(l, r, c);
    const double p = P[(long long)r + (long long)c * n]; // the upper triangle (row r <= column c), main's map_M1
    const double v = (r == c) ? p : p * kSqrt2;
    s[l] = (z_add != nullptr) ? z_add[l] + v : v;
}

// the block of a global svec index: the last b with prefix[b] <= idx (prefix has count + 1 entries)
__device__ inline int find_block(const long long *prefix, int count, long long idx)
{
    int lo = 0, hi = count; // invariant: prefix[lo] <= idx < prefix[hi]
    while (hi - lo > 1)
    {
        const int mid = (lo + hi) / 2;
        if (prefix[mid] <= idx)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

__global__ void svec_to_dense_batch_kernel(const long long *svec_offset, const long long *mat_offset, const long long *prefix,
                                           const int *size, int count, long long total, const double *z, double *mats)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int b = find_block(prefix, count, idx);
    const long long l = idx - prefix[b];
    const int n = size[b];
    int r, c;
    svec_rc(l, r, c);
    const double v = z[svec_offset[b] + l];
    double *M = mats + mat_offset[b];
    if (r == c)
        M[(long long)r + (long long)c * n] = v;
    else
    {
        const double x = v * kSqrt2Inv;
        M[(long long)r + (long long)c * n] = x;
        M[(long long)c + (long long)r * n] = x;
    }
}

__global__ void dense_to_svec_batch_kernel(const long long *svec_offset, const long long *mat_offset, const long long *prefix,
                                           const int *size, int count, long long total, const double *mats, double *s)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int b = find_block(prefix, count, idx);
    const long long l = idx - prefix[b];
    const int n = size[b];
    int r, c;
    svec_rc(l, r, c);
    const double *P = mats + mat_offset[b];
    const double p = P[(long long)r + (long long)c * n]; // the upper triangle, main's map_M1
    s[svec_offset[b] + l] = (r == c) ? p : p * kSqrt2;
}

__device__ inline double scale_factor(double w, int mode)
{
    if (mode == SCALE_RELU)
        return max(w, 0.0);
    if (mode == SCALE_SQRT_RELU)
        return sqrt(max(w, 0.0));
    return sqrt(max(-w, 0.0));
}

__global__ void scale_columns_kernel(const double *Q, double *B, int n, int first, int count, const double *w, int mode)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long len = (long long)n * count;
    if (idx >= len)
        return;
    const long long j = first + idx / n;
    const long long e = (long long)first * n + idx;
    B[e] = Q[e] * scale_factor(w[j], mode);
}

__global__ void scale_columns_batch_kernel(const double *Q, double *B, int n, int count, const double *w, int mode)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long nn = (long long)n * n;
    const long long len = nn * count;
    if (idx >= len)
        return;
    const long long k = idx / nn;
    const long long j = (idx % nn) / n;
    B[idx] = Q[idx] * scale_factor(w[k * n + j], mode);
}

unsigned grid_for(long long len)
{
    return static_cast<unsigned>((len + kBlock - 1) / kBlock);
}
} // namespace

BlockTable::BlockTable(const std::vector<BlockRef> &refs) : refs_(refs), count_(static_cast<int>(refs.size()))
{
    std::vector<long long> so(count_), mo(count_), prefix(count_ + 1, 0);
    std::vector<int> sz(count_);
    for (int b = 0; b < count_; b++)
    {
        so[b] = refs[b].svec_offset;
        mo[b] = refs[b].mat_offset;
        sz[b] = refs[b].n;
        prefix[b + 1] = prefix[b] + (long long)refs[b].n * (refs[b].n + 1) / 2;
    }
    total_ = prefix[count_];
    if (count_ == 0)
        return;
    svec_offset_.allocate(count_);
    mat_offset_.allocate(count_);
    size_.allocate(count_);
    prefix_.allocate(count_ + 1);
    CUADMM_CHECK_CUDA(cudaMemcpy(svec_offset_.data(), so.data(), sizeof(long long) * count_, cudaMemcpyHostToDevice));
    CUADMM_CHECK_CUDA(cudaMemcpy(mat_offset_.data(), mo.data(), sizeof(long long) * count_, cudaMemcpyHostToDevice));
    CUADMM_CHECK_CUDA(cudaMemcpy(size_.data(), sz.data(), sizeof(int) * count_, cudaMemcpyHostToDevice));
    CUADMM_CHECK_CUDA(cudaMemcpy(prefix_.data(), prefix.data(), sizeof(long long) * (count_ + 1), cudaMemcpyHostToDevice));
}

void svec_to_dense(const double *z_block, int n, double *M, cudaStream_t stream)
{
    const long long len = (long long)n * (n + 1) / 2;
    if (len <= 0)
        return;
    svec_to_dense_kernel<<<grid_for(len), kBlock, 0, stream>>>(z_block, n, M, len);
    CUADMM_CHECK_LAUNCH();
}

void dense_to_svec(const double *P, int n, double *s_block, const double *z_add, cudaStream_t stream)
{
    const long long len = (long long)n * (n + 1) / 2;
    if (len <= 0)
        return;
    dense_to_svec_kernel<<<grid_for(len), kBlock, 0, stream>>>(P, n, s_block, z_add, len);
    CUADMM_CHECK_LAUNCH();
}

void svec_to_dense_batch(const BlockTable &t, const double *z, double *mats, cudaStream_t stream)
{
    if (t.total() <= 0)
        return;
    svec_to_dense_batch_kernel<<<grid_for(t.total()), kBlock, 0, stream>>>(t.d_svec_offset(), t.d_mat_offset(), t.d_prefix(), t.d_size(),
                                                                          t.count(), t.total(), z, mats);
    CUADMM_CHECK_LAUNCH();
}

void dense_to_svec_batch(const BlockTable &t, const double *mats, double *s, cudaStream_t stream)
{
    if (t.total() <= 0)
        return;
    dense_to_svec_batch_kernel<<<grid_for(t.total()), kBlock, 0, stream>>>(t.d_svec_offset(), t.d_mat_offset(), t.d_prefix(), t.d_size(),
                                                                          t.count(), t.total(), mats, s);
    CUADMM_CHECK_LAUNCH();
}

void scale_columns(const double *Q, double *B, int n, int first, int count, const double *w, int mode, cudaStream_t stream)
{
    if (count <= 0 || n <= 0)
        return;
    scale_columns_kernel<<<grid_for((long long)n * count), kBlock, 0, stream>>>(Q, B, n, first, count, w, mode);
    CUADMM_CHECK_LAUNCH();
}

void scale_columns_batch(const double *Q, double *B, int n, int count, const double *w, int mode, cudaStream_t stream)
{
    if (count <= 0 || n <= 0)
        return;
    scale_columns_batch_kernel<<<grid_for((long long)n * n * count), kBlock, 0, stream>>>(Q, B, n, count, w, mode);
    CUADMM_CHECK_LAUNCH();
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
