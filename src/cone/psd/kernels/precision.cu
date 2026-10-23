/*

    cone/psd/kernels/precision.cu

*/

#include "cone/psd/kernels/precision.h"

#include "core/check.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

void symmetrize_float(
    cublasHandle_t cublasH, float* M, int n, float* workspace
) {
    const float one = 1.0, half = 0.5, zero = 0.0;

    // workspace = M^T
    CUADMM_CHECK_CUBLAS(cublasSgeam(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
        n, n,
        &one, M, n,
        &zero, M, n,
        workspace, n
    ));

    // M = M + workspace (which is M^T)
    CUADMM_CHECK_CUBLAS(cublasSgeam(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one, M, n,
        &one, workspace, n,
        M, n
    ));

    // M = 0.5 * M
    CUADMM_CHECK_CUBLAS(cublasSscal(cublasH, n * n, &half, M, 1));
}

__global__ void convert_double_to_float_kernel(const double* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<float>(in[idx]);
    }
}

__global__ void convert_float_to_double_kernel(const float* in, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<double>(in[idx]);
    }
}

void convert_double_to_float(const double* d_in, float* d_out, int n, int threads_per_block) {
    const int blocksPerGrid = (n + threads_per_block - 1) / threads_per_block;

    convert_double_to_float_kernel<<<blocksPerGrid, threads_per_block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
}

void convert_float_to_double(const float* d_in, double* d_out, int n, int threads_per_block) {
    const int blocksPerGrid = (n + threads_per_block - 1) / threads_per_block;

    convert_float_to_double_kernel<<<blocksPerGrid, threads_per_block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
}

__global__ void add_identity_kernel(float* mat, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        if (row == col) {
            mat[idx] += 1.0f; // Add 1 to the diagonal elements
        }
    }
}

void add_identity(
    cublasHandle_t cublasH,
    float* mat,
    int n
) {
    int threads_per_block = 1024;
    const int blocksPerGrid = (n * n + threads_per_block - 1) / threads_per_block;

    // Launch kernel to add identity matrix
    add_identity_kernel<<<blocksPerGrid, threads_per_block>>>(mat, n);
    CUADMM_CHECK_CUDA(cudaGetLastError());
}

__global__ void float4_to_half_kernel(
    const float4* __restrict__ A4,
    __half2 * __restrict__ B2,
    size_t N4
) {
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N4) return;

    // load 4 floats
    float4 v = A4[idx];

    // pack low two floats into half2
    B2[2*idx + 0] = __float22half2_rn(make_float2(v.x, v.y));
    // pack high two floats into half2
    B2[2*idx + 1] = __float22half2_rn(make_float2(v.z, v.w));
}

void convert_float_to_half4(const float* dA, __half* dB, size_t N) {
    size_t N4 = (N + 3)/4;  // how many float4’s
    auto A4 = reinterpret_cast<const float4*>(dA);
    auto B2 = reinterpret_cast<__half2*>(dB);

    const int blk = 1024;
    int grid = (N4 + blk - 1)/blk;
    float4_to_half_kernel<<<grid,blk>>>(A4, B2, N4);
}

__global__ void reverse_vector_kernel(const double* in, double* out, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        out[idx] = in[m - 1 - idx];
    }
}

void reverse_vector(const double* in, double* out, int m) {
    int threads = 1024;
    int blocks = (m + threads - 1) / threads;
    reverse_vector_kernel<<<blocks, threads>>>(in, out, m);
    CUADMM_CHECK_CUDA(cudaGetLastError());
}

__global__ void reverse_columns_kernel(const double* in, double* out, int n, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < m) {
        // Copy column col to column (m - 1 - col)
        out[row + (m - 1 - col) * n] = in[row + col * n];
    }
}

void reverse_columns(const double* in, double* out, int n, int m) {
    dim3 threads(32, 32);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    reverse_columns_kernel<<<blocks, threads>>>(in, out, n, m);
    CUADMM_CHECK_CUDA(cudaGetLastError());
}

__global__ void extract_columns_kernel(const double* in, double* out, int n, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < m) {
        out[row + col * n] = in[row + col * n];
    }
}

/// @brief Extract the first m columns from an n x k matrix (k >= m)
/// @param in Input matrix of size n x k (column-major)
/// @param out Output matrix of size n x m (column-major)
/// @param n Number of rows in the input matrix
/// @param m Number of columns to extract
void extract_columns(const double* in, double* out, int n, int m) {
    dim3 threads(32, 32);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    extract_columns_kernel<<<blocks, threads>>>(in, out, n, m);
    CUADMM_CHECK_CUDA(cudaGetLastError());
}

namespace
{
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long z)
{
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__global__ void fill_deterministic_kernel(double *v, int n, unsigned long long seed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const unsigned long long r = splitmix64(seed ^ (0xD1B54A32D192ED03ULL * (unsigned long long)(i + 1)));
        v[i] = ((double)(r >> 11) + 1.0) * (1.0 / 9007199254740992.0); // (0, 1]
    }
}
} // namespace

void fill_deterministic(double *v, int n, unsigned long long seed, int threads_per_block)
{
    if (n <= 0)
        return;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    fill_deterministic_kernel<<<blocks, threads_per_block>>>(v, n, seed);
    CUADMM_CHECK_LAUNCH();
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
