/*

    kernels/permutation.cu

    Defines CUDA kernel for performing a permutation on a dense vector.

*/

#include "cuadmm_legacy/kernels.h"

__global__ void perform_permutation_kernel(double *vec1_vals, double *vec2_vals, int *perm_vals, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        vec1_vals[perm_vals[idx]] = vec2_vals[idx];
    }
    return;
}

// Perform a permutation on a dense vector, such that:
// vec1[perm[i]] = vec2[i]
void perform_permutation(
    double *vec1_vals, const double *vec2_vals, const int *perm_vals, int size,
    const cudaStream_t &stream, int block_size)
{
    int num_block = (size + block_size - 1) / block_size;
    perform_permutation_kernel<<<num_block, block_size, 0, stream>>>(
        vec1_vals, const_cast<double *>(vec2_vals), const_cast<int *>(perm_vals), size);
    CHECK_KERNEL_LAUNCH();
    return;
}

void perform_permutation(
    DeviceDenseVector<double> &vec1,
    const DeviceDenseVector<double> &vec2,
    const DeviceDenseVector<int> &perm,
    const cudaStream_t &stream, int block_size)
{
    perform_permutation(vec1.vals, vec2.vals, perm.vals, perm.size, stream, block_size);
    return;
}
