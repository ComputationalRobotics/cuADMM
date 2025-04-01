#include "cuadmm/kernels.h"

__global__ void perform_permutation_kernel(double* vec1_vals, double* vec2_vals, int* perm_vals, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        vec1_vals[perm_vals[idx]] = vec2_vals[idx];
    }
    return;
}

// Perform a permutation on a dense vector, such that:
// vec1[perm[i]] = vec2[i]
void perform_permutation(
    DeviceDenseVector<double>& vec1,
    const DeviceDenseVector<double>& vec2,
    const DeviceDenseVector<int>& perm,
    const cudaStream_t& stream, int block_size
) {
    int num_block = (perm.size + block_size - 1) / block_size;
    perform_permutation_kernel<<<num_block, block_size, 0, stream>>>(
        vec1.vals, vec2.vals, perm.vals, perm.size
    );
    return;
}