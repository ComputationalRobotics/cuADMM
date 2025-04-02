/*

    kernels/sparse_scalar.cu

    Defines CUDA kernel for operations between a sparse vector (cuSPARSE) and a scalar.

*/

#include "cuadmm/memory.h"

#define BLOCK_LIMIT 1024

// vec[idx] *= scalar
__global__ void sparse_vector_mul_scalar_kernel(
    double* vec_vals, int vec_nnz, double scalar
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vec_nnz) {
        vec_vals[idx] *= scalar;
    }
    return;
}

// vec *= scalar
void sparse_vector_mul_scalar(
    DeviceSparseVector<double>& vec, double scalar,
    const cudaStream_t& stream, int block_size
) {
    int num_block;
    if (vec.nnz < BLOCK_LIMIT) {
        block_size = vec.nnz;
        num_block = 1;
    } else {
        num_block = (vec.nnz + block_size - 1) / block_size;
    }
    sparse_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(vec.vals, vec.nnz, scalar);
    return;
}

// vec /= scalar
void sparse_vector_div_scalar(
    DeviceSparseVector<double>& vec, double scalar,
    const cudaStream_t& stream, int block_size
) {
    int num_block;
    if (vec.nnz < BLOCK_LIMIT) {
        block_size = vec.nnz;
        num_block = 1;
    } else {
        num_block = (vec.nnz + block_size - 1) / block_size;
    }
    sparse_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(vec.vals, vec.nnz, 1/scalar);
    return;
}