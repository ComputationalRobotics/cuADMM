/*

    kernels/sparse_scalar.cu

    Defines CUDA kernel for operations between a sparse vector (cuSPARSE) and a scalar.

*/

#include "cuadmm_legacy/memory.h"

#define BLOCK_LIMIT 1024

// vec[idx] *= scalar
__global__ void sparse_vector_mul_scalar_kernel(
    double *vec_vals, int vec_nnz, double scalar)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vec_nnz)
    {
        vec_vals[idx] *= scalar;
    }
    return;
}

// vec *= scalar
void sparse_vector_mul_scalar(
    DeviceSparseVector<double> &vec, double scalar,
    const cudaStream_t &stream, int block_size)
{
    // An empty vector (a problem with C = 0 or b = 0, e.g. the PLATO instance cnhil10) has nothing to scale: the
    // zero-size launch that 1.x issued (and silently ignored) is rejected by the launch checks of 2.0.
    if (vec.nnz == 0)
        return;
    int num_block;
    if (vec.nnz < BLOCK_LIMIT)
    {
        block_size = vec.nnz;
        num_block = 1;
    }
    else
    {
        num_block = (vec.nnz + block_size - 1) / block_size;
    }
    sparse_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(vec.vals, vec.nnz, scalar);
    CHECK_KERNEL_LAUNCH();
    return;
}

// vec /= scalar
void sparse_vector_div_scalar(
    DeviceSparseVector<double> &vec, double scalar,
    const cudaStream_t &stream, int block_size)
{
    if (vec.nnz == 0)
        return; // see sparse_vector_mul_scalar
    int num_block;
    if (vec.nnz < BLOCK_LIMIT)
    {
        block_size = vec.nnz;
        num_block = 1;
    }
    else
    {
        num_block = (vec.nnz + block_size - 1) / block_size;
    }
    sparse_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(vec.vals, vec.nnz, 1 / scalar);
    CHECK_KERNEL_LAUNCH();
    return;
}