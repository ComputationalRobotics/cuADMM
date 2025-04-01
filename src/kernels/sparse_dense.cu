/*

    sparse_dense.cu

    Defines CUDA kernel for operations between a sparse vector (cuSPARSE) and a dense vector.

*/

#include "cuadmm/memory.h"

__global__ void sparse_vector_div_dense_vector_kernel(
    int* spvec_indices, double* spvec_vals, int spvec_nnz,
    double* dnvec_vals
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < spvec_nnz) {
        spvec_vals[idx] /= dnvec_vals[spvec_indices[idx]];
    }
    return;
}

// Divide a sparse vector by a dense vector, element-wise and in-place
// sp_vec <-- sp_vec / dn_vec
void sparse_vector_div_dense_vector(
    DeviceSparseVector<double>& spvec, const DeviceDenseVector<double>& dnvec,
    const cudaStream_t& stream, int block_size
) {
    int num_block;
    if (spvec.nnz < 1024) {
        block_size = spvec.nnz;
        num_block = 1;
    } else {
        num_block = (spvec.nnz + block_size - 1) / block_size;
    }
    sparse_vector_div_dense_vector_kernel<<<num_block, block_size, 0, stream>>>(spvec.indices, spvec.vals, spvec.nnz, dnvec.vals);
    return;
}