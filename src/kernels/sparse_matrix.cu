/*

    sparse_matrix.cu

    Computes the norm of a CSC matrix using CUDA.

*/

#include "cuadmm/kernels.h"

__global__ void get_normA_kernel(
    int* At_col_ptrs, double* At_vals,
    double* normA_vals, 
    int con_num
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < con_num) {
        double norm = 0.0;
        // sum the square of the values of the column
        for (int i = At_col_ptrs[idx]; i < At_col_ptrs[idx+1]; i++) {
            norm += (At_vals[i] * At_vals[i]);
        }
        norm = max(1.0, sqrt(norm));
        normA_vals[idx] = norm;
        // normalize the values of the column
        for (int i = At_col_ptrs[idx]; i < At_col_ptrs[idx+1]; i++) {
            At_vals[i] /= norm;
        }
    }
    return;
}

// Compute the norm (of size (con_num, 1)) of a CSC matrix At and normalize it.
void get_normA(
    DeviceSparseMatrixDoubleCSC& At, DeviceDenseVector<double>& normA,
    const cudaStream_t& stream, int block_size
) {
    const int con_num = At.col_size;
    const int num_block = (con_num + block_size - 1) / block_size;
    get_normA_kernel<<<num_block, block_size, 0, stream>>>(
        At.col_ptrs, At.vals, normA.vals, con_num
    );
    return;
}
