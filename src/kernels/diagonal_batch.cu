/*

    diagonal_batch.cu

    Computes the multiplication of a dense matrix by a batch of dense vectors as diagonal matrices.

*/

#include "cuadmm/kernels.h"

__global__ void dense_matrix_mul_diag_batch_kernel(
    double* dnmat1_vals, double* dnmat2_vals, double* dnvec_vals, 
    int mat_size, int total_len
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int k, i;
    if (idx < total_len) {
        k = idx / (mat_size * mat_size);
        i = (idx % (mat_size * mat_size)) / mat_size;
        dnmat1_vals[idx] = dnmat2_vals[idx] * dnvec_vals[k * mat_size + i];
    }
    return;
}

// Computes the multiplication of a dense matrix by a batch of dense vectors as diagonal matrices:
// mat1[i] = mat2[i] * diag(vec[i])
// shapes:
// - mat1: mat_size * mat_size
// - mat2: mat_size * mat_size
// - vec:  mat_size
// This can also be seen as multiplying the i-th column of mat2 by the i-th element of vec.
void dense_matrix_mul_diag_batch(
    DeviceDenseVector<double>& dnmat1,
    const DeviceDenseVector<double>& dnmat2,
    const DeviceDenseVector<double>& dnvec,
    const int mat_size,
    const cudaStream_t& stream, int block_size
) {
    int num_block = (dnmat1.size + block_size - 1) / block_size;
    dense_matrix_mul_diag_batch_kernel<<<num_block, block_size, 0, stream>>>(
        dnmat1.vals, dnmat2.vals, dnvec.vals,
        mat_size, dnmat1.size
    );
    return;
}