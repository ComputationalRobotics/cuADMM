/*

    kernels/diagonal_batch.cu

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
//
// Arguments:
// - dnmat1: the output dense matrix
// - dnmat2: the input dense matrix
// - dnvec: the input dense vector
// - mat_size: the size of the matrices (all matrices must be of the same size)
// - stream: the CUDA stream to use (default: 0)
// - block_size: the size of the CUDA blocks (default: 1024)
// - mat_nums: the number of matrices to process (default: -1, which means that the number of matrices is inferred from the size of dnmat1)
// - mat_offset: the offset to start processing the matrices (default: 0)
// - vec_offset: the offset to start processing the vectors (default: 0)
void dense_matrix_mul_diag_batch(
    DeviceDenseVector<double>& dnmat1,
    const DeviceDenseVector<double>& dnmat2,
    const DeviceDenseVector<double>& dnvec,
    const int mat_size,
    const int mat_nums, const int mat_offset, const int vec_offset,
    const cudaStream_t& stream, int block_size
) {
    int dnmat_size;
    if (mat_nums == -1)
        dnmat_size = dnmat1.size;
    else
        dnmat_size = mat_nums * mat_size * mat_size;
    int num_block = (dnmat_size + block_size - 1) / block_size;
    dense_matrix_mul_diag_batch_kernel<<<num_block, block_size, 0, stream>>>(
        dnmat1.vals + mat_offset, dnmat2.vals + mat_offset, dnvec.vals + vec_offset,
        mat_size, dnmat_size
    );
    return;
}