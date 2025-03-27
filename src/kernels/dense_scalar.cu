/*

    dense_scalar.cu

    Defines CUDA kernel for operations between a dense vector and a scalar.

*/

#include "cuadmm/memory.h"

/* Kernels for coefficient-wise operations */

// Multiply in place the current coefficient of a dense vector by a scalar:
// vec[idx] *= scalar
__global__ void dense_vector_mul_scalar_kernel(
    double* vec_vals, int vec_size,
    double scalar
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vec_size) {
        vec_vals[idx] *= scalar; 
    }
    return;
}

// Multiply the current coefficient of a dense vector by a scalar and store the result in another vector:
// vec1[idx] = vec2[idx] * scalar
__global__ void dense_vector_mul_scalar_kernel(
    double* vec1_vals, double* vec2_vals, int vec_size,
    double scalar
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vec_size) {
        vec1_vals[idx] = vec2_vals[idx] * scalar; 
    }
    return;
}

// Set the current coefficient of a dense vector to its positive part:
// vec[idx] = max(vec[idx], 0)
__global__ void max_dense_vector_zero_kernel(double* vec_vals, int vec_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vec_size) {
        vec_vals[idx] = max(vec_vals[idx], 0.0);
    }
    return;
}

// Set the current coefficient of a dense vector to its positive part and multiply by a mask:
// vec[idx] = max(vec[idx], 0) * mask[idx]
__global__ void max_dense_vector_zero_mask_kernel(double* vec_vals, int* mask_vals, int vec_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vec_size) {
        vec_vals[idx] = max(vec_vals[idx], 0.0) * mask_vals[idx];
    }
    return;
}


/* Kernels for vector-wise operations */

// Multiply in place a vector by a scalar:
// vec *= scalar
void dense_vector_mul_scalar(DeviceDenseVector<double>& vec, double scalar, int block_size) {
    int num_block = (vec.size + block_size - 1) / block_size;
    dense_vector_mul_scalar_kernel<<<num_block, block_size>>>(vec.vals, vec.size, scalar);
    return;
}
void dense_vector_mul_scalar(DeviceDenseVector<double>& vec, double scalar, const cudaStream_t& stream, int block_size) {
    int num_block = (vec.size + block_size - 1) / block_size;
    dense_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(vec.vals, vec.size, scalar);
    return;
}

// Divide in place a vector by a scalar:
// vec *= scalar
void dense_vector_div_scalar(DeviceDenseVector<double>& vec, double scalar, const cudaStream_t& stream, int block_size) {
    int num_block = (vec.size + block_size - 1) / block_size;
    dense_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(vec.vals, vec.size, 1/scalar);
    return;
}

// Multiply a vector by a scalar and store the result in another vector:
// vec1 = vec2 * scalar
void dense_vector_mul_scalar(DeviceDenseVector<double>& vec1, DeviceDenseVector<double>& vec2, double scalar, const cudaStream_t& stream, int block_size) {
    int num_block = (vec1.size + block_size - 1) / block_size;
    dense_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(vec1.vals, vec2.vals, vec1.size, scalar);
    return;
}

// Set a vector to its positive part coefficient-wise:
// vec = max(vec, 0)
void max_dense_vector_zero(DeviceDenseVector<double>& vec, const cudaStream_t& stream, int block_size) {
    int num_block = (vec.size + block_size - 1) / block_size;
    max_dense_vector_zero_kernel<<<num_block, block_size, 0, stream>>>(vec.vals, vec.size);
    return;
}

// Set a vector to its positive part coefficient-wise and multiply by a mask:
// vec = max(vec, 0) .* mask
void max_dense_vector_zero_mask(DeviceDenseVector<double>& vec, DeviceDenseVector<int>& mask, const cudaStream_t& stream, int block_size) {
    int num_block = (vec.size + block_size - 1) / block_size;
    max_dense_vector_zero_mask_kernel<<<num_block, block_size, 0, stream>>>(vec.vals, mask.vals, vec.size);
    return;
}