/*

    kernels/dense_dense.cu

    Defines CUDA kernel for operations between two dense vectors.

*/

#include "cuadmm/kernels.h"

/* Kernels for coefficient-wise operations */

// Add in place the current coefficient of a dense vector by a the matching coefficient of another vector (up to some scalar):
// vec1[idx] = alpha * vec1[idx] + beta + vec2[idx]
__global__ void dense_vector_add_dense_vector_kernel(
    double* vec1_vals, double* vec2_vals, 
    int size, double alpha, double beta
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        vec1_vals[idx] = alpha * vec1_vals[idx] + beta * vec2_vals[idx];
    }
    return;
}

// Store in the coefficient of a dense vector the linear combination of the coefficients of two other vectors:
// vec1[idx] = alpha * vec2[idx] + beta * vec3[idx]
__global__ void dense_vector_add_dense_vector_kernel(
    double* vec1_vals, double* vec2_vals, double* vec3_vals, 
    int size, double alpha, double beta
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        vec1_vals[idx] = alpha * vec2_vals[idx] + beta * vec3_vals[idx];
    }
    return;
}

// Multiply in place the current coefficient of a dense vector by a the matching coefficient of another vector:
// vec1[idx] *= vec2[idx]
__global__ void dense_vector_mul_dense_vector_kernel(
    double* vec1_vals, double* vec2_vals, int con_num
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < con_num) {
        vec1_vals[idx] *= vec2_vals[idx];
    }
    return;
}

// Store in the coefficient of a dense vector the product of the coefficients of two other vectors and a scalar:
// vec1[idx] = vec2[idx] * vec3[idx] * scalar
__global__ void dense_vector_mul_dense_vector_mul_scalar_kernel(
    double* vec1_vals, double* vec2_vals, double* vec3_vals, 
    double scalar, int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        vec1_vals[idx] = vec2_vals[idx] * vec3_vals[idx] * scalar;
    }
    return;
}

// Store in the coefficient of a dense vector the division of the coefficients of two other vectors and a scalar:
// vec1[idx] = vec2[idx] / vec3[idx] * scalar
__global__ void dense_vector_div_dense_vector_mul_scalar_kernel(
    double* vec1_vals, double* vec2_vals, double scalar, int con_num
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < con_num) {
        vec1_vals[idx] = vec1_vals[idx] / vec2_vals[idx] * scalar;
    }
    return;
}

// Store in the coefficient of a dense vector the sum of the coefficients of two other vectors times a scalar:
// vec1[idx] = vec2[idx] + vec3[idx] * scalar
__global__ void dense_vector_plus_dense_vector_mul_scalar_kernel(
    double* vec1_vals, double* vec2_vals, double* vec3_vals,
    double scalar, int vec_len
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vec_len) {
        vec1_vals[idx] = vec2_vals[idx] + vec3_vals[idx] * scalar;
    }
    return;
}

/* Kernels for vector-wise operations */

// vec1 = alpha * vec1 + beta * vec2
void dense_vector_add_dense_vector(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2,
    double alpha, double beta, const cudaStream_t& stream, int block_size
) { 
    const int size = vec1.size;
    const int num_block = (size + block_size - 1) / block_size;
    dense_vector_add_dense_vector_kernel<<<num_block, block_size, 0, stream>>>(vec1.vals, vec2.vals, size, alpha, beta);
    return;
}

// vec1 = alpha * vec2 + beta * vec3
void dense_vector_add_dense_vector(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2, DeviceDenseVector<double>& vec3,
    double alpha, double beta, const cudaStream_t& stream, int block_size
) { 
    const int size = vec1.size;
    const int num_block = (size + block_size - 1) / block_size;
    dense_vector_add_dense_vector_kernel<<<num_block, block_size, 0, stream>>>(vec1.vals, vec2.vals, vec3.vals, size, alpha, beta);
    return;
}

// Multiply in place coefficient-wise a vector by another vector:
// vec1 *= vec2
void dense_vector_mul_dense_vector(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2, 
    const cudaStream_t& stream, int block_size
) {
    const int con_num = vec1.size;
    const int num_block = (con_num + block_size - 1) / block_size;
    dense_vector_mul_dense_vector_kernel<<<num_block, block_size, 0, stream>>>(vec1.vals, vec2.vals, con_num);
    return;
}

// vec1 <-- vec2 * vec3 * scalar
void dense_vector_mul_dense_vector_mul_scalar(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2, const DeviceDenseVector<double>& vec3,
    const double scalar, const cudaStream_t& stream, int block_size
) {
    const int size = vec1.size;
    const int num_block = (size + block_size - 1) / block_size;
    dense_vector_mul_dense_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(
        vec1.vals, vec2.vals, vec3.vals, scalar, size
    );
    return;
}

// vec1 <-- vec1 / vec2 * scalar
void dense_vector_div_dense_vector_mul_scalar(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2, 
    const double scalar, const cudaStream_t& stream, int block_size
) {
    const int size = vec1.size;
    const int num_block = (size + block_size - 1) / block_size;
    dense_vector_div_dense_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(
        vec1.vals, vec2.vals, scalar, size
    );
    return;
}

// vec1 <-- vec2 + vec3 * scalar
void dense_vector_plus_dense_vector_mul_scalar(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2, const DeviceDenseVector<double>& vec3,
    const double scalar, const cudaStream_t& stream, int block_size
) {
    const int size = vec1.size;
    const int num_block = (size + block_size - 1) / block_size;
    dense_vector_plus_dense_vector_mul_scalar_kernel<<<num_block, block_size, 0, stream>>>(
        vec1.vals, vec2.vals, vec3.vals, scalar, size
    );
    return;
}