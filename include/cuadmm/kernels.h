#ifndef CUADMM_KERNELS_H
#define CUADMM_KERNELS_H

#include "cuadmm/memory.h"

/*
    Dense-dense operations (kernels/dense_dense.cu)
*/


/*
    Dense-scalar operations (kernels/dense_scalar.cu)
*/

// Multiply in place a vector by a scalar:
// vec *= scalar
void dense_vector_mul_scalar(DeviceDenseVector<double>& vec, double scalar, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024);

// Divide in place a vector by a scalar:
// vec *= scalar
void dense_vector_div_scalar(DeviceDenseVector<double>& vec, double scalar, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024);

// Multiply a vector by a scalar and store the result in another vector:
// vec1 = vec2 * scalar
void dense_vector_mul_scalar(DeviceDenseVector<double>& vec1, DeviceDenseVector<double>& vec2, double scalar, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024);

// Set a vector to its positive part coefficient-wise:
// vec = max(vec, 0)
void max_dense_vector_zero(DeviceDenseVector<double>& vec, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024);

// Set a vector to its positive part coefficient-wise and multiply by a mask:
// vec = max(vec, 0) .* mask
void max_dense_vector_zero_mask(DeviceDenseVector<double>& vec, DeviceDenseVector<int>& mask, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024);

#endif