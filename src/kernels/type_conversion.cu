/*

    kernels/type_conversion.cu

    Type conversion kernels for CUDA. Converts between different data types for multiple representations of vectors.

*/

#include "cuadmm/kernels.h"

__global__ void long_int_to_int_kernel(
    int* vec_int_vals, const size_t* vec_long_int_vals, const int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        vec_int_vals[idx] = static_cast<int>(vec_long_int_vals[idx]);
    }
    return;
}

__global__ void int_to_long_int_kernel(
    size_t* vec_long_int_vals, const int* vec_int_vals, const size_t size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        vec_long_int_vals[idx] = static_cast<size_t>(vec_int_vals[idx]);
    }
    return;
}

__global__ void double_to_float_kernel(
    float* vec_float_vals, const double* vec_double_vals, const int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        vec_float_vals[idx] = static_cast<float>(vec_double_vals[idx]);
    }
    return;
}

__global__ void float_to_double_kernel(
    double* vec_double_vals, const float* vec_float_vals, const int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        vec_double_vals[idx] = static_cast<double>(vec_float_vals[idx]);
    }
    return;
}

// Convert a dense vector of type int to a dense vector of type long int.
void long_int_to_int(
    DeviceDenseVector<int>& vec_int, const DeviceDenseVector<size_t>& vec_long_int,
    const cudaStream_t& stream, int block_size
) {
    int num_block;
    if (vec_int.size < 1024) {
        block_size = vec_int.size;
        num_block = 1;
    } else {
        num_block = (vec_int.size + block_size - 1) / block_size;
    }
    long_int_to_int_kernel<<<num_block, block_size, 0, stream>>>(
        vec_int.vals, vec_long_int.vals, vec_int.size
    );
    return;
}

// Convert a dense vector of type long int to a dense vector of type int.
void int_to_long_int(
    DeviceDenseVector<size_t>& vec_long_int, const DeviceDenseVector<int>& vec_int,
    const cudaStream_t& stream, int block_size
) {
    int num_block;
    if (vec_int.size < 1024) {
        block_size = vec_int.size;
        num_block = 1;
    } else {
        num_block = (vec_int.size + block_size - 1) / block_size;
    }
    int_to_long_int_kernel<<<num_block, block_size, 0, stream>>>(
        vec_long_int.vals, vec_int.vals, vec_int.size
    );
    return;
}

// Convert a dense vector of type double to a dense vector of type float.
void double_to_float(
    DeviceDenseVector<float>& vec_float, const DeviceDenseVector<double>& vec_double,
    const cudaStream_t& stream, int block_size
) {
    int num_block;
    if (vec_float.size < 1024) {
        block_size = vec_float.size;
        num_block = 1;
    } else {
        num_block = (vec_float.size + block_size - 1) / block_size;
    }
    double_to_float_kernel<<<num_block, block_size, 0, stream>>>(
        vec_float.vals, vec_double.vals, vec_float.size
    );
    return;
}

// Convert a dense vector of type float to a dense vector of type double.
void float_to_double(
    DeviceDenseVector<double>& vec_double, const DeviceDenseVector<float>& vec_float,
    const cudaStream_t& stream, int block_size
) {
    int num_block;
    if (vec_float.size < 1024) {
        block_size = vec_float.size;
        num_block = 1;
    } else {
        num_block = (vec_float.size + block_size - 1) / block_size;
    }
    float_to_double_kernel<<<num_block, block_size, 0, stream>>>(
        vec_double.vals, vec_float.vals, vec_float.size
    );
    return;
}