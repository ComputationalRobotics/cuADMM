#ifndef PSD_PROJECTION_UTILS_H
#define PSD_PROJECTION_UTILS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include "psd_projection/check.h"
#include <cublas_v2.h>
#include <curand_kernel.h>

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define D2D cudaMemcpyDeviceToDevice

// Print an n×n matrix of doubles
inline void printMatrixDouble(const double* dM, int n, int m = -1) {
    // If m is not specified, use n for a square matrix
    if (m == -1)
        m = n;

    size_t N = size_t(n) * m;
    std::vector<double> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(double), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[j*n + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Print an n×n matrix of floats
inline void printMatrixFloat(const float* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<float> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[i*n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Print an n×n matrix of __half
inline void printMatrixHalf(const __half* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<__half> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(__half), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            float v = __half2float(hM[i*n + j]);
            std::cout << std::fixed << std::setprecision(6)
                      << v << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/// @brief Symmetrize a matrix M by replacing it by (M + M^T) / 2.
/// @param cublasH cuBLAS handle
/// @param M matrix to symmetrize of size n
/// @param n size of the matrix (n x n)
/// @param workspace device memory workspace of size n x n
void symmetrizeFloat(
    cublasHandle_t cublasH, float* M, int n, float* workspace
);

__global__ void convert_double_to_float_kernel(const double* in, float* out, int n);

__global__ void convert_float_to_double_kernel(const float* in, double* out, int n);

/// @brief Convert an array of doubles to floats in device memory.
/// @param d_in the input array of doubles in device memory
/// @param d_out the output array of floats in device memory
/// @param n the number of elements in the input array
/// @param threadsPerBlock number of threads per block for the kernel launch (default is
void convert_double_to_float(const double* d_in, float* d_out, int n, const int threadsPerBlock = 1024);

/// @brief Convert an array of doubles to floats in device memory.
/// @param d_in the input array of doubles in device memory
/// @param d_out the output array of floats in device memory
/// @param n the number of elements in the input array
/// @param threadsPerBlock number of threads per block for the kernel launch (default is
void convert_float_to_double(const float* d_in, double* d_out, int n, const int threadsPerBlock = 1024);

__global__ void build_identity_kernel(float* mat, int n);

/// @brief Build an identity matrix of size n x n in device memory (single precision).
/// @param cublasH cuBLAS handle
/// @param mat the device memory pointer to store the identity matrix
/// @param n the size of the identity matrix (n x n)
/// @param threadsPerBlock number of threads per block for the kernel launch (default is 1024)
void build_identity(
    cublasHandle_t cublasH,
    float* mat,
    int n,
    const int threadsPerBlock = 1024
); // TODO: remove this function to avoid using it anywhere

__global__ void add_identity_kernel(float* mat, int n);

/// @brief Add the identity matrix of size n x n to a matrix in device memory (single precision).
/// @param cublasH cuBLAS handle
/// @param mat matrix to which the identity matrix will be added, stored in column-major order
/// @param n size of mat
void add_identity(
    cublasHandle_t cublasH,
    float* mat,
    int n
);

__global__ void float4_to_half_kernel(
    const float4* __restrict__ A4,
    __half2 * __restrict__ B2,
    size_t N4
);

/// @brief Convert an array of floats to half-precision (4 floats packed into 2 half2) in device memory.
/// @param dA the input array of floats in device memory
/// @param dB the output array of half-precision numbers in device memory
/// @param N the size of the input array (number of floats)
void convert_float_to_half4(const float* dA, __half* dB, size_t N);

__global__ void identity_minus_kernel(const float* A_in, float* A_out, const int n);

void identity_minus(
    const float* A_in,
    float* A_out,
    const int n
);

__global__ void identity_plus_kernel(const float* A_in, float* A_out, const int n);

void identity_plus(
    const float* A_in, // device pointer to matrix A
    float* A_out, // device pointer to matrix A
    const int n
);

__global__ void fill_random_kernel(double* vec, int n, unsigned long seed);

/// @brief Fills a vector with random floats in (0,1] using the CUDA random number generator.
/// @param vec Device pointer to the vector to fill
/// @param n Size of the vector
/// @param seed Seed for the random number generator
void fill_random(double* vec, int n, unsigned long seed = 0, const int threadsPerBlock = 1024);

unsigned long make_seed();

void max_dense_vector_zero(
    double* vec_vals,
    int vec_size,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

void reverse_vector(const double* in, double* out, int m);

void reverse_columns(const double* in, double* out, int n, int m);

void extract_columns(const double* in, double* out, int n, int m);

#endif // PSD_PROJECTION_UTILS_H