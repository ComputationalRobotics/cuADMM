#ifndef PSD_PROJECTION_COMPOSITE_FP32_EMULATED_H
#define PSD_PROJECTION_COMPOSITE_FP32_EMULATED_H

#include "cuda.h"

// this file is compatible with CUDA 12.9 and later
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)

#include <cublas_v2.h>

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP32 precision emulated with the BF16x9 algorithm. The matrix must be scaled such that its eigenvalues are in [-1, 1].
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param workspace optional workspace of size at least `3 * mat_size*mat_size` for intermediate computations; if not provided, a default workspace will be allocated
/// @note This function assumes that the input matrix is already scaled such that its eigenvalues are in the range [-1, 1]. If the matrix is not scaled, the computation might diverge.
void composite_FP32_emulated(
    cublasHandle_t cublasH,
    double* mat,
    const int mat_size,
    float* workspace = nullptr
);

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP32 precision emulated with the BF16x9 algorithm. The matrix is automatically scaled to ensure its eigenvalues are in [-1, 1].
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param workspace optional workspace of size at least `3 * mat_size*mat_size` for intermediate computations; if not provided, a default workspace will be allocated
void composite_FP32_emulated_auto_scale(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int mat_size,
    float* workspace = nullptr
);

#endif // defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)

#endif // PSD_PROJECTION_COMPOSITE_FP32_EMULATED_H