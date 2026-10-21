#ifndef PSD_PROJECTION_COMPOSITE_FP16_H
#define PSD_PROJECTION_COMPOSITE_FP16_H

#include <cublas_v2.h>

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP16 precision. The matrix must be scaled such that its eigenvalues are in [-1, 1].
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param workspace optional workspace of size at least `3 * mat_size*mat_size` for intermediate computations; if not provided, a default workspace will be allocated
/// @note This function assumes that the input matrix is already scaled such that its eigenvalues are in the range [-1, 1]. If the matrix is not scaled, the computation might diverge.
void composite_FP16(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    float* float_workspace = nullptr,
    __half* half_workspace = nullptr
);

/// @brief Projects a symmetric matrix to its positive semidefinite (PSD) form using FP16 precision. The matrix is automatically scaled to ensure its eigenvalues are in [-1, 1].
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
/// @param workspace optional workspace of size at least `3 * mat_size*mat_size` for intermediate computations; if not provided, a default workspace will be allocated. See the note below for details regarding alignment.
/// @note If `mat_size*mat_size` is not a multiple of 4, the workspace will be padded to the next multiple of 4 for alignment. The size of the workspace should be sized accordingly.
void composite_FP16_auto_scale(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int mat_size,
    float* float_workspace = nullptr,
    __half* half_workspace = nullptr
);

/// @brief WIP: Projects a symmetric matrix to its positive semidefinite (PSD) form using FP16 precision. The matrix is automatically scaled to ensure its eigenvalues are in [-1, 1], and the largest eigenvalues are deflated to improve numerical stability.
/// @param cublasH a cuBLAS handle
/// @param mat the matrix to be projected, stored in column-major order
/// @param mat_size the size of the matrix (assumed to be square, i.e., mat_size x mat_size)
void composite_FP16_auto_scale_deflate(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int mat_size,
    float* float_workspace = nullptr,
    __half* half_workspace = nullptr,
    const size_t k = 40,
    const int maxiter = 100,
    const double tol = 1e-10,
    const bool verbose = false
);

#endif // PSD_PROJECTION_COMPOSITE_FP16_H