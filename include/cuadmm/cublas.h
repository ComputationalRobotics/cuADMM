/*

    cublas.h

    Define wrapper for cuBLAS functions.

*/

#ifndef CUADMM_CUBLAS_H
#define CUADMM_CUBLAS_H

#include "cuadmm/check.h"
#include "cuadmm/memory.h"

// Multiply two dense vectors reprsenting matrices in batch:
// mat1[i] = mat2[i] * mat3[i]^T   (i = 0, .., batch_size-1)
// where * is the matrix multiplication operator.
inline void dense_matrix_mul_trans_batch(
    DeviceBlasHandle& cublas_H, 
    DeviceDenseVector<double>& mat1, const DeviceDenseVector<double>& mat2, const DeviceDenseVector<double>& mat3,
    const int mat_size, const int batch_size
) {
    const double alpha = 1.0;
    const double beta = 0.0;
    const long long int stride = mat_size * mat_size;
    CHECK_CUBLAS( cublasDgemmStridedBatched(
        cublas_H.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        mat_size, mat_size, mat_size, 
        &alpha, mat2.vals, mat_size, stride, mat3.vals, mat_size, stride,
        &beta, mat1.vals, mat_size, stride,
        batch_size
    ) );
    return;
}

// Multiply a dense vector by a scalar and add to another dense vector:
// y = alpha * x + y
inline void axpy_cublas(
    DeviceBlasHandle& cublas_H, const DeviceDenseVector<double>& x, DeviceDenseVector<double>& y,
    const double alpha
) {
    CHECK_CUBLAS( cublasDaxpy_v2(
        cublas_H.cublas_handle, x.size, &alpha, 
        x.vals, 1, y.vals, 1
    ) );
    return;
}

#endif // CUADMM_CUBLAS_H