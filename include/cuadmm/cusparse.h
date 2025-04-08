/*

    cusparse.h

    Wrappers for cuSPARSE functions.

*/

#ifndef CUADMM_CUSPARSE_H
#define CUADMM_CUSPARSE_H

#include "cuadmm/check.h"
#include "cuadmm/memory.h"

// Retrive the buffer size for converting cuSPARSE CSC format to cuSPARSE CSR format
inline size_t CSC_to_CSR_get_buffersize_cusparse(
    DeviceSparseHandle& cusparse_H,
    const DeviceSparseMatrixDoubleCSC& mat_csc, DeviceSparseMatrixDoubleCSR& mat_csr
) {
    size_t buffer_size;
    CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(
        cusparse_H.cusparse_handle, 
        mat_csc.col_size, mat_csc.row_size, mat_csc.nnz,
        mat_csc.vals, mat_csc.col_ptrs, mat_csc.row_ids, 
        mat_csr.vals, mat_csr.row_ptrs, mat_csr.col_ids, 
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, 
        &buffer_size
    ) );
    return buffer_size;
}

// Convert cuSPARSE CSC format to cuSPARSE CSR format
// Here we assume mat_csr's memory has already been allocated.
// Note that this routine needs additional memories on device.
inline void CSC_to_CSR_cusparse(
    DeviceSparseHandle& cusparse_H,
    const DeviceSparseMatrixDoubleCSC& mat_csc, DeviceSparseMatrixDoubleCSR& mat_csr,
    DeviceDenseVector<double>& buffer
) {
    CHECK_CUSPARSE( cusparseCsr2cscEx2(
        cusparse_H.cusparse_handle, 
        mat_csc.col_size, mat_csc.row_size, mat_csc.nnz,
        mat_csc.vals, mat_csc.col_ptrs, mat_csc.row_ids, 
        mat_csr.vals, mat_csr.row_ptrs, mat_csr.col_ids, 
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
        buffer.vals
    ) );
    return;
}


// Get the buffer size for SpMV operation
inline size_t SpMV_get_buffersize_cusparse(
    DeviceSparseHandle& cusparse_H, 
    const DeviceSparseMatrixDoubleCSR& A, const DeviceDenseVector<double>& x, DeviceDenseVector<double>& y, 
    const double alpha, const double beta
) {
    size_t buffer_size = 0;
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
        cusparse_H.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.cusparse_descr, x.cusparse_descr, 
        &beta, y.cusparse_descr,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size
    ) );
    return buffer_size;
}

// Computes the sparse MV operation. Computes:
// y <-- alpha * A * x + beta * y
inline void SpMV_cusparse(
    DeviceSparseHandle& cusparse_H, 
    const DeviceSparseMatrixDoubleCSR& A, const DeviceDenseVector<double>& x, DeviceDenseVector<double>& y, 
    const double alpha, const double beta,
    DeviceDenseVector<double>& buffer
) {
    CHECK_CUSPARSE( cusparseSpMV(
        cusparse_H.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.cusparse_descr, x.cusparse_descr, 
        &beta, y.cusparse_descr,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.vals
    ) );
    return;
}


// Computes the axpby operation. Computes:
// y <-- alpha * x + beta * y
// where x is a sparse vector and y is a dense vector
inline void axpby_cusparse(
    DeviceSparseHandle& cusparse_H,
    const DeviceSparseVector<double>& x, const DeviceDenseVector<double>& y,
    const double alpha, const double beta
) {
    CHECK_CUSPARSE( cusparseAxpby(
        cusparse_H.cusparse_handle, 
        &alpha, x.cusparse_descr, &beta, y.cusparse_descr
    ) );
    return;
}

// Get the buffer size for the sparse inner product operation.
inline size_t SparseVV_get_buffersize_cusparse(
    DeviceSparseHandle& cusparse_H, 
    const DeviceSparseVector<double>& x, const DeviceDenseVector<double>& y
) {
    double inprod;
    size_t buffer_size;
    CHECK_CUSPARSE( cusparseSpVV_bufferSize(
        cusparse_H.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        x.cusparse_descr, y.cusparse_descr, &inprod, 
        CUDA_R_64F, &buffer_size
    ) );
    return buffer_size;
}

// Computes the sparse inner product operation. Computes:
// inprod <-- x' * y
// where x is a sparse vector and y is a dense vector.
// The result (inprod) is returned.
inline double SparseVV_cusparse(
    DeviceSparseHandle& cusparse_H, 
    const DeviceSparseVector<double>& x, const DeviceDenseVector<double>& y,
    DeviceDenseVector<double>& buffer 
) {
    double inprod;
    CHECK_CUSPARSE( cusparseSpVV(
        cusparse_H.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        x.cusparse_descr, y.cusparse_descr, &inprod, 
        CUDA_R_64F, buffer.vals
    ) );
    return inprod;
}

#endif // CUADMM_CUSPARSE_H