/*

    check.h

    Defines CHECK functions for CUDA, cuBLAS, cuSOLVER, and cuSPARSE.
    These are used to check the return status of CUDA API calls.

*/

#ifndef CUADMM_CHECK_H
#define CUADMM_CHECK_H

#include <cuda_runtime_api.h>
#include <cusparse.h>

// Check if the function returns a CUDA error
#define CHECK_CUDA(func)                                                       \
do {                                                                           \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s:%d with error: %s (%d)",                 \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0) // wrap it in a do-while loop to be called with a semicolon

// Check if the function returns a cuBLAS error
#define CHECK_CUBLAS(func)                                                     \
do {                                                                           \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("cuBLAS error %d at %s:%d", status, __FILE__, __LINE__);        \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0)

// Check if the function returns a cuSPARSE error
#define CHECK_CUSOLVER(func)                                                   \
do {                                                                           \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("cuSOLVER error %d at %s:%d", status, __FILE__, __LINE__);      \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0)

// Check if the function returns a cuSPARSE error
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE error %s (%d) at %s:%d",                              \
              cusparseGetErrorString(status), status, __FILE__, __LINE__);     \
        std::cout << std::endl;                                                \
    }                                                                          \
}

#endif // CUADMM_CHECK_H