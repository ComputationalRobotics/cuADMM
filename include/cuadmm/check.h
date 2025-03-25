#ifndef CUADMM_CHECK_H
#define CUADMM_CHECK_H

#include <cuda_runtime_api.h>
#include <cusparse.h>

// Check if the function returns a CUDA error
#define CHECK_CUDA(func)                                                       \
do {                                                                           \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
} while (0) // wrap it in a do-while loop to be called with a semicolon

// Check if the function returns a cuBLAS error
#define CHECK_CUBLAS(func)                                                     \
do {                                                                           \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("cublas error %d at %s:%d\n", status, __FILE__, __LINE__);      \
    }                                                                          \
} while (0)

// Check if the function returns a cuSPARSE error
#define CHECK_CUSOLVER(func)                                                   \
do {                                                                           \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("cusolver error %d at %s:%d\n", status, __FILE__, __LINE__);    \
    }                                                                          \
} while (0)

#endif