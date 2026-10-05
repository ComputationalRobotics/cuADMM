/*

    check.h

    Defines CHECK functions for CUDA, cuBLAS, cuSOLVER, and cuSPARSE.
    These are used to check the return status of CUDA API calls.
    The CHECK_* macros throw a std::runtime_error on failure; the CHECK_*_NOTHROW
    variants only print the error and are meant for destructors (implicitly noexcept).

*/

#ifndef CUADMM_CHECK_H
#define CUADMM_CHECK_H

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

// Builds the error message "<what> at <file>:<line>".
inline std::string cuadmm_error_message(const std::string &what, const char *file, int line)
{
    return what + " at " + file + ":" + std::to_string(line);
}

// Check if the function returns a CUDA error
#define CHECK_CUDA(func)                                                                                     \
    do                                                                                                       \
    {                                                                                                        \
        cudaError_t status = (func);                                                                         \
        if (status != cudaSuccess)                                                                           \
        {                                                                                                    \
            throw std::runtime_error(cuadmm_error_message(                                                   \
                std::string("CUDA API failed with error: ") + cudaGetErrorString(status) + " (" +            \
                    std::to_string(status) + ")",                                                            \
                __FILE__, __LINE__));                                                                        \
        }                                                                                                    \
    } while (0) // wrap it in a do-while loop to be called with a semicolon

// Check if the function returns a cuBLAS error
#define CHECK_CUBLAS(func)                                                                                   \
    do                                                                                                       \
    {                                                                                                        \
        cublasStatus_t status = (func);                                                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                 \
        {                                                                                                    \
            throw std::runtime_error(cuadmm_error_message(                                                   \
                "cuBLAS error " + std::to_string(status), __FILE__, __LINE__));                              \
        }                                                                                                    \
    } while (0)

// Check if the function returns a cuSOLVER error
#define CHECK_CUSOLVER(func)                                                                                 \
    do                                                                                                       \
    {                                                                                                        \
        cusolverStatus_t status = (func);                                                                    \
        if (status != CUSOLVER_STATUS_SUCCESS)                                                               \
        {                                                                                                    \
            throw std::runtime_error(cuadmm_error_message(                                                   \
                "cuSOLVER error " + std::to_string(status), __FILE__, __LINE__));                            \
        }                                                                                                    \
    } while (0)

// Check if the function returns a cuSPARSE error
#define CHECK_CUSPARSE(func)                                                                                 \
    do                                                                                                       \
    {                                                                                                        \
        cusparseStatus_t status = (func);                                                                    \
        if (status != CUSPARSE_STATUS_SUCCESS)                                                               \
        {                                                                                                    \
            throw std::runtime_error(cuadmm_error_message(                                                   \
                std::string("cuSPARSE error ") + cusparseGetErrorString(status) + " (" +                     \
                    std::to_string(status) + ")",                                                            \
                __FILE__, __LINE__));                                                                        \
        }                                                                                                    \
    } while (0)

/* Non-throwing variants, for destructors */

#define CHECK_CUDA_NOTHROW(func)                                                \
    do                                                                          \
    {                                                                           \
        cudaError_t status = (func);                                            \
        if (status != cudaSuccess)                                              \
        {                                                                       \
            printf("CUDA API failed at %s:%d with error: %s (%d)",              \
                   __FILE__, __LINE__, cudaGetErrorString(status), status);     \
            std::cout << std::endl;                                             \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS_NOTHROW(func)                                              \
    do                                                                          \
    {                                                                           \
        cublasStatus_t status = (func);                                         \
        if (status != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                       \
            printf("cuBLAS error %d at %s:%d", status, __FILE__, __LINE__);     \
            std::cout << std::endl;                                             \
        }                                                                       \
    } while (0)

#define CHECK_CUSOLVER_NOTHROW(func)                                            \
    do                                                                          \
    {                                                                           \
        cusolverStatus_t status = (func);                                       \
        if (status != CUSOLVER_STATUS_SUCCESS)                                  \
        {                                                                       \
            printf("cuSOLVER error %d at %s:%d", status, __FILE__, __LINE__);   \
            std::cout << std::endl;                                             \
        }                                                                       \
    } while (0)

#define CHECK_CUSPARSE_NOTHROW(func)                                            \
    do                                                                          \
    {                                                                           \
        cusparseStatus_t status = (func);                                       \
        if (status != CUSPARSE_STATUS_SUCCESS)                                  \
        {                                                                       \
            printf("cuSPARSE error %s (%d) at %s:%d",                           \
                   cusparseGetErrorString(status), status, __FILE__, __LINE__); \
            std::cout << std::endl;                                             \
        }                                                                       \
    } while (0)

#endif // CUADMM_CHECK_H
