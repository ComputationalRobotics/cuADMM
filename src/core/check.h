/*

    core/check.h

    The one set of error checks of cuADMM (plan Section 14, B15): CUADMM_CHECK_CUDA / CUBLAS / CUSOLVER /
    CUSPARSE throw cuadmm::Error with the call site and a context string; CUADMM_CHECK_LAUNCH() follows every kernel
    launch (B16); the _NOTHROW variants are for destructors. The legacy header cuadmm_legacy/check.h maps its old
    macro names onto these, so every component throws the same exception.

*/

#ifndef CUADMM_CORE_CHECK_H
#define CUADMM_CORE_CHECK_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace cuadmm
{

/// @brief Error of a CUDA library call or of the solver, with the call site.
class Error : public std::runtime_error
{
public:
    Error(const std::string &what, const char *file, int line)
        : std::runtime_error(what + " at " + file + ":" + std::to_string(line)), file_(file), line_(line) {}
    const char *file() const { return file_; }
    int line() const { return line_; }

private:
    const char *file_;
    int line_;
};

inline const char *cublas_status_string(cublasStatus_t s)
{
    switch (s)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "CUBLAS_STATUS_?";
    }
}

inline const char *cusolver_status_string(cusolverStatus_t s)
{
    switch (s)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:
        return "CUSOLVER_STATUS_?";
    }
}

} // namespace cuadmm

#define CUADMM_CHECK_CUDA(call)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        const cudaError_t cuadmm_status_ = (call);                                                                     \
        if (cuadmm_status_ != cudaSuccess)                                                                             \
            throw ::cuadmm::Error(std::string("CUDA API failed with error: ") + cudaGetErrorString(cuadmm_status_) +   \
                                      " (" + std::to_string(cuadmm_status_) + ") in " #call,                           \
                                  __FILE__, __LINE__);                                                                 \
    } while (0)

#define CUADMM_CHECK_CUBLAS(call)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        const cublasStatus_t cuadmm_status_ = (call);                                                                  \
        if (cuadmm_status_ != CUBLAS_STATUS_SUCCESS)                                                                   \
            throw ::cuadmm::Error(std::string("cuBLAS error ") + ::cuadmm::cublas_status_string(cuadmm_status_) +      \
                                      " (" + std::to_string(cuadmm_status_) + ") in " #call,                           \
                                  __FILE__, __LINE__);                                                                 \
    } while (0)

#define CUADMM_CHECK_CUSOLVER(call)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        const cusolverStatus_t cuadmm_status_ = (call);                                                                \
        if (cuadmm_status_ != CUSOLVER_STATUS_SUCCESS)                                                                 \
            throw ::cuadmm::Error(std::string("cuSOLVER error ") + ::cuadmm::cusolver_status_string(cuadmm_status_) +  \
                                      " (" + std::to_string(cuadmm_status_) + ") in " #call,                           \
                                  __FILE__, __LINE__);                                                                 \
    } while (0)

#define CUADMM_CHECK_CUSPARSE(call)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        const cusparseStatus_t cuadmm_status_ = (call);                                                                \
        if (cuadmm_status_ != CUSPARSE_STATUS_SUCCESS)                                                                 \
            throw ::cuadmm::Error(std::string("cuSPARSE error ") + cusparseGetErrorString(cuadmm_status_) + " (" +     \
                                      std::to_string(cuadmm_status_) + ") in " #call,                                  \
                                  __FILE__, __LINE__);                                                                 \
    } while (0)

// after every kernel launch (B16): an invalid configuration or a sticky error surfaces here
#define CUADMM_CHECK_LAUNCH() CUADMM_CHECK_CUDA(cudaGetLastError())

/* Non-throwing variants, for destructors */

#define CUADMM_CHECK_CUDA_NOTHROW(call)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        const cudaError_t cuadmm_status_ = (call);                                                                     \
        if (cuadmm_status_ != cudaSuccess)                                                                             \
            std::fprintf(stderr, "CUDA API failed at %s:%d with error: %s (%d)\n", __FILE__, __LINE__,                 \
                         cudaGetErrorString(cuadmm_status_), static_cast<int>(cuadmm_status_));                        \
    } while (0)

#define CUADMM_CHECK_CUBLAS_NOTHROW(call)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        const cublasStatus_t cuadmm_status_ = (call);                                                                  \
        if (cuadmm_status_ != CUBLAS_STATUS_SUCCESS)                                                                   \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(cuadmm_status_), __FILE__, __LINE__);  \
    } while (0)

#define CUADMM_CHECK_CUSOLVER_NOTHROW(call)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        const cusolverStatus_t cuadmm_status_ = (call);                                                                \
        if (cuadmm_status_ != CUSOLVER_STATUS_SUCCESS)                                                                 \
            std::fprintf(stderr, "cuSOLVER error %d at %s:%d\n", static_cast<int>(cuadmm_status_), __FILE__, __LINE__); \
    } while (0)

#define CUADMM_CHECK_CUSPARSE_NOTHROW(call)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        const cusparseStatus_t cuadmm_status_ = (call);                                                                \
        if (cuadmm_status_ != CUSPARSE_STATUS_SUCCESS)                                                                 \
            std::fprintf(stderr, "cuSPARSE error %s (%d) at %s:%d\n", cusparseGetErrorString(cuadmm_status_),          \
                         static_cast<int>(cuadmm_status_), __FILE__, __LINE__);                                        \
    } while (0)

#endif // CUADMM_CORE_CHECK_H
