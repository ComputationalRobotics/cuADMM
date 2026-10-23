/*

    cuadmm_legacy/check.h

    The old macro names of the ported engine, mapped onto the one set of checks of core/check.h (plan B15: one set of
    macros throwing cuadmm::Error with the call site). Kept until the last legacy source has moved.

*/

#ifndef CUADMM_LEGACY_CHECK_H
#define CUADMM_LEGACY_CHECK_H

#include <iostream>
#include <string>

#include "core/check.h"

inline std::string cuadmm_error_message(const std::string &what, const char *file, int line)
{
    return what + " at " + file + ":" + std::to_string(line);
}

#define CHECK_CUDA(func) CUADMM_CHECK_CUDA(func)
#define CHECK_CUBLAS(func) CUADMM_CHECK_CUBLAS(func)
#define CHECK_CUSOLVER(func) CUADMM_CHECK_CUSOLVER(func)
#define CHECK_CUSPARSE(func) CUADMM_CHECK_CUSPARSE(func)
#define CHECK_KERNEL_LAUNCH() CUADMM_CHECK_LAUNCH()
#define CHECK_CUDA_NOTHROW(func) CUADMM_CHECK_CUDA_NOTHROW(func)
#define CHECK_CUBLAS_NOTHROW(func) CUADMM_CHECK_CUBLAS_NOTHROW(func)
#define CHECK_CUSOLVER_NOTHROW(func) CUADMM_CHECK_CUSOLVER_NOTHROW(func)
#define CHECK_CUSPARSE_NOTHROW(func) CUADMM_CHECK_CUSPARSE_NOTHROW(func)

#endif // CUADMM_LEGACY_CHECK_H
