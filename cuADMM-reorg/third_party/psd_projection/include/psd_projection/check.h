/*

    psd_projection/check.h  (cuADMM-reorg: mapped onto core/check.h)

    The submodule's CHECK_* macros only printed; here they are the throwing checks of cuADMM-reorg (plan B15: one set
    of macros, one include guard). The vendored kernels therefore fail with cuadmm::Error like the rest of the solver.

*/

#ifndef PSD_PROJECTION_CHECK_H
#define PSD_PROJECTION_CHECK_H

#include <iostream>

#include "core/check.h"

#ifndef CHECK_CUDA
#define CHECK_CUDA(func) CUADMM_CHECK_CUDA(func)
#endif
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(func) CUADMM_CHECK_CUBLAS(func)
#endif
#ifndef CHECK_CUSOLVER
#define CHECK_CUSOLVER(func) CUADMM_CHECK_CUSOLVER(func)
#endif
#ifndef CHECK_CUSPARSE
#define CHECK_CUSPARSE(func) CUADMM_CHECK_CUSPARSE(func)
#endif

#endif // PSD_PROJECTION_CHECK_H
