#ifndef CUADMM_KERNELS_H
#define CUADMM_KERNELS_H

#include "cuadmm/memory.h"

/*
    Dense-dense operations (kernels/dense_dense.cu)
*/


/*
    Dense-scalar operations (kernels/dense_scalar.cu)
*/
void dense_vector_mul_scalar_kernel(DeviceDenseVector<double>& vec, double scalar, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024);

#endif