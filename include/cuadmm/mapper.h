/*

    mapper.h

    Defines types conversion between C++ and CUDA for easy templating.

*/

#ifndef CUADMM_MAPPER_H
#define CUADMM_MAPPER_H

#include <cublas_v2.h>

template <typename T>
struct CudaTypeMapper;

template <>
struct CudaTypeMapper<double> {
    static const cudaDataType value = CUDA_R_64F;
};

template <>
struct CudaTypeMapper<float> {
    static const cudaDataType value = CUDA_R_32F;
};

template <>
struct CudaTypeMapper<int> {
    static const cudaDataType value = CUDA_R_32I;
};

template <>
struct CudaTypeMapper<size_t> {
    static const cudaDataType value = CUDA_R_32U;
};

#endif // CUADMM_MAPPER_H