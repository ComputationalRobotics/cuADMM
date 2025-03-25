/*

memory.h

Defines memory wrappers.

*/

#ifndef CUADMM_MEMORY_H
#define CUADMM_MEMORY_H

#include <cblas.h>
#include <cassert>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cuadmm/check.h"

// Wrapper around CUDA stream
class DeviceStream {
    public:
        int gpu_id;
        cudaStream_t stream;

        DeviceStream(): gpu_id(0), stream(NULL) {}
        DeviceStream(const int gpu_id): gpu_id(gpu_id), stream(NULL) {}

        inline void set_gpu_id(const int gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        // Set the device and create a new stream
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUDA( cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking) );
        }

        // Destroy the stream if any
        ~DeviceStream() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->stream != NULL) {
                CHECK_CUDA( cudaStreamDestroy(this->stream) );
                this->stream = NULL;
            }
            // std::cout << "DeviceStream destructor called!" << std::endl;
        }
};

// Wrapper around cuBLAS handle
class DeviceBlasHandle {
    public:
        int gpu_id;
        cublasHandle_t cublas_handle;

        DeviceBlasHandle(): gpu_id(0), cublas_handle(NULL) {}
        DeviceBlasHandle(const int gpu_id): gpu_id(gpu_id), cublas_handle(NULL) {}

        inline void set_gpu_id(const int gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        // Set the device and create a new cuBLAS handle
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasCreate_v2(&this->cublas_handle) );
            return;
        }
        // Set the device, create a new handle, and set the stream to the argument
        inline void activate(const DeviceStream& device_stream) {
            assert(device_stream.gpu_id == this->gpu_id);
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasCreate_v2(&this->cublas_handle) );
            CHECK_CUBLAS( cublasSetStream_v2(this->cublas_handle, device_stream.stream) );
            return;
        }

        // Destroy the handle if any
        ~DeviceBlasHandle() {
            if (this->cublas_handle != NULL) {
                CHECK_CUBLAS( cublasDestroy_v2(this->cublas_handle) );
                this->cublas_handle = NULL;
            }
            // std::cout << "DeviceBlasHandle destructor called!" << std::endl;
        }
};


// Wrapper around cuSOLVER dense handle
class DeviceSolverDnHandle {
    public:
        int gpu_id;
        cusolverDnHandle_t cusolver_dn_handle;

        DeviceSolverDnHandle(): gpu_id(0), cusolver_dn_handle(NULL) {}
        DeviceSolverDnHandle(const int gpu_id): gpu_id(gpu_id), cusolver_dn_handle(NULL) {}

        inline void set_gpu_id(const int gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSOLVER( cusolverDnCreate(&this->cusolver_dn_handle) );
            return;
        }
        inline void activate(const DeviceStream& device_stream) {
            assert(device_stream.gpu_id == this->gpu_id);
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSOLVER( cusolverDnCreate(&this->cusolver_dn_handle) );
            CHECK_CUSOLVER( cusolverDnSetStream(this->cusolver_dn_handle, device_stream.stream) );
            return;
        }

        ~DeviceSolverDnHandle() {
            if (this->cusolver_dn_handle != NULL) {
                CHECK_CUSOLVER( cusolverDnDestroy(this->cusolver_dn_handle) );
                this->cusolver_dn_handle = NULL;
            }
            // std::cout << "DeviceSolverDnHandle destructor called!" << std::endl;
        }
};


#endif