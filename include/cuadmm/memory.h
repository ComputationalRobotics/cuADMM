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
#include <vector>

#include "cuadmm/check.h"
#include "cuadmm/mapper.h"

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

// Wrapper around cuSPARSE handle
class DeviceSparseHandle {
    public:
        int gpu_id;
        cusparseHandle_t cusparse_handle;

        DeviceSparseHandle(): gpu_id(0), cusparse_handle(NULL) {}
        DeviceSparseHandle(const int gpu_id): gpu_id(gpu_id), cusparse_handle(NULL) {}

        inline void set_gpu_id(const int gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSPARSE( cusparseCreate(&this->cusparse_handle) );
            return;
        }
        inline void activate(const DeviceStream& device_stream) {
            assert(device_stream.gpu_id == this->gpu_id);
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSPARSE( cusparseCreate(&this->cusparse_handle) );
            CHECK_CUSPARSE( cusparseSetStream(this->cusparse_handle, device_stream.stream) );
            return;
        }

        ~DeviceSparseHandle() {
            if (this->cusparse_handle != NULL) {
                CHECK_CUSPARSE( cusparseDestroy(this->cusparse_handle) );
                this->cusparse_handle = NULL;
            }
            // std::cout << "DeviceSparseHandle destructor called!" << std::endl;
        }
};

// Dense vector on host (CPU)
template <typename T>
class HostDenseVector {
    public:
        int size;
        T* vals;

        HostDenseVector(): size(0), vals(nullptr) {}
        HostDenseVector(const int size, bool as_byte = false): size(size), vals(nullptr) {
            this->allocate(size, as_byte);
        }

        inline void allocate(const int size, bool as_byte = false) {
            if (this->vals == nullptr) {
                if (as_byte) {
                    this->size = (size + sizeof(T) - 1) / sizeof(T);
                } else {
                    this->size = size;
                }
                this->vals = (T*) malloc(sizeof(T) * this->size);
            }
            return;
        }

        ~HostDenseVector() {
            if (this->vals != nullptr) {
                free(this->vals);
                this->vals = nullptr;
            }
            // std::cout << "HostDenseVector destructor called!" << std::endl;
        }

        void print() {
            std::printf("[");
            for (size_t i = 0; i < this->size; i++) {
                std::printf("%f, ", this->vals[i]);
            }
            std::printf("]\n");
        }
};

// Dense vector on device (GPU)
template <typename T>
class DeviceDenseVector {
    public:
        int gpu_id;
        int size;
        T* vals;
        cusparseDnVecDescr_t cusparse_descr;

        DeviceDenseVector(): gpu_id(0), size(0), vals(nullptr), cusparse_descr(nullptr) {}
        DeviceDenseVector(const int gpu_id, const int size, bool as_byte = false): gpu_id(gpu_id), size(size), vals(nullptr), cusparse_descr(nullptr) {
            this->allocate(this->gpu_id, this->size, as_byte);
        }

        inline void allocate(const int gpu_id, const int size, bool as_byte = false) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->size = size;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                // as_byte is used to allocate buffer size, which is usually given in terms of bytes
                if (as_byte) {
                    this->size = (size + sizeof(T) - 1) / sizeof(T);
                } else {
                    this->size = size;
                }
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(T) * this->size) );
                CHECK_CUSPARSE( cusparseCreateDnVec(&this->cusparse_descr, this->size, this->vals, CudaTypeMapper<T>::value) );
            }
            return;
        }

        inline T get_norm(const DeviceBlasHandle& cublas_H) {
            T norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            // assert that T is double
            static_assert(std::is_same<T, double>::value, "DeviceDenseVector norm only works with double type.");
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->size, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceDenseVector() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroyDnVec(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceDenseVector destructor called!" << std::endl;
        }

        void print() {
            T host_vec[size];
            
            // copy the vector to the device
            CHECK_CUDA( cudaMemcpy(host_vec, this->vals, sizeof(T) * this->size, cudaMemcpyDeviceToHost) );
            std::printf("[");
            for (size_t i = 0; i < this->size; i++) {
                std::printf("%f, ", host_vec[i]);
            }
            std::printf("]\n");
        }
};

// Sparse vector on device (GPU): wrapper around cuSPARSE sparse vector descriptor
template <typename T>
class DeviceSparseVector {
    public:
        int gpu_id;
        int size;     // size of the vector
        int nnz;      // number of non-zero entries
        int* indices; // indices of the non-zero entries (array with `nnz` elements)
        T* vals;      // values of the non-zero entries (array with `nnz` elements)
        cusparseSpVecDescr_t cusparse_descr;

        DeviceSparseVector(): gpu_id(0), size(0), nnz(0), 
            indices(nullptr), vals(nullptr), cusparse_descr(NULL) {}
        DeviceSparseVector(const int gpu_id, const int size, const int nnz):
            gpu_id(gpu_id), size(size), nnz(nnz), 
            vals(nullptr), indices(nullptr), cusparse_descr(NULL) {
                this->allocate(gpu_id, size, nnz);
            }
        
        inline void allocate(const int gpu_id, const int size, const int nnz) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->size = size; 
                this->nnz = nnz;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->indices, sizeof(int) * this->nnz) );
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(T) * this->nnz) );
                CHECK_CUSPARSE( cusparseCreateSpVec(
                    &this->cusparse_descr, this->size, this->nnz, 
                    this->indices, this->vals, 
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CudaTypeMapper<T>::value 
                ) );
            }
            return;
        }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            T norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSparseVector() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->indices != nullptr) {
                CHECK_CUDA( cudaFree(this->indices) );
                this->indices = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpVec(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSparseVector destructor called!" << std::endl;
        }

        void print(bool show_zeros = false) {
            T vals[nnz];
            int indices[nnz];
            
            // copy the vector to the device
            CHECK_CUDA( cudaMemcpy(vals, this->vals, sizeof(T) * this->nnz, cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(indices, this->indices, sizeof(int) * this->nnz, cudaMemcpyDeviceToHost) );

            // print the vector
            if (show_zeros) {
                std::printf("[");
                int current_index = 0;
                for (size_t i = 0; i < this->size; i++) {
                    if (current_index < this->nnz && indices[current_index] == i){
                        std::printf("%f, ", vals[current_index]);
                        current_index += 1;
                    } else {
                        std::printf("0.0, ");
                    }
                }
                std::printf("]\n");
            } else {
                std::printf("[");
                for (size_t i = 0; i < this->nnz; i++) {
                    std::printf("(%d, %f), ", indices[i], vals[i]);
                }
                std::printf("]\n");
            }
        }
};

// Dense vector on device (GPU): double type and Compressed Sparse Column (CSC) format
class DeviceSparseMatrixDoubleCSC {
    public:
        int gpu_id;
        int row_size;  // rows of the matrix
        int col_size;  // columns of the matrix
        int nnz;       // number of non-zero entries
        int* col_ptrs; // column offsets of the matrix (array with `col_size + 1` elements)
        int* row_ids;  // row indices of the non-zero entries (array with `nnz` elements)
        double* vals;  // values of the non-zero entries (array with `nnz` elements)
        cusparseSpMatDescr_t cusparse_descr;

        DeviceSparseMatrixDoubleCSC(): gpu_id(0), row_size(0), col_size(0), nnz(0),
            col_ptrs(nullptr), row_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {}
            DeviceSparseMatrixDoubleCSC(const int gpu_id, const int row_size, const int col_size, const int nnz):
            gpu_id(gpu_id), row_size(row_size), col_size(col_size), nnz(nnz),
            col_ptrs(nullptr), row_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {
                this->allocate(gpu_id, row_size, col_size, nnz);
            }

        inline void allocate(const int gpu_id, const int row_size, const int col_size, const int nnz) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->row_size = row_size;
                this->col_size = col_size;
                this->nnz = nnz;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->col_ptrs, sizeof(int) * (this->col_size + 1)) );
                CHECK_CUDA( cudaMalloc((void**) &this->row_ids, sizeof(int) * this->nnz) );
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(double) * this->nnz) );
                CHECK_CUSPARSE( cusparseCreateCsc(
                    &this->cusparse_descr, this->row_size, this->col_size, this->nnz,
                    this->col_ptrs, this->row_ids, this->vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
                ) );
            }
            return;
        }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSparseMatrixDoubleCSC() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->col_ptrs != nullptr) {
                CHECK_CUDA( cudaFree(this->col_ptrs) );
                this->col_ptrs = nullptr;
            }
            if (this->row_ids != nullptr) {
                CHECK_CUDA( cudaFree(this->row_ids) );
                this->row_ids = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpMat(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSparseMatrixDoubleCSC destructor called!" << std::endl;
        }
};

// Dense vector on device (GPU): double type and Compressed Sparse Row (CSR) format
class DeviceSparseMatrixDoubleCSR {
    public:
        int gpu_id;
        int64_t row_size; // rows of the matrix
        int64_t col_size; // columns of the matrix
        int64_t nnz;      // number of non-zero entries
        int* row_ptrs;    // row offsets of the matrix (array with `row_size + 1` elements)
        int* col_ids;     // column indices of the non-zero entries (array with `nnz` elements)
        double* vals;     // values of the non-zero entries (array with `nnz` elements)
        cusparseSpMatDescr_t cusparse_descr;

        DeviceSparseMatrixDoubleCSR(): gpu_id(0), row_size(0), col_size(0), nnz(0),
            row_ptrs(nullptr), col_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {}
            DeviceSparseMatrixDoubleCSR(const int gpu_id, const int row_size, const int col_size, const int nnz):
            gpu_id(gpu_id), row_size(row_size), col_size(col_size), nnz(nnz),
            row_ptrs(nullptr), col_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {
                this->allocate(gpu_id, row_size, col_size, nnz);
            }

        inline void allocate(const int gpu_id, const int row_size, const int col_size, const int nnz) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->row_size = row_size;
                this->col_size = col_size;
                this->nnz = nnz;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                // CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->row_ptrs, sizeof(int) * (this->row_size + 1)) );
                if (nnz > 0) {
                    CHECK_CUDA( cudaMalloc((void**) &this->col_ids, sizeof(int) * this->nnz) );
                    CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(double) * this->nnz) );
                } else {
                    this->col_ids = nullptr;
                    this->vals = nullptr;
                }
                CHECK_CUSPARSE( cusparseCreateCsr(
                    &this->cusparse_descr, this->row_size, this->col_size, this->nnz,
                    this->row_ptrs, this->col_ids, this->vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
                ) );
            }
            return;
        }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSparseMatrixDoubleCSR() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->row_ptrs != nullptr) {
                CHECK_CUDA( cudaFree(this->row_ptrs) );
                this->row_ptrs = nullptr;
            }
            if (this->col_ids != nullptr) {
                CHECK_CUDA( cudaFree(this->col_ids) );
                this->col_ids = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpMat(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSparseMatrixDoubleCSR destructor called!" << std::endl;
        }
};

#endif // CUADMM_MEMORY_H