/*

    cusolver.h

    Define wrapper for cuSOLVER functions.

*/

#include "cuadmm/check.h"
#include "cuadmm/memory.h"

// Wrapper for cuSOLVER parameter for a single matrix:
// specify to compute both eigenvalues and eigenvectors,
// and to use the lower triangle of the matrix.
class SingleEigParameter {
    public:
        int gpu_id;
        cusolverEigMode_t jobz;
        cublasFillMode_t uplo;
        cusolverDnParams_t param;

        SingleEigParameter(const int gpu_id = 0): gpu_id(gpu_id) {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            this->param = NULL;
            CHECK_CUSOLVER( cusolverDnCreateParams(&this->param) );
            this->jobz = CUSOLVER_EIG_MODE_VECTOR;
            this->uplo = CUBLAS_FILL_MODE_LOWER;
        }

        ~SingleEigParameter() {
            if (this->param != NULL) {
                CHECK_CUSOLVER( cusolverDnDestroyParams(this->param) );
                this->param = NULL;
            }
        }
};

// Get the buffer size for a single matrix eigenvalue/eigenvector decomposition.
template<typename T>
inline void single_eig_get_buffersize_cusolver(
    DeviceSolverDnHandle& cusolver_H, SingleEigParameter& param,
    DeviceDenseVector<T>& mat, DeviceDenseVector<T>& W, 
    const int mat_size,
    size_t* buffer_size, size_t* buffer_size_host,
    const int mat_offset = 0, const int W_offset = 0
) {
    CHECK_CUDA( cudaSetDevice(cusolver_H.gpu_id) );
    CHECK_CUSOLVER( cusolverDnXsyevd_bufferSize(
        cusolver_H.cusolver_dn_handle, param.param, param.jobz, param.uplo,
        mat_size, CudaTypeMapper<T>::value, mat.vals + mat_offset,
        mat_size, CudaTypeMapper<T>::value, W.vals + W_offset,
        CudaTypeMapper<T>::value,
        buffer_size, buffer_size_host
    ) );
}


// Calculate the eigenvalues and eigenvectors of a single matrix
// using cuSOLVER.
// Parameters:
// - mat: input matrix (symmetric)
// - W: output eigenvalues
// - buffer: workspace for cuSOLVER
// - buffer_host: workspace for cuSOLVER (host)
// - info: output info
// - mat_size: size of the matrix
// - buffer_size: size of the buffer
// - buffer_size_host: size of the host buffer
// - offsets for mat, W, buffer, and info
// Note that buffer_size and buffer_size_host are retrieved from
// single_eig_get_buffersize_cusolver.
template<typename T>
inline void single_eig_cusolver(
    DeviceSolverDnHandle& cusolver_H, SingleEigParameter& param,
    DeviceDenseVector<T>& mat, DeviceDenseVector<T>& W, 
    DeviceDenseVector<T>& buffer, HostDenseVector<T>& buffer_host, DeviceDenseVector<int>& info,
    const int mat_size, const size_t buffer_size, const size_t buffer_size_host,
    const int mat_offset = 0, const int W_offset = 0,
    const size_t buffer_offset = 0, const size_t buffer_host_offset = 0,    // from the outside, the function receives a void* array
    const int info_offset = 0
) {
    CHECK_CUDA( cudaSetDevice(cusolver_H.gpu_id) );
    CHECK_CUSOLVER( cusolverDnXsyevd(
        cusolver_H.cusolver_dn_handle, param.param, param.jobz, param.uplo,
        mat_size, CudaTypeMapper<T>::value, mat.vals + mat_offset,
        mat_size, CudaTypeMapper<T>::value, W.vals + W_offset,
        CudaTypeMapper<T>::value, 
        buffer.vals + buffer_offset / sizeof(double), buffer_size,
        buffer_host.vals + buffer_host_offset / sizeof(double), buffer_size_host,
        info.vals + info_offset
    ) );
}


// Wrapper for cuSOLVER parameter for a batch of matrices:
// specify to compute both eigenvalues and eigenvectors,
// and to use the lower triangle of the matrices.
// WARNING: this is a legacy interface!
class BatchEigParameter {
    public:
        int gpu_id;
        cusolverEigMode_t jobz;
        cublasFillMode_t uplo;
        double eig_tol;
        int eig_max_sweeps;
        int eig_sort;
        syevjInfo_t syevj_param;

        BatchEigParameter(
            int gpu_id = 0, double eig_tol = 1e-6, int eig_max_sweeps = 15, int eig_sort = 1
        ): gpu_id(gpu_id), eig_tol(eig_tol), eig_max_sweeps(eig_max_sweeps), eig_sort(eig_sort) {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            this->jobz = CUSOLVER_EIG_MODE_VECTOR;
            this->uplo = CUBLAS_FILL_MODE_LOWER;
            this->syevj_param = NULL;
            CHECK_CUSOLVER( cusolverDnCreateSyevjInfo(&this->syevj_param) );
            CHECK_CUSOLVER( cusolverDnXsyevjSetTolerance(this->syevj_param, this->eig_tol) );
            CHECK_CUSOLVER( cusolverDnXsyevjSetMaxSweeps(this->syevj_param, this->eig_max_sweeps) );
            CHECK_CUSOLVER( cusolverDnXsyevjSetSortEig(this->syevj_param, this->eig_sort) );
        }

        ~BatchEigParameter() {
            if (this->syevj_param != NULL) {
                CHECK_CUSOLVER( cusolverDnDestroySyevjInfo(this->syevj_param) );
                this->syevj_param = NULL;
            }
        }
};

// Get the buffer size for the eigenvalue/eigenvector decomposition
// of a batch of matrices.
template<typename T>
inline size_t batch_eig_get_buffersize_cusolver(
    DeviceSolverDnHandle& cusolver_H, BatchEigParameter& param, 
    DeviceDenseVector<T>& mat, DeviceDenseVector<T>& W,
    const int mat_size, const int batch_size
) {
    int buffer_len; // this size is len, not byte!
    CHECK_CUDA( cudaSetDevice(cusolver_H.gpu_id) );
    CHECK_CUSOLVER( cusolverDnDsyevjBatched_bufferSize(
        cusolver_H.cusolver_dn_handle, param.jobz, param.uplo,
        mat_size, mat.vals, mat_size, W.vals,
        &buffer_len, param.syevj_param, batch_size
    ) );
    return (buffer_len * sizeof(double));
}

// Calculate the eigenvalues and eigenvectors of a batch of matrices
template<typename T>
inline void batch_eig_cusolver(
    DeviceSolverDnHandle& cusolver_H, BatchEigParameter& param, 
    DeviceDenseVector<T>& mat, DeviceDenseVector<T>& W, DeviceDenseVector<T>& buffer, DeviceDenseVector<int>& info, 
    const int mat_size, const int batch_size, const size_t buffer_size    
) {
    int buffer_len = buffer_size / sizeof(double); // this size is len, not byte!
    CHECK_CUDA( cudaSetDevice(cusolver_H.gpu_id) );
    CHECK_CUSOLVER( cusolverDnDsyevjBatched(
        cusolver_H.cusolver_dn_handle, param.jobz, param.uplo,
        mat_size, mat.vals, mat_size, W.vals,
        buffer.vals, buffer_len,
        info.vals, param.syevj_param, batch_size
    ) );
    return;
}