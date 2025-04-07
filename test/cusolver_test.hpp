#include "cuadmm/cusolver.h"

TEST(CuSOLVER, SingleEigGetBufferSize)
{
    DeviceSolverDnHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    SingleEigParameter param(GPU0);

    const int mat_size = 2;
    DeviceDenseVector<double> mat(GPU0, mat_size * mat_size);
    DeviceDenseVector<double> W(GPU0, mat_size);
    DeviceDenseVector<double> buffer(GPU0, 1);
    HostDenseVector<double> buffer_host;
    DeviceDenseVector<int> info(GPU0, 1);

    size_t buffer_size = 0;
    size_t buffer_size_host = 0;

    single_eig_get_buffersize_cusolver(handle, param, mat, W, 
        mat_size, &buffer_size, &buffer_size_host
    );
}

TEST(CuSOLVER, SingleEig)
{
    
}