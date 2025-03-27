#include "cuadmm/memory.h"
#include "cuadmm/check.h"

/* DeviceStream */ 
TEST(Memory, DeviceStream)
{
    DeviceStream device_stream;
    device_stream.set_gpu_id(0);
    device_stream.activate();
    device_stream.~DeviceStream();
}

/* DeviceBlasHandle */ 
TEST(Memory, SimpleDeviceBlasHandle)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate();
    handle.~DeviceBlasHandle();
}

TEST(Memory, DeviceBlasHandleWithStream)
{
    DeviceStream device_stream;
    device_stream.set_gpu_id(0);
    device_stream.activate();

    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate(device_stream);
    handle.~DeviceBlasHandle();
    device_stream.~DeviceStream();
}

/* DeviceSolverDnHandle */
TEST(Memory, DeviceSolverDnHandle)
{
    DeviceSolverDnHandle handle;
    handle.set_gpu_id(0);
    handle.activate();
    handle.~DeviceSolverDnHandle();
}

TEST(Memory, DeviceSolverDnHandleWithStream)
{
    DeviceStream device_stream;
    device_stream.set_gpu_id(0);
    device_stream.activate();

    DeviceSolverDnHandle handle;
    handle.set_gpu_id(0);
    handle.activate(device_stream);
    handle.~DeviceSolverDnHandle();
    device_stream.~DeviceStream();
}

/* DeviceSparseHandle */
TEST(Memory, DeviceSparseHandle)
{
    DeviceSparseHandle handle;
    handle.set_gpu_id(0);
    handle.activate();
    handle.~DeviceSparseHandle();
}

TEST(Memory, DeviceSparseHandleWithStream)
{
    DeviceStream device_stream;
    device_stream.set_gpu_id(0);
    device_stream.activate();

    DeviceSparseHandle handle;
    handle.set_gpu_id(0);
    handle.activate(device_stream);
    handle.~DeviceSparseHandle();
    device_stream.~DeviceStream();
}

/* HostDenseVector */
TEST(Memory, HostDenseVector)
{
    HostDenseVector<double> host_dense_vector_double;
    host_dense_vector_double.allocate(10);
    host_dense_vector_double.~HostDenseVector();

    HostDenseVector<float> host_dense_vector_float;
    host_dense_vector_float.allocate(10);
    host_dense_vector_float.~HostDenseVector();

    HostDenseVector<int> host_dense_vector_int;
    host_dense_vector_int.allocate(10);
    host_dense_vector_int.~HostDenseVector();
}

/* DeviceDenseVector */
TEST(Memory, DeviceDenseVector)
{
    // create a handle
    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate();

    // test with different types
    DeviceDenseVector<double> device_dense_vector_double;
    device_dense_vector_double.allocate(0, 10);
    device_dense_vector_double.~DeviceDenseVector();

    DeviceDenseVector<float> device_dense_vector_float;
    device_dense_vector_float.allocate(0, 10);
    device_dense_vector_float.~DeviceDenseVector();

    DeviceDenseVector<int> device_dense_vector_int;
    device_dense_vector_int.allocate(0, 10);
    device_dense_vector_int.~DeviceDenseVector();

    DeviceDenseVector<size_t> device_dense_vector_size_t;
    device_dense_vector_size_t.allocate(0, 10);
    device_dense_vector_size_t.~DeviceDenseVector();
}

TEST(Memory, DeviceDenseVectorNormZero)
{
    // create a handle
    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate();

    // create a vector of zeros
    const int SIZE = 10;
    std::vector<double> vector(SIZE, 0.0);
    
    // copy the vector to the device
    DeviceDenseVector<double> dense_vector(0, SIZE);
    dense_vector.allocate(0, SIZE);
    CHECK_CUDA( cudaMemcpy(&dense_vector, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), 0.0);
}

TEST(Memory, DeviceDenseVectorNormNonZero)
{
    // create a handle
    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vector(SIZE, 1.0);
    
    // copy the vector to the device
    DeviceDenseVector<double> dense_vector(0, SIZE);
    CHECK_CUDA( cudaMemcpy(dense_vector.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), std::sqrt(SIZE));
}

/* DeviceSpMatDoubleCSR */
TEST(Memory, DeviceSpMatDoubleCSC)
{
    DeviceSpMatDoubleCSC mat;
    mat.allocate(0, 10, 10, 10);
    mat.~DeviceSpMatDoubleCSC();
}

/* DeviceSpMatDoubleCSR */
TEST(Memory, DeviceSpMatDoubleCSR)
{
    DeviceSpMatDoubleCSR mat;
    mat.allocate(0, 10, 10, 10);
    mat.~DeviceSpMatDoubleCSR();
}