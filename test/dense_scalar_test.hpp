#include <math.h>

#include "cuadmm/kernels.h"

TEST(DenseScalar, Simple)
{
    DeviceStream stream;
    stream.set_gpu_id(0);
    stream.activate();

    DeviceBlasHandle handle;
    handle.set_gpu_id(0);
    handle.activate(stream);

    DeviceDenseVector<double> dvec;
    dvec.allocate(0, 10);
}

TEST(DenseScalar, Norm)
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

    // multiply the vector by 2
    dense_vector_mul_scalar_kernel(dense_vector, 2.0);
    //check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), 2.0 * sqrt(SIZE));

    // multiply the vector by -pi
    dense_vector_mul_scalar_kernel(dense_vector, -M_PI);
    //check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), 2 * M_PI * sqrt(SIZE));
}