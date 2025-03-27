#include <math.h>

#include "cuadmm/kernels.h"

TEST(DenseScalar, Mul1Vec)
{
    // create a handle
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vector(SIZE, 1.0);
    
    // copy the vector to the device
    DeviceDenseVector<double> dense_vector(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(dense_vector.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // multiply the vector by 2
    dense_vector_mul_scalar(dense_vector, 2.0);
    //check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), 2.0 * sqrt(SIZE));

    // multiply the vector by -pi
    dense_vector_mul_scalar(dense_vector, -M_PI);
    //check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), 2 * M_PI * sqrt(SIZE));
}

TEST(DenseScalar, Div1Vec)
{
    // create a handle
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vector(SIZE, 1.0);
    
    // copy the vector to the device
    DeviceDenseVector<double> dense_vector(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(dense_vector.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // divide the vector by 2
    dense_vector_div_scalar(dense_vector, 2.0);
    //check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), 1/2.0 * sqrt(SIZE));

    // multiply the vector by -pi
    dense_vector_div_scalar(dense_vector, -M_PI);
    //check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), 1/2.0 * 1/M_PI * sqrt(SIZE));
}

TEST(DenseScalar, Mul2Vec)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vector(SIZE, 1.0);
    
    // copy the vector to the device (in vec2)
    DeviceDenseVector<double> vec1(GPU0, SIZE);
    DeviceDenseVector<double> vec2(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(vec2.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // multiply vec2 by 2 and store the result in vec1
    dense_vector_mul_scalar(vec1, vec2, 2.0);
    //check the norms
    EXPECT_DOUBLE_EQ(vec1.get_norm(handle), 2.0 * sqrt(SIZE));
    EXPECT_DOUBLE_EQ(vec2.get_norm(handle), sqrt(SIZE));
}

TEST(DenseScalar, PositivePart)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vector = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    ASSERT_EQ(vector.size(), SIZE);
    
    // copy the vector to the device
    DeviceDenseVector<double> dense_vector(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(dense_vector.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // set the vector to its positive part
    max_dense_vector_zero(dense_vector);
    //check the norm
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), sqrt(SIZE/2));

    // multiply the vector by 2.0 and check the norm
    dense_vector_mul_scalar(dense_vector, 2.0);
    EXPECT_DOUBLE_EQ(dense_vector.get_norm(handle), 2.0 * sqrt(SIZE/2));
}