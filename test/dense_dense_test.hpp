#include "cuadmm/kernels.h"

TEST(DenseDense, AddInPlace)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vector(SIZE, 1.0);
    
    // copy the vector to the device
    DeviceDenseVector<double> vec1(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(vec1.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // copy the vector to the device
    DeviceDenseVector<double> vec2(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(vec2.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // add the vectors in place
    dense_vector_add_dense_vector(vec1, vec2, 2.0, 3.0);
    
    // retrieve the result
    std::vector<double> result(SIZE);
    CHECK_CUDA( cudaMemcpy(result.data(), vec1.vals, sizeof(double) * SIZE, cudaMemcpyDeviceToHost) );

    // check the result
    for (size_t i = 0; i < SIZE; i++) {
        EXPECT_DOUBLE_EQ(result[i], 5.0);
    }
}

TEST(DenseDense, Add3)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vector(SIZE, 1.0);
    
    // initialize the result vector
    DeviceDenseVector<double> vec1(GPU0, SIZE);

    // copy the vector to the device
    DeviceDenseVector<double> vec2(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(vec2.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // copy the vector to the device
    DeviceDenseVector<double> vec3(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(vec3.vals, vector.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // add the vectors in place
    dense_vector_add_dense_vector(vec1, vec2, vec3, 2.0, 3.0);
    
    // retrieve the result
    std::vector<double> result(SIZE);
    CHECK_CUDA( cudaMemcpy(result.data(), vec1.vals, sizeof(double) * SIZE, cudaMemcpyDeviceToHost) );

    // check the result
    for (size_t i = 0; i < SIZE; i++) {
        EXPECT_DOUBLE_EQ(result[i], 5.0);
    }
}

TEST(DenseDense, Add3Scalar)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create two vectors
    std::vector<double> vector1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> vector2 = {4.0, 5.0, 6.0, 7.0, 8.0};
    
    // initialize the result vector
    DeviceDenseVector<double> vec1(GPU0, 5);
    
    // copy the vectors to the device
    DeviceDenseVector<double> vec2(GPU0, 5);
    CHECK_CUDA( cudaMemcpy(vec2.vals, vector1.data(), sizeof(double) * 5, cudaMemcpyHostToDevice) );
    DeviceDenseVector<double> vec3(GPU0, 5);
    CHECK_CUDA( cudaMemcpy(vec3.vals, vector2.data(), sizeof(double) * 5, cudaMemcpyHostToDevice) );

    dense_vector_plus_dense_vector_mul_scalar(vec1, vec2, vec3, 2.0);
    
    // retrieve the result
    std::vector<double> result(5);
    CHECK_CUDA( cudaMemcpy(result.data(), vec1.vals, sizeof(double) * 5, cudaMemcpyDeviceToHost) );

    ASSERT_EQ(result, std::vector<double>({1.0 + 4.0*2.0, 2.0 + 5.0*2.0, 3.0 + 6.0*2.0, 4.0 + 7.0*2.0, 5.0 + 8.0*2.0}));
}

TEST(DenseDense, MulInPlace)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create two vectors
    std::vector<double> vector1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> vector2 = {4.0, 5.0, 6.0, 7.0, 8.0};
    
    // copy the vector to the device
    DeviceDenseVector<double> vec1(GPU0, 5);
    CHECK_CUDA( cudaMemcpy(vec1.vals, vector1.data(), sizeof(double) * 5, cudaMemcpyHostToDevice) );

    // copy the vector to the device
    DeviceDenseVector<double> vec2(GPU0, 5);
    CHECK_CUDA( cudaMemcpy(vec2.vals, vector2.data(), sizeof(double) * 5, cudaMemcpyHostToDevice) );

    // multiply the vectors in place
    dense_vector_mul_dense_vector(vec1, vec2);
    
    // retrieve the result
    std::vector<double> result(5);
    CHECK_CUDA( cudaMemcpy(result.data(), vec1.vals, sizeof(double) * 5, cudaMemcpyDeviceToHost) );

    ASSERT_EQ(result, std::vector<double>({4.0, 10.0, 18.0, 28.0, 40.0}));
}

TEST(DenseDense, Mul3)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create two vectors
    std::vector<double> vector1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> vector2 = {4.0, 5.0, 6.0, 7.0, 8.0};
    
    // initialize the result vector
    DeviceDenseVector<double> vec1(GPU0, 5);

    // copy the vector to the device
    DeviceDenseVector<double> vec2(GPU0, 5);
    CHECK_CUDA( cudaMemcpy(vec2.vals, vector1.data(), sizeof(double) * 5, cudaMemcpyHostToDevice) );

    // copy the vector to the device
    DeviceDenseVector<double> vec3(GPU0, 5);
    CHECK_CUDA( cudaMemcpy(vec3.vals, vector2.data(), sizeof(double) * 5, cudaMemcpyHostToDevice) );

    // multiply the vectors in place
    dense_vector_mul_dense_vector_mul_scalar(vec1, vec2, vec3, 2.0);
    
    // retrieve the result
    std::vector<double> result(5);
    CHECK_CUDA( cudaMemcpy(result.data(), vec1.vals, sizeof(double) * 5, cudaMemcpyDeviceToHost) );

    ASSERT_EQ(result, std::vector<double>({2.0*4.0, 2.0*10.0, 2.0*18.0, 2.0*28.0, 2.0*40.0}));
}

TEST(DenseDense, DivInPlace)
{
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create two vectors
    std::vector<double> vector1 = {1.0, 4.0, 9.0, 16.0, 25.0};
    std::vector<double> vector2 = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // copy the vectors to the device
    DeviceDenseVector<double> vec1(GPU0, 5);
    CHECK_CUDA( cudaMemcpy(vec1.vals, vector1.data(), sizeof(double) * 5, cudaMemcpyHostToDevice) );

    DeviceDenseVector<double> vec2(GPU0, 5);
    CHECK_CUDA( cudaMemcpy(vec2.vals, vector2.data(), sizeof(double) * 5, cudaMemcpyHostToDevice) );

    // multiply the vectors in place
    dense_vector_div_dense_vector_mul_scalar(vec1, vec2, 2.0);
    
    // retrieve the result
    std::vector<double> result(5);
    CHECK_CUDA( cudaMemcpy(result.data(), vec1.vals, sizeof(double) * 5, cudaMemcpyDeviceToHost) );

    ASSERT_EQ(result, std::vector<double>({2.0*1.0, 2.0*2.0, 2.0*3.0, 2.0*4.0, 2.0*5.0}));
}