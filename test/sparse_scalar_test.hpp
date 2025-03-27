#include <math.h>

#include "cuadmm/kernels.h"

TEST(SparseScalar, Mul)
{
    // create a handle
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vals(SIZE, 1.0);
    std::vector<int> indices;
    for (int i = 0; i < SIZE; ++i) {
        indices.push_back(2*i);
    }
    
    // copy the vector to the device
    DeviceSparseVector<double> sparse_vector(GPU0, 2*SIZE, SIZE);
    CHECK_CUDA( cudaMemcpy(sparse_vector.vals, vals.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );
    sparse_vector.indices = indices.data();

    // multiply the vector by 2
    sparse_vector_mul_scalar(sparse_vector, 2.0);
    //check the norm
    EXPECT_DOUBLE_EQ(sparse_vector.get_norm(handle), 2.0 * sqrt(SIZE));

    // multiply the vector by -pi
    sparse_vector_mul_scalar(sparse_vector, -M_PI);
    //check the norm
    EXPECT_DOUBLE_EQ(sparse_vector.get_norm(handle), 2 * M_PI * sqrt(SIZE));
}

TEST(SparseScalar, Div)
{
    // create a handle
    DeviceBlasHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // create a vector of ones
    const int SIZE = 10;
    std::vector<double> vals(SIZE, 1.0);
    std::vector<int> indices;
    for (int i = 0; i < SIZE; ++i) {
        indices.push_back(2*i);
    }
    
    // copy the vector to the device
    DeviceSparseVector<double> sparse_vector(GPU0, 2*SIZE, SIZE);
    CHECK_CUDA( cudaMemcpy(sparse_vector.vals, vals.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );
    sparse_vector.indices = indices.data();

    // divide the vector by 2
    sparse_vector_div_scalar(sparse_vector, 2.0);
    //check the norm
    EXPECT_DOUBLE_EQ(sparse_vector.get_norm(handle), 1/2.0 * sqrt(SIZE));

    // divide the vector by -pi
    sparse_vector_div_scalar(sparse_vector, -M_PI);
    //check the norm
    EXPECT_DOUBLE_EQ(sparse_vector.get_norm(handle), 1/2.0 * 1/M_PI * sqrt(SIZE));
}