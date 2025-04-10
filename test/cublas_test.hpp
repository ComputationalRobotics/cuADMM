#include "cuadmm/cublas.h"

TEST(CuBLAS, DenseMatrixMulBatch)
{
    const int batch_size = 2;
    const int mat_size = 2;
    DeviceBlasHandle handle(GPU0);
    handle.activate();

    // create a batch of matrices
    std::vector<double> mat2_host = {
        1.0, 3.0, 2.0, 4.0,
        5.0, 7.0, 6.0, 8.0,
    };
    std::vector<double> mat3_host = {
        1.0, 3.0, 2.0, 4.0,
        5.0, 7.0, 6.0, 8.0,
    };
    DeviceDenseVector<double> mat2(GPU0, mat_size * mat_size * batch_size);
    DeviceDenseVector<double> mat3(GPU0, mat_size * mat_size * batch_size);
    DeviceDenseVector<double> mat1(GPU0, mat_size * mat_size * batch_size);
    CHECK_CUDA( cudaMemcpy(mat2.vals, mat2_host.data(), sizeof(double) * mat_size * mat_size * batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(mat3.vals, mat3_host.data(), sizeof(double) * mat_size * mat_size * batch_size, cudaMemcpyHostToDevice) );

    // batch-wise multiply
    dense_matrix_mul_trans_batch(handle, mat1, mat2, mat3, mat_size, batch_size);

    // retrieve result
    std::vector<double> mat1_host(mat_size * mat_size * batch_size);
    CHECK_CUDA( cudaMemcpy(mat1_host.data(), mat1.vals, sizeof(double) * mat_size * mat_size * batch_size, cudaMemcpyDeviceToHost) );
    std::vector<double> expected_result = {
         5.0, 11.0,
        11.0, 25.0,

        5.0*5.0+6.0*6.0, 5.0*7.0+8.0*6.0,
        5.0*7.0+8.0*6.0, 7.0*7.0+8.0*8.0,
    };

    ASSERT_EQ(mat1_host, expected_result);
}

TEST(CuBLAS, AXpYCuBLAS)
{
    DeviceBlasHandle handle(GPU0);
    handle.activate();

    const int SIZE = 5;
    std::vector<double> x_host = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y_host = {5.0, 4.0, 3.0, 2.0, 1.0};
    
    // copy the vector to the device
    DeviceDenseVector<double> x(GPU0, SIZE);
    DeviceDenseVector<double> y(GPU0, SIZE);
    CHECK_CUDA( cudaMemcpy(x.vals, x_host.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(y.vals, y_host.data(), sizeof(double) * SIZE, cudaMemcpyHostToDevice) );

    // perform y = alpha * x + y
    const double alpha = 3.0;
    axpy_cublas(handle, x, y, alpha);

    // retrieve result
    std::vector<double> y_result(SIZE);
    CHECK_CUDA( cudaMemcpy(y_result.data(), y.vals, sizeof(double) * SIZE, cudaMemcpyDeviceToHost) );
    
    // expected result: y = 3.0 * x + y
    std::vector<double> expected_y = {
        3.0 * 1.0 + 5.0,
        3.0 * 2.0 + 4.0,
        3.0 * 3.0 + 3.0,
        3.0 * 4.0 + 2.0,
        3.0 * 5.0 + 1.0
    };
    
    ASSERT_EQ(y_result, expected_y);
}