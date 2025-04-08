#include "cuadmm/cusparse.h"

TEST(CuSPARSE, CSCtoCSR)
{
    DeviceSparseHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // Create a sparse matrix in CSC format
    std::vector<int> col_ptrs = {0, 2, 4, 5, 6};
    std::vector<int> rows = {0, 2, 1, 3, 2, 2};
    std::vector<double> vals = {10.0, 30.0, 20.0, 60.0, 40.0, 50.0};
    DeviceSparseMatrixDoubleCSC csc_matrix(GPU0, 4, 4, 6);
    CHECK_CUDA( cudaMemcpy(csc_matrix.col_ptrs, col_ptrs.data(), sizeof(int) * (csc_matrix.col_size + 1), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(csc_matrix.row_ids, rows.data(), sizeof(int) * csc_matrix.nnz, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(csc_matrix.vals, vals.data(), sizeof(double) * csc_matrix.nnz, cudaMemcpyHostToDevice) );

    // Convert it to CSR format using cuSPARSE
    DeviceSparseMatrixDoubleCSR csr_matrix(GPU0, 4, 4, 6);
    size_t buffer_size = CSC_to_CSR_get_buffersize_cusparse(handle, csc_matrix, csr_matrix);
    DeviceDenseVector<double> buffer(GPU0, buffer_size);
    CSC_to_CSR_cusparse(handle, csc_matrix, csr_matrix, buffer);

    // Retrieve the CSR format data
    std::vector<int> csr_row_ptrs(5);
    std::vector<int> csr_col_ids(6);
    std::vector<double> csr_vals(6);
    CHECK_CUDA( cudaMemcpy(csr_row_ptrs.data(), csr_matrix.row_ptrs, sizeof(int) * (csr_matrix.row_size + 1), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(csr_col_ids.data(), csr_matrix.col_ids, sizeof(int) * csr_matrix.nnz, cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(csr_vals.data(), csr_matrix.vals, sizeof(double) * csr_matrix.nnz, cudaMemcpyDeviceToHost) );

    // Check the values of the CSR format
    std::vector<int> expected_row_ptrs = {0, 1, 2, 5, 6};
    std::vector<int> expected_col_ids = {0, 1, 0, 2, 3, 1};
    std::vector<double> expected_vals = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    EXPECT_EQ(csr_row_ptrs, expected_row_ptrs);
    EXPECT_EQ(csr_col_ids, expected_col_ids);
    EXPECT_EQ(csr_vals, expected_vals);
}

TEST(CuSPARSE, SpMV)
{
    DeviceSparseHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // Create a sparse matrix in CSR format
    std::vector<int> row_ptrs = {0, 1, 2, 5, 6};
    std::vector<int> col_ids = {0, 1, 0, 2, 3, 1};
    std::vector<double> vals = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    DeviceSparseMatrixDoubleCSR csr_matrix(GPU0, 4, 4, 6);
    CHECK_CUDA( cudaMemcpy(csr_matrix.row_ptrs, row_ptrs.data(), sizeof(int) * (csr_matrix.row_size + 1), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(csr_matrix.col_ids, col_ids.data(), sizeof(int) * csr_matrix.nnz, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(csr_matrix.vals, vals.data(), sizeof(double) * csr_matrix.nnz, cudaMemcpyHostToDevice) );

    // hence A is:
    // [[10,  0,  0,  0]
    //  [ 0, 20,  0,  0]
    //  [30,  0, 40, 50]
    // [ 0, 60,  0,  0]]

    // Create a dense vector
    std::vector<double> x_vals = {1.0, 2.0, 3.0, 4.0};
    DeviceDenseVector<double> x(GPU0, x_vals.size());
    CHECK_CUDA( cudaMemcpy(x.vals, x_vals.data(), sizeof(double) * x_vals.size(), cudaMemcpyHostToDevice) );

    // Create another vector (input/output)
    DeviceDenseVector<double> y(GPU0, x_vals.size());
    std::vector<double> y_vals = {5.0, 6.0, 7.0, 8.0};
    CHECK_CUDA( cudaMemcpy(y.vals, y_vals.data(), sizeof(double) * y_vals.size(), cudaMemcpyHostToDevice) );

    // Perform the SpMV operation
    double alpha = 2.0;
    double beta = 3.0;
    size_t buffer_size = SpMV_get_buffersize_cusparse(handle, csr_matrix, x, y, alpha, beta);
    DeviceDenseVector<double> buffer(GPU0, buffer_size);
    SpMV_cusparse(
        handle, 
        csr_matrix, x, y,
        alpha, beta, buffer
    );

    // Retrieve the result
    CHECK_CUDA( cudaMemcpy(y_vals.data(), y.vals, sizeof(double) * y_vals.size(), cudaMemcpyDeviceToHost) );

    // Check the result
    // Expected result: y = 2 * A * x + 3 * y
    std::vector<double> expected_y_vals = {
        2 * (10.0 * 1.0) + 3 * 5.0,
        2 * (20.0 * 2.0) + 3 * 6.0,
        2 * (30.0 * 1.0 + 40.0 * 3.0 + 50.0 * 4.0) + 3 * 7.0,
        2 * (60.0 * 1.0) + 3 * 8.0
    };
}

TEST(CuSPARSE, axpby)
{
    DeviceSparseHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // Create a sparse vector
    std::vector<int> indices = {0, 2, 4};
    std::vector<double> vals = {1.0, 2.0, 3.0};
    DeviceSparseVector<double> x(GPU0, 5, 3);
    CHECK_CUDA( cudaMemcpy(x.vals, vals.data(), sizeof(double) * vals.size(), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(x.indices, indices.data(), sizeof(int) * indices.size(), cudaMemcpyHostToDevice) );

    // Create a dense vector
    std::vector<double> y_vals = {4.0, 5.0, 6.0, 7.0, 8.0};
    DeviceDenseVector<double> y(GPU0, y_vals.size());
    CHECK_CUDA( cudaMemcpy(y.vals, y_vals.data(), sizeof(double) * y_vals.size(), cudaMemcpyHostToDevice) );

    // Perform the axpby operation
    double alpha = 2.0;
    double beta = 3.0;
    axpby_cusparse(handle, x, y, alpha, beta);

    // Retrieve the result
    CHECK_CUDA( cudaMemcpy(y_vals.data(), y.vals, sizeof(double) * y_vals.size(), cudaMemcpyDeviceToHost) );

    // Check the result
    // Expected result: y = alpha * x + beta * y
    std::vector<double> expected_y_vals = {
        2.0 * 1.0 + 3.0 * 4.0,
        2.0 * 0.0 + 3.0 * 5.0,
        2.0 * 2.0 + 3.0 * 6.0,
        2.0 * 0.0 + 3.0 * 7.0,
        2.0 * 3.0 + 3.0 * 8.0
    };
}

TEST(CuSPARSE, SpVV)
{
    DeviceSparseHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    // Create a sparse vector
    std::vector<int> indices = {0, 2, 4};
    std::vector<double> vals = {1.0, 2.0, 3.0};
    DeviceSparseVector<double> x(GPU0, 5, 3);
    CHECK_CUDA( cudaMemcpy(x.vals, vals.data(), sizeof(double) * vals.size(), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(x.indices, indices.data(), sizeof(int) * indices.size(), cudaMemcpyHostToDevice) );

    // Create a dense vector
    std::vector<double> y_vals = {4.0, 5.0, 6.0, 7.0, 8.0};
    DeviceDenseVector<double> y(GPU0, y_vals.size());
    CHECK_CUDA( cudaMemcpy(y.vals, y_vals.data(), sizeof(double) * y_vals.size(), cudaMemcpyHostToDevice) );

    // Perform the SpVV operation
    double buffer_size = SparseVV_get_buffersize_cusparse(handle, x, y);
    DeviceDenseVector<double> buffer(GPU0, buffer_size);

    double inprod = SparseVV_cusparse(handle, x, y, buffer);

    // Check the result
    // Expected result: inprod = sum(x[i] * y[i])
    double expected_inprod =
          1.0 * 4.0 
        + 0.0 * 5.0
        + 2.0 * 6.0 
        + 0.0 * 7.0
        + 3.0 * 8.0
    ;
    EXPECT_DOUBLE_EQ(inprod, expected_inprod);
}