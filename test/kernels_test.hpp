#include "cuadmm/kernels.h"
#include "cuadmm/utils.h"

TEST(Kernels, Permutation)
{
    // Create host vectors
    std::vector<double> vec1_host(10);
    std::vector<double> vec2_host = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> perm_host = {6, 4, 1, 3, 0, 5, 2, 8, 7, 9};


    // Create a dense vector
    DeviceDenseVector<double> vec1(GPU0, 10);
    DeviceDenseVector<double> vec2(GPU0, 10);
    DeviceDenseVector<int> perm(GPU0, 10);

    // Copy data to the device
    CHECK_CUDA( cudaMemcpy(vec1.vals, vec1_host.data(), sizeof(double) * 10, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(vec2.vals, vec2_host.data(), sizeof(double) * 10, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(perm.vals, perm_host.data(), sizeof(int) * 10, cudaMemcpyHostToDevice) );

    // Perform the permutation
    // i.e. vec1[perm[i]] = vec2[i]
    perform_permutation(vec1, vec2, perm);

    // Retrieve the result from the device
    CHECK_CUDA( cudaMemcpy(vec1_host.data(), vec1.vals, sizeof(double) * 10, cudaMemcpyDeviceToHost) );

    // Check the result
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(vec1_host[perm_host[i]], vec2_host[i]);
    }
}

TEST(Kernels, SparseMatrixNorm)
{
    std::string filename = "../test/data/sparse_matrix_coo.txt";
    int col_num = 4; // Number of columns in the matrix
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    read_COO_sparse_matrix_data(filename, rows, cols, vals);

    // Convert COO to CSC
    std::vector<int> col_ptrs(col_num + 1, 0);
    COO_to_CSC(col_ptrs, cols, rows, vals, vals.size(), col_num);

    // Create a sparse matrix by copying the data to the device
    DeviceSparseMatrixDoubleCSC At(GPU0, 4, 4, vals.size());
    CHECK_CUDA( cudaMemcpy(At.col_ptrs, col_ptrs.data(), sizeof(int) * (col_num + 1), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(At.row_ids, rows.data(), sizeof(int) * vals.size(), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(At.vals, vals.data(), sizeof(double) * vals.size(), cudaMemcpyHostToDevice) );

    // Create a norm vector
    DeviceDenseVector<double> normA(GPU0, 4);

    // Compute the norm and normalize
    get_normA(At, normA);
    normA.print();

    // Retrieve the result from the device
    std::vector<double> normA_host(4);
    CHECK_CUDA( cudaMemcpy(normA_host.data(), normA.vals, sizeof(double) * 4, cudaMemcpyDeviceToHost) );
    std::vector<double> At_host(vals.size());
    CHECK_CUDA( cudaMemcpy(At_host.data(), At.vals, sizeof(double) * vals.size(), cudaMemcpyDeviceToHost) );

    // Check the results
    EXPECT_EQ(normA_host, std::vector<double>({
        std::sqrt(10.0*10.0 + 30.0*30.0),
        std::sqrt(20.0*20.0 + 60.0*60.0),
        40.0,
        50.0
    }));
    EXPECT_EQ(At_host, std::vector<double>({
        10.0/std::sqrt(10.0*10.0 + 30.0*30.0),
        30.0/std::sqrt(10.0*10.0 + 30.0*30.0),
        20.0/std::sqrt(20.0*20.0 + 60.0*60.0),
        60.0/std::sqrt(20.0*20.0 + 60.0*60.0),
        40.0/40.0,
        50.0/50.0
    }));

}

TEST(Kernels, DiagonalBatch)
{
    // Create a 5x5 dense matrix
    const int mat_size = 5;
    std::vector<double> mat2_host = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0
    };
    DeviceDenseVector<double> mat2(GPU0, mat_size * mat_size);
    CHECK_CUDA( cudaMemcpy(mat2.vals, mat2_host.data(), sizeof(double) * mat_size * mat_size, cudaMemcpyHostToDevice) );

    // Create a dense vector
    std::vector<double> vec_host = {1.0, 2.0, 3.0, 4.0, 5.0};
    DeviceDenseVector<double> vec(GPU0, mat_size);
    CHECK_CUDA( cudaMemcpy(vec.vals, vec_host.data(), sizeof(double) * mat_size, cudaMemcpyHostToDevice) );

    // Create a result matrix
    DeviceDenseVector<double> mat1(GPU0, mat_size * mat_size);
    dense_matrix_mul_diag_batch(mat1, mat2, vec, mat_size);

    // Retrieve the result from the device
    std::vector<double> mat1_host(mat_size * mat_size);
    CHECK_CUDA( cudaMemcpy(mat1_host.data(), mat1.vals, sizeof(double) * mat_size * mat_size, cudaMemcpyDeviceToHost) );

    // Check the result
    std::vector<double> expected_result = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0 * 2.0, 7.0 * 2.0, 8.0 * 2.0, 9.0 * 2.0, 10.0 * 2.0,
        11.0 * 3.0, 12.0 * 3.0, 13.0 * 3.0, 14.0 * 3.0, 15.0 * 3.0,
        16.0 * 4.0, 17.0 * 4.0, 18.0 * 4.0, 19.0 * 4.0, 20.0 * 4.0,
        21.0 * 5.0, 22.0 * 5.0, 23.0 * 5.0, 24.0 * 5.0, 25.0 * 5.0
    };
}