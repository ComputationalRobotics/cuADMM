#include "cuadmm/io.h"

TEST(IO, ReadDenseVectorDataDouble)
{
    std::string filename = "../test/data/dense_vector.txt";
    std::vector<double> vals;
    read_dense_vector_data(filename, vals);
    
    // Check the size of the vector
    EXPECT_EQ(vals, std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0}));
}

TEST(IO, ReadDenseVectorDataInt)
{
    std::string filename = "../test/data/dense_vector.txt";
    std::vector<int> vals;
    read_dense_vector_data(filename, vals);
    
    // Check the size of the vector
    EXPECT_EQ(vals, std::vector<int>({1, 2, 3, 4, 5}));
}

TEST(IO, ReadSparseVectorData)
{
    std::string filename = "../test/data/sparse_vector.txt";
    std::vector<int> rows;
    std::vector<double> vals;
    read_sparse_vector_data(filename, rows, vals);
    
    // Check the values of the vectors
    EXPECT_EQ(rows, std::vector<int>({0, 2, 4, 6, 8}));
    EXPECT_EQ(vals, std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0}));
}

TEST(IO, ReadCOOSparseMatrixData)
{
    std::string filename = "../test/data/sparse_matrix.txt";
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    read_COO_sparse_matrix_data(filename, rows, cols, vals);
    
    // Check the values of the vectors
    EXPECT_EQ(rows, std::vector<int>({0, 1, 0, 1, 0}));
    EXPECT_EQ(cols, std::vector<int>({0, 0, 1, 1, 2}));
    EXPECT_EQ(vals, std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0}));
}

TEST(IO, WriteDenseVectorDataDouble)
{
    std::string filename = "../test/data/dense_vector_out.txt";
    std::vector<double> vals = {1.0, 2.0, 3.0, 4.0, 5.0};
    write_dense_vector_data(filename, vals);
    
    // Read the data back and check
    std::vector<double> read_vals;
    read_dense_vector_data(filename, read_vals);
    EXPECT_EQ(read_vals, vals);
}

TEST(IO, WriteDenseVectorDataInt)
{
    std::string filename = "../test/data/dense_vector_out.txt";
    std::vector<int> vals = {1, 2, 3, 4, 5};
    write_dense_vector_data(filename, vals);
    
    // Read the data back and check
    std::vector<int> read_vals;
    read_dense_vector_data(filename, read_vals);
    EXPECT_EQ(read_vals, vals);
}

TEST(IO, WriteSparseMatrixData)
{
    std::string filename = "../test/data/sparse_matrix_out.txt";
    std::vector<int> rows = {0, 1, 0, 1, 0};
    std::vector<int> cols = {0, 0, 1, 1, 2};
    std::vector<double> vals = {1.0, 2.0, 3.0, 4.0, 5.0};
    write_sparse_matrix_data(filename, rows, cols, vals);
    
    // Read the data back and check
    std::vector<int> read_rows;
    std::vector<int> read_cols;
    std::vector<double> read_vals;
    read_COO_sparse_matrix_data(filename, read_rows, read_cols, read_vals);
    
    EXPECT_EQ(read_rows, rows);
    EXPECT_EQ(read_cols, cols);
    EXPECT_EQ(read_vals, vals);
}