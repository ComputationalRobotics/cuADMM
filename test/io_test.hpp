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

TEST(IO, COOtoCSC)
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

    // Check the values of the vectors
    EXPECT_EQ(col_ptrs, std::vector<int>({0, 2, 4, 5, 6}));
    EXPECT_EQ(rows, std::vector<int>({0, 2, 1, 3, 2, 2}));
    EXPECT_EQ(vals, std::vector<double>({10.0, 30.0, 20.0, 60.0, 40.0, 50.0}));
}

TEST(IO, CSCtoCOO)
{
    std::string filename = "../test/data/sparse_matrix_coo_sorted.txt";
    int col_num = 4; // Number of columns in the matrix
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    read_COO_sparse_matrix_data(filename, rows, cols, vals);
    std::vector<int> new_rows = rows;
    std::vector<int> new_cols = cols;
    std::vector<double> new_vals = vals;
    
    // Convert COO to CSC
    std::vector<int> col_ptrs(col_num + 1, 0);
    COO_to_CSC(col_ptrs, cols, rows, vals, vals.size(), col_num);

    // Convert CSC back to COO
    COO_to_CSC(col_ptrs, cols, rows, vals, vals.size(), col_num);

    // Check the values of the vectors
    EXPECT_EQ(rows, new_rows);
    EXPECT_EQ(cols, new_cols);
    EXPECT_EQ(vals, new_vals);
}