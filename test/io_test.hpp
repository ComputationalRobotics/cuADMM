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