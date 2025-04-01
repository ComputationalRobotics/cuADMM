#ifndef CUADMM_IO_H
#define CUADMM_IO_H

#include <string>
#include <vector>
#include <iostream>

// Read dense vector data from a .txt file.
void read_dense_vector_data(
    const std::string& filename,
    std::vector<double>& vals
);
void read_dense_vector_data(
    const std::string& filename,
    std::vector<int>& vals
);

// Read sparse vector data from a .txt file.
void read_sparse_vector_data(
    const std::string& filename,
    std::vector<int>& rows, std::vector<double>& vals
);

// Read sparse matrix data from a .txt file.
void read_COO_sparse_matrix_data(
    const std::string& filename,
    std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals
);

// Write dense vector data from to a .txt file.
void write_dense_vector_data(
    const std::string& filename, const std::vector<double>& vals,
    const int precision = 20
);
void write_dense_vector_data(
    const std::string& filename, const std::vector<int>& vals,
    const int precision = 30
);

// Write sparse matrix to a .txt file.
void write_sparse_matrix_data(
    const std::string& filename,
    const std::vector<int>& rows, const std::vector<int>& cols, const std::vector<double>& vals,
    const int precision = 20
);

#endif // CUADMM_IO_H