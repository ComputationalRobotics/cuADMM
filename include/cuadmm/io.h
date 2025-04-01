#ifndef CUADMM_IO_H
#define CUADMM_IO_H

#include <string>
#include <vector>
#include <iostream>

// Read dense vector data from a .txt file.
void read_dense_vector_data(const std::string& filename, std::vector<double>& vals);
void read_dense_vector_data(const std::string& filename, std::vector<int>& vals);

// Read dense matrix data from a .txt file.
void read_sparse_vector_data(
    const std::string& filename,
    std::vector<int>& rows, std::vector<double>& vals
);

#endif // CUADMM_IO_H