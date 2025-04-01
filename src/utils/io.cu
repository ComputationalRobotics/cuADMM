/*

    io.cu

    Defines functions for reading data from files.
    The data is read into vectors of doubles or integers, depending on the type of data.

    For dense vectors, the data is read from a file where each line contains a single value.
    For sparse vectors and matrices, the data is read from a file where each line contains three values: row index, column index, and value.

*/

#include <fstream>
#include <iomanip>

#include "cuadmm/io.h"

// Read dense vector data from a .txt file.
// type: double
void read_dense_vector_data(const std::string& filename, std::vector<double>& vals)
{
    if (!vals.empty()) {
        std::cout << "WARNING: buffer for data reading is not empty!" << std::endl;
        vals.clear();
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: could not open file " << filename << std::endl;
        throw std::runtime_error("Failed to open file.");
        return;
    }

    double val;
    while (file >> val) {
        vals.push_back(val);
    }
    return;
}

// Read dense vector data from a .txt file.
// type: int
void read_dense_vector_data(const std::string& filename, std::vector<int>& vals)
{
    if (!vals.empty()) {
        std::cout << "WARNING: buffer for data reading is not empty!" << std::endl;
        vals.clear();
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: could not open file " << filename << std::endl;
        throw std::runtime_error("Failed to open file.");
        return;
    }

    double val;
    while (file >> val) {
        vals.push_back(val);
    }
    return;
}

// Read sparse vector data from a .txt file.
// type: double
void read_sparse_vector_data(
    const std::string& filename,
    std::vector<int>& rows, std::vector<double>& vals
) {
    if (!rows.empty() || !vals.empty()) {
        std::cout << "WARNING: buffer for data reading is not empty!" << std::endl;
        rows.clear();
        vals.clear();
    }
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        throw std::runtime_error("Failed to open file.");
    }
    int row, col; 
    double val;
    while (file >> row >> col >> val) {
        if (col != 0) {
            std::cerr << "WARNING: sparse vector data has a non-zero column index." << std::endl;
        }
        rows.push_back(row);
        vals.push_back(val);
    }
    file.close();
    return;
}

// Read sparse matrix data from a .txt file.
void read_COO_sparse_matrix_data(
    const std::string& filename,
    std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals
)
{
    if (!rows.empty() || !cols.empty() || !vals.empty()) {
        std::cout << "WARNING: buffer for data reading is not empty!" << std::endl;
        rows.clear();
        cols.clear();
        vals.clear();
    }
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        throw std::runtime_error("Failed to open file.");
    }
    int row, col; 
    double val;
    while (file >> row >> col >> val) {
        rows.push_back(row); 
        cols.push_back(col);
        vals.push_back(val);
    }
    file.close();
    return;
}

// Write dense vector data from to a .txt file.
// type: double
void write_dense_vector_data(
    const std::string& filename, const std::vector<double>& vals,
    const int precision
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        throw std::runtime_error("Failed to open file.");
    }
    file << std::setprecision(precision);
    for (int i = 0; i < vals.size(); i++) {
        file << vals[i] << std::endl;
    }
    file.close();
    return;
}

// Write dense vector data to a .txt file.
// type: int
void write_dense_vector_data(
    const std::string& filename, const std::vector<int>& vals,
    const int precision
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        throw std::runtime_error("Failed to open file.");
    }
    file << std::setprecision(precision);
    for (int i = 0; i < vals.size(); i++) {
        file << vals[i] << std::endl;
    }
    file.close();
    return;
}

// Write sparse matrix to a .txt file.
void write_sparse_matrix_data(
    const std::string& filename,
    const std::vector<int>& rows, const std::vector<int>& cols, const std::vector<double>& vals,
    const int precision
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        throw std::runtime_error("Failed to open file.");
    }
    file << std::setprecision(precision);
    for (int i = 0; i < vals.size(); i++) {
        file << rows[i] << " " << cols[i] << " " << vals[i] << std::endl;
    }
    file.close();
    return;
}