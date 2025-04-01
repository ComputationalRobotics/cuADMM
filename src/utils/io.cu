#include <fstream>

#include "cuadmm/io.h"

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
        rows.push_back(row);
        vals.push_back(val);
    }
    file.close();
    return;
}