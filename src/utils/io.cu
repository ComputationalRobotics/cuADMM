/*

    utils/io.cu

    Defines functions for reading data from files.
    The data is read into vectors of doubles or integers, depending on the type of data.

    For dense vectors, the data is read from a file where each line contains a single value.
    For sparse vectors and matrices, the data is read from a file where each line contains three values: row index, column index, and value.

*/

#include <fstream>
#include <iomanip>
#include <algorithm>

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

// Read sparse matrix in COOrdinate format from a .txt file.
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

// Convert COO sparse matrix format to CSC sparse matrix format.
// Note: the CSC format generated will always be sorted,
// hence this function can also help to sort the COO format data.
void COO_to_CSC(
    std::vector<int>& col_ptrs, // pointers of col in CSC, of size (col_num+1, 0)
    std::vector<int>& col_ids, std::vector<int>& row_ids, std::vector<double>& vals, // triplets for the COO format
    const int nnz, const int col_num 
) {
    // check the input
    if (col_ptrs.size() != (col_num+1)) {
        std::cout << "WARNING: col_ptrs size is wrong and was resized to col_num+1" << std::endl;
        col_ptrs.clear();
        col_ptrs.resize(col_num+1);
    }

    // create nnz triplets, each containing (col_id, row_id, val)
    std::vector<Triplet> triplets;
    for (int i = 0; i < nnz; i++) {
        triplets.emplace_back(Triplet(col_ids[i], row_ids[i], vals[i]));
    }
    // sort the triplets by col_id and row_id (in lexicographical order)
    std::sort(
        triplets.begin(), triplets.end(),
        [](const Triplet& t1, const Triplet& t2) {
            if (t1.col_id != t2.col_id) {
                return t1.col_id < t2.col_id;
            } else {
                return t1.row_id < t2.row_id;
            }
        }
    );
    // fill the col_ptrs
    int id = 0;
    for (int i = 1; i < nnz; i++) {
        if (triplets[i-1].col_id < triplets[i].col_id) {
            int tmp = triplets[i-1].col_id;
            while (tmp < triplets[i].col_id) {
                id = id + 1;
                col_ptrs[id] = i;
                tmp = tmp + 1;
            }
        }
    }
    id = id + 1;
    // fill the remaining col_ptrs
    while (id <= col_num) {
        col_ptrs[id] = nnz;
        id = id + 1;
    }
    // permutate the col_ids, row_ids, and vals
    for (int i = 0; i < nnz; i++) {
        col_ids[i] = triplets[i].col_id;
        row_ids[i] = triplets[i].row_id;
        vals[i] = triplets[i].val;
    }
    return;
}


// Convert CSC sparse matrix format to COO sparse matrix format.
// Note: CSC data is sorted by column, while COO data is sorted by row
// After this function, CSC data's col_ptrs will be set as empty.
void CSC_to_COO(
    std::vector<int>& col_ptrs, // pointers of col in CSC, of size (col_num+1, 0)
    std::vector<int>& col_ids, std::vector<int>& row_ids, std::vector<double>& vals, // triplets for the COO format
    const int nnz, const int col_num 
) {
    // check the input
    if (col_ids.size() != nnz) {
        col_ids.clear();
        col_ids.resize(nnz);
    }
    // in CSC format, row_ids and vals is sorted by column
    // first fill the col_ids by column
    int id = 0;
    for (int i = 0; i < nnz; i++) {
        if (i == col_ptrs[id+1]) {
            id++;
        }
        col_ids[i] = id;
    }
    // generate nnz triplets, each containing (col_id, row_id, val)
    std::vector<Triplet> triplets;
    for (int i = 0; i < nnz; i++) {
        triplets.emplace_back(Triplet(col_ids[i], row_ids[i], vals[i]));
    }
    // sort by row
    std::sort(
        triplets.begin(), triplets.end(),
        [](const Triplet& t1, const Triplet& t2) {
            if (t1.row_id != t2.row_id) {
                return t1.row_id < t2.row_id;
            } else {
                return t1.col_id < t2.col_id;
            }
        }
    );
    // permutate the col_ids, row_ids, and vals
    for (int i = 0; i < nnz; i++) {
        col_ids[i] = triplets[i].col_id;
        row_ids[i] = triplets[i].row_id;
        vals[i] = triplets[i].val;
    }
    // set col_ptrs to empty
    col_ptrs.clear();
    return;
}