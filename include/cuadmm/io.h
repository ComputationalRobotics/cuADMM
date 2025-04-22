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
    std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals,
    const bool transpose = false
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

// Triplet containing the column index, row index, and values
// (This class helps to convert COO sparse matrix data to CSC sparse matrix data.)
class Triplet {
    public:
        Triplet(int col_id, int row_id, double val): 
            col_id(col_id), row_id(row_id), val(val) {}

        int col_id;
        int row_id;
        double val;
};

// Convert COO sparse matrix format to CSC sparse matrix format.
// Note: the CSC format generated will always be sorted,
// hence this function can also help to sort the COO format data.
void COO_to_CSC(
    std::vector<int>& col_ptrs, // pointers of col in CSC, of size (col_num+1, 0)
    std::vector<int>& col_ids, std::vector<int>& row_ids, std::vector<double>& vals, // triplets for the COO format
    const int nnz, const int col_num 
);

// Convert CSC sparse matrix format to COO sparse matrix format.
// Note: CSC data is sorted by column, while COO data is sorted by row
// After this function, CSC data's col_ptrs will be set as empty.
void CSC_to_COO(
    std::vector<int>& col_ptrs, // pointers of col in CSC, of size (col_num+1, 0)
    std::vector<int>& col_ids, std::vector<int>& row_ids, std::vector<double>& vals, // triplets for the COO format
    const int nnz, const int col_num 
);

#endif // CUADMM_IO_H