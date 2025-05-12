/*

    problem.cu

*/

#include "cuadmm/problem.h"
#include "cuadmm/io.h"
#include <algorithm>

// TODO: throw an error if the files do not exist
void Problem::from_txt(const std::string& prefix, bool warm_start) {
    read_dense_vector_data(prefix + "blk.txt", this->blk_vals);
    this->mat_num = this->blk_vals.size();

    if (warm_start) {
        read_dense_vector_data(prefix + "X.txt", this->X_vals);
        this->vec_len = this->X_vals.size();
    
        read_dense_vector_data(prefix + "y.txt", this->y_vals);
        this->con_num = this->y_vals.size();
    
        read_dense_vector_data(prefix + "S.txt", this->S_vals);
    } else {
        this->vec_len = 0;
        for (int i = 0; i < this->blk_vals.size(); i++) {
            // each matrix is symmetric
            this->vec_len += this->blk_vals[i] * (this->blk_vals[i] + 1) / 2;
        }

        // we have to read the con_num from the file
        // since the vectors do not contain the information
        // (they are sparse)
        std::vector<int> con_num_vec;
        read_dense_vector_data(prefix + "con_num.txt", con_num_vec);
        this->con_num = con_num_vec[0];
    }

    read_COO_sparse_matrix_data(prefix + "At.txt", this->At_csc_row_ids, this->At_coo_col_ids, this->At_csc_vals);
    this->At_nnz = this->At_csc_vals.size();
    COO_to_CSC(this->At_csc_col_ptrs, this->At_coo_col_ids, this->At_csc_row_ids, this->At_csc_vals, this->At_nnz, this->con_num);

    read_sparse_vector_data(prefix + "b.txt", this->b_indices, this->b_vals);
    this->b_nnz = this->b_vals.size();

    read_sparse_vector_data(prefix + "C.txt", this->C_indices, C_vals);
    this->C_nnz = this->C_vals.size();

    /* check if the dimensions are correct */
    int max_col_id = *std::max_element(this->At_csc_row_ids.begin(), this->At_csc_row_ids.end());
    if (max_col_id != this->vec_len - 1) {
        std::cerr << "WARNING: the largest column index in At is different from the specified column number!\n" << std::endl;
    }

    int max_row_id = *std::max_element(this->At_coo_col_ids.begin(), this->At_coo_col_ids.end());
    if (max_row_id != this->con_num - 1) {
        std::cerr << "WARNING: the largest row index in At is different from the SDP vector length!\n" << std::endl;
    }

    if (warm_start && this->vec_len != this->X_vals.size()) {
        std::cerr << "ERROR: the length of warmstarted X does not match the vector length." << std::endl;
        exit(1);
    }

    /* display problem stats */
    std::cout << "Loaded problem from " << prefix << std::endl;
    std::cout << "              vector length: " << this->vec_len << std::endl;
    std::cout << "      number of constraints: " << this->con_num << std::endl;
    std::cout << "           number of blocks: " << this->mat_num << std::endl;
    std::cout << "  number of non-zeros in At: " << this->At_nnz << std::endl;
    std::cout << "   number of non-zeros in b: " << this->b_nnz << std::endl;
    std::cout << "   number of non-zeros in C: " << this->C_nnz << std::endl;

    // read_dense_vector_data(prefix + "sig.txt",this-> sig_vals);
}

// void Problem::from_sedumi_txt(const std::string& prefix) {
//     read_COO_sparse_matrix_data(prefix + "At.txt", this->At_csc_row_ids, this->At_coo_col_ids, this->At_csc_vals);
//     this->At_nnz = this->At_csc_vals.size();
//     COO_to_CSC(this->At_csc_col_ptrs, this->At_coo_col_ids, this->At_csc_row_ids, this->At_csc_vals, this->At_nnz, this->con_num);
    
//     read_sparse_vector_data(prefix + "b.txt", this->b_indices, this->b_vals);
//     this->b_nnz = this->b_vals.size();

//     read_sparse_vector_data(prefix + "c.txt", this->C_indices, C_vals);
//     this->C_nnz = this->C_vals.size();

//     read_dense_vector_data(prefix + "blk.txt", this->blk_vals);
//     this->mat_num = this->blk_vals.size();
//     // this->vec_len = ;
// }