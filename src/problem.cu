/*

    problem.cu

*/

#include "cuadmm/problem.h"
#include "cuadmm/io.h"

void Problem::from_txt(const std::string& prefix) {
    read_dense_vector_data(prefix + "X.txt", this->X_vals);
    this->vec_len = this->X_vals.size();
    read_dense_vector_data(prefix + "y.txt", this->y_vals);
    this->con_num = this->y_vals.size();
    read_dense_vector_data(prefix + "S.txt", this->S_vals);
    read_COO_sparse_matrix_data(prefix + "At.txt", this->At_csc_row_ids, this->At_coo_col_ids, this->At_csc_vals);
    this->At_nnz = this->At_csc_vals.size();
    COO_to_CSC(this->At_csc_col_ptrs, this->At_coo_col_ids, this->At_csc_row_ids, this->At_csc_vals, this->At_nnz, this->con_num);
    read_sparse_vector_data(prefix + "b.txt", this->b_indices, this->b_vals);
    this->b_nnz = this->b_vals.size();
    read_sparse_vector_data(prefix + "C.txt", this->C_indices, C_vals);
    this->C_nnz = this->C_vals.size();
    read_dense_vector_data(prefix + "blk.txt", this->blk_vals);
    this->mat_num = this->blk_vals.size();
    read_dense_vector_data(prefix + "sig.txt",this-> sig_vals);
}