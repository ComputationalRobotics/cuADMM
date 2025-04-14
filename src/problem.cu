/*

    problem.cu

*/

#include "cuadmm/problem.h"
#include "cuadmm/io.h"

void Problem::from_txt(const std::string& prefix) {
    read_dense_vector_data(prefix + "X.txt", this->cpu_X_vals);
    this->vec_len = this->cpu_X_vals.size();
    read_dense_vector_data(prefix + "y.txt", this->cpu_y_vals);
    this->con_num = this->cpu_y_vals.size();
    read_dense_vector_data(prefix + "S.txt", this->cpu_S_vals);
    read_COO_sparse_matrix_data(prefix + "At.txt", this->cpu_At_csc_row_ids, this->cpu_At_coo_col_ids, this->cpu_At_csc_vals);
    this->At_nnz = this->cpu_At_csc_vals.size();
    COO_to_CSC(this->cpu_At_csc_col_ptrs, this->cpu_At_coo_col_ids, this->cpu_At_csc_row_ids, this->cpu_At_csc_vals, this->At_nnz, this->con_num);
    read_sparse_vector_data(prefix + "b.txt", this->cpu_b_indices, this->cpu_b_vals);
    this->b_nnz = this->cpu_b_vals.size();
    read_sparse_vector_data(prefix + "C.txt", this->cpu_C_indices, cpu_C_vals);
    this->C_nnz = this->cpu_C_vals.size();
    read_dense_vector_data(prefix + "blk.txt", this->cpu_blk_vals);
    this->mat_num = this->cpu_blk_vals.size();
    read_dense_vector_data(prefix + "sig.txt",this-> cpu_sig_vals);
}