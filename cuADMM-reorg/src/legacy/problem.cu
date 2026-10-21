/*

    problem.cu

*/

#include "cuadmm/problem.h"
#include "cuadmm/io.h"
#include <algorithm>
#include <stdexcept>
#include <string>

void Problem::from_txt(std::string &prefix, bool warm_start)
{
    // if prefix does not end with a slash, add it
    if (prefix.back() != '/')
    {
        prefix += '/';
    }

    read_blk(prefix + "blk.txt", this->blk_vals);
    this->mat_num = this->blk_vals.size(); // TODO

    this->vec_len = 0;
    for (int i = 0; i < this->blk_vals.size(); i++)
    {
        auto [blk_type, blk_size] = this->blk_vals[i];

        switch (blk_type)
        {
        case 's': // symmetric matrix
            // size of the upper triangular part of a symmetric matrix
            this->vec_len += blk_size * (blk_size + 1) / 2;
            break;
        case 'u': // free variable
            this->vec_len += blk_size;
            break;
        case 'l': // non-negative variable
            this->vec_len += blk_size;
            break;
        default:
            throw std::invalid_argument(std::string("unknown block type '") + blk_type + "' in " + prefix + "blk.txt");
        }
    }

    // we have to read the con_num from the file
    // since the vectors do not contain the information
    // (they are sparse)
    std::vector<int> con_num_vec;
    read_dense_vector_data(prefix + "con_num.txt", con_num_vec);
    if (con_num_vec.empty() || con_num_vec[0] <= 0)
        throw std::invalid_argument("con_num.txt in " + prefix + " must contain a positive number of constraints");
    this->con_num = con_num_vec[0];

    if (warm_start)
    {
        read_dense_vector_data(prefix + "X.txt", this->X_vals);
        read_dense_vector_data(prefix + "y.txt", this->y_vals);
        read_dense_vector_data(prefix + "S.txt", this->S_vals);
        if ((int)this->X_vals.size() != this->vec_len || (int)this->S_vals.size() != this->vec_len ||
            (int)this->y_vals.size() != this->con_num)
            throw std::invalid_argument(
                "warm start in " + prefix + ": X.txt, S.txt must have " + std::to_string(this->vec_len) +
                " entries and y.txt " + std::to_string(this->con_num));
    }

    read_COO_sparse_matrix_data(prefix + "At.txt", this->At_csc_row_ids, this->At_coo_col_ids, this->At_csc_vals);
    this->At_nnz = this->At_csc_vals.size();
    COO_to_CSC(this->At_csc_col_ptrs, this->At_coo_col_ids, this->At_csc_row_ids, this->At_csc_vals, this->At_nnz, this->con_num);

    read_sparse_vector_data(prefix + "b.txt", this->b_indices, this->b_vals);
    this->b_nnz = this->b_vals.size();

    read_sparse_vector_data(prefix + "C.txt", this->C_indices, C_vals);
    this->C_nnz = this->C_vals.size();

    /* check the dimensions: indices outside the problem would be read out of bounds on the GPU */
    // (the constraint indices of At are checked by COO_to_CSC)
    auto check_indices = [&](const std::vector<int> &ids, int size, const std::string &what)
    {
        for (int id : ids)
            if (id < 0 || id >= size)
                throw std::invalid_argument(
                    what + " in " + prefix + " has index " + std::to_string(id) + " outside [0, " + std::to_string(size) + ")");
    };
    check_indices(this->At_csc_row_ids, this->vec_len, "At.txt (svec index)");
    check_indices(this->C_indices, this->vec_len, "C.txt");
    check_indices(this->b_indices, this->con_num, "b.txt");

    if (!this->At_csc_row_ids.empty())
    {
        int max_col_id = *std::max_element(this->At_csc_row_ids.begin(), this->At_csc_row_ids.end());
        if (max_col_id != this->vec_len - 1)
        {
            std::cerr << "WARNING: the largest column index in At is smaller than the vector length (unused trailing entries)\n"
                      << std::endl;
        }

        int max_row_id = *std::max_element(this->At_coo_col_ids.begin(), this->At_coo_col_ids.end());
        if (max_row_id != this->con_num - 1)
        {
            std::cerr << "WARNING: the largest row index in At is smaller than the number of constraints (empty constraints)\n"
                      << std::endl;
        }
    }

    /* display problem stats */
    std::cout << "Loaded problem from " << prefix << std::endl;
    std::cout << "              vector length: " << this->vec_len << std::endl;
    std::cout << "      number of constraints: " << this->con_num << std::endl;
    std::cout << "           number of blocks: " << this->mat_num << std::endl;
    std::cout << "  number of non-zeros in At: " << this->At_nnz << std::endl;
    std::cout << "  number of non-zeros in  b: " << this->b_nnz << std::endl;
    std::cout << "  number of non-zeros in  C: " << this->C_nnz << std::endl;

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