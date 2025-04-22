/*

    problem.h

    Defines an SDP problem.

*/

#ifndef CUADMM_PROBLEM_H
#define CUADMM_PROBLEM_H

#include <vector>
#include <string>

class Problem{
    public:
        std::vector<double> X_vals; // warm start for X
        std::vector<double> y_vals; // warm start for y
        std::vector<double> S_vals; // warm start for S
        std::vector<int> At_csc_col_ptrs; // |- At in COO format
        std::vector<int> At_coo_col_ids;  // |
        std::vector<int> At_csc_row_ids;  // |- At in CSC format
        std::vector<double> At_csc_vals;  // |
        int At_nnz;                       // |
        std::vector<int> b_indices; // |- b in COO format
        std::vector<double> b_vals; // |
        int b_nnz;                  // |
        std::vector<int> C_indices; // |- C in COO format
        std::vector<double> C_vals; // |
        int C_nnz;                  // |
        std::vector<int> blk_vals; // size of each block
        // std::vector<double> sig_vals;

        int vec_len; // length of X in vector form
        int mat_num; // number of blocks in the matrix
        int con_num; // number of constraints

        // Load a problem from .txt files stored in the same directory.
        void from_txt(const std::string& prefix, const bool warm_start = false);
};

#endif // CUADMM_PROBLEM_H