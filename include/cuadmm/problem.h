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
        std::vector<double> X_vals;
        std::vector<double> y_vals;
        std::vector<double> S_vals;
        std::vector<int> At_csc_col_ptrs;
        std::vector<int> At_coo_col_ids;
        std::vector<int> At_csc_row_ids;
        std::vector<double> At_csc_vals;
        std::vector<int> b_indices;
        std::vector<double> b_vals;
        std::vector<int> C_indices;
        std::vector<double> C_vals;
        std::vector<int> blk_vals;
        std::vector<double> sig_vals;

        int vec_len;
        int col_num;
        int At_nnz;
        int b_nnz;
        int C_nnz;
        int mat_num;
        int con_num;

        // Load a problem from .txt files stored in the same directory.
        void from_txt(const std::string& prefix);
};

#endif // CUADMM_PROBLEM_H