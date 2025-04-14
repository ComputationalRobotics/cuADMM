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
        std::vector<double> cpu_X_vals;
        std::vector<double> cpu_y_vals;
        std::vector<double> cpu_S_vals;
        std::vector<int> cpu_At_csc_col_ptrs;
        std::vector<int> cpu_At_coo_col_ids;
        std::vector<int> cpu_At_csc_row_ids;
        std::vector<double> cpu_At_csc_vals;
        std::vector<int> cpu_b_indices;
        std::vector<double> cpu_b_vals;
        std::vector<int> cpu_C_indices;
        std::vector<double> cpu_C_vals;
        std::vector<int> cpu_blk_vals;
        std::vector<double> cpu_sig_vals;

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