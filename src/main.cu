#include <cusparse.h>
#include <stdio.h>
#include <iostream>

#include "cuadmm/solver.h"
#include "cuadmm/problem.h"

int main(int argc, char* argv[]) {
    std::string prefix = argv[1];
    int eig_stream_num_per_gpu = 15;

    Problem problem;
    problem.from_txt(prefix);

    // extract the second element of each tuple in blk_vals
    std::vector<char> blk_sizes;
    std::vector<int> blk_vals;
    for (const auto& blk : problem.blk_vals) {
        blk_sizes.push_back(std::get<0>(blk));
        blk_vals.push_back(std::get<1>(blk));
    }
    
    SDPSolver solver;
    double sig = 1e0;
    solver.init(
        eig_stream_num_per_gpu,
        problem.vec_len, problem.con_num,
        problem.At_csc_col_ptrs.data(), problem.At_csc_row_ids.data(), problem.At_csc_vals.data(), problem.At_nnz,
        problem.b_indices.data(), problem.b_vals.data(), problem.b_nnz,
        problem.C_indices.data(), problem.C_vals.data(), problem.C_nnz,
        blk_sizes.data(), blk_vals.data(), problem.mat_num,
        ProjectionMethod::COMPOSITE_FP16,
        ProjectionMethod::EIG_FP64,
        problem.X_vals.data(), problem.y_vals.data(), problem.S_vals.data(),
        sig
    );

    // sGS-ADMM
    // solver.solve((int) 1e6, 1e-4, 500, 50, 100, 5000);

    // standard ADMM
    solver.solve((int) 1e6, 1e-4, 500, 50, 100, 0);

    // solver.X.to_txt(prefix + "X_opt.txt");
    
    return 0;
}