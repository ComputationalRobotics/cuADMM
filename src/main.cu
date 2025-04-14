#include <cusparse.h>
#include <stdio.h>
#include <iostream>

#include "cuadmm/solver.h"
#include "cuadmm/problem.h"

int main(int argc, char* argv[]) {
    std::string prefix = argv[1];
    // bool if_gpu_eig_mom = true;
    // int device_num_requested = 1;
    int eig_stream_num_per_gpu = 15;
    int cpu_eig_thread_num = 30;

    Problem problem;
    problem.from_txt(prefix);
    // double sig_1 = cpu_sig_vals[0];
    
    SDPSolver solver;
    double sig = 1e0;
    solver.init(
        // if_gpu_eig_mom, device_num_requested,
        eig_stream_num_per_gpu, cpu_eig_thread_num,
        problem.vec_len, problem.con_num,
        problem.At_csc_col_ptrs.data(), problem.At_csc_row_ids.data(), problem.At_csc_vals.data(), problem.At_nnz,
        problem.b_indices.data(), problem.b_vals.data(), problem.b_nnz,
        problem.C_indices.data(), problem.C_vals.data(), problem.C_nnz,
        problem.blk_vals.data(), problem.mat_num,
        problem.X_vals.data(), problem.y_vals.data(), problem.S_vals.data(),
        sig
    );

    solver.solve((int) 1e4, 1e-2, false);
    
    return 0;
}