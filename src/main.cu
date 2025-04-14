#include <cusparse.h>
#include <stdio.h>
#include <iostream>

#include "cuadmm/duo_solver.h"
#include "cuadmm/problem.h"

int main() {
    std::string prefix = "/home/jordan/ksc/2023-traopt/pendulum/cartpole/data/debug/Cpp/";
    bool if_gpu_eig_mom = true;
    int device_num_requested = 1;
    int eig_stream_num_per_gpu = 15;
    int cpu_eig_thread_num = 30;

    Problem problem;
    problem.from_txt(prefix);
    // double sig_1 = cpu_sig_vals[0];
    
    SDPDuoSolver solver;
    double sig = 1e0;
    solver.init(
        if_gpu_eig_mom, device_num_requested, eig_stream_num_per_gpu, cpu_eig_thread_num,
        problem.vec_len, problem.con_num,
        problem.cpu_At_csc_col_ptrs.data(), problem.cpu_At_csc_row_ids.data(), problem.cpu_At_csc_vals.data(), problem.At_nnz,
        problem.cpu_b_indices.data(), problem.cpu_b_vals.data(), problem.b_nnz,
        problem.cpu_C_indices.data(), problem.cpu_C_vals.data(), problem.C_nnz,
        problem.cpu_blk_vals.data(), problem.mat_num,
        problem.cpu_X_vals.data(), problem.cpu_y_vals.data(), problem.cpu_S_vals.data(),
        sig
    );

    solver.solve((int) 1e4, 1e-4, false);
    
    return 0;
}