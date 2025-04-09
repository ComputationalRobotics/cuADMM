#include "cuadmm/solver.h"
#include "cuadmm/io.h"

TEST(Solver, Init)
{
    // TODO: change data path
    std::string prefix = "/home/jordan/ksc/2023-traopt/pendulum/cartpole/data/debug/Cpp/";
    bool if_gpu_eig_mom = true;
    int device_num_requested = 2;
    int eig_stream_num_per_gpu = 15;
    int cpu_eig_thread_num = 30;

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

    read_dense_vector_data(prefix + "X.txt", cpu_X_vals);
    int vec_len = cpu_X_vals.size();
    read_dense_vector_data(prefix + "y.txt", cpu_y_vals);
    int con_num = cpu_y_vals.size();
    read_dense_vector_data(prefix + "S.txt", cpu_S_vals);
    read_COO_sparse_matrix_data(prefix + "At.txt", cpu_At_csc_row_ids, cpu_At_coo_col_ids, cpu_At_csc_vals);
    int At_nnz = cpu_At_csc_vals.size();
    COO_to_CSC(cpu_At_csc_col_ptrs, cpu_At_coo_col_ids, cpu_At_csc_row_ids, cpu_At_csc_vals, At_nnz, con_num);
    read_sparse_vector_data(prefix + "b.txt", cpu_b_indices, cpu_b_vals);
    int b_nnz = cpu_b_vals.size();
    read_sparse_vector_data(prefix + "C.txt", cpu_C_indices, cpu_C_vals);
    int C_nnz = cpu_C_vals.size();
    read_dense_vector_data(prefix + "blk.txt", cpu_blk_vals);
    int mat_num = cpu_blk_vals.size();
    read_dense_vector_data(prefix + "sig.txt", cpu_sig_vals);
    double sig_1 = cpu_sig_vals[0];

    SDPSolver solver1;
    solver1.init(
        true,
        2,
        15,
        30,

        vec_len, con_num,
        cpu_At_csc_col_ptrs.data(), cpu_At_csc_row_ids.data(), cpu_At_csc_vals.data(), At_nnz,
        cpu_b_indices.data(), cpu_b_vals.data(), b_nnz,
        cpu_C_indices.data(), cpu_C_vals.data(), C_nnz,
        cpu_blk_vals.data(), mat_num,
        cpu_X_vals.data(), cpu_y_vals.data(), cpu_S_vals.data(),

        1.0
    );

    SDPSolver solver2;
    solver2.init(
        false,
        2,
        15,
        30,

        vec_len, con_num,
        cpu_At_csc_col_ptrs.data(), cpu_At_csc_row_ids.data(), cpu_At_csc_vals.data(), At_nnz,
        cpu_b_indices.data(), cpu_b_vals.data(), b_nnz,
        cpu_C_indices.data(), cpu_C_vals.data(), C_nnz,
        cpu_blk_vals.data(), mat_num,
        cpu_X_vals.data(), cpu_y_vals.data(), cpu_S_vals.data(),

        1.0
    );
}