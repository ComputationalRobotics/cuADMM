#include "cuadmm/cholesky_cpu.h"

TEST(CholeskyCPU, Dense)
{
    // create a dense matrix
    const int N = 4;
    const int nnz = N * N;
    std::vector<int> A_col_ptrs = {0, N, 2*N, 3*N, 4*N};
    std::vector<int> A_row_ids = {0, 1, 2, 3,
                                   0, 1, 2, 3,
                                   0, 1, 2, 3,
                                   0, 1, 2, 3};
    std::vector<double> A_vals = {
        2.0, 1.0, 1.0, 1.0,
        1.0, 2.0, 1.0, 1.0,
        1.0, 1.0, 2.0, 1.0,
        1.0, 1.0, 1.0, 2.0,
    };

    // hence, AA^T is:
    // [[7, 6, 6, 6]
    //  [6, 7, 6, 6]
    //  [6, 6, 7, 6]
    //  [6, 6, 6, 7]]

    ASSERT_EQ(A_col_ptrs.size(), N + 1);
    ASSERT_EQ(A_row_ids.size(), nnz);
    ASSERT_EQ(A_vals.size(), nnz);

    // create a solver
    CholeskySolverCPU solver(
        A_col_ptrs.data(), A_row_ids.data(), A_vals.data(),
        N, N, nnz
    );
    solver.factorize();

    // create a rhs vector
    std::vector<double> rhs = {25.0, 25.0, 25.0, 25.0};
    std::copy(rhs.begin(), rhs.end(), (double*)solver.chol_dn_rhs->x);
    
    // solve the system
    solver.solve();

    // normalize the expected result
    std::vector<double> expected_res = {
        1.0, 1.0, 1.0, 1.0
    };

    // check the result
    std::vector<double> res(N);
    std::copy((double*)solver.chol_dn_res->x, (double*)solver.chol_dn_res->x + N, res.data());
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(res[i], expected_res[i], 1e-5);
    }
}

TEST(CholeskyCPU, Sparse)
{
    // create a sparse symmetric positive definite matrix
    // in CSC format
    const int N = 4;
    const int nnz = 8;
    std::vector<int> A_col_ptrs = {0, 2, 4, 6, 8};
    std::vector<int> A_row_ids = {0, 2, 1, 3, 0, 2, 1, 3};
    std::vector<double> A_vals = {3.0, 2.0, 4.0, 1.0, 2.0, 5.0, 1.0, 6.0};

    // hence, AA^T is:
    // [[13,  0, 16,  0],
    //  [ 0, 17,  0, 10],
    //  [16,  0, 29,  0],
    //  [ 0, 10,  0, 37]]

    ASSERT_EQ(A_col_ptrs.size(), N + 1);
    ASSERT_EQ(A_row_ids.size(), nnz);
    ASSERT_EQ(A_vals.size(), nnz);

    // create a solver
    CholeskySolverCPU solver(
        A_col_ptrs.data(), A_row_ids.data(), A_vals.data(),
        N, N, nnz,
        false, 1e-5
    );
    solver.factorize();

    // create a rhs vector
    std::vector<double> rhs = {1.0, 2.0, 3.0, 4.0};
    std::copy(rhs.begin(), rhs.end(), (double*)solver.chol_dn_rhs->x);
    
    // solve the system
    solver.solve();

    // check the result
    std::vector<double> expected_res = {
        -19.0/121.0, 34.0/529.0, 23.0/121.0, 48.0/529.0
    };
    std::vector<double> res(N);
    std::copy((double*)solver.chol_dn_res->x, (double*)solver.chol_dn_res->x + N, res.data());
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(res[i], expected_res[i], 0.2);
    }
}