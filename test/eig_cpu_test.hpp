#include "cuadmm/eig_cpu.h"

TEST(EigCPU, SingleMatrix)
{
    // create an input matrix
    const int N = 4;
    std::vector<double> mat_vals = {4.0, 1.0, 2.0, 2.0,
                                    1.0, 4.0, 1.0, 2.0,
                                    2.0, 1.0, 4.0, 1.0,
                                    2.0, 2.0, 1.0, 4.0};
    HostDenseVector<double> mat(N * N);
    std::copy(mat_vals.begin(), mat_vals.end(), mat.vals);
    HostDenseVector<double> W(N);

    mat.print();

    const int lwork = 1 + 6 * N + 2 * N * N;
    const int lwork2 = 2 * (3 + 5 * N);
    HostDenseVector<double> workspace(lwork);
    HostDenseVector<ptrdiff_t> workspace2(lwork2);

    HostDenseVector<ptrdiff_t> info(1);

    single_eig_lapack(
        mat, W,
        workspace, workspace2, info,
        N, lwork, lwork2,
        0, 0,
        0, 0, 0
    );

    W.print();
    mat.print();

    // Check the eigenvalues
    std::vector<double> expected_eigenvalues = {1.38197, 2.45862, 3.61803, 8.54138};
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(W.vals[i], expected_eigenvalues[i], 1e-5);
    }

    // Check the eigenvectors
    std::vector<double> expected_eigenvectors = {
        -1.0, -0.618034, 0.618034, 1.0,
        1.0, -1.18046, -1.18046, 1.0,
        -1.0, 1.61803, -1.61803, 1.0,
        1.0, 0.847127, 0.847127, 1.0
    };
    // Normalize the eigenvectors
    std::vector<double> norms;
    for (int i = 0; i < N; ++i) {
        double norm = 0;
        for (int j = 0; j < N; ++j) {
            norm += expected_eigenvectors[i * N + j] * expected_eigenvectors[i * N + j];
        }
        norms.push_back(sqrt(norm));
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            expected_eigenvectors[i * N + j] /= norms[i];
        }
    }

    // Check the eigenvectors
    for (int i = 0; i < N * N; ++i) {
        EXPECT_NEAR(std::abs(mat.vals[i]), std::abs(expected_eigenvectors[i]), 1e-5);
    }

    // Check the info
    EXPECT_EQ(info.vals[0], 0);
}