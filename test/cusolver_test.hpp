#include "cuadmm/cusolver.h"

TEST(CuSOLVER, SingleEigGetBufferSize)
{
    DeviceSolverDnHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    SingleEigParameter param(GPU0);

    const int mat_size = 2;
    DeviceDenseVector<double> mat(GPU0, mat_size * mat_size);
    DeviceDenseVector<double> W(GPU0, mat_size);

    size_t buffer_size = 0;
    size_t buffer_size_host = 0;

    single_eig_get_buffersize_cusolver(handle, param, mat, W, 
        mat_size, &buffer_size, &buffer_size_host
    );
}

TEST(CuSOLVER, SingleEig)
{
    DeviceSolverDnHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    SingleEigParameter param(GPU0);

    const int mat_size = 4;
    DeviceDenseVector<double> mat(GPU0, mat_size * mat_size);
    std::vector<double> mat_vals = {4.0, 1.0, 2.0, 2.0,
                                    1.0, 4.0, 1.0, 2.0,
                                    2.0, 1.0, 4.0, 1.0,
                                    2.0, 2.0, 1.0, 4.0};
    CHECK_CUDA ( cudaMemcpy(mat.vals, mat_vals.data(), mat_size * mat_size * sizeof(double), cudaMemcpyHostToDevice) );

    DeviceDenseVector<double> W(GPU0, mat_size); // eigenvalues
    DeviceDenseVector<int> info(GPU0, 1);

    // Create buffer for eigenvalues and eigenvectors
    size_t buffer_size = 0;
    size_t buffer_size_host = 0;
    single_eig_get_buffersize_cusolver(handle, param, mat, W, 
        mat_size, &buffer_size, &buffer_size_host
    );
    DeviceDenseVector<double> buffer(GPU0, buffer_size);
    HostDenseVector<double> buffer_host(buffer_size_host);

    // Compute eigenvalues and eigenvectors
    single_eig_cusolver(
        handle, param,
        mat, W,
        buffer, buffer_host, info,
        mat_size, buffer_size, buffer_size_host
    );

    // Copy eigenvalues and eigenvectors back to host
    std::vector<double> W_host(mat_size);
    CHECK_CUDA ( cudaMemcpy(W_host.data(), W.vals, mat_size * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA ( cudaMemcpy(mat_vals.data(), mat.vals, mat_size * mat_size * sizeof(double), cudaMemcpyDeviceToHost) );

    // Check the eigenvalues
    std::vector<double> expected_eigenvalues = {1.38197, 2.45862, 3.61803, 8.54138};
    for (int i = 0; i < mat_size; ++i) {
        EXPECT_NEAR(W_host[i], expected_eigenvalues[i], 1e-5);
    }

    std::vector<double> expected_eigenvectors = {
        -1.0, -0.618034, 0.618034, 1.0,
        1.0, -1.18046, -1.18046, 1.0,
        -1.0, 1.61803, -1.61803, 1.0,
        1.0, 0.847127, 0.847127, 1.0
    };
    // Normalize the expected eigenvectors
    std::vector<double> norms;
    for (int i = 0; i < mat_size; ++i) {
        double norm = 0;
        for (int j = 0; j < mat_size; ++j) {
            norm += expected_eigenvectors[i * mat_size + j] * expected_eigenvectors[i * mat_size + j];
        }
        norms.push_back(sqrt(norm));
    }
    for (int i = 0; i < mat_size; ++i) {
        for (int j = 0; j < mat_size; ++j) {
            expected_eigenvectors[i * mat_size + j] /= norms[i];
        }
    }
    // Check the eigenvectors
    for (int i = 0; i < mat_size * mat_size; ++i) {
        EXPECT_NEAR(std::abs(mat_vals[i]), std::abs(expected_eigenvectors[i]), 1e-5);
    }

    // Check the info
    int info_host = 0;
    CHECK_CUDA ( cudaMemcpy(&info_host, info.vals, sizeof(int), cudaMemcpyDeviceToHost) );
    EXPECT_EQ(info_host, 0) << "Eigenvalue computation failed with info = " << info_host;
}

TEST(CuSOLVER, MultipleSingleEig)
{
    DeviceSolverDnHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    SingleEigParameter param(GPU0);

    const std::vector<int> mat_sizes = {2, 3, 4};
    std::vector<DeviceDenseVector<double>> matrices;
    matrices.reserve(mat_sizes.size()); // important
    for (const auto& size : mat_sizes) {
        matrices.emplace_back(GPU0, size * size);
    }

    std::vector<std::vector<double>> matrices_vals = {
        {2.0, 1.0,
         1.0, 3.0},
        {3.0, 1.0, 2.0,
         1.0, 3.0, 1.0,
         2.0, 1.0, 3.0},
        {4.0, 1.0, 2.0, 2.0,
         1.0, 4.0, 1.0, 2.0,
         2.0, 1.0, 4.0, 1.0,
         2.0, 2.0, 1.0, 4.0}
    };

    for (int i = 0; i < 3; i++) {
        CHECK_CUDA ( cudaMemcpy(matrices[i].vals, matrices_vals[i].data(), mat_sizes[i] * mat_sizes[i] * sizeof(double), cudaMemcpyHostToDevice) );
    }

    std::vector<DeviceDenseVector<double>> Ws; // eigenvalues
    std::vector<DeviceDenseVector<int>> infos;
    Ws.reserve(mat_sizes.size());
    infos.reserve(mat_sizes.size());
    for (const auto& size : mat_sizes) {
        Ws.emplace_back(GPU0, size);
        infos.emplace_back(GPU0, 1);
    }

    // Create buffers for eigenvalues and eigenvectors
    std::vector<size_t> buffer_sizes(3, 0);
    std::vector<size_t> buffer_size_hosts(3, 0);
    for (int i = 0; i < 3; i++) {
        single_eig_get_buffersize_cusolver(handle, param, matrices[i], Ws[i], 
            mat_sizes[i], &buffer_sizes[i], &buffer_size_hosts[i]
        );
    }
    std::vector<DeviceDenseVector<double>> buffers;
    std::vector<HostDenseVector<double>> buffer_hosts;
    buffers.reserve(mat_sizes.size());
    buffer_hosts.reserve(mat_sizes.size());
    for (int i = 0; i < 3; i++) {
        buffers.emplace_back(GPU0, buffer_sizes[i]);
        buffer_hosts.emplace_back(buffer_size_hosts[i]);
    }

    // Compute eigenvalues and eigenvectors
    for (int i = 0; i < 3; i++) {
        single_eig_cusolver(
            handle, param,
            matrices[i], Ws[i],
            buffers[i], buffer_hosts[i], infos[i],
            mat_sizes[i], buffer_sizes[i], buffer_size_hosts[i]
        );
    }

    // Copy eigenvalues and eigenvectors back to host
    std::vector<std::vector<double>> W_hosts;
    W_hosts.reserve(mat_sizes.size());
    for (int i = 0; i < 3; i++) {
        W_hosts.emplace_back(mat_sizes[i]);
        CHECK_CUDA ( cudaMemcpy(W_hosts[i].data(), Ws[i].vals, mat_sizes[i] * sizeof(double), cudaMemcpyDeviceToHost) );
        CHECK_CUDA ( cudaMemcpy(matrices_vals[i].data(), matrices[i].vals, mat_sizes[i] * mat_sizes[i] * sizeof(double), cudaMemcpyDeviceToHost) );
    }

    // Check the eigenvalues
    std::vector<std::vector<double>> expected_eigenvalues = {
        {0.5*(5-std::sqrt(5.0)), 0.5*(5+std::sqrt(5.0))},
        {1.0, 4-std::sqrt(3.0), 4+std::sqrt(3.0)},
        {0.5*(5-std::sqrt(5.0)), 0.5*(11-std::sqrt(37.0)), 0.5*(5+std::sqrt(5.0)), 0.5*(11+std::sqrt(37.0))}
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < mat_sizes[i]; ++j) {
            EXPECT_NEAR(W_hosts[i][j], expected_eigenvalues[i][j], 1e-5);
        }
    }
}

TEST(CuSOLVER, MultipleSingleEigOneVector)
{
    DeviceSolverDnHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    SingleEigParameter param(GPU0);

    const std::vector<int> mat_sizes = {2, 3, 4};
    std::vector<int> mat_starts_indices; // indices for the start of each matrix in the vector
    int total_size = 0;
    for (const auto& size : mat_sizes) {
        mat_starts_indices.push_back(total_size);
        total_size += size * size;
    }
    // mat_starts_indices.push_back(total_size);

    DeviceDenseVector<double> matrices(GPU0, total_size);

    std::vector<std::vector<double>> matrices_vals = {
        {2.0, 1.0,
         1.0, 3.0},
        {3.0, 1.0, 2.0,
         1.0, 3.0, 1.0,
         2.0, 1.0, 3.0},
        {4.0, 1.0, 2.0, 2.0,
         1.0, 4.0, 1.0, 2.0,
         2.0, 1.0, 4.0, 1.0,
         2.0, 2.0, 1.0, 4.0}
    };

    for (int i = 0; i < mat_sizes.size(); i++) {
        CHECK_CUDA ( cudaMemcpy(matrices.vals + mat_starts_indices[i], matrices_vals[i].data(), mat_sizes[i] * mat_sizes[i] * sizeof(double), cudaMemcpyHostToDevice) );
    }

    std::vector<DeviceDenseVector<double>> Ws; // eigenvalues
    std::vector<DeviceDenseVector<int>> infos;
    Ws.reserve(mat_sizes.size());
    infos.reserve(mat_sizes.size());
    for (const auto& size : mat_sizes) {
        Ws.emplace_back(GPU0, size);
        infos.emplace_back(GPU0, 1);
    }

    // Create buffers for eigenvalues and eigenvectors
    std::vector<size_t> buffer_sizes(3, 0);
    std::vector<size_t> buffer_size_hosts(3, 0);
    for (int i = 0; i < 3; i++) {
        single_eig_get_buffersize_cusolver(handle, param, matrices, Ws[i], 
            mat_sizes[i], &buffer_sizes[i], &buffer_size_hosts[i],
            mat_starts_indices[i], 0 // offsets
        );
    }
    std::vector<DeviceDenseVector<double>> buffers;
    std::vector<HostDenseVector<double>> buffer_hosts;
    buffers.reserve(mat_sizes.size());
    buffer_hosts.reserve(mat_sizes.size());
    for (int i = 0; i < 3; i++) {
        buffers.emplace_back(GPU0, buffer_sizes[i]);
        buffer_hosts.emplace_back(buffer_size_hosts[i]);
    }

    // Compute eigenvalues and eigenvectors
    for (int i = 0; i < 3; i++) {
        single_eig_cusolver(
            handle, param,
            matrices, Ws[i],
            buffers[i], buffer_hosts[i], infos[i],
            mat_sizes[i], buffer_sizes[i], buffer_size_hosts[i],
            mat_starts_indices[i], 0 // offsets
        );
    }

    // Copy eigenvalues and eigenvectors back to host
    std::vector<std::vector<double>> W_hosts;
    W_hosts.reserve(mat_sizes.size());
    for (int i = 0; i < 3; i++) {
        W_hosts.emplace_back(mat_sizes[i]);
        CHECK_CUDA ( cudaMemcpy(W_hosts[i].data(), Ws[i].vals, mat_sizes[i] * sizeof(double), cudaMemcpyDeviceToHost) );
        CHECK_CUDA ( cudaMemcpy(matrices_vals[i].data(), matrices.vals + mat_starts_indices[i], mat_sizes[i] * mat_sizes[i] * sizeof(double), cudaMemcpyDeviceToHost) );
    }

    // Check the eigenvalues
    std::vector<std::vector<double>> expected_eigenvalues = {
        {0.5*(5-std::sqrt(5.0)), 0.5*(5+std::sqrt(5.0))},
        {1.0, 4-std::sqrt(3.0), 4+std::sqrt(3.0)},
        {0.5*(5-std::sqrt(5.0)), 0.5*(11-std::sqrt(37.0)), 0.5*(5+std::sqrt(5.0)), 0.5*(11+std::sqrt(37.0))}
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < mat_sizes[i]; ++j) {
            EXPECT_NEAR(W_hosts[i][j], expected_eigenvalues[i][j], 1e-5);
        }
    }
}

TEST(CuSOLVER, BatchEig)
{
    DeviceSolverDnHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();

    BatchEigParameter param(GPU0);

    const int mat_size = 4;
    const int batch_size = 2;
    DeviceDenseVector<double> mat(GPU0, mat_size * mat_size * batch_size);
    std::vector<double> mat_vals = {
        4.0, 1.0, 2.0, 2.0,
        1.0, 4.0, 1.0, 2.0,
        2.0, 1.0, 4.0, 1.0,
        2.0, 2.0, 1.0, 4.0,

        4.0, 1.0, 2.0, 2.0,
        1.0, 4.0, 1.0, 2.0,
        2.0, 1.0, 4.0, 1.0,
        2.0, 2.0, 1.0, 4.0,
    };
    ASSERT_EQ(mat_vals.size(), mat_size * mat_size * batch_size);
    CHECK_CUDA ( cudaMemcpy(mat.vals, mat_vals.data(), mat_size * mat_size * batch_size * sizeof(double), cudaMemcpyHostToDevice) );

    DeviceDenseVector<double> W(GPU0, mat_size * batch_size); // eigenvalues
    DeviceDenseVector<int> info(GPU0, batch_size);

    // Create buffer for eigenvalues and eigenvectors
    size_t buffer_size_host = 0;
    size_t buffer_size = batch_eig_get_buffersize_cusolver(
        handle, param, 
        mat, W, 
        mat_size, batch_size
    );
    DeviceDenseVector<double> buffer(GPU0, buffer_size);

    // Compute eigenvalues and eigenvectors
    batch_eig_cusolver(
        handle, param,
        mat, W,
        buffer, info,
        mat_size, batch_size, buffer_size
    );

    // Copy eigenvalues and eigenvectors back to host
    std::vector<double> W_host(mat_size * batch_size);
    CHECK_CUDA ( cudaMemcpy(W_host.data(), W.vals, mat_size * batch_size * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA ( cudaMemcpy(mat_vals.data(), mat.vals, mat_size * mat_size * batch_size * sizeof(double), cudaMemcpyDeviceToHost) );

    // Check the eigenvalues
    std::vector<double> expected_eigenvalues = {1.38197, 2.45862, 3.61803, 8.54138};
    for (int i = 0; i < mat_size * batch_size; ++i) {
        EXPECT_NEAR(W_host[i], expected_eigenvalues[i % mat_size], 1e-5);
    }

    std::vector<double> expected_eigenvectors = {
        -1.0, -0.618034, 0.618034, 1.0,
        1.0, -1.18046, -1.18046, 1.0,
        -1.0, 1.61803, -1.61803, 1.0,
        1.0, 0.847127, 0.847127, 1.0
    };
    // Normalize the expected eigenvectors
    std::vector<double> norms;
    for (int i = 0; i < mat_size; ++i) {
        double norm = 0;
        for (int j = 0; j < mat_size; ++j) {
            norm += expected_eigenvectors[i * mat_size + j] * expected_eigenvectors[i * mat_size + j];
        }
        norms.push_back(sqrt(norm));
    }
    for (int i = 0; i < mat_size; ++i) {
        for (int j = 0; j < mat_size; ++j) {
            expected_eigenvectors[i * mat_size + j] /= norms[i];
        }
    }
    // Check the eigenvectors
    for (int i = 0; i < mat_size * mat_size * batch_size; ++i) {
        EXPECT_NEAR(std::abs(mat_vals[i]), std::abs(expected_eigenvectors[i % (mat_size * mat_size)]), 1e-5);
    }

    // Check the infos
    std::vector<int> info_host(batch_size);
    CHECK_CUDA ( cudaMemcpy(info_host.data(), info.vals, batch_size * sizeof(int), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(info_host[i], 0) << "Eigenvalue computation failed with info = " << info_host[i];
    }
}