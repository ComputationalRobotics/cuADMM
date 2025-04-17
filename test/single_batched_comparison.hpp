#include <iostream>
#include <fstream>
#include <string>
#include <random>

#include <condition_variable>
#include <mutex>
#include <thread>

#include "cuadmm/cusolver.h"

#define SINGLE_BATCHED_COMP_RESTARTS 1
#define SINGLE_BATCHED_COMP_MAT_SIZES {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21}
#define SINGLE_BATCHED_COMP_MAT_NUMS {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
#define EIG_STREAM_NUM_PER_GPU 15

float test_single_performance(
    std::vector<DeviceSolverDnHandle>& handle_arr,
    std::vector<DeviceStream>& stream_arr,
    SingleEigParameter& param,
    const int mat_size,
    const int mat_num,
    std::vector<double> mat_vals
) {
    DeviceDenseVector<double> mat(GPU0, mat_size * mat_size * mat_num);
    DeviceDenseVector<double> W(GPU0, mat_size * mat_num); // eigenvalues
    DeviceDenseVector<int> info(GPU0, mat_num);

    // copy matrices from CPU to GPU
    CHECK_CUDA ( cudaMemcpy(mat.vals, mat_vals.data(), mat_size * mat_size * mat_num * sizeof(double), cudaMemcpyHostToDevice) );

    // compute the buffer size and allocate the buffers
    // (we do not time this part since it is done only once)
    size_t buffer_size;
    size_t cpu_buffer_size;
    single_eig_get_buffersize_cusolver(
        handle_arr[0], param, mat, W,
        mat_size, &buffer_size, &cpu_buffer_size
    ); // buffer size per moment matrix
    DeviceDenseVector<double> buffer = DeviceDenseVector<double>(GPU0, buffer_size * mat_num, true);
    HostDenseVector<double> cpu_buffer = HostDenseVector<double>(cpu_buffer_size * mat_num, true);

    // timing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // compute the eig decomposition
    cudaEventRecord(start);

    // for each moment matrix on this GPU, compute the eig decomposition
    int stream_id;
    for (int i = 0; i < mat_num; i++) {
        stream_id = i % EIG_STREAM_NUM_PER_GPU;
        // simply calls the cuSOLVER wrapper
        single_eig_cusolver(
            handle_arr[stream_id], param, mat, W,
            buffer, cpu_buffer, info,
            mat_size, buffer_size, cpu_buffer_size,
            i * mat_size * mat_size, i * mat_size,
            i * buffer_size, i * cpu_buffer_size, i
        );
    }

    // for each stream, synchronize
    for (int stream_id = 0; stream_id < EIG_STREAM_NUM_PER_GPU; stream_id++) {
        CHECK_CUDA( cudaStreamSynchronize(stream_arr[stream_id].stream) );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // end
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

float test_batched_performance(
    DeviceSolverDnHandle& handle,
    BatchEigParameter& param,
    const int mat_size,
    const int mat_num,
    std::vector<double> mat_vals
) {
    // compute the buffer size and allocate the buffers
    // (we do not time this part since it is done only once)
    DeviceDenseVector<double> mat(GPU0, mat_size * mat_size * mat_num);
    DeviceDenseVector<double> W(GPU0, mat_size * mat_num); // eigenvalues
    DeviceDenseVector<int> info(GPU0, mat_num);
    int buffer_size = batch_eig_get_buffersize_cusolver(
        handle, param, mat, W,
        mat_size, mat_num
    );
    DeviceDenseVector<double> buffer(GPU0, buffer_size);

    // copy matrices from CPU to 
    CHECK_CUDA ( cudaMemcpy(mat.vals, mat_vals.data(), mat_size * mat_size * mat_num * sizeof(double), cudaMemcpyHostToDevice) );

    // timing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // compute the eigenvalues and eigenvectors
    cudaEventRecord(start);

    batch_eig_cusolver(
        handle, param,
        mat, W,
        buffer, info,
        mat_size, mat_num, buffer_size
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // end
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

TEST(SingleBatchedComparison, Default)
{
    // retrive the output path from the environment variable
    const char* output_env = std::getenv("CUADMM_SOLVER_OUTPUT_PATH");
    std::string output_path;
    if (!output_env) {
        GTEST_SKIP() << "CUADMM_SOLVER_OUTPUT_PATH environment variable not set. Skipping test.";
    } else {
        output_path = std::string(output_env);
    }

    // if the output file exists, fail the test
    std::ifstream check_file(output_path + "single_batched_comparison.txt");
    if (check_file.good() && check_file.peek() != std::ifstream::traits_type::eof()) {
        GTEST_SKIP_("SingleBatchedComparison: output file already exists. Please remove it before running the test.\n");
    }

    // open the output file
    std::ofstream output_file(output_path + "single_batched_comparison.txt", std::ios::out);
    if (!output_file.is_open()) {
        std::cerr << "Unable to open file.\n";
        ASSERT_TRUE(false);
    }

    std::vector<int> mat_sizes = SINGLE_BATCHED_COMP_MAT_SIZES;
    std::vector<int> mat_nums = SINGLE_BATCHED_COMP_MAT_NUMS;

    // batched handle
    DeviceSolverDnHandle handle(GPU0);
    handle.activate();

    // single handles and streams
    std::vector<DeviceStream> stream_arr = std::vector<DeviceStream>(EIG_STREAM_NUM_PER_GPU);
    std::vector<DeviceSolverDnHandle> handle_arr = std::vector<DeviceSolverDnHandle>(EIG_STREAM_NUM_PER_GPU);
    
    for (int stream_id = 0; stream_id < EIG_STREAM_NUM_PER_GPU; stream_id++) {
        // ininitialize and activate the streams and handles
        stream_arr[stream_id].set_gpu_id(GPU0);
        stream_arr[stream_id].activate();
        handle_arr[stream_id].set_gpu_id(GPU0);
        handle_arr[stream_id].activate(stream_arr[stream_id]);
    }

    SingleEigParameter single_param(GPU0);
    BatchEigParameter batch_param(GPU0);

    // warm-up the GPU
    test_single_performance(
        handle_arr, stream_arr, single_param, 10, 10, std::vector<double>(10 * 10 * 10, 0.0)
    );
    test_batched_performance(
        handle, batch_param, 10, 10, std::vector<double>(10 * 10 * 10, 0.0)
    );

    for (int r = 0; r < SINGLE_BATCHED_COMP_RESTARTS; r++) {
        for (auto mat_size : mat_sizes) {
            for (auto mat_num : mat_nums) {
                // generate random matrices
                std::vector<double> mat_vals = std::vector<double>(mat_size * mat_size * mat_num, 0.0);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dis(0.0, 1.0);
                for (int n = 0; n < mat_num; n++) {
                    for (int i = 0; i < mat_size; i++) {
                        for (int j = i; j < mat_size; j++) {
                            double val = dis(gen);
                            // matrix must be symmetric
                            mat_vals[n * mat_size * mat_size + i * mat_size + j] = val;
                            mat_vals[n * mat_size * mat_size + j * mat_size + i] = val;
                        }
                    }
                }
    
                // test the single QR performance
                float single = test_single_performance(
                    handle_arr, stream_arr, single_param, mat_size, mat_num, mat_vals
                );
    
                // test the batched Jacobi performance
                float batched = test_batched_performance(
                    handle, batch_param, mat_size, mat_num, mat_vals
                );
    
                // write the results to the file
                output_file << mat_size << " " << mat_num << " "
                            << single << " " << batched << "\n";
                
                std::cout << "mat_size: " << mat_size << ", mat_num: " << mat_num
                          << ", single: " << single << ", batched: " << batched << "\n";
            }
        }
    }
    
}