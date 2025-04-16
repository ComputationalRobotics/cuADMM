#include <iostream>
#include <fstream>
#include <string>

#define SINGLE_BATCHED_COMP_MAT_SIZES {2, 3, 4}
#define SINGLE_BATCHED_COMP_MAT_NUMS {1, 2, 3}
#define EIG_STREAM_NUM_PER_GPU 15

float test_single_performance(
    DeviceSolverDnHandle& handle,
    SingleEigParameter& param,
    const int mat_size,
    const int mat_num
) {
    // compute the buffer size
    // (we do not time this part since it is done only once)

    return std::sqrt(2);
}

float test_batched_performance(
    DeviceSolverDnHandle& handle,
    BatchEigParameter& param,
    const int mat_size,
    const int mat_num
) {
    return std::sqrt(3);
}

TEST(SingleBatchedComparison, Default)
{
    // GTEST_SKIP_("");

    // retrive the output path from the environment variable
    const char* output_env = std::getenv("CUADMM_SOLVER_OUTPUT_PATH");
    std::string output_path;
    if (!output_env) {
        GTEST_SKIP() << "CUADMM_SOLVER_OUTPUT_PATH environment variable not set. Skipping test.";
    } else {
        output_path = std::string(output_env);
    }

    // if the output file exists, fail the test
    std::ifstream input_file(output_path + "single_batched_comparison.txt");
    if (input_file.good()) {
        std::cerr << "File already exists. Please remove it before running the test.\n";
        ASSERT_TRUE(false);
    }

    // open the output file
    std::ofstream output_file(output_path + "single_batched_comparison.txt", std::ios::out);
    if (!output_file.is_open()) {
        std::cerr << "Unable to open file.\n";
        ASSERT_TRUE(false);
    }

    std::vector<int> mat_sizes = SINGLE_BATCHED_COMP_MAT_SIZES;
    std::vector<int> mat_nums = SINGLE_BATCHED_COMP_MAT_NUMS;

    DeviceSolverDnHandle handle;
    handle.set_gpu_id(GPU0);
    handle.activate();
    SingleEigParameter single_param(GPU0);
    BatchEigParameter batch_param(GPU0);

    for (auto mat_size : mat_sizes) {
        for (auto mat_num : mat_nums) {
            /* Test the single QR performance */
            float single = test_single_performance(
                handle, single_param, mat_size, mat_num
            );

            /* Test the batched Jacobi performance */
            float batched = test_batched_performance(
                handle, batch_param, mat_size, mat_num
            );

            output_file << mat_size << " " << mat_num << " "
                        << single << " " << batched << "\n";
            
            std::cout << "mat_size: " << mat_size << ", mat_num: " << mat_num
                      << ", single: " << single << ", batched: " << batched << "\n";
        }
    }
}