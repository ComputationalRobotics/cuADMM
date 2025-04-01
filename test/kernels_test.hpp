#include "cuadmm/kernels.h"
#include "cuadmm/utils.h"

TEST(Kernels, Permutation)
{
    // Create host vectors
    std::vector<double> vec1_host(10);
    std::vector<double> vec2_host = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> perm_host = {6, 4, 1, 3, 0, 5, 2, 8, 7, 9};


    // Create a dense vector
    DeviceDenseVector<double> vec1(GPU0, 10);
    DeviceDenseVector<double> vec2(GPU0, 10);
    DeviceDenseVector<int> perm(GPU0, 10);

    // Copy data to the device
    CHECK_CUDA( cudaMemcpy(vec1.vals, vec1_host.data(), sizeof(double) * 10, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(vec2.vals, vec2_host.data(), sizeof(double) * 10, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(perm.vals, perm_host.data(), sizeof(int) * 10, cudaMemcpyHostToDevice) );

    // Perform the permutation
    // i.e. vec1[perm[i]] = vec2[i]
    perform_permutation(vec1, vec2, perm);

    // Retrieve the result from the device
    CHECK_CUDA( cudaMemcpy(vec1_host.data(), vec1.vals, sizeof(double) * 10, cudaMemcpyDeviceToHost) );

    // Check the result
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(vec1_host[perm_host[i]], vec2_host[i]);
    }
}