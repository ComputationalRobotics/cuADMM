#include "cuadmm/rank.h"

TEST(Rank, Standard)
{
    // create a vector of ones on GPU
    std::vector<double> h_A = {1.0, -1.0, 2.0, -2.0, -3.0};
    double* d_A;
    cudaMalloc((void**)&d_A, h_A.size() * sizeof(double));
    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(double), cudaMemcpyHostToDevice);

    // create a vector to hold the rank result
    int *positive_ranks, *negative_ranks;
    cudaMalloc((void**)&positive_ranks, sizeof(int));
    cudaMalloc((void**)&negative_ranks, sizeof(int));
    cudaMemset(positive_ranks, 0, sizeof(int));
    cudaMemset(negative_ranks, 0, sizeof(int));

    // compute the rank
    compute_ranks(d_A, h_A.size(), positive_ranks, negative_ranks);

    // copy the result back to host
    int h_positive_ranks, h_negative_ranks;
    cudaMemcpy(&h_positive_ranks, positive_ranks, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_negative_ranks, negative_ranks, sizeof(int), cudaMemcpyDeviceToHost);

    ASSERT_EQ(h_positive_ranks, 2);
    ASSERT_EQ(h_negative_ranks, 3);

    // free
    cudaFree(d_A);
    cudaFree(positive_ranks);
    cudaFree(negative_ranks);
}