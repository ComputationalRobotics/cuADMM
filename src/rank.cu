#include <cuadmm/rank.h>
#include <cstdio>

__global__ void compute_ranks_kernel(
    const double* eigenvalues,
    const int mat_size,
    int* positive_rank,
    int* negative_rank,
    const double tol
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat_size) {
        if (eigenvalues[idx] > tol)
            atomicAdd(positive_rank, 1);
        else if (eigenvalues[idx] < -tol)
            atomicAdd(negative_rank, 1);
    }
}

void compute_ranks(
    const double* eigenvalues,
    const int mat_size,
    int* positive_rank,
    int* negative_rank,
    const double tol,
    const int block_size
) {
    int num_blocks = (mat_size + block_size - 1) / block_size;

    // set positive and negative ranks to zero
    cudaMemset(positive_rank, 0, sizeof(int));
    cudaMemset(negative_rank, 0, sizeof(int));

    // launch kernel to compute ranks
    compute_ranks_kernel<<<num_blocks, block_size>>>(
        eigenvalues,
        mat_size,
        positive_rank,
        negative_rank,
        tol
    );
}