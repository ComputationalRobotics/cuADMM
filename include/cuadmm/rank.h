/*

    cuadmm/rank.h

    Rank computation of large matrices.

*/

#ifndef RANK_H
#define RANK_H

__global__ void compute_ranks_kernel(
    const double* eigenvalues,
    const int mat_size,
    int* positive_rank,
    int* negative_rank,
    double tol
);

void compute_ranks(
    const double* eigenvalues,
    const int mat_size,
    int* positive_rank,
    int* negative_rank,
    const double tol = 1e-8, // should be 1e-12 * n * max(M)
    const int block_size = 256
);

#endif // RANK_H