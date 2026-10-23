/*

    cuadmm/rank.h

    Rank computation of large matrices.

*/

#ifndef RANK_H
#define RANK_H

__global__ void compute_ranks_kernel(
    const double *eigenvalues,
    const int mat_size,
    int *positive_rank,
    int *negative_rank,
    double tol);

void compute_ranks(
    const double *eigenvalues,
    const int mat_size,
    int *positive_rank,
    int *negative_rank,
    const double tol = 1e-8, // eigenvalues in [-tol, tol] are not counted
    const int block_size = 256);

#endif // RANK_H