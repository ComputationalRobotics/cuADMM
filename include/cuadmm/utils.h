#ifndef CUADMM_UTILS_H
#define CUADMM_UTILS_H

#define GPU0 0

#define D2D cudaMemcpyDeviceToDevice
#define H2D cudaMemcpyHostToDevice
#define D2D cudaMemcpyDeviceToDevice

#include <vector>
#include "cuadmm/memory.h"

// Check GPU numbers and whether they have peer-to-peer access
int check_gpus();

// Compute the inverse of a permutation vector.
// The inverse permutation is a vector such that if perm[i] = j, then perm_inv[j] = i.
void get_inverse_permutation(std::vector<int>& perm_inv, const std::vector<int>& perm);

// Computes the maps for the vectorized representation of symmetric matrices.
// - blk: input block sizes
// - LARGE: the size of the large blocks
// - SMALL: the size of the small blocks
// - vec_len: length of the vectorized representation
// - map_B_tmp: output map, where 0 is for large blocks and 1 is for small blocks
// - map_M1_tmp: output map for M1 (horizontal count of lower triangle)
// - map_M2_tmp: output map for M2 (vertical count of upper triangle)
void get_maps(
    const HostDenseVector<int>& blk, 
    const int LARGE, const int SMALL, const int vec_len,
    std::vector<int>& map_B_tmp, std::vector<int>& map_M1_tmp, std::vector<int>& map_M2_tmp
);

// Analyze the blk vector to determine the following hyperparameters:
// - LARGE: the size of the large blocks (moment matrices)
// - SMALL: the size of the small blocks (localizing matrices)
// - mom_mat_num: the number of moment matrices
// - loc_mat_num: the number of localizing matrices
void analyze_blk(
    HostDenseVector<int>& blk, 
    int* LARGE, int* SMALL, int* mom_mat_num, int* loc_mat_num
);

// Compute a mask based on the rank limit for batched
// eigenvalue decomposition.
// The full matrix is of size batch_size * mat_size.
// Eigenvalues are sorted in ascending order.
void get_eig_rank_mask(
    std::vector<int>& eig_rank_mask,
    int batch_size, int mat_size, int eig_rank
);

#endif // CUADMM_UTILS_H