#ifndef CUADMM_UTILS_H
#define CUADMM_UTILS_H

#define GPU0 0

#include <vector>
#include "cuadmm/memory.h"

// Check GPU numbers and whether they have peer-to-peer access
int check_gpus();

// Compute the inverse of a permutation vector.
// The inverse permutation is a vector such that if perm[i] = j, then perm_inv[j] = i.
void get_inverse_permutation(std::vector<int>& perm_inv, const std::vector<int>& perm);

void get_maps(
    const HostDenseVector<int>& blk, 
    const int LARGE, const int SMALL, const int vec_len,
    std::vector<int>& map_B_tmp, std::vector<int>& map_M1_tmp, std::vector<int>& map_M2_tmp
);

#endif // CUADMM_UTILS_H