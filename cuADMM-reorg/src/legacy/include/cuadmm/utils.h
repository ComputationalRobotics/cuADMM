#ifndef CUADMM_UTILS_H
#define CUADMM_UTILS_H

#define GPU0 0

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define D2D cudaMemcpyDeviceToDevice

#include <vector>
#include <unordered_map>
#include "cuadmm/memory.h"
#include "cuadmm/matrix_sizes.h"

// Compute the inverse of a permutation vector.
// The inverse permutation is a vector such that if perm[i] = j, then perm_inv[j] = i.
void get_inverse_permutation(std::vector<int> &perm_inv, const std::vector<int> &perm);

// (analyze_blk, get_maps, get_eig_rank_mask and check_gpus were deleted in Phase 1: superseded by BlockStructure or
// unused, plan P4; the first two live on as a test oracle in test/legacy_block_layout.hpp)

#endif // CUADMM_UTILS_H