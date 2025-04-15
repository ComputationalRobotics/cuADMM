/*

    utils/get_maps.cu

    Computes the maps for the vectorized representation of symmetric matrices.

*/

#include "cuadmm/memory.h"
#include "cuadmm/utils.h"

// Computes the maps for the vectorized representation of symmetric matrices.
// - blk: input block sizes
// - LARGE: the size of the large blocks
// - SMALL: the size of the small blocks
// - vec_len: length of the vectorized representation
// - map_B_tmp: output map, where 0 is for large blocks and 1 is for small blocks
// - map_M1_tmp: output map for M1 (horizontal count of lower triangle)
// - map_M2_tmp: output map for M2 (vertical count of upper triangle)
void get_maps_duo(
    const HostDenseVector<int>& blk, 
    const int LARGE, const int SMALL, const int vec_len,
    std::vector<int>& map_B_tmp, std::vector<int>& map_M1_tmp, std::vector<int>& map_M2_tmp
) {
    // Reserve space for the maps
    map_B_tmp.clear();
    map_B_tmp.resize(vec_len);
    map_M1_tmp.clear();
    map_M1_tmp.resize(vec_len);
    map_M2_tmp.clear();
    map_M2_tmp.resize(vec_len);

    int idx = 0;    // current index in the maps
    int k_Xmom = 0; // current large block index
    int k_Xloc = 0; // current small block index
    int s;          // block size
    int b;         // block type (0 for large, 1 for small)
    for (int k = 0; k < blk.size; ++k) { // for each block
        s = blk.vals[k];
        if (s == LARGE) {
            b = 0;
            ++k_Xmom;
        } else {
            b = 1;
            ++k_Xloc;
        }
        for (int i = 1; i <= s; ++i) {      // for each coefficient
            for (int j = 1; j <= i; ++j) {  // in the triangle
                map_B_tmp[idx] = b;
                if (s == LARGE) {
                    // count horizontally
                    map_M1_tmp[idx] = s * s * (k_Xmom - 1) + s * (i-1) + j-1;
                    // count vertically
                    map_M2_tmp[idx] = s * s * (k_Xmom - 1) + s * (j-1) + i-1;
                } else {
                    // count horizontally
                    map_M1_tmp[idx] = s * s * (k_Xloc - 1) + s * (i-1) + j-1;
                    // count vertically
                    map_M2_tmp[idx] = s * s * (k_Xloc - 1) + s * (j-1) + i-1;
                }
                ++idx;
            }
        }
    }
    return;
}

// Computes the maps for the vectorized representation of symmetric matrices.
// Matrices are split into to groups depending on their sizes and numbers:
// large and small ones.
//
// - blk: input block sizes
// - blk_sizes: sizes of the blocks
// - blk_nums: number of blocks of each size
// - vec_len: length of the vectorized representation
// - map_B_tmp: output map, where 0 is for large blocks and 1 is for small blocks
// - map_M1_tmp: output map for M1 (horizontal count of lower triangle)
// - map_M2_tmp: output map for M2 (vertical count of upper triangle)
void get_maps(
    const HostDenseVector<int>& blk, 
    const std::vector<int>& blk_sizes, const std::unordered_map<int, int>& blk_nums,
    const int vec_len,
    std::vector<int>& map_B_tmp, std::vector<int>& map_M1_tmp, std::vector<int>& map_M2_tmp
) {
    // Reserve space for the maps
    map_B_tmp.clear();
    map_B_tmp.resize(vec_len);
    map_M1_tmp.clear();
    map_M1_tmp.resize(vec_len);
    map_M2_tmp.clear();
    map_M2_tmp.resize(vec_len);

    // for each matrix size, determine if it is large or small
    // i.e. if we put it in M1 or M2
    // this determines if we use single QR or batched Jacobi for eig
    std::unordered_map<int, bool> is_large;
    for (const auto& pair : blk_nums) {
        is_large[pair.first] = (pair.first >= 20);
    }

    int idx = 0;    // current index in the maps
    int k_Xmom = 0; // current large block index
    int k_Xloc = 0; // current small block index
    int s;          // block size
    int b;         // block type (0 for large, 1 for small)
    for (int k = 0; k < blk.size; ++k) { // for each block
        s = blk.vals[k];
        if (is_large.at(s)) {
            b = 0;
            ++k_Xmom;
        } else {
            b = 1;
            ++k_Xloc;
        }
        for (int i = 1; i <= s; ++i) {      // for each coefficient
            for (int j = 1; j <= i; ++j) {  // in the triangle
                map_B_tmp[idx] = b;
                if (is_large.at(s)) {
                    // count horizontally
                    map_M1_tmp[idx] = s * s * (k_Xmom - 1) + s * (i-1) + j-1;
                    // count vertically
                    map_M2_tmp[idx] = s * s * (k_Xmom - 1) + s * (j-1) + i-1;
                } else {
                    // count horizontally
                    map_M1_tmp[idx] = s * s * (k_Xloc - 1) + s * (i-1) + j-1;
                    // count vertically
                    map_M2_tmp[idx] = s * s * (k_Xloc - 1) + s * (j-1) + i-1;
                }
                ++idx;
            }
        }
    }
    return;
}