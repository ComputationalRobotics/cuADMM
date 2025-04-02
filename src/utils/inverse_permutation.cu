/*

    utils/inverse_permutation.cu

    Defines functions for computing the inverse of a permutation vector.

*/

#include <vector>
#include <algorithm>
#include <numeric>

#include "cuadmm/utils.h"

// Compute the inverse of a permutation vector.
// The inverse permutation is a vector such that if perm[i] = j, then perm_inv[j] = i.
void get_inverse_permutation(std::vector<int>& perm_inv, const std::vector<int>& perm) {
    int size = perm.size();
    if (!perm_inv.empty()) {
        perm_inv.clear();
    }
    perm_inv.resize(size);
    for (int i = 0; i < size; i++) {
        perm_inv[i] = i;
    }
    std::sort(
        perm_inv.begin(), perm_inv.end(),
        [&perm](int i, int j) { return perm[i] < perm[j]; }
    );
}