#define GPU0 0

#include <vector>

// Check GPU numbers and whether they have peer-to-peer access
int check_gpus();

// Compute the inverse of a permutation vector.
// The inverse permutation is a vector such that if perm[i] = j, then perm_inv[j] = i.
void get_inverse_permutation(std::vector<int>& perm_inv, const std::vector<int>& perm);