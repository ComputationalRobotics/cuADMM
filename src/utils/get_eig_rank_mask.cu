/*

    utils/get_eig_rank_mask.cu

    Compute a mask based on the rank limit for batched
    eigenvalue decomposition.

*/

#include "cuadmm/utils.h"

// Compute a mask based on the rank limit for batched
// eigenvalue decomposition.
// The full matrix is of size batch_size * mat_size.
// Eigenvalues are sorted in ascending order.
void get_eig_rank_mask(
    std::vector<int>& eig_rank_mask,
    int batch_size, int mat_size, int eig_rank
) {
    // Create a mask of size batch_size * mat_size
    // and initialize it to 0.
    eig_rank_mask.clear();
    eig_rank_mask.resize(batch_size * mat_size);
    for (int i = 0; i < eig_rank_mask.size(); i++) {
        eig_rank_mask[i] = 0;
    }

    // Set the last eig_rank elements of each batch
    // to 1, starting from the end of the matrix.
    int idx;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < eig_rank; j++) {
            idx = i * mat_size + (mat_size - 1 - j);
            eig_rank_mask[idx] = 1;
        }
    }
    return;
}