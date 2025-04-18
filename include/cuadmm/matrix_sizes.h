/*

    matrix_sizes.h

    Defines a structure to hold the sizes of matrices used in the solver.

*/

#ifndef CUADMM_MATRIX_SIZES_H
#define CUADMM_MATRIX_SIZES_H

#include <vector>
#include <unordered_map>

// Heuristics to determine if a matrix is large or small.
// This is used to determine if we use single QR or batched Jacobi for eig.
// - mat_size: size of the matrix
// - mat_num: number of matrices of this size
// Returns true if the matrix is large, false otherwise.
bool is_large_mat(int mat_size, int mat_num);

class MatrixSizes {
private:
    std::unordered_map<int, bool> is_large_map; // map of sizes to whether they are large or small
public:
    int unique_large_mat_num; // number of sizes of large matrices (ex: for blk = [40, 50, 50], this is 2)
    int unique_small_mat_num; // number of sizes of large matrices (ex: for blk = [3, 4, 4], this is 2)
    int large_mat_num; // number of large matrices (nb)
    int small_mat_num; // number of small matrices (nb)
    int sum_large_mat_size; // sum of sizes of large matrices (nb * side)
    int sum_small_mat_size; // sum of sizes of small matrices (nb * side)
    int total_large_mat_size; // total size of large matrices (nb * side * side)
    int total_small_mat_size; // total size of large matrices (nb * side * side)

    MatrixSizes() {}

    void init(const std::vector<int>& blk_sizes, const std::unordered_map<int, int>& blk_nums);

    bool is_large(int mat_size);
};

#endif // CUADMM_MATRIX_SIZES_H