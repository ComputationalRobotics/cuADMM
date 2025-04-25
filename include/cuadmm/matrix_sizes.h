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
    /* large */
    int large_mat_num;        // number of large matrices (with multiplicity)
    int sum_large_mat_size;   // sum of sizes of large matrices (nb * side)
    int total_large_mat_size; // total size of large matrices (nb * side * side)
    std::vector<int> large_mat_sizes;
    std::vector<int> large_mat_nums;
    std::vector<int> large_mat_start_indices; // start indices of large matrices in the vectorized representation
    std::vector<int> large_W_start_indices; // start indices of large matrices in the W vector
    std::vector<int> large_buffer_start_indices; // start indices of GPU buffers for large matrices
    std::vector<int> large_cpu_buffer_start_indices; // start indices of CPU buffers for large matrices
    
    /* small */
    int small_mat_num;        // number of small matrices (with multiplicity)
    int sum_small_mat_size;   // sum of sizes of small matrices (nb * side)
    int total_small_mat_size; // total size of large matrices (nb * side * side)
    std::vector<int> small_mat_sizes;
    std::vector<int> small_mat_nums;
    std::vector<int> small_mat_start_indices; // start indices of small matrices in the vectorized representation
    std::vector<int> small_W_start_indices; // start indices of small matrices in the W vector
    std::vector<int> small_buffer_start_indices; // start indices of buffers for small matrices

    MatrixSizes() {}

    void init(const std::vector<int>& blk_sizes, const std::vector<int>& blk_nums);

    // Given a matrix size, returns true if it is large, false otherwise.
    bool is_large(const int mat_size) const;

    // Given a matrix size index and an index i, returns the offset of the i-th matrix of size mat_size (to which mat_size_index corresponds) in the vectorized representation.
    int large_mat_offset(int mat_size_index, int mat_index) const;

    // Given a matrix size and an index i, returns the offset of the i-th matrix of size mat_size in the W vector.
    int large_W_offset(int mat_size_index, int mat_index) const;

    // Given a matrix size and an index i, returns the offset of the i-th GPU buffer for matrices of size mat_size.
    int large_buffer_offset(int mat_size_index, int mat_index, std::vector<size_t>& buffer_sizes) const;

    // Given a matrix size and an index i, returns the offset of the i-th CPU buffer for matrices of size mat_size.
    int large_cpu_buffer_offset(int mat_size_index, int mat_index, std::vector<size_t>& eig_large_cpu_buffer_size) const;

    // Given a matrix size index, returns the offset of the matrices of size mat_size (to which mat_size_index corresponds) in the vectorized representation.
    int small_mat_offset(int mat_size_index, int same_size_idx = 0) const;

    // Given a matrix size index, returns the offset of the matrices of size mat_size in the W vector.
    int small_W_offset(int mat_size_index) const;

    // Given a matrix size and an index i, returns the offset of the i-th GPU buffer for matrices of size mat_size.
    int small_buffer_offset(int mat_size_index, std::vector<size_t>& buffer_sizes) const;
};

#endif // CUADMM_MATRIX_SIZES_H