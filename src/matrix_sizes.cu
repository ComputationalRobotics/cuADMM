#include "cuadmm/matrix_sizes.h"

#include <vector>
#include <unordered_map>

// Heuristics to determine if a matrix is large or small.
// This is used to determine if we use single QR or batched Jacobi for eig.
// - mat_size: size of the matrix
// - mat_num: number of matrices of this size
// Returns true if the matrix is large, false otherwise.
bool is_large_mat(int mat_size, int mat_num) {
    if (mat_size > 32) {
        return true; // single is faster
    }
    return ((double) mat_size - 17.0 > (double) mat_num * 1.4); // approximation of the slope
}


void MatrixSizes::init(const std::vector<int>& blk_sizes, const std::unordered_map<int, int>& blk_nums) {
    // initialize the sizes and numbers
    this->total_large_mat_size = 0;
    this->total_small_mat_size = 0;
    this->sum_large_mat_size = 0;
    this->sum_small_mat_size = 0;
    this->large_mat_num = 0;
    this->small_mat_num = 0;
    this->unique_large_mat_num = 0;
    this->unique_small_mat_num = 0;

    // for each matrix size, determine if it is large or small
    // i.e. if we put it in M1 or M2
    // this determines if we use single QR or batched Jacobi for eig
    ;
    for (const auto& pair : blk_nums) {
        int mat_size = pair.first; // size of the matrix
        int mat_num = pair.second; // number of matrices of this size

        this->is_large_map[mat_size] = is_large_mat(mat_size, mat_num);

        if (this->is_large(mat_size)) {
            this->sum_large_mat_size += mat_size * mat_num;
            this->large_mat_num += mat_num;
            this->unique_large_mat_num += 1;
            this->total_large_mat_size += mat_num * mat_size * mat_size;
        } else {
            this->sum_small_mat_size += mat_size * mat_num;
            this->small_mat_num += mat_num;
            this->unique_small_mat_num += 1;
            this->total_small_mat_size += mat_num * mat_size * mat_size;
        }
    }
}

bool MatrixSizes::is_large(int mat_size) {
    return this->is_large_map.at(mat_size);
}