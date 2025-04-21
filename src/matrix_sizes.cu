#include "cuadmm/matrix_sizes.h"

#include <vector>
#include <unordered_map>
#include <cassert>
#include <iostream>
#include <iomanip>

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


void MatrixSizes::init(const std::vector<int>& blk_sizes, const std::vector<int>& blk_nums) {
    assert(blk_sizes.size() == blk_nums.size());

    // initialize the sizes and numbers
    this->total_large_mat_size = 0;
    this->total_small_mat_size = 0;
    this->sum_large_mat_size = 0;
    this->sum_small_mat_size = 0;
    this->large_mat_num = 0;
    this->small_mat_num = 0;

    // first matrix starts at index 0
    this->large_mat_start_indices.push_back(0);
    this->large_W_start_indices.push_back(0);
    this->small_mat_start_indices.push_back(0);
    this->small_W_start_indices.push_back(0);

    // for each matrix size, determine if it is large or small
    // i.e. if we put it in M1 or M2
    // this determines if we use single QR or batched Jacobi for eig
    for (int i = 0; i < blk_sizes.size(); i++) {
        int mat_size = blk_sizes[i]; // size of the matrix
        int mat_num = blk_nums[i]; // number of matrices of this size
        
        this->is_large_map[mat_size] = is_large_mat(mat_size, mat_num);
        
        if (this->is_large(mat_size)) {
            this->large_mat_num += mat_num;
            this->sum_large_mat_size += mat_size * mat_num;
            this->total_large_mat_size += mat_num * mat_size * mat_size;

            this->large_mat_sizes.push_back(mat_size);
            this->large_mat_nums.push_back(mat_num);

            this->large_mat_start_indices.push_back(this->total_large_mat_size);
            this->large_W_start_indices.push_back(this->sum_large_mat_size);
        } else {
            this->sum_small_mat_size += mat_size * mat_num;
            this->small_mat_num += mat_num;
            this->total_small_mat_size += mat_num * mat_size * mat_size;

            this->small_mat_sizes.push_back(mat_size);
            this->small_mat_nums.push_back(mat_num);

            this->small_mat_start_indices.push_back(this->total_small_mat_size);
            this->small_W_start_indices.push_back(this->sum_small_mat_size);
        }
    }

    this->large_buffer_start_indices.reserve(this->large_mat_sizes.size() + 1);
    this->large_cpu_buffer_start_indices.reserve(this->large_mat_sizes.size() + 1);
    this->small_buffer_start_indices.reserve(this->small_mat_sizes.size() + 1);

    std::cout << "\nAnalysis of the large matrices sizes:" << std::endl;
    std::cout << "    size of large matrices: ";
    for (int i = 0; i < this->large_mat_sizes.size(); i++) {
        std::cout << std::setw(3) << this->large_mat_sizes[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  number of large matrices: ";
    for (int i = 0; i < this->large_mat_nums.size(); i++) {
        std::cout << std::setw(3) << this->large_mat_nums[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "    total size of large matrices: " << this->total_large_mat_size << std::endl;
    std::cout << "  sum of sizes of large matrices: " << this->sum_large_mat_size << std::endl;
    std::cout << "    nb large (with multiplicity): " << this->large_mat_num << std::endl;
    std::cout << "  large matrices start indices: ";
    for (int i = 0; i < this->large_mat_start_indices.size(); i++) {
        std::cout << this->large_mat_start_indices[i] << " ";
    }
    std::cout << std::endl << std::endl;
    std::cout << "    size of small matrices: ";
    for (int i = 0; i < this->small_mat_sizes.size(); i++) {
        std::cout << std::setw(3) << this->small_mat_sizes[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  number of small matrices: ";
    for (int i = 0; i < this->small_mat_nums.size(); i++) {
        std::cout << std::setw(3) << this->small_mat_nums[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "    total size of small matrices: " << this->total_small_mat_size << std::endl;
    std::cout << "  sum of sizes of small matrices: " << this->sum_small_mat_size << std::endl;
    std::cout << "    nb small (with multiplicity): " << this->small_mat_num << std::endl;
    std::cout << "  small matrices start indices: ";
    for (int i = 0; i < this->small_mat_start_indices.size(); i++) {
        std::cout << this->small_mat_start_indices[i] << " ";
    }
    std::cout << std::endl;
}

int MatrixSizes::large_mat_offset(int large_idx, int same_size_idx) {
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    return this->large_mat_start_indices[large_idx] + same_size_idx * this->large_mat_sizes[large_idx] * this->large_mat_sizes[large_idx];
}

int MatrixSizes::large_W_offset(int large_idx, int same_size_idx) {
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    return this->large_W_start_indices[large_idx] + same_size_idx * this->large_mat_sizes[large_idx];
}

int MatrixSizes::large_buffer_offset(int large_idx, int same_size_idx, std::vector<size_t>& eig_large_buffer_size) {
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->large_buffer_start_indices[large_idx] + eig_large_buffer_size[large_idx] * same_size_idx;
}

int MatrixSizes::large_cpu_buffer_offset(int large_idx, int same_size_idx, std::vector<size_t>& eig_large_cpu_buffer_size) {
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->large_cpu_buffer_start_indices[large_idx] + eig_large_cpu_buffer_size[large_idx] * same_size_idx;
}

int MatrixSizes::small_mat_offset(int mat_size_index) {
    assert(mat_size_index < this->large_mat_sizes.size());

    return this->large_mat_start_indices[mat_size_index];
}

int MatrixSizes::small_W_offset(int mat_size_index) {
    assert(mat_size_index < this->large_mat_sizes.size());

    return this->large_W_start_indices[mat_size_index];
}

int MatrixSizes::small_buffer_offset(int small_idx, std::vector<size_t>& eig_small_buffer_size) {
    assert(small_idx < this->small_mat_sizes.size());

    // note: this is for the case where we only use a single vector as buffer
    return this->small_buffer_start_indices[small_idx];
}

bool MatrixSizes::is_large(int mat_size) {
    return this->is_large_map.at(mat_size);
}