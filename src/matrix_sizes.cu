#include "cuadmm/matrix_sizes.h"

#include <vector>
#include <unordered_map>
#include <cassert>
#include <iostream>
#include <iomanip>

void MatrixSizes::init(const std::vector<int>& psd_blk_sizes, const std::vector<int>& psd_blk_nums) {
    assert(psd_blk_sizes.size() == psd_blk_nums.size());

    // initialize the sizes and numbers
    this->total_large_mat_size = 0;
    this->total_medium_mat_size = 0;
    this->total_small_mat_size = 0;

    this->sum_large_mat_size = 0;
    this->sum_medium_mat_size = 0;
    this->sum_small_mat_size = 0;

    this->large_mat_num = 0;
    this->medium_mat_num = 0;
    this->small_mat_num = 0;

    this->max_large_mat_size = 0;

    // first matrix starts at index 0
    this->large_mat_start_indices.push_back(0);
    this->large_W_start_indices.push_back(0);
    this->medium_mat_start_indices.push_back(0);
    this->medium_W_start_indices.push_back(0);
    this->small_mat_start_indices.push_back(0);
    this->small_W_start_indices.push_back(0);

    // for each matrix size, determine if it is large, medium or small
    for (int i = 0; i < psd_blk_sizes.size(); i++) {
        int mat_size = psd_blk_sizes[i]; // size of the matrix
        int mat_num = psd_blk_nums[i]; // number of matrices of this size

        if (this->get_size_category(mat_size) == MatrixSizeCategory::LARGE) {
            this->large_mat_num += mat_num;
            this->sum_large_mat_size += mat_size * mat_num;
            this->total_large_mat_size += mat_num * mat_size * mat_size;

            this->large_mat_sizes.push_back(mat_size);
            this->large_mat_nums.push_back(mat_num);

            this->large_mat_start_indices.push_back(this->total_large_mat_size);
            this->large_W_start_indices.push_back(this->sum_large_mat_size);

            if (mat_size > this->max_large_mat_size)
                this->max_large_mat_size = mat_size;
        } else if (this->get_size_category(mat_size) == MatrixSizeCategory::MEDIUM) {
            this->medium_mat_num += mat_num;
            this->sum_medium_mat_size += mat_size * mat_num;
            this->total_medium_mat_size += mat_num * mat_size * mat_size;

            this->medium_mat_sizes.push_back(mat_size);
            this->medium_mat_nums.push_back(mat_num);

            this->medium_mat_start_indices.push_back(this->total_medium_mat_size);
            this->medium_W_start_indices.push_back(this->sum_medium_mat_size);
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

    // std::cout << "\nAnalysis of the large matrices sizes:" << std::endl;
    // std::cout << "    size of large matrices: ";
    // for (int i = 0; i < this->large_mat_sizes.size(); i++) {
    //     std::cout << std::setw(3) << this->large_mat_sizes[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "  number of large matrices: ";
    // for (int i = 0; i < this->large_mat_nums.size(); i++) {
    //     std::cout << std::setw(3) << this->large_mat_nums[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "    total size of large matrices: " << this->total_large_mat_size << std::endl;
    // std::cout << "  sum of sizes of large matrices: " << this->sum_large_mat_size << std::endl;
    // std::cout << "    nb large (with multiplicity): " << this->large_mat_num << std::endl;
    // std::cout << "  large matrices start indices: ";
    // for (int i = 0; i < this->large_mat_start_indices.size(); i++) {
    //     std::cout << this->large_mat_start_indices[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "\nAnalysis of the small matrices sizes:" << std::endl;
    // std::cout << "    size of small matrices: ";
    // for (int i = 0; i < this->small_mat_sizes.size(); i++) {
    //     std::cout << std::setw(3) << this->small_mat_sizes[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "  number of small matrices: ";
    // for (int i = 0; i < this->small_mat_nums.size(); i++) {
    //     std::cout << std::setw(3) << this->small_mat_nums[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "    total size of small matrices: " << this->total_small_mat_size << std::endl;
    // std::cout << "  sum of sizes of small matrices: " << this->sum_small_mat_size << std::endl;
    // std::cout << "    nb small (with multiplicity): " << this->small_mat_num << std::endl;
    // std::cout << "  small matrices start indices: ";
    // for (int i = 0; i < this->small_mat_start_indices.size(); i++) {
    //     std::cout << this->small_mat_start_indices[i] << " ";
    // }
    // std::cout << std::endl;
}

int MatrixSizes::large_mat_offset(int large_idx, int same_size_idx) const {
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    return this->large_mat_start_indices[large_idx] + same_size_idx * this->large_mat_sizes[large_idx] * this->large_mat_sizes[large_idx];
}

int MatrixSizes::large_W_offset(int large_idx, int same_size_idx) const {
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    return this->large_W_start_indices[large_idx] + same_size_idx * this->large_mat_sizes[large_idx];
}

int MatrixSizes::large_buffer_offset(int large_idx, int same_size_idx, std::vector<size_t>& eig_large_buffer_size) const {
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->large_buffer_start_indices[large_idx] + eig_large_buffer_size[large_idx] * same_size_idx;
}

int MatrixSizes::large_cpu_buffer_offset(int large_idx, int same_size_idx, std::vector<size_t>& eig_large_cpu_buffer_size) const {
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->large_cpu_buffer_start_indices[large_idx] + eig_large_cpu_buffer_size[large_idx] * same_size_idx;
}

int MatrixSizes::medium_mat_offset(int medium_idx, int same_size_idx) const {
    assert(medium_idx < this->medium_mat_sizes.size());
    assert(same_size_idx < this->medium_mat_nums[medium_idx]);

    return this->medium_mat_start_indices[medium_idx] + same_size_idx * this->medium_mat_sizes[medium_idx] * this->medium_mat_sizes[medium_idx];
}

int MatrixSizes::medium_W_offset(int medium_idx, int same_size_idx) const {
    assert(medium_idx < this->medium_mat_sizes.size());
    assert(same_size_idx < this->medium_mat_nums[medium_idx]);

    return this->medium_W_start_indices[medium_idx] + same_size_idx * this->medium_mat_sizes[medium_idx];
}

int MatrixSizes::medium_buffer_offset(int medium_idx, int same_size_idx, std::vector<size_t>& eig_medium_buffer_size) const {
    assert(medium_idx < this->medium_mat_sizes.size());
    assert(same_size_idx < this->medium_mat_nums[medium_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->medium_buffer_start_indices[medium_idx] + eig_medium_buffer_size[medium_idx] * same_size_idx;
}

int MatrixSizes::medium_cpu_buffer_offset(int medium_idx, int same_size_idx, std::vector<size_t>& eig_medium_cpu_buffer_size) const {
    assert(medium_idx < this->medium_mat_sizes.size());
    assert(same_size_idx < this->medium_mat_nums[medium_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->medium_cpu_buffer_start_indices[medium_idx] + eig_medium_cpu_buffer_size[medium_idx] * same_size_idx;
}

int MatrixSizes::small_mat_offset(int mat_size_index, int same_size_idx) const {
    assert(mat_size_index < this->small_mat_sizes.size());
    assert(same_size_idx < this->small_mat_nums[mat_size_index]);

    return this->small_mat_start_indices[mat_size_index] + same_size_idx * this->small_mat_sizes[mat_size_index] * this->small_mat_sizes[mat_size_index];
}

int MatrixSizes::small_W_offset(int mat_size_index) const {
    assert(mat_size_index < this->small_mat_sizes.size());

    return this->small_W_start_indices[mat_size_index];
}

int MatrixSizes::small_buffer_offset(int small_idx, std::vector<size_t>& eig_small_buffer_size) const {
    assert(small_idx < this->small_mat_sizes.size());

    // note: this is for the case where we only use a single vector as buffer
    return this->small_buffer_start_indices[small_idx];
}

MatrixSizeCategory MatrixSizes::get_size_category(const int mat_size) {
    if (mat_size <= SMALL_MAT_LIMIT) {
        return MatrixSizeCategory::SMALL;
    } else if (mat_size <= MEDIUM_MAT_LIMIT) {
        return MatrixSizeCategory::MEDIUM;
    } else {
        return MatrixSizeCategory::LARGE;
    }
}