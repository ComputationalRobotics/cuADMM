#include "cuadmm_legacy/matrix_sizes.h"

#include <vector>
#include <unordered_map>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

// MatrixSizes::init was deleted in Phase 1 (superseded by BlockStructure::init, plan P4; kept as a test oracle in
// test/legacy_block_layout.hpp)

int MatrixSizes::large_mat_offset(int large_idx, int same_size_idx) const
{
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    return this->large_mat_start_indices[large_idx] + same_size_idx * this->large_mat_sizes[large_idx] * this->large_mat_sizes[large_idx];
}

int MatrixSizes::large_W_offset(int large_idx, int same_size_idx) const
{
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    return this->large_W_start_indices[large_idx] + same_size_idx * this->large_mat_sizes[large_idx];
}

size_t MatrixSizes::large_buffer_offset(int large_idx, int same_size_idx, std::vector<size_t> &eig_large_buffer_size) const
{
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->large_buffer_start_indices[large_idx] + eig_large_buffer_size[large_idx] * same_size_idx;
}

size_t MatrixSizes::large_cpu_buffer_offset(int large_idx, int same_size_idx, std::vector<size_t> &eig_large_cpu_buffer_size) const
{
    assert(large_idx < this->large_mat_sizes.size());
    assert(same_size_idx < this->large_mat_nums[large_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->large_cpu_buffer_start_indices[large_idx] + eig_large_cpu_buffer_size[large_idx] * same_size_idx;
}

int MatrixSizes::medium_mat_offset(int medium_idx, int same_size_idx) const
{
    assert(medium_idx < this->medium_mat_sizes.size());
    assert(same_size_idx < this->medium_mat_nums[medium_idx]);

    return this->medium_mat_start_indices[medium_idx] + same_size_idx * this->medium_mat_sizes[medium_idx] * this->medium_mat_sizes[medium_idx];
}

int MatrixSizes::medium_W_offset(int medium_idx, int same_size_idx) const
{
    assert(medium_idx < this->medium_mat_sizes.size());
    assert(same_size_idx < this->medium_mat_nums[medium_idx]);

    return this->medium_W_start_indices[medium_idx] + same_size_idx * this->medium_mat_sizes[medium_idx];
}

size_t MatrixSizes::medium_buffer_offset(int medium_idx, int same_size_idx, std::vector<size_t> &eig_medium_buffer_size) const
{
    assert(medium_idx < this->medium_mat_sizes.size());
    assert(same_size_idx < this->medium_mat_nums[medium_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->medium_buffer_start_indices[medium_idx] + eig_medium_buffer_size[medium_idx] * same_size_idx;
}

size_t MatrixSizes::medium_cpu_buffer_offset(int medium_idx, int same_size_idx, std::vector<size_t> &eig_medium_cpu_buffer_size) const
{
    assert(medium_idx < this->medium_mat_sizes.size());
    assert(same_size_idx < this->medium_mat_nums[medium_idx]);

    // note: this is for the case where we only use a single vector as buffer
    return this->medium_cpu_buffer_start_indices[medium_idx] + eig_medium_cpu_buffer_size[medium_idx] * same_size_idx;
}

int MatrixSizes::small_mat_offset(int mat_size_index, int same_size_idx) const
{
    assert(mat_size_index < this->small_mat_sizes.size());
    assert(same_size_idx < this->small_mat_nums[mat_size_index]);

    return this->small_mat_start_indices[mat_size_index] + same_size_idx * this->small_mat_sizes[mat_size_index] * this->small_mat_sizes[mat_size_index];
}

int MatrixSizes::small_W_offset(int mat_size_index) const
{
    assert(mat_size_index < this->small_mat_sizes.size());

    return this->small_W_start_indices[mat_size_index];
}

size_t MatrixSizes::small_buffer_offset(int small_idx, std::vector<size_t> &eig_small_buffer_size) const
{
    assert(small_idx < this->small_mat_sizes.size());

    // note: this is for the case where we only use a single vector as buffer
    return this->small_buffer_start_indices[small_idx];
}

int MatrixSizes::small_limit = SMALL_MAT_LIMIT;
int MatrixSizes::medium_limit = MEDIUM_MAT_LIMIT;

void MatrixSizes::set_limits(int small_max, int medium_max)
{
    // the small tier is cusolverDnDsyevjBatched (n <= 32); the medium tier starts above the small one
    if (small_max < 1 || small_max > 32 || medium_max <= small_max || medium_max > MAX_PSD_BLOCK_SIZE)
        throw std::invalid_argument("MatrixSizes::set_limits: invalid tier cutoffs " + std::to_string(small_max) + " / " + std::to_string(medium_max));
    small_limit = small_max;
    medium_limit = medium_max;
}

MatrixSizeCategory MatrixSizes::get_size_category(const int mat_size)
{
    if (mat_size <= small_limit)
    {
        return MatrixSizeCategory::SMALL;
    }
    else if (mat_size <= medium_limit)
    {
        return MatrixSizeCategory::MEDIUM;
    }
    else
    {
        return MatrixSizeCategory::LARGE;
    }
}