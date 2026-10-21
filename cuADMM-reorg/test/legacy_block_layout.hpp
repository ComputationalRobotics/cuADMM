/*

    legacy_block_layout.hpp

    Test-only oracle for BlockStructure: the historical analyze_blk + MatrixSizes::init + get_maps path of
    SDPSolver::init, copied verbatim from src/utils/analyze_blk.cu, src/matrix_sizes.cu and src/utils/get_maps.cu of
    main (c23e67e) when Phase 1 deleted them from the library (plan P4: superseded by BlockStructure). Kept here so that
    block_structure_test.hpp can still check BlockStructure against the layout the solver used before.

*/

#ifndef CUADMM_TEST_LEGACY_BLOCK_LAYOUT_HPP
#define CUADMM_TEST_LEGACY_BLOCK_LAYOUT_HPP

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuadmm/matrix_sizes.h"
#include "cuadmm/memory.h"

namespace legacy_block_layout
{

// Analyze the blk vector to determine the size and number of PSD matrices (src/utils/analyze_blk.cu)
inline void analyze_blk(
    char *cpu_blk_types,
    HostDenseVector<int> &blk,
    std::vector<int> &psd_blk_sizes,
    std::vector<int> &psd_blk_nums)
{
    // first pass: get PSD matrix sizes
    std::set<int> size_set;
    for (int i = 0; i < blk.size; i++)
    {
        if (cpu_blk_types[i] == 's')
        {
            size_set.insert(blk.vals[i]);
        }
    }
    psd_blk_sizes = std::vector<int>(size_set.begin(), size_set.end());

    // determine the size of the small and large matrices
    std::cout << "\nAnalysis of the blk vector:" << std::endl;

    // second pass: get PSD matrix numbers
    psd_blk_nums = std::vector<int>(psd_blk_sizes.size(), 0);
    for (int i = 0; i < blk.size; i++)
    {
        for (int j = 0; j < psd_blk_sizes.size(); j++)
        {
            if (cpu_blk_types[i] == 's' && blk.vals[i] == psd_blk_sizes[j])
            {
                psd_blk_nums[j] = psd_blk_nums[j] + 1;
            }
        }
    }

    // print unconstrained variables
    for (int i = 0; i < blk.size; i++)
    {
        if (cpu_blk_types[i] == 'u')
        {
            std::cout << "     " << std::setw(4) << 1 << " u. block of size " << std::setw(3) << blk.vals[i];
            std::cout << std::endl;
        }
    }

    // print unconstrained variables
    for (int i = 0; i < blk.size; i++)
    {
        if (cpu_blk_types[i] == 'l')
        {
            std::cout << "     " << std::setw(4) << 1 << " l. block of size " << std::setw(3) << blk.vals[i];
            std::cout << std::endl;
        }
    }

    // print the PSD matrices of the map
    for (int i = 0; i < psd_blk_sizes.size(); i++)
    {
        std::cout << "     " << std::setw(4) << psd_blk_nums[i] << " matrices of size " << std::setw(4) << psd_blk_sizes[i];
        if (MatrixSizes::get_size_category(psd_blk_sizes[i]) == MatrixSizeCategory::LARGE)
        {
            std::cout << " (large)";
        }
        else if (MatrixSizes::get_size_category(psd_blk_sizes[i]) == MatrixSizeCategory::MEDIUM)
        {
            std::cout << " (medium)";
        }
        else
        {
            std::cout << " (small)";
        }
        std::cout << std::endl;
    }

    return;
}

// MatrixSizes::init of src/matrix_sizes.cu, as a free function on a fresh MatrixSizes
inline void matrix_sizes_init(MatrixSizes &sizes, const std::vector<int> &psd_blk_sizes, const std::vector<int> &psd_blk_nums)
{
    assert(psd_blk_sizes.size() == psd_blk_nums.size());

    // initialize the sizes and numbers
    sizes.total_large_mat_size = 0;
    sizes.total_medium_mat_size = 0;
    sizes.total_small_mat_size = 0;

    sizes.sum_large_mat_size = 0;
    sizes.sum_medium_mat_size = 0;
    sizes.sum_small_mat_size = 0;

    sizes.large_mat_num = 0;
    sizes.medium_mat_num = 0;
    sizes.small_mat_num = 0;

    sizes.max_large_mat_size = 0;

    // first matrix starts at index 0
    sizes.large_mat_start_indices.push_back(0);
    sizes.large_W_start_indices.push_back(0);
    sizes.medium_mat_start_indices.push_back(0);
    sizes.medium_W_start_indices.push_back(0);
    sizes.small_mat_start_indices.push_back(0);
    sizes.small_W_start_indices.push_back(0);

    // all sizes and offsets below are int: check that the totals fit
    long long total_mat_size[3] = {0, 0, 0}; // indexed by MatrixSizeCategory SMALL, MEDIUM, LARGE
    for (int i = 0; i < psd_blk_sizes.size(); i++)
    {
        if (psd_blk_sizes[i] > MAX_PSD_BLOCK_SIZE)
            throw std::overflow_error(
                "PSD block of size " + std::to_string(psd_blk_sizes[i]) + " > " + std::to_string(MAX_PSD_BLOCK_SIZE) +
                ": its number of entries does not fit in an int");
        total_mat_size[sizes.get_size_category(psd_blk_sizes[i])] +=
            (long long)psd_blk_nums[i] * psd_blk_sizes[i] * psd_blk_sizes[i];
    }
    for (int c = 0; c < 3; c++)
        if (total_mat_size[c] > std::numeric_limits<int>::max())
            throw std::overflow_error("the total number of entries of the PSD blocks of one size class does not fit in an int");

    // for each matrix size, determine if it is large, medium or small
    for (int i = 0; i < psd_blk_sizes.size(); i++)
    {
        int mat_size = psd_blk_sizes[i]; // size of the matrix
        int mat_num = psd_blk_nums[i];   // number of matrices of this size

        if (sizes.get_size_category(mat_size) == MatrixSizeCategory::LARGE)
        {
            sizes.large_mat_num += mat_num;
            sizes.sum_large_mat_size += mat_size * mat_num;
            sizes.total_large_mat_size += mat_num * mat_size * mat_size;

            sizes.large_mat_sizes.push_back(mat_size);
            sizes.large_mat_nums.push_back(mat_num);

            sizes.large_mat_start_indices.push_back(sizes.total_large_mat_size);
            sizes.large_W_start_indices.push_back(sizes.sum_large_mat_size);

            if (mat_size > sizes.max_large_mat_size)
                sizes.max_large_mat_size = mat_size;
        }
        else if (sizes.get_size_category(mat_size) == MatrixSizeCategory::MEDIUM)
        {
            sizes.medium_mat_num += mat_num;
            sizes.sum_medium_mat_size += mat_size * mat_num;
            sizes.total_medium_mat_size += mat_num * mat_size * mat_size;

            sizes.medium_mat_sizes.push_back(mat_size);
            sizes.medium_mat_nums.push_back(mat_num);

            sizes.medium_mat_start_indices.push_back(sizes.total_medium_mat_size);
            sizes.medium_W_start_indices.push_back(sizes.sum_medium_mat_size);
        }
        else
        {
            sizes.sum_small_mat_size += mat_size * mat_num;
            sizes.small_mat_num += mat_num;
            sizes.total_small_mat_size += mat_num * mat_size * mat_size;

            sizes.small_mat_sizes.push_back(mat_size);
            sizes.small_mat_nums.push_back(mat_num);

            sizes.small_mat_start_indices.push_back(sizes.total_small_mat_size);
            sizes.small_W_start_indices.push_back(sizes.sum_small_mat_size);
        }
    }

    sizes.large_buffer_start_indices.reserve(sizes.large_mat_sizes.size() + 1);
    sizes.large_cpu_buffer_start_indices.reserve(sizes.large_mat_sizes.size() + 1);
    sizes.small_buffer_start_indices.reserve(sizes.small_mat_sizes.size() + 1);
}

// Computes the maps for the vectorized representation of symmetric matrices (src/utils/get_maps.cu)
inline void get_maps(
    const char *blk_types,
    const HostDenseVector<int> &blk_sizes,
    const int vec_len,
    std::vector<int> &map_B_tmp, std::vector<int> &map_M1_tmp, std::vector<int> &map_M2_tmp,
    const MatrixSizes &sizes)
{
    // Reserve space for the maps
    map_B_tmp.clear();
    map_B_tmp.resize(vec_len);
    // fill map_B_tmp with -1, for error checking
    std::fill(map_B_tmp.begin(), map_B_tmp.end(), -1);
    map_M1_tmp.clear();
    map_M1_tmp.resize(vec_len);
    map_M2_tmp.clear();
    map_M2_tmp.resize(vec_len);

    int idx = 0;         // current index in the maps
    int mat_size_index;  // n-th size of matrix in sizes.large_mat_sizes or sizes.small_mat_sizes
    int same_size_index; // n-th matrix of the same size
    std::vector<int> large_mat_nb_encoutered(sizes.large_mat_sizes.size(), 0);
    std::vector<int> medium_mat_nb_encoutered(sizes.medium_mat_sizes.size(), 0);
    std::vector<int> small_mat_nb_encoutered(sizes.small_mat_sizes.size(), 0);
    int s; // block size
    int b; // block type (0 for large, 1 for small)
    for (int k = 0; k < blk_sizes.size; ++k)
    { // for each block
        if (blk_types[k] == 'u')
        {
            std::fill(map_B_tmp.begin() + idx, map_B_tmp.begin() + idx + blk_sizes.vals[k], MatrixSizeCategory::FREE);
            idx += blk_sizes.vals[k];
            continue;
        }

        if (blk_types[k] == 'l')
        {
            std::fill(map_B_tmp.begin() + idx, map_B_tmp.begin() + idx + blk_sizes.vals[k], MatrixSizeCategory::NONNEGATIVE);
            idx += blk_sizes.vals[k];
            continue;
        }

        s = blk_sizes.vals[k];
        if (sizes.get_size_category(s) == MatrixSizeCategory::LARGE)
        {
            b = MatrixSizeCategory::LARGE;
            auto findex = std::find(sizes.large_mat_sizes.begin(), sizes.large_mat_sizes.end(), s);
            mat_size_index = std::distance(sizes.large_mat_sizes.begin(), findex);
            same_size_index = large_mat_nb_encoutered[mat_size_index];
            ++large_mat_nb_encoutered[mat_size_index];
        }
        else if (sizes.get_size_category(s) == MatrixSizeCategory::MEDIUM)
        {
            b = MatrixSizeCategory::MEDIUM;
            auto findex = std::find(sizes.medium_mat_sizes.begin(), sizes.medium_mat_sizes.end(), s);
            mat_size_index = std::distance(sizes.medium_mat_sizes.begin(), findex);
            same_size_index = medium_mat_nb_encoutered[mat_size_index];
            ++medium_mat_nb_encoutered[mat_size_index];
        }
        else
        {
            b = MatrixSizeCategory::SMALL;
            auto findex = std::find(sizes.small_mat_sizes.begin(), sizes.small_mat_sizes.end(), s);
            mat_size_index = std::distance(sizes.small_mat_sizes.begin(), findex);
            same_size_index = small_mat_nb_encoutered[mat_size_index];
            ++small_mat_nb_encoutered[mat_size_index];
        }
        for (int i = 1; i <= s; ++i)
        { // for each coefficient
            for (int j = 1; j <= i; ++j)
            { // in the triangle
                map_B_tmp[idx] = b;
                if (sizes.get_size_category(s) == MatrixSizeCategory::LARGE)
                {
                    // count horizontally
                    map_M1_tmp[idx] = sizes.large_mat_offset(mat_size_index, same_size_index) + s * (i - 1) + j - 1;
                    // count vertically
                    map_M2_tmp[idx] = sizes.large_mat_offset(mat_size_index, same_size_index) + s * (j - 1) + i - 1;
                }
                else if (sizes.get_size_category(s) == MatrixSizeCategory::MEDIUM)
                {
                    // count horizontally
                    map_M1_tmp[idx] = sizes.medium_mat_offset(mat_size_index, same_size_index) + s * (i - 1) + j - 1;
                    // count vertically
                    map_M2_tmp[idx] = sizes.medium_mat_offset(mat_size_index, same_size_index) + s * (j - 1) + i - 1;
                }
                else
                {
                    // count horizontally
                    map_M1_tmp[idx] = sizes.small_mat_offset(mat_size_index, same_size_index) + s * (i - 1) + j - 1;
                    // count vertically
                    map_M2_tmp[idx] = sizes.small_mat_offset(mat_size_index, same_size_index) + s * (j - 1) + i - 1;
                }
                ++idx;
            }
        }
    }
    return;
}

} // namespace legacy_block_layout

#endif // CUADMM_TEST_LEGACY_BLOCK_LAYOUT_HPP
