/*

    utils/block_structure.cu

    Analyzes the blk vector: sizes and numbers of the PSD blocks, small/medium/large matrix
    layouts, and maps for the vectorized representation of symmetric matrices.

*/

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "cuadmm/block_structure.h"

static const char *category_name(const MatrixSizeCategory category)
{
    switch (category)
    {
    case MatrixSizeCategory::LARGE:
        return "large";
    case MatrixSizeCategory::MEDIUM:
        return "medium";
    default:
        return "small";
    }
}

static std::string type_name(const char type)
{
    if (std::isprint(static_cast<unsigned char>(type)))
        return std::string("'") + type + "'";
    return "(char code " + std::to_string(static_cast<int>(type)) + ")";
}

// Append mat_num matrices of size n to one of the small/medium/large layouts of MatrixSizes
static void add_matrices(
    const int n, const int mat_num,
    int &total_mat_num, int &sum_mat_size, int &total_mat_size,
    std::vector<int> &mat_sizes, std::vector<int> &mat_nums,
    std::vector<int> &mat_start_indices, std::vector<int> &W_start_indices)
{
    total_mat_num += mat_num;
    sum_mat_size += n * mat_num;
    total_mat_size += mat_num * n * n;
    mat_sizes.push_back(n);
    mat_nums.push_back(mat_num);
    mat_start_indices.push_back(total_mat_size);
    W_start_indices.push_back(sum_mat_size);
}

BlockStructure::BlockStructure() : vec_len(0)
{
    // MatrixSizes() leaves its members uninitialized: set them as MatrixSizes::init does for no PSD block
    this->sizes.large_mat_num = 0;
    this->sizes.sum_large_mat_size = 0;
    this->sizes.total_large_mat_size = 0;
    this->sizes.max_large_mat_size = 0;
    this->sizes.medium_mat_num = 0;
    this->sizes.sum_medium_mat_size = 0;
    this->sizes.total_medium_mat_size = 0;
    this->sizes.small_mat_num = 0;
    this->sizes.sum_small_mat_size = 0;
    this->sizes.total_small_mat_size = 0;
    // first matrix starts at index 0
    this->sizes.large_mat_start_indices = {0};
    this->sizes.large_W_start_indices = {0};
    this->sizes.medium_mat_start_indices = {0};
    this->sizes.medium_W_start_indices = {0};
    this->sizes.small_mat_start_indices = {0};
    this->sizes.small_W_start_indices = {0};
}

BlockStructure::BlockStructure(const char *blk_types, const int *blk_sizes, const int blk_num) : BlockStructure()
{
    this->init(blk_types, blk_sizes, blk_num);
}

void BlockStructure::init(const char *blk_types, const int *blk_sizes, const int blk_num)
{
    if (blk_num < 0)
        throw std::invalid_argument("BlockStructure: negative number of blocks (" + std::to_string(blk_num) + ")");
    if (blk_num > 0 && (blk_types == nullptr || blk_sizes == nullptr))
        throw std::invalid_argument("BlockStructure: blk_types or blk_sizes is null");

    // First pass: validate the blocks, and check in int64 that vec_len and the total size of the
    // small/medium/large matrix buffers fit in an int. Every other size and offset (W vectors,
    // start indices, map_M1 and map_M2 values) is bounded by one of these totals.
    const int64_t int_max = std::numeric_limits<int>::max();
    int64_t vec_len_64 = 0;
    int64_t total_mat_size_64[3] = {0, 0, 0}; // indexed by SMALL, MEDIUM, LARGE
    for (int k = 0; k < blk_num; k++)
    {
        const char type = blk_types[k];
        const int64_t n = blk_sizes[k];
        if (type != 's' && type != 'u' && type != 'l')
            throw std::invalid_argument("BlockStructure: block " + std::to_string(k) + " has unknown type " +
                                        type_name(type) + " (expected 's', 'u' or 'l')");
        if (n <= 0)
            throw std::invalid_argument("BlockStructure: block " + std::to_string(k) + " has non-positive size " +
                                        std::to_string(n));
        if (type == 's')
        {
            if (n > MAX_PSD_BLK_SIZE)
                throw std::overflow_error("BlockStructure: PSD block " + std::to_string(k) + " has size " +
                                          std::to_string(n) + " > " + std::to_string(MAX_PSD_BLK_SIZE) +
                                          " (n * n overflows int)");
            const MatrixSizeCategory category = MatrixSizes::get_size_category(static_cast<int>(n));
            total_mat_size_64[category] += n * n;
            if (total_mat_size_64[category] > int_max)
                throw std::overflow_error("BlockStructure: total size of the " + std::string(category_name(category)) +
                                          " matrices overflows int (at block " + std::to_string(k) + ")");
            vec_len_64 += n * (n + 1) / 2;
        }
        else
        {
            vec_len_64 += n;
        }
        if (vec_len_64 > int_max)
            throw std::overflow_error("BlockStructure: length of the vectorized representation overflows int (at block " +
                                      std::to_string(k) + ")");
    }

    // build in a fresh object (the structure of an empty blk), so that init() can be called again
    BlockStructure res;

    // distinct PSD sizes in increasing order (as analyze_blk) and their numbers
    std::map<int, int> psd_count;
    for (int k = 0; k < blk_num; k++)
    {
        if (blk_types[k] == 's')
            psd_count[blk_sizes[k]]++;
    }
    for (const auto &[n, num] : psd_count)
    {
        res.psd_blk_sizes.push_back(n);
        res.psd_blk_nums.push_back(num);
    }

    // small/medium/large layouts (as MatrixSizes::init): in each category, the matrices are
    // grouped by increasing size, and the matrices of the same size are stored in input order
    MatrixSizes &sz = res.sizes;
    std::map<int, int> size_index; // PSD block size -> index in sz.{small,medium,large}_mat_sizes
    for (size_t i = 0; i < res.psd_blk_sizes.size(); i++)
    {
        const int n = res.psd_blk_sizes[i];
        const int num = res.psd_blk_nums[i];
        switch (MatrixSizes::get_size_category(n))
        {
        case MatrixSizeCategory::LARGE:
            size_index[n] = sz.large_mat_sizes.size();
            add_matrices(n, num, sz.large_mat_num, sz.sum_large_mat_size, sz.total_large_mat_size,
                         sz.large_mat_sizes, sz.large_mat_nums, sz.large_mat_start_indices, sz.large_W_start_indices);
            sz.max_large_mat_size = std::max(sz.max_large_mat_size, n);
            break;
        case MatrixSizeCategory::MEDIUM:
            size_index[n] = sz.medium_mat_sizes.size();
            add_matrices(n, num, sz.medium_mat_num, sz.sum_medium_mat_size, sz.total_medium_mat_size,
                         sz.medium_mat_sizes, sz.medium_mat_nums, sz.medium_mat_start_indices, sz.medium_W_start_indices);
            break;
        default:
            size_index[n] = sz.small_mat_sizes.size();
            add_matrices(n, num, sz.small_mat_num, sz.sum_small_mat_size, sz.total_small_mat_size,
                         sz.small_mat_sizes, sz.small_mat_nums, sz.small_mat_start_indices, sz.small_W_start_indices);
            break;
        }
    }

    // per-block information, in input order
    std::map<int, int> same_size_count; // PSD block size -> number of blocks of that size seen so far
    res.blocks.resize(blk_num);
    int vec_offset = 0;
    for (int k = 0; k < blk_num; k++)
    {
        BlockInfo &b = res.blocks[k];
        b.type = blk_types[k];
        b.size = blk_sizes[k];
        b.vec_offset = vec_offset;
        b.size_index = -1;
        b.same_size_index = -1;
        b.mat_offset = -1;
        b.W_offset = -1;
        if (b.type == 's')
        {
            const int n = b.size;
            b.category = MatrixSizes::get_size_category(n);
            b.vec_len = n * (n + 1) / 2;
            b.size_index = size_index[n];
            b.same_size_index = same_size_count[n]++;
            const bool large = b.category == MatrixSizeCategory::LARGE;
            const bool medium = b.category == MatrixSizeCategory::MEDIUM;
            const std::vector<int> &mat_start = large ? sz.large_mat_start_indices : (medium ? sz.medium_mat_start_indices : sz.small_mat_start_indices);
            const std::vector<int> &W_start = large ? sz.large_W_start_indices : (medium ? sz.medium_W_start_indices : sz.small_W_start_indices);
            b.mat_offset = mat_start[b.size_index] + b.same_size_index * n * n;
            b.W_offset = W_start[b.size_index] + b.same_size_index * n;
        }
        else
        {
            b.category = (b.type == 'u') ? MatrixSizeCategory::FREE : MatrixSizeCategory::NONNEGATIVE;
            b.vec_len = b.size;
        }
        vec_offset += b.vec_len;
    }
    res.vec_len = vec_offset;

    // maps for the vectorized representation (as get_maps): the entries of a PSD block list its
    // upper triangle column by column, (1,1), (1,2), (2,2), (1,3), ...
    res.map_B.assign(res.vec_len, 0);
    res.map_M1.assign(res.vec_len, 0);
    res.map_M2.assign(res.vec_len, 0);
    for (const BlockInfo &b : res.blocks)
    {
        int idx = b.vec_offset;
        if (b.type != 's')
        {
            std::fill(res.map_B.begin() + idx, res.map_B.begin() + idx + b.vec_len, b.category);
            continue;
        }
        const int n = b.size;
        for (int i = 0; i < n; i++)
        { // for each column
            for (int j = 0; j <= i; j++)
            { // in the upper triangle
                res.map_B[idx] = b.category;
                res.map_M1[idx] = b.mat_offset + n * i + j; // coefficient (j, i), column-major
                res.map_M2[idx] = b.mat_offset + n * j + i; // mirrored coefficient (i, j)
                ++idx;
            }
        }
    }

    *this = std::move(res);
}

void BlockStructure::print(std::ostream &os) const
{
    os << "\nAnalysis of the blk vector:" << std::endl;

    // unconstrained, then nonnegative variables
    for (const char type : {'u', 'l'})
    {
        for (const BlockInfo &b : this->blocks)
        {
            if (b.type == type)
                os << "     " << std::setw(4) << 1 << " " << type << ". block of size " << std::setw(3) << b.size << std::endl;
        }
    }

    // PSD matrices, grouped by size
    for (size_t i = 0; i < this->psd_blk_sizes.size(); i++)
    {
        const int n = this->psd_blk_sizes[i];
        os << "     " << std::setw(4) << this->psd_blk_nums[i] << " matrices of size " << std::setw(4) << n
           << " (" << category_name(MatrixSizes::get_size_category(n)) << ")" << std::endl;
    }
}
