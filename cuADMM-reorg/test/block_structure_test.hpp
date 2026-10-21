#include <gtest/gtest.h>
#include <algorithm>
#include <climits>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuadmm/block_structure.h"
#include "cuadmm/matrix_sizes.h"
#include "cuadmm/memory.h"
#include "cuadmm/utils.h"

namespace block_structure_test
{
    // Output of the legacy analyze_blk + MatrixSizes::init + get_maps path, as SDPSolver::init runs it
    struct Legacy
    {
        int vec_len;
        std::vector<int> psd_blk_sizes;
        std::vector<int> psd_blk_nums;
        MatrixSizes sizes;
        std::vector<int> map_B, map_M1, map_M2;
        std::string printed; // what analyze_blk prints
    };

    inline void run_legacy(const std::vector<char> &types, const std::vector<int> &blk_sizes, Legacy &ref)
    {
        const int blk_num = types.size();
        HostDenseVector<int> blk(blk_num);
        std::copy(blk_sizes.begin(), blk_sizes.end(), blk.vals);
        std::vector<char> types_copy(types); // analyze_blk takes a non-const pointer

        // vec_len as Problem::from_txt computes it
        ref.vec_len = 0;
        for (int k = 0; k < blk_num; k++)
            ref.vec_len += (types[k] == 's') ? blk_sizes[k] * (blk_sizes[k] + 1) / 2 : blk_sizes[k];

        testing::internal::CaptureStdout();
        analyze_blk(types_copy.data(), blk, ref.psd_blk_sizes, ref.psd_blk_nums);
        ref.printed = testing::internal::GetCapturedStdout();
        ref.sizes.init(ref.psd_blk_sizes, ref.psd_blk_nums);
        get_maps(types.data(), blk, ref.vec_len, ref.map_B, ref.map_M1, ref.map_M2, ref.sizes);
    }

    // Compare two int vectors, reporting the first mismatch rather than both vectors
    inline ::testing::AssertionResult same_ints(const std::vector<int> &expected, const std::vector<int> &actual)
    {
        if (expected.size() != actual.size())
            return ::testing::AssertionFailure() << "sizes differ: expected " << expected.size() << ", got " << actual.size();
        for (size_t i = 0; i < expected.size(); i++)
        {
            if (expected[i] != actual[i])
                return ::testing::AssertionFailure() << "first mismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i];
        }
        return ::testing::AssertionSuccess();
    }

    inline void expect_same_sizes(const MatrixSizes &e, const MatrixSizes &a)
    {
#define EXPECT_SAME_MEMBER(member) EXPECT_EQ(e.member, a.member) << #member
        EXPECT_SAME_MEMBER(large_mat_num);
        EXPECT_SAME_MEMBER(sum_large_mat_size);
        EXPECT_SAME_MEMBER(total_large_mat_size);
        EXPECT_SAME_MEMBER(large_mat_sizes);
        EXPECT_SAME_MEMBER(large_mat_nums);
        EXPECT_SAME_MEMBER(large_mat_start_indices);
        EXPECT_SAME_MEMBER(large_W_start_indices);
        EXPECT_SAME_MEMBER(large_buffer_start_indices);
        EXPECT_SAME_MEMBER(large_cpu_buffer_start_indices);
        EXPECT_SAME_MEMBER(max_large_mat_size);

        EXPECT_SAME_MEMBER(medium_mat_num);
        EXPECT_SAME_MEMBER(sum_medium_mat_size);
        EXPECT_SAME_MEMBER(total_medium_mat_size);
        EXPECT_SAME_MEMBER(medium_mat_sizes);
        EXPECT_SAME_MEMBER(medium_mat_nums);
        EXPECT_SAME_MEMBER(medium_mat_start_indices);
        EXPECT_SAME_MEMBER(medium_W_start_indices);
        EXPECT_SAME_MEMBER(medium_buffer_start_indices);
        EXPECT_SAME_MEMBER(medium_cpu_buffer_start_indices);

        EXPECT_SAME_MEMBER(small_mat_num);
        EXPECT_SAME_MEMBER(sum_small_mat_size);
        EXPECT_SAME_MEMBER(total_small_mat_size);
        EXPECT_SAME_MEMBER(small_mat_sizes);
        EXPECT_SAME_MEMBER(small_mat_nums);
        EXPECT_SAME_MEMBER(small_mat_start_indices);
        EXPECT_SAME_MEMBER(small_W_start_indices);
        EXPECT_SAME_MEMBER(small_buffer_start_indices);
#undef EXPECT_SAME_MEMBER

        // the eig buffer start indices are filled later by SDPSolver::init
        EXPECT_TRUE(a.large_buffer_start_indices.empty());
        EXPECT_TRUE(a.large_cpu_buffer_start_indices.empty());
        EXPECT_TRUE(a.medium_buffer_start_indices.empty());
        EXPECT_TRUE(a.medium_cpu_buffer_start_indices.empty());
        EXPECT_TRUE(a.small_buffer_start_indices.empty());
    }

    // Check the per-block information against the offsets given by the legacy MatrixSizes
    inline void expect_same_blocks(const std::vector<char> &types, const std::vector<int> &blk_sizes,
                                   const MatrixSizes &ref, const BlockStructure &bs)
    {
        ASSERT_EQ(bs.blocks.size(), types.size());
        int vec_offset = 0;
        for (size_t k = 0; k < types.size(); k++)
        {
            SCOPED_TRACE("block " + std::to_string(k));
            const BlockInfo &b = bs.blocks[k];
            const int n = blk_sizes[k];
            const int len = (types[k] == 's') ? n * (n + 1) / 2 : n;
            EXPECT_EQ(b.type, types[k]);
            EXPECT_EQ(b.size, n);
            EXPECT_EQ(b.vec_offset, vec_offset);
            EXPECT_EQ(b.vec_len, len);
            vec_offset += len;

            if (types[k] != 's')
            {
                EXPECT_EQ(b.category, (types[k] == 'u') ? MatrixSizeCategory::FREE : MatrixSizeCategory::NONNEGATIVE);
                EXPECT_EQ(b.size_index, -1);
                EXPECT_EQ(b.same_size_index, -1);
                EXPECT_EQ(b.mat_offset, -1);
                EXPECT_EQ(b.W_offset, -1);
                continue;
            }

            // number of earlier PSD blocks of the same size, and index of the size in its category
            int same_size_index = 0;
            for (size_t l = 0; l < k; l++)
                same_size_index += (types[l] == 's' && blk_sizes[l] == n);
            const MatrixSizeCategory category = MatrixSizes::get_size_category(n);
            const std::vector<int> &category_sizes = (category == MatrixSizeCategory::LARGE)    ? ref.large_mat_sizes
                                                     : (category == MatrixSizeCategory::MEDIUM) ? ref.medium_mat_sizes
                                                                                                : ref.small_mat_sizes;
            const int size_index = std::find(category_sizes.begin(), category_sizes.end(), n) - category_sizes.begin();
            ASSERT_LT(size_index, (int)category_sizes.size());

            EXPECT_EQ(b.category, category);
            EXPECT_EQ(b.size_index, size_index);
            EXPECT_EQ(b.same_size_index, same_size_index);
            if (category == MatrixSizeCategory::LARGE)
            {
                EXPECT_EQ(b.mat_offset, ref.large_mat_offset(size_index, same_size_index));
                EXPECT_EQ(b.W_offset, ref.large_W_offset(size_index, same_size_index));
            }
            else if (category == MatrixSizeCategory::MEDIUM)
            {
                EXPECT_EQ(b.mat_offset, ref.medium_mat_offset(size_index, same_size_index));
                EXPECT_EQ(b.W_offset, ref.medium_W_offset(size_index, same_size_index));
            }
            else
            {
                // small matrices of one size are decomposed as a batch: their eigenvalues are contiguous
                EXPECT_EQ(b.mat_offset, ref.small_mat_offset(size_index, same_size_index));
                EXPECT_EQ(b.W_offset, ref.small_W_offset(size_index) + same_size_index * n);
            }
        }
        EXPECT_EQ(bs.vec_len, vec_offset);
    }

    // Check that bs, the BlockStructure of blk, reproduces analyze_blk + MatrixSizes::init + get_maps
    inline void expect_same_as_legacy(const std::vector<char> &types, const std::vector<int> &blk_sizes,
                                      const BlockStructure &bs)
    {
        Legacy ref;
        run_legacy(types, blk_sizes, ref);
        std::ostringstream printed;
        bs.print(printed);

        EXPECT_EQ(bs.vec_len, ref.vec_len);
        EXPECT_EQ(bs.psd_blk_sizes, ref.psd_blk_sizes);
        EXPECT_EQ(bs.psd_blk_nums, ref.psd_blk_nums);
        expect_same_sizes(ref.sizes, bs.sizes);
        expect_same_blocks(types, blk_sizes, ref.sizes, bs);
        EXPECT_TRUE(same_ints(ref.map_B, bs.map_B)) << "map_B";
        EXPECT_TRUE(same_ints(ref.map_M1, bs.map_M1)) << "map_M1";
        EXPECT_TRUE(same_ints(ref.map_M2, bs.map_M2)) << "map_M2";
        // get_maps initializes map_B to -1: every entry must have been assigned
        EXPECT_EQ(std::count(ref.map_B.begin(), ref.map_B.end(), -1), 0);
        EXPECT_EQ(printed.str(), ref.printed);
    }

    // Draw a random blk vector: 1-12 blocks of types s/u/l, PSD sizes in the small (<= 32), medium
    // (33..1000) and, rarely, large (1001..1500) regimes, regime boundaries and repeated sizes.
    inline void random_blk(std::mt19937 &rng, std::vector<char> &types, std::vector<int> &blk_sizes)
    {
        // keep the maps of one mixture below 4M entries so that the test stays fast
        const long long max_vec_len = 4000000;
        const int boundary_sizes[] = {1, 2, 32, 33, 1000, 1001};
        auto uniform = [&rng](int lo, int hi)
        { return std::uniform_int_distribution<int>(lo, hi)(rng); };

        types.clear();
        blk_sizes.clear();
        const int blk_num = uniform(1, 12);
        long long vec_len = 0;
        for (int k = 0; k < blk_num; k++)
        {
            const int type_draw = uniform(0, 99);
            if (type_draw < 30)
            {
                // free or nonnegative block
                const int n = (uniform(0, 9) == 0) ? uniform(1, 500) : uniform(1, 50);
                types.push_back((type_draw < 15) ? 'u' : 'l');
                blk_sizes.push_back(n);
                vec_len += n;
                continue;
            }

            std::vector<int> earlier_psd_sizes;
            for (size_t l = 0; l < types.size(); l++)
            {
                if (types[l] == 's')
                    earlier_psd_sizes.push_back(blk_sizes[l]);
            }
            int n;
            if (!earlier_psd_sizes.empty() && uniform(0, 99) < 35)
            {
                n = earlier_psd_sizes[uniform(0, earlier_psd_sizes.size() - 1)]; // repeated size
            }
            else
            {
                const int size_draw = uniform(0, 99);
                if (size_draw < 10)
                    n = boundary_sizes[uniform(0, 5)];
                else if (size_draw < 55)
                    n = uniform(1, 32); // small
                else if (size_draw < 82)
                    n = uniform(33, 200); // medium
                else if (size_draw < 95)
                    n = uniform(201, 1000); // medium
                else
                    n = uniform(1001, 1500); // large
            }
            if (vec_len + (long long)n * (n + 1) / 2 > max_vec_len)
                n = uniform(1, 32);
            types.push_back('s');
            blk_sizes.push_back(n);
            vec_len += (long long)n * (n + 1) / 2;
        }
    }

    inline std::string describe_blk(const std::vector<char> &types, const std::vector<int> &blk_sizes)
    {
        std::ostringstream os;
        for (size_t k = 0; k < types.size(); k++)
            os << (k ? ", " : "") << types[k] << " " << blk_sizes[k];
        return os.str();
    }
}

TEST(BlockStructure, MatchesLegacyOnRandomMixtures)
{
    using namespace block_structure_test;
    const int trial_num = 300;
    std::mt19937 rng(20260925);

    // count what the mixtures exercise, to check that every regime is covered
    int with_u = 0, with_l = 0, with_small = 0, with_medium = 0, with_large = 0;
    int with_repeated_size = 0, with_repeated_large = 0, with_several_large_sizes = 0, with_boundary = 0;
    std::vector<char> types;
    std::vector<int> blk_sizes;
    BlockStructure bs; // reused: init() must not keep anything from the previous mixture
    for (int trial = 0; trial < trial_num; trial++)
    {
        random_blk(rng, types, blk_sizes);
        SCOPED_TRACE("trial " + std::to_string(trial) + ", blk = {" + describe_blk(types, blk_sizes) + "}");
        bs.init(types.data(), blk_sizes.data(), types.size());
        expect_same_as_legacy(types, blk_sizes, bs);
        if (HasFailure())
            return; // report the first failing mixture only

        with_u += std::count(types.begin(), types.end(), 'u') > 0;
        with_l += std::count(types.begin(), types.end(), 'l') > 0;
        with_small += bs.sizes.small_mat_num > 0;
        with_medium += bs.sizes.medium_mat_num > 0;
        with_large += bs.sizes.large_mat_num > 0;
        with_repeated_size += !bs.psd_blk_nums.empty() && *std::max_element(bs.psd_blk_nums.begin(), bs.psd_blk_nums.end()) > 1;
        with_repeated_large += bs.sizes.large_mat_num > (int)bs.sizes.large_mat_sizes.size();
        with_several_large_sizes += bs.sizes.large_mat_sizes.size() > 1;
        bool boundary = false;
        for (const int n : bs.psd_blk_sizes)
            boundary = boundary || n == 32 || n == 33 || n == 1000 || n == 1001;
        with_boundary += boundary;
    }

    std::cout << "[ coverage ] " << trial_num << " mixtures, with: u " << with_u << ", l " << with_l
              << ", small " << with_small << ", medium " << with_medium << ", large " << with_large
              << ", repeated PSD size " << with_repeated_size << ", repeated large size " << with_repeated_large
              << ", several large sizes " << with_several_large_sizes << ", size 32/33/1000/1001 " << with_boundary
              << std::endl;
    EXPECT_GE(with_u, 50);
    EXPECT_GE(with_l, 50);
    EXPECT_GE(with_small, 50);
    EXPECT_GE(with_medium, 50);
    EXPECT_GE(with_large, 20);
    EXPECT_GE(with_repeated_size, 50);
    EXPECT_GE(with_repeated_large, 3);
    EXPECT_GE(with_several_large_sizes, 3);
    EXPECT_GE(with_boundary, 20);
}

TEST(BlockStructure, EmptyBlkMatchesLegacy)
{
    using namespace block_structure_test;
    const BlockStructure bs(nullptr, nullptr, 0);
    expect_same_as_legacy({}, {}, bs);
    // a default-constructed BlockStructure is fully initialized, as the structure of the empty blk
    const BlockStructure default_bs;
    expect_same_as_legacy({}, {}, default_bs);
}

TEST(BlockStructure, SmallExample)
{
    // two 2x2 PSD blocks around a free block of size 2, then a nonnegative block of size 1
    const std::vector<char> types = {'s', 'u', 's', 'l'};
    const std::vector<int> blk_sizes = {2, 2, 2, 1};
    const BlockStructure bs(types.data(), blk_sizes.data(), types.size());

    EXPECT_EQ(bs.vec_len, 9);
    EXPECT_EQ(bs.psd_blk_sizes, std::vector<int>({2}));
    EXPECT_EQ(bs.psd_blk_nums, std::vector<int>({2}));
    EXPECT_EQ(bs.sizes.small_mat_num, 2);
    EXPECT_EQ(bs.sizes.small_mat_start_indices, std::vector<int>({0, 8}));
    EXPECT_EQ(bs.sizes.small_W_start_indices, std::vector<int>({0, 4}));
    EXPECT_EQ(bs.sizes.medium_mat_num, 0);
    EXPECT_EQ(bs.sizes.large_mat_num, 0);

    EXPECT_EQ(bs.blocks[1].vec_offset, 3);
    EXPECT_EQ(bs.blocks[2].vec_offset, 5);
    EXPECT_EQ(bs.blocks[2].same_size_index, 1);
    EXPECT_EQ(bs.blocks[2].mat_offset, 4);
    EXPECT_EQ(bs.blocks[2].W_offset, 2);
    EXPECT_EQ(bs.blocks[3].vec_offset, 8);

    // svec order (1,1), (1,2), (2,2); M1 points to (j, i) and M2 to (i, j) in column-major order
    EXPECT_EQ(bs.map_B, std::vector<int>({0, 0, 0, 3, 3, 0, 0, 0, 4}));
    EXPECT_EQ(bs.map_M1, std::vector<int>({0, 2, 3, 0, 0, 4, 6, 7, 0}));
    EXPECT_EQ(bs.map_M2, std::vector<int>({0, 1, 3, 0, 0, 4, 5, 7, 0}));
}

TEST(BlockStructure, RejectsInvalidBlocks)
{
    BlockStructure bs;
    const std::vector<char> ok_types = {'s', 'u'};
    const std::vector<int> ok_sizes = {3, 2};
    bs.init(ok_types.data(), ok_sizes.data(), 2);
    ASSERT_EQ(bs.vec_len, 8);

    // unknown types
    for (const char bad_type : {'x', 'S', 'q', '\0'})
    {
        const std::vector<char> types = {'s', bad_type};
        const std::vector<int> sizes = {3, 2};
        EXPECT_THROW(bs.init(types.data(), sizes.data(), 2), std::invalid_argument) << "type code " << (int)bad_type;
    }
    // non-positive sizes
    for (const char type : {'s', 'u', 'l'})
    {
        for (const int bad_size : {0, -1, INT_MIN})
        {
            const std::vector<char> types = {'u', type};
            const std::vector<int> sizes = {4, bad_size};
            EXPECT_THROW(bs.init(types.data(), sizes.data(), 2), std::invalid_argument) << type << " " << bad_size;
        }
    }
    // bad arguments
    EXPECT_THROW(bs.init(ok_types.data(), ok_sizes.data(), -1), std::invalid_argument);
    EXPECT_THROW(bs.init(nullptr, ok_sizes.data(), 2), std::invalid_argument);
    EXPECT_THROW(bs.init(ok_types.data(), nullptr, 2), std::invalid_argument);

    // a failed init leaves the structure unchanged
    EXPECT_EQ(bs.vec_len, 8);
    EXPECT_EQ(bs.blocks.size(), 2u);
    EXPECT_EQ(bs.map_B.size(), 8u);
}

TEST(BlockStructure, RejectsIntOverflow)
{
    BlockStructure bs;
    auto init = [&bs](const std::vector<char> &types, const std::vector<int> &sizes)
    { bs.init(types.data(), sizes.data(), types.size()); };

    // n * n overflows int for n > 46340
    EXPECT_THROW(init({'s'}, {46341}), std::overflow_error);
    EXPECT_THROW(init({'u', 's'}, {10, INT_MAX}), std::overflow_error);
    // 46340 passes the size and overflow checks: the unknown type of the next block is what gets
    // rejected (building the maps of a 46340 block would need ~13 GB)
    EXPECT_THROW(init({'s', 'x'}, {46340, 1}), std::invalid_argument);
    // the limit only applies to PSD blocks
    EXPECT_NO_THROW(init({'u', 'l'}, {46341, 46341}));
    EXPECT_EQ(bs.vec_len, 2 * 46341);

    // total size of the large matrices: 2 * 40000^2 > INT_MAX although 40000^2 fits
    EXPECT_THROW(init({'s', 's'}, {40000, 40000}), std::overflow_error);
    // total size of the medium matrices: 2200 * 1000^2 > INT_MAX
    EXPECT_THROW(init(std::vector<char>(2200, 's'), std::vector<int>(2200, 1000)), std::overflow_error);
    // vec_len: 46340 * 46341 / 2 + (INT_MAX - 46340 * 46341 / 2 + 1) = INT_MAX + 1
    const int psd_len = 46340 * 46341 / 2;
    EXPECT_THROW(init({'s', 'u'}, {46340, INT_MAX - psd_len + 1}), std::overflow_error);
    EXPECT_THROW(init({'u', 'l'}, {2000000000, 2000000000}), std::overflow_error);

    // a failed init leaves the structure unchanged
    EXPECT_EQ(bs.vec_len, 2 * 46341);
}
