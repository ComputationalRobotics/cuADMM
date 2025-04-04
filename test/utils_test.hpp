#include "cuadmm/utils.h"

TEST(Utils, CheckGPUs)
{
    check_gpus();
}

TEST(Utils, InversePermutation)
{
    std::vector<int> perm = {10, 6, 2, 4, 0, 8, 1, 3, 5, 7, 9};
    std::vector<int> inv_perm;
    get_inverse_permutation(inv_perm, perm);
    
    for (int i = 0; i < perm.size(); i++) {
        EXPECT_EQ(perm[inv_perm[i]], i);
    }
}

TEST(Utils, GetMaps)
{
    const int LARGE = 5;
    const int SMALL = 4;
    const int vec_len = (LARGE * (LARGE + 1) / 2) + 
                        (SMALL * (SMALL + 1) / 2);

    HostDenseVector<int> blk(2);
    blk.vals[0] = LARGE;
    blk.vals[1] = SMALL;

    std::vector<int> map_B_tmp;
    std::vector<int> map_M1_tmp;
    std::vector<int> map_M2_tmp;

    get_maps(blk, LARGE, SMALL, vec_len, map_B_tmp, map_M1_tmp, map_M2_tmp);

    for (int i = 0; i < vec_len; i++) {
        EXPECT_EQ(map_B_tmp[i], i < LARGE * (LARGE + 1) / 2 ? 0 : 1);
    }

    EXPECT_EQ(map_M1_tmp, std::vector<int>({
        0,
        5,   6,
        10, 11, 12,
        15, 16, 17, 18,
        20, 21, 22, 23, 24,
        0,
        4,  5,
        8,  9,  10,
        12, 13, 14, 15
    }));

    EXPECT_EQ(map_M2_tmp, std::vector<int>({
        0,
        1, 6,
        2, 7, 12,
        3, 8, 13, 18,
        4, 9, 14, 19, 24,
        0,
        1, 5,
        2, 6, 10,
        3, 7, 11, 15
    }));
}

TEST(Utils, AnalyzeBlk)
{
    int LARGE;
    int SMALL;
    int mom_mat_num;
    int loc_mat_num;

    HostDenseVector<int> blk(5);
    std::vector<int> vals = {5, 4, 4, 5, 5};
    std::copy(vals.begin(), vals.end(), blk.vals);

    analyze_blk(blk, &LARGE, &SMALL, &mom_mat_num, &loc_mat_num);

    EXPECT_EQ(LARGE, 5);
    EXPECT_EQ(SMALL, 4);
    EXPECT_EQ(mom_mat_num, 3);
    EXPECT_EQ(loc_mat_num, 2);
}

TEST(Utils, GetEigRankMask)
{
    int batch_size = 2;
    int mat_size = 4;
    int eig_rank = 2;

    std::vector<int> eig_rank_mask;
    get_eig_rank_mask(eig_rank_mask, batch_size, mat_size, eig_rank);

    EXPECT_EQ(eig_rank_mask.size(), batch_size * mat_size);
    EXPECT_EQ(eig_rank_mask, std::vector<int>({
        0, 0, 1, 1,
        0, 0, 1, 1
    }));
}