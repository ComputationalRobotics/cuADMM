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