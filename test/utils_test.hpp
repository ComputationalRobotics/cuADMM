#include "cuadmm/utils.h"

TEST(UtilsTest, CheckGPUs)
{
    check_gpus();
}

TEST(UtilsTest, InversePermutation)
{
    std::vector<int> perm = {10, 6, 2, 4, 0, 8, 1, 3, 5, 7, 9};
    std::vector<int> inv_perm;
    get_inverse_permutation(inv_perm, perm);
    
    for (int i = 0; i < perm.size(); i++) {
        EXPECT_EQ(perm[inv_perm[i]], i);
    }
}