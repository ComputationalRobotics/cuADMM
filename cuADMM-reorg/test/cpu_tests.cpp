// CPU-only tests of cuADMM-reorg (no GPU needed: options, penalty policies on scripted residuals, readers and writers)
#include <gtest/gtest.h>

#include "options_test.hpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
