// CPU tests of the Phase 7 presolve for dependent constraint rows (problem/presolve.h); no GPU needed
#include <gtest/gtest.h>

#include "presolve_test.hpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
