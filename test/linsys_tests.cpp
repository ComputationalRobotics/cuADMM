// GPU tests of the Phase 7 linear-system backends (NormalSystem, DenseCusolverSolver, CudssSolver when compiled in),
// a runner of their own so that test/tests.cpp stays as it is (branch reorg-linsys)
#include <gtest/gtest.h>

#include "linsys_test.hpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
