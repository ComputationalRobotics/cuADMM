// CPU-only tests of cuADMM-reorg (no GPU needed: options, penalty policies on scripted residuals, readers and writers)
#include <gtest/gtest.h>

#include "options_test.hpp"
#include "penalty_policy_test.hpp"
#include "large_block_policy_test.hpp"
#include "workspace_arena_test.hpp"
#include "io_builder_test.hpp"
#include "io_sdpa_test.hpp"
#include "io_simplify_test.hpp"
#include "io_h5_test.hpp"
#include "io_mat_test.hpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
