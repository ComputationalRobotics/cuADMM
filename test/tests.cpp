#include <gtest/gtest.h>

#include "meta_test.hpp"
#include "memory_test.hpp"
#include "dense_dense_test.hpp"
#include "dense_scalar_test.hpp"

int main(int argc, char **argv)
{
    // Execute all the included tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}