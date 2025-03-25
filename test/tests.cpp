#include <gtest/gtest.h>

#include "meta_test.hpp"
#include "memory_test.hpp"

// Execute all the included tests
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}