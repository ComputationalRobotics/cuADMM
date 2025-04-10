#include <gtest/gtest.h>

#include "cuadmm/utils.h"
#include "cuadmm/check.h"

#include "meta_test.hpp"
#include "memory_test.hpp"
#include "utils_test.hpp"
#include "dense_scalar_test.hpp"
#include "dense_dense_test.hpp"
#include "sparse_scalar_test.hpp"
#include "sparse_dense_test.hpp"
#include "io_test.hpp"
#include "kernels_test.hpp"
#include "eig_cpu_test.hpp"
#include "cholesky_cpu_test.hpp"
#include "cublas_test.hpp"
#include "cusolver_test.hpp"
#include "cusparse_test.hpp"
// #include "solver_test.hpp"

int main(int argc, char **argv)
{
    // Execute all the included tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}