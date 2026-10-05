#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuadmm/io.h"
#include "cuadmm/memory.h"

// Regression tests for COO_to_CSC (CPU only, no data files).
// Triplets are passed as (col_ids, row_ids, vals); after the call they are
// sorted by (col, row) in place and col_ptrs holds the CSC column pointers.

// Column 0 is empty: the pointers of every later column used to be shifted
// by one column (col_ptrs = {0, 2, 2, 3, 3}).
TEST(IORegression, EmptyLeadingColumn)
{
    std::vector<int> col_ptrs;
    std::vector<int> cols = {1, 1, 3};
    std::vector<int> rows = {2, 0, 1};
    std::vector<double> vals = {10.0, 20.0, 30.0};

    COO_to_CSC(col_ptrs, cols, rows, vals, vals.size(), 4);

    // column 0: empty, column 1: entries [0, 2), column 2: empty, column 3: entry [2, 3)
    EXPECT_EQ(col_ptrs, std::vector<int>({0, 0, 2, 2, 3}));
    EXPECT_EQ(cols, std::vector<int>({1, 1, 3}));
    EXPECT_EQ(rows, std::vector<int>({0, 2, 1}));
    EXPECT_EQ(vals, std::vector<double>({20.0, 10.0, 30.0}));
}

// Only the last column is non-empty: all leading pointers must stay 0.
TEST(IORegression, OnlyLastColumnNonEmpty)
{
    std::vector<int> col_ptrs;
    std::vector<int> cols = {3, 3};
    std::vector<int> rows = {5, 1};
    std::vector<double> vals = {-1.0, 2.5};

    COO_to_CSC(col_ptrs, cols, rows, vals, vals.size(), 4);

    EXPECT_EQ(col_ptrs, std::vector<int>({0, 0, 0, 0, 2}));
    EXPECT_EQ(cols, std::vector<int>({3, 3}));
    EXPECT_EQ(rows, std::vector<int>({1, 5}));
    EXPECT_EQ(vals, std::vector<double>({2.5, -1.0}));
}

// Columns 2..4 are empty: their pointers must all equal nnz.
TEST(IORegression, TrailingEmptyColumns)
{
    std::vector<int> col_ptrs;
    std::vector<int> cols = {0, 1, 0};
    std::vector<int> rows = {1, 0, 0};
    std::vector<double> vals = {1.0, 2.0, 3.0};

    COO_to_CSC(col_ptrs, cols, rows, vals, vals.size(), 5);

    EXPECT_EQ(col_ptrs, std::vector<int>({0, 2, 3, 3, 3, 3}));
    EXPECT_EQ(cols, std::vector<int>({0, 0, 1}));
    EXPECT_EQ(rows, std::vector<int>({0, 1, 0}));
    EXPECT_EQ(vals, std::vector<double>({3.0, 1.0, 2.0}));
}

// Dense 2x2 [1 3; 2 4] given in scrambled order: the triplets are sorted
// column-major and the values are permuted along with the indices.
TEST(IORegression, Dense2x2)
{
    std::vector<int> col_ptrs;
    std::vector<int> cols = {1, 0, 1, 0};
    std::vector<int> rows = {1, 1, 0, 0};
    std::vector<double> vals = {4.0, 2.0, 3.0, 1.0};

    COO_to_CSC(col_ptrs, cols, rows, vals, vals.size(), 2);

    EXPECT_EQ(col_ptrs, std::vector<int>({0, 2, 4}));
    EXPECT_EQ(cols, std::vector<int>({0, 0, 1, 1}));
    EXPECT_EQ(rows, std::vector<int>({0, 1, 0, 1}));
    EXPECT_EQ(vals, std::vector<double>({1.0, 2.0, 3.0, 4.0}));
}

// A correctly sized but stale col_ptrs buffer must be fully overwritten,
// including col_ptrs[0].
TEST(IORegression, StaleColPtrsOverwritten)
{
    std::vector<int> col_ptrs(3, -7);
    std::vector<int> cols = {1, 0};
    std::vector<int> rows = {0, 0};
    std::vector<double> vals = {2.0, 1.0};

    COO_to_CSC(col_ptrs, cols, rows, vals, vals.size(), 2);

    EXPECT_EQ(col_ptrs, std::vector<int>({0, 1, 2}));
    EXPECT_EQ(rows, std::vector<int>({0, 0}));
    EXPECT_EQ(vals, std::vector<double>({1.0, 2.0}));
}

// nnz == 0 (with and without columns) gives all-zero column pointers.
TEST(IORegression, EmptyMatrix)
{
    std::vector<int> col_ptrs;
    std::vector<int> cols, rows;
    std::vector<double> vals;

    COO_to_CSC(col_ptrs, cols, rows, vals, 0, 3);
    EXPECT_EQ(col_ptrs, std::vector<int>({0, 0, 0, 0}));
    EXPECT_TRUE(cols.empty());
    EXPECT_TRUE(rows.empty());
    EXPECT_TRUE(vals.empty());

    COO_to_CSC(col_ptrs, cols, rows, vals, 0, 0);
    EXPECT_EQ(col_ptrs, std::vector<int>({0}));
}

// Checks that COO_to_CSC throws std::invalid_argument whose message contains
// `needle`, and leaves the triplets and col_ptrs untouched.
static void expect_COO_to_CSC_rejects(
    std::vector<int> cols, std::vector<int> rows, std::vector<double> vals,
    const int nnz, const int col_num, const std::string &needle)
{
    const std::vector<int> cols0 = cols, rows0 = rows;
    const std::vector<double> vals0 = vals;
    std::vector<int> col_ptrs = {-7};
    try
    {
        COO_to_CSC(col_ptrs, cols, rows, vals, nnz, col_num);
        ADD_FAILURE() << "COO_to_CSC did not throw (expected message containing '" << needle << "')";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << "message: " << e.what();
    }
    EXPECT_EQ(cols, cols0);
    EXPECT_EQ(rows, rows0);
    EXPECT_EQ(vals, vals0);
    EXPECT_EQ(col_ptrs, std::vector<int>({-7}));
}

// Column indices outside [0, col_num) are rejected, naming the offending index.
TEST(IORegression, OutOfRangeColumnRejected)
{
    expect_COO_to_CSC_rejects({2, 8}, {0, 0}, {1.0, 2.0}, 2, 8, "column index 8");   // col == col_num
    expect_COO_to_CSC_rejects({2, 9}, {0, 0}, {1.0, 2.0}, 2, 8, "column index 9");   // col > col_num
    expect_COO_to_CSC_rejects({-1, 0}, {0, 0}, {1.0, 2.0}, 2, 2, "column index -1"); // col < 0
    expect_COO_to_CSC_rejects({0}, {0}, {1.0}, 1, 0, "column index 0");              // no columns at all
}

// Negative sizes and nnz larger than the triplet vectors are rejected.
TEST(IORegression, InvalidSizesRejected)
{
    expect_COO_to_CSC_rejects({0}, {0}, {1.0}, 1, -1, "col_num (-1)");
    expect_COO_to_CSC_rejects({0}, {0}, {1.0}, -1, 1, "nnz (-1)");
    expect_COO_to_CSC_rejects({0}, {0}, {1.0}, 2, 1, "nnz = 2");
}

// DeviceDenseVector::to_txt (GPU): the vector used to be copied into a stack array (overflow at about 1M doubles)
// and written with '%.17f' (entries below 5e-18 written as 0, few significant digits for small entries).
TEST(IORegression, DeviceVectorToTxtRoundTrip)
{
    const int n = 1 << 21;
    std::vector<double> v(n);
    for (int i = 0; i < n; i++)
        v[i] = std::sin(0.37 * i) * std::pow(10.0, (i % 41) - 20);
    v[0] = 1e-20;
    v[1] = -3.5e300;
    v[2] = 0.1;
    v[3] = 0.0;
    DeviceDenseVector<double> d_v(GPU0, n);
    ASSERT_EQ(cudaMemcpy(d_v.vals, v.data(), sizeof(double) * n, cudaMemcpyHostToDevice), cudaSuccess);
    const std::string file = "io_regression_to_txt.txt";
    d_v.to_txt(file);
    std::vector<double> r;
    read_dense_vector_data(file, r);
    ASSERT_EQ(r.size(), v.size());
    int mismatches = 0;
    for (int i = 0; i < n; i++)
        mismatches += (r[i] != v[i]);
    EXPECT_EQ(mismatches, 0);
    std::remove(file.c_str());
}
