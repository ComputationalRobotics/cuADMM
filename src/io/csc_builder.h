/*

    csc_builder.h

    Builds the CSC matrix At of a ProblemData directly from entries in any order, without a COO copy (plan Section
    12, B27): the entries are appended to the final rowind/val arrays with a parallel column array, then sorted in
    place (counting sort by column with the in-place bucket permutation, then by row within each column); duplicates
    are merged by summation or rejected, and entries that are exactly zero are dropped. The column array is the only
    temporary (8 bytes per entry) and is released at the end.

*/

#ifndef CUADMM_IO_CSC_BUILDER_H
#define CUADMM_IO_CSC_BUILDER_H

#include <cstdint>
#include <string>
#include <vector>

#include "cuadmm/problem.h"

namespace cuadmm
{
namespace io
{
namespace detail
{

enum class DuplicatePolicy { Sum, Reject };

struct BuildStats
{
    std::int64_t entries = 0;           // entries added
    std::int64_t duplicates = 0;        // entries that shared (column, row) with an earlier one
    std::int64_t zeros_dropped = 0;     // entries equal to zero after merging
    std::int64_t first_duplicate_col = -1, first_duplicate_row = -1;
    bool sorted_on_arrival = true;      // the columns arrived in non-decreasing order
};

class CscBuilder
{
  public:
    CscBuilder(std::int64_t rows, std::int64_t cols);

    void reserve(std::size_t entries);
    /// @brief Appends one entry; throws std::invalid_argument when the indices are out of range.
    void add(std::int64_t col, std::int64_t row, double val);
    std::size_t size() const { return val_.size(); }

    /// @brief Sorts into CSC and merges (Sum) or detects (Reject: the first duplicate is reported in `stats`, the
    /// duplicates are kept so that the caller can name the item) duplicates, drops exact zeros when asked.
    CscMatrix finish(DuplicatePolicy policy, bool drop_zeros, BuildStats &stats);

  private:
    std::int64_t rows_, cols_;
    std::vector<std::int64_t> col_, row_;
    std::vector<double> val_;
};

/// @brief One-column convenience: a sparse vector from entries in any order (same rules as the matrix).
class SparseVectorBuilder
{
  public:
    explicit SparseVectorBuilder(std::int64_t size) : builder_(size, 1) {}
    void reserve(std::size_t entries) { builder_.reserve(entries); }
    void add(std::int64_t index, double val) { builder_.add(0, index, val); }
    std::size_t size() const { return builder_.size(); }
    SparseVector finish(DuplicatePolicy policy, bool drop_zeros, BuildStats &stats);

  private:
    CscBuilder builder_;
};

} // namespace detail
} // namespace io
} // namespace cuadmm

#endif // CUADMM_IO_CSC_BUILDER_H
