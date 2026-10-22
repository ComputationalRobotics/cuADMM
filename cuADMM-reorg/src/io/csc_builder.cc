/*

    csc_builder.cc

*/

#include "io/csc_builder.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cuadmm
{
namespace io
{
namespace detail
{

CscBuilder::CscBuilder(std::int64_t rows, std::int64_t cols) : rows_(rows), cols_(cols)
{
    if (rows < 0 || cols < 0)
        throw std::invalid_argument("CscBuilder: negative sizes");
}

void CscBuilder::reserve(std::size_t entries)
{
    col_.reserve(entries);
    row_.reserve(entries);
    val_.reserve(entries);
}

void CscBuilder::add(std::int64_t col, std::int64_t row, double val)
{
    if (col < 0 || col >= cols_)
        throw std::invalid_argument("column index " + std::to_string(col) + " outside [0, " + std::to_string(cols_) + ")");
    if (row < 0 || row >= rows_)
        throw std::invalid_argument("row index " + std::to_string(row) + " outside [0, " + std::to_string(rows_) + ")");
    col_.push_back(col);
    row_.push_back(row);
    val_.push_back(val);
}

CscMatrix CscBuilder::finish(DuplicatePolicy policy, bool drop_zeros, BuildStats &stats)
{
    const std::int64_t n = static_cast<std::int64_t>(val_.size());
    stats.entries = n;
    CscMatrix M;
    M.rows = rows_;
    M.cols = cols_;
    M.colptr.assign(static_cast<std::size_t>(cols_) + 1, 0);
    for (std::int64_t i = 0; i < n; i++)
        M.colptr[col_[i] + 1]++;
    for (std::int64_t c = 0; c < cols_; c++)
        M.colptr[c + 1] += M.colptr[c];

    // 1. in-place counting sort by column (bucket permutation: every swap places one entry for good)
    stats.sorted_on_arrival = std::is_sorted(col_.begin(), col_.end());
    if (!stats.sorted_on_arrival)
    {
        std::vector<std::int64_t> next(M.colptr.begin(), M.colptr.end() - 1);
        for (std::int64_t c = 0; c < cols_; c++)
        {
            const std::int64_t end = M.colptr[c + 1];
            while (next[c] < end)
            {
                const std::int64_t i = next[c];
                const std::int64_t cb = col_[i];
                if (cb == c)
                    next[c]++;
                else
                {
                    const std::int64_t d = next[cb]++;
                    std::swap(col_[i], col_[d]);
                    std::swap(row_[i], row_[d]);
                    std::swap(val_[i], val_[d]);
                }
            }
        }
    }
    std::vector<std::int64_t>().swap(col_); // released: the column of an entry is now its position

    // 2. rows sorted within each column (a temporary of the column length only when a column is unsorted)
    std::vector<std::pair<std::int64_t, double>> tmp;
    for (std::int64_t c = 0; c < cols_; c++)
    {
        const std::int64_t lo = M.colptr[c], hi = M.colptr[c + 1];
        if (hi - lo < 2 || std::is_sorted(row_.begin() + lo, row_.begin() + hi))
            continue;
        tmp.resize(static_cast<std::size_t>(hi - lo));
        for (std::int64_t i = lo; i < hi; i++)
            tmp[i - lo] = {row_[i], val_[i]};
        std::stable_sort(tmp.begin(), tmp.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (std::int64_t i = lo; i < hi; i++)
        {
            row_[i] = tmp[i - lo].first;
            val_[i] = tmp[i - lo].second;
        }
    }

    // 3. duplicates and zeros, compacting in place
    stats.duplicates = 0;
    stats.zeros_dropped = 0;
    stats.first_duplicate_col = stats.first_duplicate_row = -1;
    std::int64_t w = 0;
    std::vector<std::int64_t> new_colptr(M.colptr.size(), 0);
    for (std::int64_t c = 0; c < cols_; c++)
    {
        const std::int64_t lo = M.colptr[c], hi = M.colptr[c + 1];
        std::int64_t i = lo;
        while (i < hi)
        {
            const std::int64_t r = row_[i];
            double v = val_[i];
            std::int64_t j = i + 1;
            while (j < hi && row_[j] == r)
            {
                if (stats.duplicates == 0)
                {
                    stats.first_duplicate_col = c;
                    stats.first_duplicate_row = r;
                }
                stats.duplicates++;
                v += val_[j];
                j++;
            }
            if (policy == DuplicatePolicy::Reject && j > i + 1)
            {
                // keep the entries as they are: the caller reports the item and throws
                for (std::int64_t k = i; k < j; k++)
                {
                    row_[w] = row_[k];
                    val_[w] = val_[k];
                    w++;
                }
            }
            else if (drop_zeros && v == 0.0)
                stats.zeros_dropped++;
            else
            {
                row_[w] = r;
                val_[w] = v;
                w++;
            }
            i = j;
        }
        new_colptr[c + 1] = w;
    }
    row_.resize(static_cast<std::size_t>(w));
    val_.resize(static_cast<std::size_t>(w));
    row_.shrink_to_fit();
    val_.shrink_to_fit();
    M.colptr = std::move(new_colptr);
    M.rowind = std::move(row_);
    M.val = std::move(val_);
    row_.clear();
    val_.clear();
    return M;
}

SparseVector SparseVectorBuilder::finish(DuplicatePolicy policy, bool drop_zeros, BuildStats &stats)
{
    CscMatrix M = builder_.finish(policy, drop_zeros, stats);
    SparseVector v;
    v.size = M.rows;
    v.index = std::move(M.rowind);
    v.val = std::move(M.val);
    return v;
}

} // namespace detail
} // namespace io
} // namespace cuadmm
