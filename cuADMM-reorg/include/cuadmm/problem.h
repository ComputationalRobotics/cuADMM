/*

    problem.h

    ProblemData: the host-side problem in original units with 64-bit sizes (plan Section 6.3 and Section 12), produced
    by every reader of src/io/ and consumed by Solver::setup(), which checks that the sizes fit the solver's current
    32-bit indexing.

        min <C, X>  s.t.  A X = b,  X in K          (reported in `sense`, with `objective_offset` added)

    At is the transpose of A in svec form: vec_len x m in CSC, one column per constraint, row indices sorted within
    each column. PSD blocks are stored as svec (upper triangle, column by column, off-diagonal entries times sqrt 2).

*/

#ifndef CUADMM_PROBLEM_DATA_H
#define CUADMM_PROBLEM_DATA_H

#include <cstdint>
#include <string>
#include <vector>

#include "cuadmm/options.h" // Sense

namespace cuadmm
{

enum class ConeKind { Free, Nonnegative, Psd, SecondOrder, RotatedSecondOrder, Exponential, Power };

struct ConeSpec
{
    ConeKind kind = ConeKind::Psd;
    std::int64_t size = 0;  // side of a PSD block, length of a free or nonnegative block
    double parameter = 0.0; // unused by the cones of Milestone 1
};

/// @brief Legacy block code of a cone kind ('s', 'l', 'u'); throws for the cones the engine does not support.
char cone_code(ConeKind kind);
ConeKind cone_kind_of(char code);
const char *name_of(ConeKind kind);

/// @brief Compressed sparse columns with 64-bit sizes.
struct CscMatrix
{
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<std::int64_t> colptr; // cols + 1
    std::vector<std::int64_t> rowind; // nnz, sorted within each column
    std::vector<double> val;          // nnz
    std::int64_t nnz() const { return static_cast<std::int64_t>(val.size()); }
};

struct SparseVector
{
    std::int64_t size = 0;
    std::vector<std::int64_t> index; // as the reader produced them: the TXT reader keeps the file order (as today), the other readers sort and merge
    std::vector<double> val;
    std::int64_t nnz() const { return static_cast<std::int64_t>(val.size()); }
};

namespace io
{
/// @brief Where a problem came from and what the reader did to it (plan Section 12: conversion report).
struct SourceInfo
{
    std::string format;                      // "txt", "sdpa", "sedumi", "sdpt3", "mosek", "h5"
    std::string path;
    std::vector<std::string> report;         // conversion report, one line per item (mapped, mirrored, merged, simplified)
    std::vector<char> dimacs_kinds;          // block kinds of the original data for the DIMACS norms ('s' / 'l'); empty: from the cones
    // mapping back to the file's block structure (filled by the readers that simplify or reorder blocks)
    std::vector<int> original_block_of;      // for each cone, the index of the original block (-1: new block)
    std::vector<std::int64_t> original_block_sizes;
    std::vector<char> original_block_kinds;
};
} // namespace io

struct ProblemData
{
    CscMatrix At;                  // vec_len x m
    SparseVector b;                // m
    SparseVector C;                // vec_len
    std::vector<ConeSpec> cones;   // in input order
    Sense sense = Sense::Minimize; // results are reported in this sense
    double objective_offset = 0.0; // e.g. MOSEK cfix
    io::SourceInfo source;

    std::int64_t vec_len() const { return At.rows; }
    std::int64_t num_constraints() const { return At.cols; }

    // Checks the structure (sizes, sorted indices, cone sizes against vec_len); throws std::invalid_argument.
    void validate() const;
};

/// @brief Warm start in original units (plan Section 6.3).
struct WarmStart
{
    std::vector<double> X, y, S;
    double sigma = -1.0; // < 0: use penalty.sigma0
};

} // namespace cuadmm

#endif // CUADMM_PROBLEM_DATA_H
