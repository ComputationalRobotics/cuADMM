/*

    readers.h

    Internal interface of src/io/ (plan Section 12): one reader per format, the exact simplifications, the common
    finalization, the solution writers per convention, and the expansion of a solution vector into the file's
    block structure. Everything here is implementation detail behind include/cuadmm/io.h.

*/

#ifndef CUADMM_IO_READERS_H
#define CUADMM_IO_READERS_H

#include <cstdint>
#include <string>
#include <vector>

#include "cuadmm/io.h"
#include "cuadmm/options.h"
#include "cuadmm/problem.h"
#include "cuadmm/result.h"

namespace cuadmm
{
namespace io
{
namespace detail
{

// what this build can do
bool have_hdf5();
bool have_matio();
bool have_zlib();

// readers (each fills source.format/path/report/dimacs_kinds/original_block_*, applies the simplifications and
// validates; they throw std::invalid_argument / std::runtime_error with the file and the item named)
ProblemData read_sdpa(const std::string &path, const IoOptions &options);
ProblemData read_h5(const std::string &path, const IoOptions &options);
ProblemData read_sedumi(const std::string &path, const IoOptions &options);
ProblemData read_sdpt3(const std::string &path, const IoOptions &options);
ProblemData read_mosek(const std::string &path, const IoOptions &options);

/// @brief The variables of a .mat file (top level), for the detection; throws when the file cannot be opened.
std::vector<std::string> mat_variable_names(const std::string &path);
/// @brief sedumi / sdpt3 / mosek from the variables present (plan Section 12); throws when none matches.
IoFormat detect_mat_format(const std::string &path);

/// @brief The original blocks are the cones themselves (readers whose block structure is the file's).
void set_identity_original_blocks(ProblemData &problem);

/// @brief io.simplify: 1x1 PSD blocks and PSD blocks whose A and C data touch only the diagonal become `l` blocks
/// (exact; each one reported); the svec indices are remapped. Never called for TXT input.
void apply_simplifications(ProblemData &problem, bool simplify);

/// @brief io.sense against a format whose sense is fixed by the file: a contradiction is an error.
void check_sense_option(const IoOptions &options, Sense file_sense, const std::string &path, const char *format_name);

/// @brief Common end of every reader: non-finite values rejected, structure validated (ProblemData::validate).
void finish_problem(ProblemData &problem, const std::string &path);

/// @brief The DIMACS kinds from the original block kinds ('s' for PSD-origin blocks, 'l' otherwise).
void set_dimacs_kinds_from_original(ProblemData &problem);

// solution output -------------------------------------------------------------------------------------------------

/// @brief The values of a svec-layout vector (X or S) arranged by the file's original blocks: a full column-major
/// n x n matrix for an 's' block (simplified or split cones are expanded back), a vector otherwise.
struct OriginalBlockValues
{
    char kind = 's';
    std::int64_t size = 0;
    std::vector<double> values; // n*n (column-major, symmetric) for 's', n for 'l' / 'u'
};
std::vector<OriginalBlockValues> expand_to_original_blocks(const ProblemData &problem, const std::vector<double> &v);

/// @brief A block layout of the file for solution vectors in svec form: every cone sits in one block at an offset
/// (a simplified cone covers the diagonal of its block, split cones of an SDPT3 group are consecutive).
struct BlockLayout
{
    std::vector<char> kinds;                 // of the file's blocks
    std::vector<std::int64_t> sizes;
    std::vector<std::int64_t> offsets;       // svec offset of each block (sizes.size() + 1 entries)
    std::vector<std::int64_t> cone_block;    // per cone: its block
    std::vector<std::int64_t> cone_offset;   // per cone: position of its first row/column inside the block
    std::int64_t vec_len() const { return offsets.back(); }
};
/// @brief The layout of the file's blocks: the native file's own /blocks for .h5 input, else the original blocks of
/// the reader (cones without one become blocks of their own, in order).
BlockLayout file_block_layout(const ProblemData &problem);
/// @brief A solver vector (cones) written into the layout's svec vector (exact copies; a simplified cone fills the
/// diagonal), and the reverse (entries outside the diagonal of a simplified cone are dropped).
std::vector<double> to_file_layout(const ProblemData &problem, const BlockLayout &layout, const std::vector<double> &v);
std::vector<double> from_file_layout(const ProblemData &problem, const BlockLayout &layout, const std::vector<double> &v);

/// @brief `path` is the file when it ends with one of the extensions, else a directory receiving solution.<ext>.
std::string solution_file_path(const std::string &path, const std::vector<std::string> &extensions, const std::string &default_name);

void write_solution_txt(const std::string &directory, const ProblemData &problem, const SolveResult &result);
void write_solution_h5(const std::string &path, const ProblemData &problem, const SolveResult &result);
void write_solution_sdpa(const std::string &path, const ProblemData &problem, const SolveResult &result);
void write_solution_sedumi(const std::string &path, const ProblemData &problem, const SolveResult &result);
void write_solution_sdpt3(const std::string &path, const ProblemData &problem, const SolveResult &result);
void write_solution_mosek(const std::string &path, const ProblemData &problem, const SolveResult &result);

WarmStart read_h5_warm_start(const std::string &path, const ProblemData &problem);
void write_h5_warm_start(const std::string &path, const WarmStart &warm_start);
void write_problem_h5(const std::string &path, const ProblemData &problem, int compression);
/// @brief Reads back /solution of a native file (tests, tools).
void read_h5_solution(const std::string &path, std::vector<double> &X, std::vector<double> &y, std::vector<double> &S, double *pobj, double *dobj);
void read_h5_solution_blocks(const std::string &path, std::vector<char> &kinds, std::vector<std::int64_t> &sizes);

// text helpers
std::string format_double(double v); // shortest round-trip decimal
std::string lower(std::string s);
bool ends_with(const std::string &s, const std::string &suffix);

} // namespace detail
} // namespace io
} // namespace cuadmm

#endif // CUADMM_IO_READERS_H
