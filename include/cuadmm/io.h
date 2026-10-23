/*

    io.h

    Readers and writers of problem data (plan Section 12): the native HDF5 file, SDPA sparse (.dat-s, .dat-s.gz),
    SeDuMi, SDPT3 and MOSEK prob .mat files (through matio), and the legacy TXT directory (read exactly as today's
    Problem::from_txt, never simplified). Every reader produces one ProblemData with 64-bit sizes, fills the CSC of
    At directly, and leaves a conversion report in problem.source.report. Solutions are written in the input's
    convention.

*/

#ifndef CUADMM_IO_PUBLIC_H
#define CUADMM_IO_PUBLIC_H

#include <string>
#include <vector>

#include "cuadmm/options.h"
#include "cuadmm/problem.h"
#include "cuadmm/result.h"

namespace cuadmm
{
namespace io
{

/// @brief Reads a problem; the format comes from io.format, else from the extension and, for .mat, the variables.
ProblemData read_problem(const std::string &path, const IoOptions &options);

/// @brief The legacy TXT directory (blk.txt, At.txt, b.txt, C.txt, con_num.txt): exactly today's arrays, no
/// simplification, 0-based indices, a bare size in blk.txt meaning 's'.
ProblemData read_txt(const std::string &directory);

/// @brief Warm start X.txt, y.txt, S.txt (dense, one value per line) from a directory; sizes are checked.
WarmStart read_txt_warm_start(const std::string &directory, const ProblemData &problem);

/// @brief Warm start for a problem: a directory holds X.txt, y.txt, S.txt; a .h5 file holds the group /start.
WarmStart read_warm_start(const std::string &path, const ProblemData &problem);

/// @brief Writes the returned iterate in the input's convention (plan Section 12): TXT directory (X.txt, y.txt,
/// S.txt, 17 digits, certificate.txt), .h5 (group /solution), SDPA (x = -y, X, Y in the sparse initial-point
/// layout), SeDuMi (x, y, z as full vectors), SDPT3 (cell arrays X, y, Z), MOSEK (res.sol.itr with barx, bars as
/// lower triangles, xx, slx, y). `format` forces a convention ("" : the input's). For the file formats `path` is
/// the file when it carries the format's extension, otherwise a directory that receives solution.<ext>.
void write_solution(const std::string &path, const ProblemData &problem, const SolveResult &result);
void write_solution(const std::string &path, const ProblemData &problem, const SolveResult &result, const std::string &format);

/// @brief The native HDF5 problem file (schema version 1); compression_level 0 (none) to 9 (deflate).
void write_problem(const std::string &path_h5, const ProblemData &problem);
void write_problem(const std::string &path_h5, const ProblemData &problem, int compression_level);

/// @brief The legacy TXT directory (blk.txt with explicit types, At.txt, b.txt, C.txt, con_num.txt, and
/// dimacs_blocks.txt when the source kinds are known), readable by read_txt and by today's solver.
void write_problem_txt(const std::string &directory, const ProblemData &problem);

/// @brief The formats compiled into this build: "txt", "sdpa" always; "sdpa.gz" with zlib; "h5" with HDF5;
/// "sedumi", "sdpt3", "mosek" with matio.
std::vector<std::string> available_formats();
bool format_available(const std::string &name);

} // namespace io
} // namespace cuadmm

#endif // CUADMM_IO_PUBLIC_H
