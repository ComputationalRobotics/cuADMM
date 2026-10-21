/*

    io.h

    Readers and writers of problem data (plan Section 12). Phase 2: the legacy TXT directory (read exactly as today's
    Problem::from_txt, never simplified) and the solution writer in TXT form; Phase 2b adds the other formats, the
    detection and the conversion tool.

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

/// @brief Writes the returned iterate in the input's convention (Phase 2: TXT X.txt, y.txt, S.txt, 17 digits, and
/// certificate.txt); Phase 2b adds the other conventions.
void write_solution(const std::string &path, const ProblemData &problem, const SolveResult &result);

/// @brief The native HDF5 problem file (Phase 2b).
void write_problem(const std::string &path_h5, const ProblemData &problem);

/// @brief The formats compiled into this build ("txt", and from Phase 2b "sdpa", "h5", "sedumi", ...).
std::vector<std::string> available_formats();

} // namespace io
} // namespace cuadmm

#endif // CUADMM_IO_PUBLIC_H
