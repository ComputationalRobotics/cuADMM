/*

    txt.cc

    The legacy TXT directory, read exactly as today's Problem::from_txt (src/legacy/problem.cu) reads it: the same
    files, parsing, index checks and warnings, and the same arrays (the CSC of At as COO_to_CSC sorts it, b and C in
    file order). Never simplified (plan Section 12). Also the TXT warm start and the TXT solution writer
    (detail::write_solution_txt, reached through io::write_solution).

*/

#include "cuadmm/io.h"
#include "io/readers.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

#include "cuadmm_legacy/io.h"
#include "cuadmm_legacy/problem.h"

namespace cuadmm
{
namespace io
{

ProblemData read_txt(const std::string &directory)
{
    std::string prefix = directory;
    Problem legacy;
    legacy.from_txt(prefix, false);

    ProblemData p;
    p.At.rows = legacy.vec_len;
    p.At.cols = legacy.con_num;
    p.At.colptr.assign(legacy.At_csc_col_ptrs.begin(), legacy.At_csc_col_ptrs.end());
    p.At.rowind.assign(legacy.At_csc_row_ids.begin(), legacy.At_csc_row_ids.end());
    p.At.val = legacy.At_csc_vals;
    p.b.size = legacy.con_num;
    p.b.index.assign(legacy.b_indices.begin(), legacy.b_indices.end());
    p.b.val = legacy.b_vals;
    p.C.size = legacy.vec_len;
    p.C.index.assign(legacy.C_indices.begin(), legacy.C_indices.end());
    p.C.val = legacy.C_vals;
    for (const BlkVal &blk : legacy.blk_vals)
    {
        ConeSpec c;
        c.kind = cone_kind_of(std::get<0>(blk));
        c.size = std::get<1>(blk);
        p.cones.push_back(c);
    }
    p.sense = Sense::Minimize;
    p.objective_offset = 0.0;
    p.source.format = "txt";
    p.source.path = prefix;
    p.source.report.push_back("legacy TXT directory read as today's Problem::from_txt (no simplification)");

    // block kinds of the original data for the DIMACS norms, written by the SDPA converter (optional)
    std::ifstream kf(prefix + "dimacs_blocks.txt");
    char kind;
    int size;
    while (kf >> kind >> size)
        p.source.dimacs_kinds.push_back(kind);
    if (!p.source.dimacs_kinds.empty() && p.source.dimacs_kinds.size() != p.cones.size())
        throw std::invalid_argument("dimacs_blocks.txt lists " + std::to_string(p.source.dimacs_kinds.size()) + " blocks, blk.txt " +
                                    std::to_string(p.cones.size()));
    return p;
}

WarmStart read_txt_warm_start(const std::string &directory, const ProblemData &problem)
{
    const std::string prefix = directory.empty() || directory.back() == '/' ? directory : directory + "/";
    WarmStart w;
    read_dense_vector_data(prefix + "X.txt", w.X);
    read_dense_vector_data(prefix + "y.txt", w.y);
    read_dense_vector_data(prefix + "S.txt", w.S);
    if (static_cast<std::int64_t>(w.X.size()) != problem.vec_len() || static_cast<std::int64_t>(w.S.size()) != problem.vec_len() ||
        static_cast<std::int64_t>(w.y.size()) != problem.num_constraints())
        throw std::invalid_argument("warm start in " + prefix + " does not match the problem sizes: X.txt and S.txt need " +
                                    std::to_string(problem.vec_len()) + " entries, y.txt " + std::to_string(problem.num_constraints()));
    return w;
}

namespace
{
void write_vector(const std::string &filename, const std::vector<double> &v)
{
    std::ofstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("could not open " + filename + " for writing");
    f << std::setprecision(17);
    for (double x : v)
        f << x << "\n";
    if (!f.good())
        throw std::runtime_error("could not write " + filename);
}
} // namespace

namespace detail
{
void write_solution_txt(const std::string &path, const ProblemData &problem, const SolveResult &result)
{
    std::filesystem::create_directories(path);
    write_vector(path + "/X.txt", result.best.X);
    write_vector(path + "/y.txt", result.best.y);
    write_vector(path + "/S.txt", result.best.S);
    // the block layout of X.txt and S.txt (the solver's cones: for a non-TXT input with simplifications they are
    // not the file's blocks; solution_blocks.txt names them)
    {
        std::ofstream blocks(path + "/solution_blocks.txt");
        for (const ConeSpec &c : problem.cones)
            blocks << cone_code(c.kind) << " " << c.size << "\n";
        if (!blocks.good())
            throw std::runtime_error("could not write " + path + "/solution_blocks.txt");
    }
    std::ofstream f(path + "/certificate.txt");
    f << std::setprecision(17) << "# b_y = " << result.dobj << "\n"
      << "# block type size lambda_min c_b   (C - A^T y, original units; lower bound = b_y + sum_b R_b c_b)\n";
    for (std::size_t b = 0; b < problem.cones.size() && b < result.dual_block_min.size(); b++)
        f << b << " " << cone_code(problem.cones[b].kind) << " " << problem.cones[b].size << " " << result.dual_block_min[b] << " "
          << result.dual_cone_coef[b] << "\n";
    if (!f.good())
        throw std::runtime_error("could not write " + path + "/certificate.txt");
}
} // namespace detail

} // namespace io
} // namespace cuadmm
