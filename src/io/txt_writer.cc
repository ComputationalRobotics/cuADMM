/*

    txt_writer.cc

    The legacy TXT directory as output (cuadmm-convert, and the TXT solution convention): blk.txt with explicit
    types, con_num.txt, At.txt (row = svec index, col = constraint, 0-based, in CSC order), b.txt and C.txt
    ("index 0 value"), dimacs_blocks.txt when the source kinds are known. Numbers are the shortest decimals that
    round-trip exactly, so read_txt gives back the same arrays.

*/

#include "io/readers.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace cuadmm
{
namespace io
{

namespace
{
std::ofstream open_out(const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("could not open " + filename + " for writing");
    return f;
}
} // namespace

void write_problem_txt(const std::string &directory, const ProblemData &problem)
{
    problem.validate();
    std::filesystem::create_directories(directory);
    {
        std::ofstream f = open_out(directory + "/blk.txt");
        for (const ConeSpec &c : problem.cones)
            f << cone_code(c.kind) << " " << c.size << "\n";
    }
    {
        std::ofstream f = open_out(directory + "/con_num.txt");
        f << problem.num_constraints() << "\n";
    }
    {
        std::ofstream f = open_out(directory + "/At.txt");
        std::string buf;
        for (std::int64_t j = 0; j < problem.At.cols; j++)
            for (std::int64_t k = problem.At.colptr[j]; k < problem.At.colptr[j + 1]; k++)
            {
                buf += std::to_string(problem.At.rowind[k]);
                buf += ' ';
                buf += std::to_string(j);
                buf += ' ';
                buf += detail::format_double(problem.At.val[k]);
                buf += '\n';
                if (buf.size() > (1u << 20))
                {
                    f << buf;
                    buf.clear();
                }
            }
        f << buf;
        if (!f.good())
            throw std::runtime_error("could not write " + directory + "/At.txt");
    }
    auto write_vector = [&](const std::string &name, const SparseVector &v)
    {
        std::ofstream f = open_out(directory + "/" + name);
        for (std::size_t k = 0; k < v.val.size(); k++)
            f << v.index[k] << " 0 " << detail::format_double(v.val[k]) << "\n";
        if (!f.good())
            throw std::runtime_error("could not write " + directory + "/" + name);
    };
    write_vector("b.txt", problem.b);
    write_vector("C.txt", problem.C);
    if (!problem.source.dimacs_kinds.empty())
    {
        std::ofstream f = open_out(directory + "/dimacs_blocks.txt");
        for (std::size_t b = 0; b < problem.cones.size(); b++)
            f << problem.source.dimacs_kinds[b] << " " << problem.cones[b].size << "\n";
    }
}

} // namespace io
} // namespace cuadmm
