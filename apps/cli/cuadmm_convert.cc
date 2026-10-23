/*

    cuadmm_convert.cc

    cuadmm-convert: any supported input format to the native HDF5 file (or a TXT directory), once, for problems
    too large to parse at every run (plan Section 12). Prints the conversion report of the reader.

    usage: cuadmm-convert <input> <output.h5 | output directory> [options]
      --format auto|h5|sdpa|sedumi|sdpt3|mosek|txt   input format (default: detected)
      --sense min|max                                objective sense (MOSEK files without a sense variable)
      --simplify 0|1                                 1x1 and diagonal-only PSD blocks become l blocks (default 1; never for TXT)
      --to h5|txt                                    output format (default: h5 for a .h5/.hdf5 output name, txt otherwise)
      --compress N                                   deflate level 0-9 of the HDF5 datasets (default 0)
      --quiet                                        no report on stdout

*/

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cuadmm/io.h"
#include "cuadmm/options.h"

using namespace cuadmm;

namespace
{
void usage(const char *exe)
{
    std::cerr << "usage: " << exe << " <input> <output.h5 | output directory> [--format F] [--sense min|max] [--simplify 0|1] [--to h5|txt] [--compress N] [--quiet]\n"
              << "  formats compiled in:";
    for (const std::string &f : io::available_formats())
        std::cerr << " " << f;
    std::cerr << "\n";
}

bool ends_with(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
} // namespace

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        usage(argv[0]);
        return 1;
    }
    const std::string input = argv[1], output = argv[2];
    IoOptions io;
    std::string to;
    int compress = 0;
    bool quiet = false;
    try
    {
        for (int i = 3; i < argc; i++)
        {
            const std::string key = argv[i];
            if (key == "--quiet")
            {
                quiet = true;
                continue;
            }
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + key);
            const std::string value = argv[++i];
            if (key == "--format")
                io.format = parse_io_format(value);
            else if (key == "--sense")
                io.sense = parse_sense(value);
            else if (key == "--simplify")
                io.simplify = value == "1" || value == "true";
            else if (key == "--to")
                to = value;
            else if (key == "--compress")
                compress = std::stoi(value);
            else
                throw std::invalid_argument("unknown option " + key);
        }
        if (to.empty())
            to = ends_with(output, ".h5") || ends_with(output, ".hdf5") ? "h5" : "txt";
        if (to != "h5" && to != "txt")
            throw std::invalid_argument("--to must be h5 or txt");
        const auto t0 = std::chrono::steady_clock::now();
        const ProblemData problem = io::read_problem(input, io);
        const double t_read = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!quiet)
        {
            std::cout << "read " << problem.source.format << " problem " << input << " in " << t_read << " s: vec_len " << problem.vec_len() << ", m "
                      << problem.num_constraints() << ", " << problem.cones.size() << " blocks, nnz(At) " << problem.At.nnz() << ", nnz(b) " << problem.b.nnz()
                      << ", nnz(C) " << problem.C.nnz() << ", sense " << name_of(problem.sense) << ", offset " << problem.objective_offset << "\n";
            for (const std::string &line : problem.source.report)
                std::cout << "  [io] " << line << "\n";
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (to == "h5")
            io::write_problem(output, problem, compress);
        else
            io::write_problem_txt(output, problem);
        const double t_write = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        if (!quiet)
            std::cout << "wrote " << to << " " << output << " in " << t_write << " s" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
