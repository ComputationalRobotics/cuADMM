/*

    io_read_bench.cc

    Reading rate of the problem formats (plan Phase 2b acceptance: 1e8 nonzeros from .h5 limited by the disk, not
    by parsing). Generates a synthetic problem in the native format and times read_problem on any path.

    usage: cuadmm_io_bench generate <file.h5> <m> <nnz per column> [blocks] [block side] [compress]
           cuadmm_io_bench read <problem path> [repeat] [--format F]
    The synthetic problem has `blocks` PSD blocks of side `block side` (default 100 x 1000: vec_len 5.005e7), m
    constraints with `nnz per column` distinct sorted entries each (deterministic pseudo-random), a dense b and a
    sparse C. `read` reports seconds, nnz/s, and MB/s of the file (or directory) read.

*/

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cuadmm/io.h"

using namespace cuadmm;

namespace
{

std::uint64_t splitmix(std::uint64_t &x)
{
    x += 0x9e3779b97f4a7c15ull;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

std::uintmax_t size_of(const std::string &path)
{
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec))
    {
        std::uintmax_t total = 0;
        for (const auto &e : std::filesystem::directory_iterator(path))
            if (e.is_regular_file())
                total += e.file_size();
        return total;
    }
    return std::filesystem::file_size(path, ec);
}

int generate(int argc, char *argv[])
{
    if (argc < 5)
        throw std::invalid_argument("generate needs <file.h5> <m> <nnz per column> [blocks] [block side] [compress]");
    const std::string path = argv[2];
    const std::int64_t m = std::stoll(argv[3]), per_col = std::stoll(argv[4]);
    const std::int64_t blocks = argc > 5 ? std::stoll(argv[5]) : 100, side = argc > 6 ? std::stoll(argv[6]) : 1000;
    const int compress = argc > 7 ? std::stoi(argv[7]) : 0;
    ProblemData p;
    for (std::int64_t b = 0; b < blocks; b++)
        p.cones.push_back({ConeKind::Psd, side, 0.0});
    const std::int64_t vec_len = blocks * side * (side + 1) / 2;
    if (per_col > vec_len)
        throw std::invalid_argument("nnz per column exceeds vec_len");
    p.At.rows = vec_len;
    p.At.cols = m;
    p.At.colptr.resize(static_cast<std::size_t>(m) + 1);
    p.At.rowind.resize(static_cast<std::size_t>(m * per_col));
    p.At.val.resize(static_cast<std::size_t>(m * per_col));
    const std::int64_t stride = vec_len / per_col;
    std::uint64_t state = 12345;
    for (std::int64_t j = 0; j < m; j++)
    {
        p.At.colptr[j] = j * per_col;
        for (std::int64_t k = 0; k < per_col; k++)
        {
            const std::uint64_t r = splitmix(state);
            p.At.rowind[j * per_col + k] = k * stride + static_cast<std::int64_t>(r % static_cast<std::uint64_t>(stride));
            p.At.val[j * per_col + k] = static_cast<double>(static_cast<std::int64_t>(r >> 11) % 2000001 - 1000000) / 1000.0 + 0.001;
        }
    }
    p.At.colptr[m] = m * per_col;
    p.b.size = m;
    for (std::int64_t j = 0; j < m; j++)
    {
        p.b.index.push_back(j);
        p.b.val.push_back(static_cast<double>(j % 97) + 1.0);
    }
    p.C.size = vec_len;
    for (std::int64_t i = 0; i < vec_len; i += 1000)
    {
        p.C.index.push_back(i);
        p.C.val.push_back(1.0);
    }
    p.source.format = "synthetic";
    p.source.report.push_back("synthetic problem of cuadmm_io_bench");
    const auto t0 = std::chrono::steady_clock::now();
    io::write_problem(path, p, compress);
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "wrote " << path << ": vec_len " << vec_len << ", m " << m << ", nnz " << p.At.nnz() << ", " << size_of(path) / 1048576.0 << " MB in " << t
              << " s (" << size_of(path) / 1048576.0 / t << " MB/s)" << std::endl;
    return 0;
}

int read(int argc, char *argv[])
{
    if (argc < 3)
        throw std::invalid_argument("read needs <problem path> [repeat] [--format F]");
    const std::string path = argv[2];
    int repeat = 1;
    IoOptions io;
    for (int i = 3; i < argc; i++)
    {
        const std::string a = argv[i];
        if (a == "--format" && i + 1 < argc)
            io.format = parse_io_format(argv[++i]);
        else
            repeat = std::stoi(a);
    }
    const double mb = size_of(path) / 1048576.0;
    for (int r = 0; r < repeat; r++)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const ProblemData p = io::read_problem(path, io);
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "read " << p.source.format << " " << path << ": nnz(At) " << p.At.nnz() << ", vec_len " << p.vec_len() << ", m " << p.num_constraints()
                  << ": " << t << " s, " << mb / t << " MB/s (" << mb << " MB), " << p.At.nnz() / t / 1e6 << " Mnnz/s" << std::endl;
    }
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        if (argc >= 2 && std::string(argv[1]) == "generate")
            return generate(argc, argv);
        if (argc >= 2 && std::string(argv[1]) == "read")
            return read(argc, argv);
        std::cerr << "usage: cuadmm_io_bench generate <file.h5> <m> <nnz per column> [blocks] [block side] [compress]\n"
                     "       cuadmm_io_bench read <problem path> [repeat] [--format F]\n";
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
