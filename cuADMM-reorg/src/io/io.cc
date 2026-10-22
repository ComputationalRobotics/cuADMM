/*

    io.cc

    Format detection and dispatch (plan Section 12): the format comes from io.format, else from the extension (a
    directory is TXT; .h5/.hdf5; .dat-s/.dat-s.gz; .mat) or the file's signature, and for .mat files from the
    variables present (prob -> MOSEK; blk with At, C, b -> SDPT3; K with A or At -> SeDuMi). Also the list of the
    formats compiled in, the warm start entry point, and the TXT solution writer kept from Phase 2.

*/

#include "cuadmm/io.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "io/readers.h"

namespace cuadmm
{
namespace io
{

namespace
{

IoFormat detect_format(const std::string &path)
{
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec))
        return IoFormat::Txt;
    const std::string p = detail::lower(path);
    if (detail::ends_with(p, ".h5") || detail::ends_with(p, ".hdf5"))
        return IoFormat::H5;
    if (detail::ends_with(p, ".dat-s") || detail::ends_with(p, ".dat-s.gz"))
        return IoFormat::Sdpa;
    if (detail::ends_with(p, ".mat"))
        return IoFormat::Auto; // sedumi / sdpt3 / mosek: decided by the variables present
    if (detail::ends_with(p, ".cbf") || detail::ends_with(p, ".cbf.gz"))
        throw std::invalid_argument(path + ": the CBF format (CBLIB, with second-order cones) comes with Phase 9");
    // no known extension: the file's signature
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (f)
    {
        unsigned char head[8] = {0};
        const std::size_t n = std::fread(head, 1, 8, f);
        std::fclose(f);
        static const unsigned char kHdf5[8] = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
        if (n == 8 && std::memcmp(head, kHdf5, 8) == 0)
            return IoFormat::H5;
        if (n >= 6 && std::memcmp(head, "MATLAB", 6) == 0)
            return IoFormat::Auto;
        if (n >= 2 && head[0] == 0x1f && head[1] == 0x8b)
            return IoFormat::Sdpa;
    }
    throw std::invalid_argument("cannot detect the format of " + path +
                                " (a directory is read as TXT; otherwise .h5, .dat-s, .dat-s.gz, .mat, or a recognizable file signature); use io.format");
}

} // namespace

ProblemData read_problem(const std::string &path, const IoOptions &options)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
        throw std::runtime_error("no such file or directory: " + path);
    IoFormat format = options.format == IoFormat::Auto ? detect_format(path) : options.format;
    if (format == IoFormat::Auto)
    {
        if (!detail::have_matio())
            throw std::invalid_argument(path + ": reading .mat files (SeDuMi, SDPT3, MOSEK) needs matio, which is not compiled into this build "
                                               "(CUADMM_WITH_MATIO); convert the file with MATLAB or Python to the native .h5 format");
        format = detail::detect_mat_format(path);
    }
    switch (format)
    {
    case IoFormat::Txt:
        if (!std::filesystem::is_directory(path, ec))
            throw std::invalid_argument(path + " is not a directory (the TXT format is a directory with blk.txt, At.txt, b.txt, C.txt, con_num.txt)");
        detail::check_sense_option(options, Sense::Minimize, path, "TXT");
        return read_txt(path);
    case IoFormat::H5:
        return detail::read_h5(path, options);
    case IoFormat::Sdpa:
        return detail::read_sdpa(path, options);
    case IoFormat::Sedumi:
        return detail::read_sedumi(path, options);
    case IoFormat::Sdpt3:
        return detail::read_sdpt3(path, options);
    case IoFormat::Mosek:
        return detail::read_mosek(path, options);
    default:
        throw std::invalid_argument("unknown input format");
    }
}

std::vector<std::string> available_formats()
{
    std::vector<std::string> f = {"txt", "sdpa"};
    if (detail::have_zlib())
        f.push_back("sdpa.gz");
    if (detail::have_hdf5())
        f.push_back("h5");
    if (detail::have_matio())
    {
        f.push_back("sedumi");
        f.push_back("sdpt3");
        f.push_back("mosek");
    }
    return f;
}

bool format_available(const std::string &name)
{
    for (const std::string &f : available_formats())
        if (f == name)
            return true;
    return false;
}

WarmStart read_warm_start(const std::string &path, const ProblemData &problem)
{
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec))
        return read_txt_warm_start(path, problem);
    const std::string p = detail::lower(path);
    if (detail::ends_with(p, ".h5") || detail::ends_with(p, ".hdf5"))
        return detail::read_h5_warm_start(path, problem);
    throw std::invalid_argument("warm start: " + path + " is neither a directory (X.txt, y.txt, S.txt) nor a .h5 file (/start)");
}

namespace detail
{
bool have_zlib()
{
#ifdef CUADMM_HAVE_ZLIB
    return true;
#else
    return false;
#endif
}
} // namespace detail

} // namespace io
} // namespace cuadmm
