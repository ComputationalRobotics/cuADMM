/*

    io.cc

    Format detection and dispatch (plan Section 12). Phase 2: the legacy TXT directory; Phase 2b adds the other
    readers behind the same entry point.

*/

#include "cuadmm/io.h"

#include <filesystem>
#include <stdexcept>

namespace cuadmm
{
namespace io
{

namespace
{
IoFormat detect_format(const std::string &path)
{
    if (std::filesystem::is_directory(path))
        return IoFormat::Txt;
    std::string p = path;
    for (char &c : p)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto ends_with = [&p](const std::string &suffix)
    { return p.size() >= suffix.size() && p.compare(p.size() - suffix.size(), suffix.size(), suffix) == 0; };
    if (ends_with(".h5") || ends_with(".hdf5"))
        return IoFormat::H5;
    if (ends_with(".dat-s") || ends_with(".dat-s.gz"))
        return IoFormat::Sdpa;
    if (ends_with(".mat"))
        return IoFormat::Auto; // sedumi / sdpt3 / mosek: decided by the variables present (Phase 2b)
    throw std::invalid_argument("cannot detect the format of " + path + " (a directory is read as TXT; otherwise .h5, .dat-s, .dat-s.gz, .mat); use io.format");
}
} // namespace

ProblemData read_problem(const std::string &path, const IoOptions &options)
{
    IoFormat format = options.format == IoFormat::Auto ? detect_format(path) : options.format;
    switch (format)
    {
    case IoFormat::Txt:
        return read_txt(path);
    case IoFormat::Auto:
        throw std::invalid_argument("reading .mat files (SeDuMi, SDPT3, MOSEK) comes with Phase 2b");
    default:
        throw std::invalid_argument(std::string("the ") + name_of(format) + " reader comes with Phase 2b");
    }
}

void write_problem(const std::string &path_h5, const ProblemData &)
{
    throw std::invalid_argument("writing " + path_h5 + ": the HDF5 writer comes with Phase 2b");
}

std::vector<std::string> available_formats()
{
    return {"txt"};
}

} // namespace io
} // namespace cuadmm
