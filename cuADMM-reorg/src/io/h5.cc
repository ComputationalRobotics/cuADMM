/*

    h5.cc

    The native HDF5 problem format, schema version 1 (plan Section 12, "Native format"):

        /                  attrs: format = "cuadmm", version = 1, sense = "min" | "max", objective_offset,
                                  source, source_sha256, source_format, svec = "upper, column-major, off-diagonal x sqrt(2)"
        /blocks/kind       uint8 (B)       's', 'l', 'u'
        /blocks/size       int64 (B)
        /At/colptr         int64 (m+1)     CSC of A^T: one column per constraint
        /At/rowind         int64 (nnz)     svec index, 0-based, sorted within each column
        /At/val            float64 (nnz)
        /b/index, /b/val   int64, float64  sparse
        /C/index, /C/val   int64, float64  sparse
        /meta/...          optional: dimacs_kinds (uint8), report (strings), original_block_* , the MOSEK mapping
        /start/...         optional warm start: X, y, S (float64), sigma (attr)
        /solution/...      written by the solver: X, y, S, status, residuals, objective values

    Datasets are chunked (2^20 entries per chunk) and optionally deflate-compressed; the reader reads every dataset
    straight into the CSC arrays. Without HDF5 (CUADMM_WITH_HDF5=OFF) every entry point fails with a clear message.

*/

#include "io/readers.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "io/sha256.h"
#include "io/svec.h"

#ifdef CUADMM_HAVE_HDF5
#include <hdf5.h>
#endif

namespace cuadmm
{
namespace io
{
namespace detail
{

bool have_hdf5()
{
#ifdef CUADMM_HAVE_HDF5
    return true;
#else
    return false;
#endif
}

#ifdef CUADMM_HAVE_HDF5

namespace
{

constexpr std::int64_t kSchemaVersion = 1;
constexpr hsize_t kChunk = 1u << 20;
const char *const kSvecDescription = "upper, column-major, off-diagonal x sqrt(2)";

// RAII over an HDF5 identifier
class Hid
{
  public:
    Hid() = default;
    Hid(hid_t id, herr_t (*closer)(hid_t), const std::string &what) : id_(id), closer_(closer)
    {
        if (id_ < 0)
            throw std::runtime_error("HDF5: " + what + " failed");
    }
    ~Hid()
    {
        if (id_ >= 0 && closer_)
            closer_(id_);
    }
    Hid(const Hid &) = delete;
    Hid &operator=(const Hid &) = delete;
    Hid(Hid &&o) noexcept : id_(o.id_), closer_(o.closer_) { o.id_ = -1; }
    operator hid_t() const { return id_; }

  private:
    hid_t id_ = -1;
    herr_t (*closer_)(hid_t) = nullptr;
};

struct QuietErrors
{
    QuietErrors() { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
};

void check(herr_t status, const std::string &what)
{
    if (status < 0)
        throw std::runtime_error("HDF5: " + what + " failed");
}

bool link_exists(hid_t loc, const char *name)
{
    const htri_t r = H5Lexists(loc, name, H5P_DEFAULT);
    return r > 0;
}

bool attribute_exists(hid_t loc, const char *name)
{
    const htri_t r = H5Aexists(loc, name);
    return r > 0;
}

void write_string_attribute(hid_t loc, const char *name, const std::string &value)
{
    Hid type(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
    check(H5Tset_size(type, value.empty() ? 1 : value.size()), "H5Tset_size");
    check(H5Tset_strpad(type, H5T_STR_NULLTERM), "H5Tset_strpad");
    Hid space(H5Screate(H5S_SCALAR), H5Sclose, "H5Screate");
    Hid attr(H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, std::string("creating attribute ") + name);
    const std::string v = value.empty() ? std::string("\0", 1) : value;
    check(H5Awrite(attr, type, v.data()), std::string("writing attribute ") + name);
}

void write_scalar_attribute(hid_t loc, const char *name, hid_t type, const void *value)
{
    Hid space(H5Screate(H5S_SCALAR), H5Sclose, "H5Screate");
    Hid attr(H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, std::string("creating attribute ") + name);
    check(H5Awrite(attr, type, value), std::string("writing attribute ") + name);
}

std::string read_string_attribute(hid_t loc, const char *name)
{
    Hid attr(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose, std::string("opening attribute ") + name);
    Hid type(H5Aget_type(attr), H5Tclose, "H5Aget_type");
    if (H5Tget_class(type) != H5T_STRING)
        throw std::invalid_argument(std::string("attribute ") + name + " is not a string");
    if (H5Tis_variable_str(type) > 0)
    {
        Hid mem(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
        check(H5Tset_size(mem, H5T_VARIABLE), "H5Tset_size");
        char *s = nullptr;
        check(H5Aread(attr, mem, &s), std::string("reading attribute ") + name);
        std::string out = s ? s : "";
        Hid space(H5Aget_space(attr), H5Sclose, "H5Aget_space");
        H5Treclaim(mem, space, H5P_DEFAULT, &s);
        return out;
    }
    const std::size_t size = H5Tget_size(type);
    std::string buf(size + 1, '\0');
    Hid mem(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
    check(H5Tset_size(mem, size + 1), "H5Tset_size");
    check(H5Aread(attr, mem, buf.data()), std::string("reading attribute ") + name);
    return std::string(buf.c_str());
}

double read_double_attribute(hid_t loc, const char *name)
{
    Hid attr(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose, std::string("opening attribute ") + name);
    double v = 0.0;
    check(H5Aread(attr, H5T_NATIVE_DOUBLE, &v), std::string("reading attribute ") + name);
    return v;
}

std::int64_t read_int_attribute(hid_t loc, const char *name)
{
    Hid attr(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose, std::string("opening attribute ") + name);
    std::int64_t v = 0;
    check(H5Aread(attr, H5T_NATIVE_INT64, &v), std::string("reading attribute ") + name);
    return v;
}

void write_1d(hid_t loc, const char *name, hid_t file_type, hid_t mem_type, const void *data, hsize_t n, int compression)
{
    hsize_t dims[1] = {n};
    Hid space(H5Screate_simple(1, dims, nullptr), H5Sclose, "H5Screate_simple");
    Hid dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "H5Pcreate");
    if (n > 0)
    {
        hsize_t chunk[1] = {n < kChunk ? n : kChunk};
        check(H5Pset_chunk(dcpl, 1, chunk), "H5Pset_chunk");
        if (compression > 0)
        {
            check(H5Pset_shuffle(dcpl), "H5Pset_shuffle");
            check(H5Pset_deflate(dcpl, static_cast<unsigned>(compression)), "H5Pset_deflate");
        }
    }
    Hid dset(H5Dcreate2(loc, name, file_type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT), H5Dclose, std::string("creating dataset ") + name);
    if (n > 0)
        check(H5Dwrite(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), std::string("writing dataset ") + name);
}

void write_strings(hid_t loc, const char *name, const std::vector<std::string> &lines)
{
    std::vector<const char *> ptrs;
    for (const std::string &s : lines)
        ptrs.push_back(s.c_str());
    hsize_t dims[1] = {ptrs.size()};
    Hid type(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
    check(H5Tset_size(type, H5T_VARIABLE), "H5Tset_size");
    Hid space(H5Screate_simple(1, dims, nullptr), H5Sclose, "H5Screate_simple");
    Hid dset(H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose, std::string("creating dataset ") + name);
    if (!ptrs.empty())
        check(H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptrs.data()), std::string("writing dataset ") + name);
}

hsize_t dataset_length(hid_t dset, const char *name)
{
    Hid space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    const int rank = H5Sget_simple_extent_ndims(space);
    if (rank == 0)
        return 1;
    if (rank != 1)
        throw std::invalid_argument(std::string("dataset ") + name + " must be one-dimensional (rank " + std::to_string(rank) + ")");
    hsize_t dims[1];
    check(H5Sget_simple_extent_dims(space, dims, nullptr), "H5Sget_simple_extent_dims");
    return dims[0];
}

template <typename T>
std::vector<T> read_1d(hid_t loc, const char *name, hid_t mem_type)
{
    Hid dset(H5Dopen2(loc, name, H5P_DEFAULT), H5Dclose, std::string("opening dataset ") + name);
    const hsize_t n = dataset_length(dset, name);
    std::vector<T> out(static_cast<std::size_t>(n));
    if (n > 0)
        check(H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()), std::string("reading dataset ") + name);
    return out;
}

std::vector<std::string> read_strings(hid_t loc, const char *name)
{
    Hid dset(H5Dopen2(loc, name, H5P_DEFAULT), H5Dclose, std::string("opening dataset ") + name);
    const hsize_t n = dataset_length(dset, name);
    std::vector<std::string> out;
    if (n == 0)
        return out;
    Hid type(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
    check(H5Tset_size(type, H5T_VARIABLE), "H5Tset_size");
    std::vector<char *> ptrs(static_cast<std::size_t>(n), nullptr);
    check(H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptrs.data()), std::string("reading dataset ") + name);
    for (char *p : ptrs)
        out.push_back(p ? p : "");
    Hid space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    H5Treclaim(type, space, H5P_DEFAULT, ptrs.data());
    return out;
}

template <typename T>
std::vector<T> read_optional_1d(hid_t loc, const char *name, hid_t mem_type)
{
    if (!link_exists(loc, name))
        return {};
    return read_1d<T>(loc, name, mem_type);
}

std::vector<std::uint8_t> to_u8(const std::vector<char> &v) { return std::vector<std::uint8_t>(v.begin(), v.end()); }

std::string source_hash(const std::string &path)
{
    if (path.empty())
        return "";
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec))
    {
        // a TXT directory: the five files in a fixed order
        Sha256 h;
        for (const char *name : {"blk.txt", "At.txt", "b.txt", "C.txt", "con_num.txt"})
        {
            const std::string f = (path.back() == '/' ? path : path + "/") + name;
            if (!std::filesystem::exists(f, ec))
                continue;
            const std::string d = sha256_of_file(f);
            h.update(reinterpret_cast<const unsigned char *>(d.data()), d.size());
        }
        return h.hex_digest();
    }
    if (!std::filesystem::is_regular_file(path, ec))
        return "";
    return sha256_of_file(path);
}

} // namespace

void write_problem_h5(const std::string &path, const ProblemData &problem, int compression)
{
    static QuietErrors quiet;
    if (compression < 0 || compression > 9)
        throw std::invalid_argument("write_problem: the compression level must be in [0, 9]");
    problem.validate();
    Hid file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose, "creating " + path);
    write_string_attribute(file, "format", "cuadmm");
    write_scalar_attribute(file, "version", H5T_NATIVE_INT64, &kSchemaVersion);
    write_string_attribute(file, "sense", problem.sense == Sense::Maximize ? "max" : "min");
    write_scalar_attribute(file, "objective_offset", H5T_NATIVE_DOUBLE, &problem.objective_offset);
    write_string_attribute(file, "source", problem.source.path);
    write_string_attribute(file, "source_format", problem.source.format);
    write_string_attribute(file, "source_sha256", source_hash(problem.source.path));
    write_string_attribute(file, "svec", kSvecDescription);
    {
        Hid g(H5Gcreate2(file, "blocks", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating /blocks");
        std::vector<std::uint8_t> kinds;
        std::vector<std::int64_t> sizes;
        for (const ConeSpec &c : problem.cones)
        {
            kinds.push_back(static_cast<std::uint8_t>(cone_code(c.kind)));
            sizes.push_back(c.size);
        }
        write_1d(g, "kind", H5T_STD_U8LE, H5T_NATIVE_UINT8, kinds.data(), kinds.size(), 0);
        write_1d(g, "size", H5T_STD_I64LE, H5T_NATIVE_INT64, sizes.data(), sizes.size(), 0);
    }
    {
        Hid g(H5Gcreate2(file, "At", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating /At");
        write_1d(g, "colptr", H5T_STD_I64LE, H5T_NATIVE_INT64, problem.At.colptr.data(), problem.At.colptr.size(), compression);
        write_1d(g, "rowind", H5T_STD_I64LE, H5T_NATIVE_INT64, problem.At.rowind.data(), problem.At.rowind.size(), compression);
        write_1d(g, "val", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, problem.At.val.data(), problem.At.val.size(), compression);
        const std::int64_t rows = problem.At.rows, cols = problem.At.cols;
        write_scalar_attribute(g, "rows", H5T_NATIVE_INT64, &rows);
        write_scalar_attribute(g, "cols", H5T_NATIVE_INT64, &cols);
    }
    auto write_vec = [&](const char *name, const SparseVector &v)
    {
        Hid g(H5Gcreate2(file, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, std::string("creating /") + name);
        write_1d(g, "index", H5T_STD_I64LE, H5T_NATIVE_INT64, v.index.data(), v.index.size(), compression);
        write_1d(g, "val", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, v.val.data(), v.val.size(), compression);
        write_scalar_attribute(g, "size", H5T_NATIVE_INT64, &v.size);
    };
    write_vec("b", problem.b);
    write_vec("C", problem.C);
    {
        Hid g(H5Gcreate2(file, "meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating /meta");
        const SourceInfo &s = problem.source;
        if (!s.dimacs_kinds.empty())
        {
            const std::vector<std::uint8_t> k = to_u8(s.dimacs_kinds);
            write_1d(g, "dimacs_kinds", H5T_STD_U8LE, H5T_NATIVE_UINT8, k.data(), k.size(), 0);
        }
        write_strings(g, "report", s.report);
        if (!s.original_block_sizes.empty())
        {
            std::vector<std::int64_t> of(s.original_block_of.begin(), s.original_block_of.end());
            write_1d(g, "original_block_of", H5T_STD_I64LE, H5T_NATIVE_INT64, of.data(), of.size(), 0);
            write_1d(g, "original_block_sizes", H5T_STD_I64LE, H5T_NATIVE_INT64, s.original_block_sizes.data(), s.original_block_sizes.size(), 0);
            const std::vector<std::uint8_t> k = to_u8(s.original_block_kinds);
            write_1d(g, "original_block_kinds", H5T_STD_U8LE, H5T_NATIVE_UINT8, k.data(), k.size(), 0);
        }
        if (!s.mosek_var_position.empty() || !s.mosek_row_position.empty())
        {
            write_1d(g, "mosek_var_position", H5T_STD_I64LE, H5T_NATIVE_INT64, s.mosek_var_position.data(), s.mosek_var_position.size(), 0);
            write_1d(g, "mosek_var_shift", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, s.mosek_var_shift.data(), s.mosek_var_shift.size(), 0);
            write_1d(g, "mosek_var_sign", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, s.mosek_var_sign.data(), s.mosek_var_sign.size(), 0);
            write_1d(g, "mosek_row_position", H5T_STD_I64LE, H5T_NATIVE_INT64, s.mosek_row_position.data(), s.mosek_row_position.size(), 0);
        }
        write_string_attribute(g, "mat_version", s.mat_version);
    }
}

ProblemData read_h5(const std::string &path, const IoOptions &options)
{
    static QuietErrors quiet;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        throw std::runtime_error("could not open " + path);
    if (H5Fis_hdf5(path.c_str()) <= 0)
        throw std::invalid_argument(path + " is not an HDF5 file");
    Hid file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + path);
    auto fail = [&](const std::string &why) { throw std::invalid_argument(path + ": " + why); };
    if (!attribute_exists(file, "format") || read_string_attribute(file, "format") != "cuadmm")
        fail("not a cuADMM problem file (attribute format = \"cuadmm\" missing)");
    const std::int64_t version = attribute_exists(file, "version") ? read_int_attribute(file, "version") : -1;
    if (version != kSchemaVersion)
        fail("schema version " + std::to_string(version) + " is not supported (this build reads version " + std::to_string(kSchemaVersion) + ")");
    for (const char *name : {"blocks", "At", "b", "C"})
        if (!link_exists(file, name))
            fail(std::string("group /") + name + " is missing");

    ProblemData p;
    const std::string sense = attribute_exists(file, "sense") ? read_string_attribute(file, "sense") : "min";
    if (sense == "min" || sense == "minimize")
        p.sense = Sense::Minimize;
    else if (sense == "max" || sense == "maximize")
        p.sense = Sense::Maximize;
    else
        fail("attribute sense must be \"min\" or \"max\" (got \"" + sense + "\")");
    p.objective_offset = attribute_exists(file, "objective_offset") ? read_double_attribute(file, "objective_offset") : 0.0;
    check_sense_option(options, p.sense, path, "stored");
    {
        Hid g(H5Gopen2(file, "blocks", H5P_DEFAULT), H5Gclose, "opening /blocks");
        const std::vector<std::uint8_t> kinds = read_1d<std::uint8_t>(g, "kind", H5T_NATIVE_UINT8);
        const std::vector<std::int64_t> sizes = read_1d<std::int64_t>(g, "size", H5T_NATIVE_INT64);
        if (kinds.size() != sizes.size())
            fail("/blocks/kind and /blocks/size differ in length");
        for (std::size_t b = 0; b < kinds.size(); b++)
        {
            ConeSpec c;
            try
            {
                c.kind = cone_kind_of(static_cast<char>(kinds[b]));
            }
            catch (const std::invalid_argument &)
            {
                fail("block " + std::to_string(b) + " has the unknown kind " + std::to_string(kinds[b]));
            }
            if (c.kind != ConeKind::Psd && c.kind != ConeKind::Nonnegative && c.kind != ConeKind::Free)
                fail("block " + std::to_string(b) + " has kind '" + static_cast<char>(kinds[b]) + "', which comes with Phase 9");
            if (sizes[b] <= 0)
                fail("block " + std::to_string(b) + " has size " + std::to_string(sizes[b]));
            c.size = sizes[b];
            p.cones.push_back(c);
        }
    }
    const std::vector<std::int64_t> off = cone_offsets(p.cones);
    {
        Hid g(H5Gopen2(file, "At", H5P_DEFAULT), H5Gclose, "opening /At");
        p.At.colptr = read_1d<std::int64_t>(g, "colptr", H5T_NATIVE_INT64);
        p.At.rowind = read_1d<std::int64_t>(g, "rowind", H5T_NATIVE_INT64);
        p.At.val = read_1d<double>(g, "val", H5T_NATIVE_DOUBLE);
        if (p.At.colptr.empty())
            fail("/At/colptr is empty");
        p.At.cols = static_cast<std::int64_t>(p.At.colptr.size()) - 1;
        p.At.rows = off.back();
        if (attribute_exists(g, "rows") && read_int_attribute(g, "rows") != p.At.rows)
            fail("/At rows attribute " + std::to_string(read_int_attribute(g, "rows")) + " does not match the blocks (" + std::to_string(p.At.rows) + ")");
        if (p.At.rowind.size() != p.At.val.size())
            fail("/At/rowind and /At/val differ in length");
    }
    auto read_vec = [&](const char *name, SparseVector &v, std::int64_t size)
    {
        Hid g(H5Gopen2(file, name, H5P_DEFAULT), H5Gclose, std::string("opening /") + name);
        v.index = read_1d<std::int64_t>(g, "index", H5T_NATIVE_INT64);
        v.val = read_1d<double>(g, "val", H5T_NATIVE_DOUBLE);
        v.size = size;
        if (v.index.size() != v.val.size())
            fail(std::string("/") + name + "/index and /" + name + "/val differ in length");
        for (std::size_t k = 1; k < v.index.size(); k++)
            if (v.index[k] <= v.index[k - 1])
                fail(std::string("/") + name + "/index must be strictly increasing");
    };
    read_vec("b", p.b, p.At.cols);
    read_vec("C", p.C, p.At.rows);
    p.source.format = "h5";
    p.source.path = path;
    std::vector<std::string> stored_report;
    std::string source_format, source;
    if (attribute_exists(file, "source_format"))
        source_format = read_string_attribute(file, "source_format");
    if (attribute_exists(file, "source"))
        source = read_string_attribute(file, "source");
    if (link_exists(file, "meta"))
    {
        Hid g(H5Gopen2(file, "meta", H5P_DEFAULT), H5Gclose, "opening /meta");
        const std::vector<std::uint8_t> dk = read_optional_1d<std::uint8_t>(g, "dimacs_kinds", H5T_NATIVE_UINT8);
        p.source.dimacs_kinds.assign(dk.begin(), dk.end());
        if (link_exists(g, "report"))
            stored_report = read_strings(g, "report");
        const std::vector<std::int64_t> of = read_optional_1d<std::int64_t>(g, "original_block_of", H5T_NATIVE_INT64);
        p.source.original_block_of.assign(of.begin(), of.end());
        p.source.original_block_sizes = read_optional_1d<std::int64_t>(g, "original_block_sizes", H5T_NATIVE_INT64);
        const std::vector<std::uint8_t> ok = read_optional_1d<std::uint8_t>(g, "original_block_kinds", H5T_NATIVE_UINT8);
        p.source.original_block_kinds.assign(ok.begin(), ok.end());
        p.source.mosek_var_position = read_optional_1d<std::int64_t>(g, "mosek_var_position", H5T_NATIVE_INT64);
        p.source.mosek_var_shift = read_optional_1d<double>(g, "mosek_var_shift", H5T_NATIVE_DOUBLE);
        p.source.mosek_var_sign = read_optional_1d<double>(g, "mosek_var_sign", H5T_NATIVE_DOUBLE);
        p.source.mosek_row_position = read_optional_1d<std::int64_t>(g, "mosek_row_position", H5T_NATIVE_INT64);
        if (attribute_exists(g, "mat_version"))
            p.source.mat_version = read_string_attribute(g, "mat_version");
    }
    if (p.source.original_block_of.size() != p.cones.size() || p.source.original_block_sizes.size() != p.source.original_block_kinds.size())
    {
        if (!p.source.original_block_of.empty())
            fail("/meta/original_block_* do not match the blocks");
        set_identity_original_blocks(p);
    }
    p.source.report.push_back("native HDF5 (schema 1): m = " + std::to_string(p.At.cols) + ", vec_len = " + std::to_string(p.At.rows) + ", " +
                              std::to_string(p.cones.size()) + " blocks, nnz(At) = " + std::to_string(p.At.nnz()) +
                              (source_format.empty() ? "" : "; converted from " + source_format + (source.empty() ? "" : " " + source)));
    for (const std::string &line : stored_report)
        p.source.report.push_back("  [conversion] " + line);
    // the file's structure is trusted only after the checks of finish_problem (sorted columns, ranges)
    const std::size_t before = p.source.report.size();
    apply_simplifications(p, options.simplify);
    if (p.source.report.size() != before && !p.source.dimacs_kinds.empty() && p.source.dimacs_kinds.size() != p.cones.size())
        p.source.dimacs_kinds.clear();
    finish_problem(p, path);
    return p;
}

void write_solution_h5(const std::string &path, const ProblemData &problem, const SolveResult &result)
{
    static QuietErrors quiet;
    std::error_code ec;
    Hid file = std::filesystem::is_regular_file(path, ec) && H5Fis_hdf5(path.c_str()) > 0
                   ? Hid(H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening " + path + " for writing")
                   : Hid(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose, "creating " + path);
    if (link_exists(file, "solution"))
        check(H5Ldelete(file, "solution", H5P_DEFAULT), "removing the previous /solution");
    Hid g(H5Gcreate2(file, "solution", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating /solution");
    write_1d(g, "X", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, result.best.X.data(), result.best.X.size(), 0);
    write_1d(g, "y", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, result.best.y.data(), result.best.y.size(), 0);
    write_1d(g, "S", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, result.best.S.data(), result.best.S.size(), 0);
    write_string_attribute(g, "status", name_of(result.status));
    write_string_attribute(g, "stop_reason", result.stop_reason);
    write_string_attribute(g, "sense", problem.sense == Sense::Maximize ? "max" : "min");
    const std::int64_t iterations = result.iterations, best_iteration = result.best.iteration;
    write_scalar_attribute(g, "iterations", H5T_NATIVE_INT64, &iterations);
    write_scalar_attribute(g, "best_iteration", H5T_NATIVE_INT64, &best_iteration);
    const std::int64_t converged = result.converged ? 1 : 0;
    write_scalar_attribute(g, "converged", H5T_NATIVE_INT64, &converged);
    write_scalar_attribute(g, "eta_p", H5T_NATIVE_DOUBLE, &result.eta_p);
    write_scalar_attribute(g, "eta_d", H5T_NATIVE_DOUBLE, &result.eta_d);
    write_scalar_attribute(g, "eta_g", H5T_NATIVE_DOUBLE, &result.eta_g);
    write_scalar_attribute(g, "pobj", H5T_NATIVE_DOUBLE, &result.pobj);
    write_scalar_attribute(g, "dobj", H5T_NATIVE_DOUBLE, &result.dobj);
    write_scalar_attribute(g, "objective_offset", H5T_NATIVE_DOUBLE, &problem.objective_offset);
    write_scalar_attribute(g, "sigma_final", H5T_NATIVE_DOUBLE, &result.sigma_final_original);
    write_scalar_attribute(g, "cone_violation_X", H5T_NATIVE_DOUBLE, &result.cone_violation_X);
    write_scalar_attribute(g, "cone_violation_S", H5T_NATIVE_DOUBLE, &result.cone_violation_S);
    write_scalar_attribute(g, "solve_time_s", H5T_NATIVE_DOUBLE, &result.solve_time_s);
}

void read_h5_solution(const std::string &path, std::vector<double> &X, std::vector<double> &y, std::vector<double> &S, double *pobj, double *dobj)
{
    static QuietErrors quiet;
    Hid file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + path);
    if (!link_exists(file, "solution"))
        throw std::invalid_argument(path + " has no /solution group");
    Hid g(H5Gopen2(file, "solution", H5P_DEFAULT), H5Gclose, "opening /solution");
    X = read_1d<double>(g, "X", H5T_NATIVE_DOUBLE);
    y = read_1d<double>(g, "y", H5T_NATIVE_DOUBLE);
    S = read_1d<double>(g, "S", H5T_NATIVE_DOUBLE);
    if (pobj)
        *pobj = read_double_attribute(g, "pobj");
    if (dobj)
        *dobj = read_double_attribute(g, "dobj");
}

WarmStart read_h5_warm_start(const std::string &path, const ProblemData &problem)
{
    static QuietErrors quiet;
    Hid file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose, "opening " + path);
    if (!link_exists(file, "start"))
        throw std::invalid_argument(path + " has no /start group (warm start)");
    Hid g(H5Gopen2(file, "start", H5P_DEFAULT), H5Gclose, "opening /start");
    WarmStart w;
    w.X = read_1d<double>(g, "X", H5T_NATIVE_DOUBLE);
    w.y = read_1d<double>(g, "y", H5T_NATIVE_DOUBLE);
    w.S = read_1d<double>(g, "S", H5T_NATIVE_DOUBLE);
    if (attribute_exists(g, "sigma"))
        w.sigma = read_double_attribute(g, "sigma");
    else if (link_exists(g, "sigma"))
    {
        const std::vector<double> s = read_1d<double>(g, "sigma", H5T_NATIVE_DOUBLE);
        if (!s.empty())
            w.sigma = s[0];
    }
    if (static_cast<std::int64_t>(w.X.size()) != problem.vec_len() || static_cast<std::int64_t>(w.S.size()) != problem.vec_len() ||
        static_cast<std::int64_t>(w.y.size()) != problem.num_constraints())
        throw std::invalid_argument("warm start in " + path + " does not match the problem sizes: X and S need " + std::to_string(problem.vec_len()) +
                                    " entries, y " + std::to_string(problem.num_constraints()));
    return w;
}

void write_h5_warm_start(const std::string &path, const WarmStart &w)
{
    static QuietErrors quiet;
    std::error_code ec;
    Hid file = std::filesystem::is_regular_file(path, ec) && H5Fis_hdf5(path.c_str()) > 0
                   ? Hid(H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose, "opening " + path + " for writing")
                   : Hid(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose, "creating " + path);
    if (link_exists(file, "start"))
        check(H5Ldelete(file, "start", H5P_DEFAULT), "removing the previous /start");
    Hid g(H5Gcreate2(file, "start", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "creating /start");
    write_1d(g, "X", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, w.X.data(), w.X.size(), 0);
    write_1d(g, "y", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, w.y.data(), w.y.size(), 0);
    write_1d(g, "S", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, w.S.data(), w.S.size(), 0);
    if (w.sigma > 0)
        write_scalar_attribute(g, "sigma", H5T_NATIVE_DOUBLE, &w.sigma);
}

#else // no HDF5

namespace
{
[[noreturn]] void no_hdf5(const std::string &path)
{
    throw std::invalid_argument(path + ": the native HDF5 format is not compiled into this build (CUADMM_WITH_HDF5); "
                                       "available formats: see cuadmm_exe --formats");
}
} // namespace

void write_problem_h5(const std::string &path, const ProblemData &, int) { no_hdf5(path); }
ProblemData read_h5(const std::string &path, const IoOptions &) { no_hdf5(path); }
void write_solution_h5(const std::string &path, const ProblemData &, const SolveResult &) { no_hdf5(path); }
void read_h5_solution(const std::string &path, std::vector<double> &, std::vector<double> &, std::vector<double> &, double *, double *) { no_hdf5(path); }
WarmStart read_h5_warm_start(const std::string &path, const ProblemData &) { no_hdf5(path); }
void write_h5_warm_start(const std::string &path, const WarmStart &) { no_hdf5(path); }

#endif

} // namespace detail

void write_problem(const std::string &path_h5, const ProblemData &problem, int compression_level)
{
    detail::write_problem_h5(path_h5, problem, compression_level);
}

void write_problem(const std::string &path_h5, const ProblemData &problem) { write_problem(path_h5, problem, 0); }

} // namespace io
} // namespace cuadmm
