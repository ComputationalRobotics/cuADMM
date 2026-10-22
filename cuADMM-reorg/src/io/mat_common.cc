/*

    mat_common.cc

*/

#include "io/mat_common.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "io/readers.h"

namespace cuadmm
{
namespace io
{
namespace detail
{

bool have_matio()
{
#ifdef CUADMM_HAVE_MATIO
    return true;
#else
    return false;
#endif
}

#ifdef CUADMM_HAVE_MATIO

MatFile::MatFile(const std::string &path, int mode) : path_(path)
{
    mat_ = Mat_Open(path.c_str(), mode);
    if (!mat_)
        throw std::runtime_error("could not open " + path + " as a MATLAB file");
}

MatFile MatFile::create(const std::string &path, bool v73)
{
    MatFile f;
    f.path_ = path;
    f.mat_ = Mat_CreateVer(path.c_str(), nullptr, v73 ? MAT_FT_MAT73 : MAT_FT_MAT5);
    if (!f.mat_)
        throw std::runtime_error("could not create " + path + (v73 ? " (MATLAB v7.3; matio built without HDF5?)" : ""));
    return f;
}

MatFile::~MatFile()
{
    if (mat_)
        Mat_Close(mat_);
}

std::string MatFile::version() const
{
    switch (Mat_GetVersion(mat_))
    {
    case MAT_FT_MAT73:
        return "7.3";
    case MAT_FT_MAT5:
        return "5";
    case MAT_FT_MAT4:
        return "4";
    default:
        return "";
    }
}

MatVar read_variable(const MatFile &file, const char *name)
{
    return MatVar(Mat_VarRead(file.get(), name));
}

std::size_t numel(const matvar_t *v)
{
    if (!v || v->rank <= 0 || !v->dims)
        return 0;
    std::size_t n = 1;
    for (int d = 0; d < v->rank; d++)
        n *= v->dims[d];
    return n;
}

bool is_empty(const matvar_t *v) { return !v || v->class_type == MAT_C_EMPTY || numel(v) == 0; }

std::string class_name(const matvar_t *v)
{
    if (!v)
        return "absent";
    switch (v->class_type)
    {
    case MAT_C_EMPTY:
        return "empty";
    case MAT_C_CELL:
        return "cell";
    case MAT_C_STRUCT:
        return "struct";
    case MAT_C_CHAR:
        return "char";
    case MAT_C_SPARSE:
        return "sparse";
    case MAT_C_DOUBLE:
        return "double";
    case MAT_C_SINGLE:
        return "single";
    case MAT_C_INT8:
    case MAT_C_UINT8:
    case MAT_C_INT16:
    case MAT_C_UINT16:
    case MAT_C_INT32:
    case MAT_C_UINT32:
    case MAT_C_INT64:
    case MAT_C_UINT64:
        return "integer";
    default:
        return "other";
    }
}

std::string dims_string(const matvar_t *v)
{
    if (!v)
        return "absent";
    std::string s;
    for (int d = 0; d < v->rank; d++)
        s += (d ? "x" : "") + std::to_string(v->dims[d]);
    return s;
}

const matvar_t *struct_field(const matvar_t *s, const char *name, const std::string &what)
{
    if (!s || s->class_type != MAT_C_STRUCT)
        throw std::invalid_argument(what + " is not a struct (" + class_name(s) + ")");
    return Mat_VarGetStructFieldByName(s, name, 0);
}

const matvar_t *cell_at(const matvar_t *c, std::size_t i, const std::string &what)
{
    if (!c || c->class_type != MAT_C_CELL)
        throw std::invalid_argument(what + " is not a cell array (" + class_name(c) + ")");
    if (i >= numel(c))
        throw std::invalid_argument(what + ": cell index " + std::to_string(i) + " out of range");
    const matvar_t *e = Mat_VarGetCell(c, static_cast<int>(i));
    return e;
}

namespace
{

bool is_numeric_class(const matvar_t *v)
{
    switch (v->class_type)
    {
    case MAT_C_DOUBLE:
    case MAT_C_SINGLE:
    case MAT_C_INT8:
    case MAT_C_UINT8:
    case MAT_C_INT16:
    case MAT_C_UINT16:
    case MAT_C_INT32:
    case MAT_C_UINT32:
    case MAT_C_INT64:
    case MAT_C_UINT64:
    case MAT_C_SPARSE:
        return true;
    default:
        return false;
    }
}

double element(const void *data, enum matio_types t, std::size_t i, const std::string &what)
{
    switch (t)
    {
    case MAT_T_DOUBLE:
        return static_cast<const double *>(data)[i];
    case MAT_T_SINGLE:
        return static_cast<const float *>(data)[i];
    case MAT_T_INT8:
        return static_cast<const std::int8_t *>(data)[i];
    case MAT_T_UINT8:
        return static_cast<const std::uint8_t *>(data)[i];
    case MAT_T_INT16:
        return static_cast<const std::int16_t *>(data)[i];
    case MAT_T_UINT16:
        return static_cast<const std::uint16_t *>(data)[i];
    case MAT_T_INT32:
        return static_cast<const std::int32_t *>(data)[i];
    case MAT_T_UINT32:
        return static_cast<const std::uint32_t *>(data)[i];
    case MAT_T_INT64:
        return static_cast<double>(static_cast<const std::int64_t *>(data)[i]);
    case MAT_T_UINT64:
        return static_cast<double>(static_cast<const std::uint64_t *>(data)[i]);
    default:
        throw std::invalid_argument(what + ": unsupported numeric storage type " + std::to_string(static_cast<int>(t)));
    }
}

void require_real_numeric(const matvar_t *v, const std::string &what)
{
    if (!v)
        throw std::invalid_argument(what + " is missing");
    if (!is_numeric_class(v) && v->class_type != MAT_C_EMPTY)
        throw std::invalid_argument(what + " must be numeric (it is " + class_name(v) + ")");
    if (v->isComplex)
        throw std::invalid_argument(what + " is complex; complex data is not supported");
}

} // namespace

std::vector<double> numeric_values(const matvar_t *v, const std::string &what)
{
    require_real_numeric(v, what);
    std::vector<double> out;
    const std::size_t n = numel(v);
    if (n == 0 || v->class_type == MAT_C_EMPTY)
        return out;
    if (v->class_type == MAT_C_SPARSE)
    {
        const SparseMatrix m = sparse_values(v, what);
        out.assign(static_cast<std::size_t>(m.rows * m.cols), 0.0);
        for (std::int64_t j = 0; j < m.cols; j++)
            for (std::int64_t k = m.colptr[j]; k < m.colptr[j + 1]; k++)
                out[static_cast<std::size_t>(m.rowind[k] + j * m.rows)] = m.val[k];
        return out;
    }
    if (!v->data)
        throw std::invalid_argument(what + ": no data was read");
    out.resize(n);
    for (std::size_t i = 0; i < n; i++)
        out[i] = element(v->data, v->data_type, i, what);
    return out;
}

std::vector<std::int64_t> integer_values(const matvar_t *v, const std::string &what)
{
    const std::vector<double> d = numeric_values(v, what);
    std::vector<std::int64_t> out(d.size());
    for (std::size_t i = 0; i < d.size(); i++)
    {
        if (!std::isfinite(d[i]) || d[i] != std::floor(d[i]) || std::fabs(d[i]) > 9.0e15)
            throw std::invalid_argument(what + ": entry " + std::to_string(i + 1) + " = " + format_double(d[i]) + " is not an integer");
        out[i] = static_cast<std::int64_t>(d[i]);
    }
    return out;
}

double scalar_value(const matvar_t *v, const std::string &what)
{
    const std::vector<double> d = numeric_values(v, what);
    if (d.size() != 1)
        throw std::invalid_argument(what + " must be a scalar (it has " + std::to_string(d.size()) + " elements)");
    return d[0];
}

std::string string_value(const matvar_t *v, const std::string &what)
{
    if (!v)
        throw std::invalid_argument(what + " is missing");
    if (v->class_type != MAT_C_CHAR)
        throw std::invalid_argument(what + " must be a character array (it is " + class_name(v) + ")");
    const std::size_t n = numel(v);
    std::string s;
    if (n == 0 || !v->data)
        return s;
    for (std::size_t i = 0; i < n; i++)
    {
        unsigned code = 0;
        switch (v->data_size)
        {
        case 1:
            code = static_cast<const unsigned char *>(v->data)[i];
            break;
        case 2:
            code = static_cast<const std::uint16_t *>(v->data)[i];
            break;
        case 4:
            code = static_cast<const std::uint32_t *>(v->data)[i];
            break;
        default:
            throw std::invalid_argument(what + ": unsupported character storage");
        }
        s += static_cast<char>(code < 128 ? code : '?');
    }
    return s;
}

SparseMatrix sparse_values(const matvar_t *v, const std::string &what)
{
    require_real_numeric(v, what);
    if (v->class_type != MAT_C_SPARSE)
        throw std::invalid_argument(what + " must be sparse (it is " + class_name(v) + ")");
    if (v->rank != 2)
        throw std::invalid_argument(what + " must be two-dimensional");
    const mat_sparse_t *sp = static_cast<const mat_sparse_t *>(v->data);
    SparseMatrix m;
    m.rows = static_cast<std::int64_t>(v->dims[0]);
    m.cols = static_cast<std::int64_t>(v->dims[1]);
    m.colptr.assign(static_cast<std::size_t>(m.cols) + 1, 0);
    if (!sp)
        return m; // an empty sparse matrix
    if (sp->njc != static_cast<mat_uint32_t>(m.cols) + 1)
        throw std::invalid_argument(what + ": sparse column pointers have " + std::to_string(sp->njc) + " entries for " + std::to_string(m.cols) + " columns");
    for (std::int64_t j = 0; j <= m.cols; j++)
        m.colptr[static_cast<std::size_t>(j)] = sp->jc[j];
    const std::int64_t nnz = m.colptr.back();
    if (nnz > static_cast<std::int64_t>(sp->nir) || nnz > static_cast<std::int64_t>(sp->ndata))
        throw std::invalid_argument(what + ": sparse structure inconsistent (nnz " + std::to_string(nnz) + ", nir " + std::to_string(sp->nir) + ")");
    m.rowind.resize(static_cast<std::size_t>(nnz));
    m.val.resize(static_cast<std::size_t>(nnz));
    for (std::int64_t k = 0; k < nnz; k++)
    {
        m.rowind[static_cast<std::size_t>(k)] = sp->ir[k];
        if (m.rowind[static_cast<std::size_t>(k)] >= m.rows)
            throw std::invalid_argument(what + ": sparse row index " + std::to_string(sp->ir[k]) + " outside " + std::to_string(m.rows) + " rows");
        m.val[static_cast<std::size_t>(k)] = element(sp->data, v->data_type, static_cast<std::size_t>(k), what);
    }
    for (std::int64_t j = 0; j < m.cols; j++)
        if (m.colptr[static_cast<std::size_t>(j)] > m.colptr[static_cast<std::size_t>(j) + 1])
            throw std::invalid_argument(what + ": sparse column pointers are not non-decreasing");
    return m;
}

std::size_t nonzero_capacity(const matvar_t *v)
{
    if (!v)
        return 0;
    if (v->class_type == MAT_C_SPARSE)
    {
        const mat_sparse_t *sp = static_cast<const mat_sparse_t *>(v->data);
        return sp ? sp->ndata : 0;
    }
    return numel(v);
}

void for_each_nonzero(const matvar_t *v, const std::string &what, const std::function<void(std::int64_t, std::int64_t, double)> &f)
{
    require_real_numeric(v, what);
    if (v->class_type == MAT_C_EMPTY || numel(v) == 0)
        return;
    if (v->rank != 2)
        throw std::invalid_argument(what + " must be two-dimensional");
    if (v->class_type == MAT_C_SPARSE)
    {
        const SparseMatrix m = sparse_values(v, what);
        for (std::int64_t j = 0; j < m.cols; j++)
            for (std::int64_t k = m.colptr[j]; k < m.colptr[j + 1]; k++)
                if (m.val[k] != 0.0)
                    f(m.rowind[k], j, m.val[k]);
        return;
    }
    const std::int64_t rows = static_cast<std::int64_t>(v->dims[0]), cols = static_cast<std::int64_t>(v->dims[1]);
    for (std::int64_t j = 0; j < cols; j++)
        for (std::int64_t i = 0; i < rows; i++)
        {
            const double x = element(v->data, v->data_type, static_cast<std::size_t>(i + j * rows), what);
            if (x != 0.0)
                f(i, j, x);
        }
}

matvar_t *make_double_matrix(const char *name, std::int64_t rows, std::int64_t cols, const double *data)
{
    size_t dims[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
    matvar_t *v = Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, rows * cols > 0 ? data : nullptr, 0);
    if (!v)
        throw std::runtime_error(std::string("matio: could not create the variable ") + name);
    return v;
}

matvar_t *make_string(const char *name, const std::string &s)
{
    size_t dims[2] = {1, s.size()};
    matvar_t *v = Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UTF8, 2, dims, s.c_str(), 0);
    if (!v)
        throw std::runtime_error(std::string("matio: could not create the variable ") + name);
    return v;
}

matvar_t *make_sparse(const char *name, const SparseMatrix &m)
{
    const std::size_t nnz = m.val.size();
    std::vector<mat_uint32_t> ir(nnz), jc(static_cast<std::size_t>(m.cols) + 1);
    for (std::size_t k = 0; k < nnz; k++)
        ir[k] = static_cast<mat_uint32_t>(m.rowind[k]);
    for (std::size_t j = 0; j < jc.size(); j++)
        jc[j] = static_cast<mat_uint32_t>(m.colptr[j]);
    mat_sparse_t sp;
    sp.nzmax = static_cast<mat_uint32_t>(nnz);
    sp.ir = ir.data();
    sp.nir = static_cast<mat_uint32_t>(nnz);
    sp.jc = jc.data();
    sp.njc = static_cast<mat_uint32_t>(jc.size());
    sp.ndata = static_cast<mat_uint32_t>(nnz);
    sp.data = const_cast<double *>(m.val.data());
    size_t dims[2] = {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols)};
    matvar_t *v = Mat_VarCreate(name, MAT_C_SPARSE, MAT_T_DOUBLE, 2, dims, &sp, 0);
    if (!v)
        throw std::runtime_error(std::string("matio: could not create the sparse variable ") + name);
    return v;
}

matvar_t *make_cell(const char *name, std::int64_t rows, std::int64_t cols, std::vector<matvar_t *> &cells)
{
    size_t dims[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
    if (cells.size() != static_cast<std::size_t>(rows * cols))
        throw std::invalid_argument(std::string("make_cell ") + name + ": wrong number of cells");
    matvar_t *v = Mat_VarCreate(name, MAT_C_CELL, MAT_T_CELL, 2, dims, cells.data(), 0);
    if (!v)
        throw std::runtime_error(std::string("matio: could not create the cell array ") + name);
    cells.clear(); // owned by v now
    return v;
}

matvar_t *make_struct(const char *name, std::vector<std::pair<std::string, matvar_t *>> &fields)
{
    std::vector<const char *> names;
    for (const auto &f : fields)
        names.push_back(f.first.c_str());
    names.push_back(nullptr);
    size_t dims[2] = {1, 1};
    matvar_t *s = Mat_VarCreateStruct2(name, 2, dims, names.data());
    if (!s)
        throw std::runtime_error(std::string("matio: could not create the struct ") + name);
    for (auto &f : fields)
    {
        Mat_VarSetStructFieldByName(s, f.first.c_str(), 0, f.second);
        f.second = nullptr;
    }
    fields.clear();
    return s;
}

void write_variable(const MatFile &file, matvar_t *var)
{
    const int rc = Mat_VarWrite(file.get(), var, MAT_COMPRESSION_ZLIB);
    const std::string name = var->name ? var->name : "";
    Mat_VarFree(var);
    if (rc != 0)
        throw std::runtime_error("matio: could not write the variable " + name + " to " + file.path());
}

#endif // CUADMM_HAVE_MATIO

} // namespace detail
} // namespace io
} // namespace cuadmm

// detection of the .mat conventions from the variables present (plan Section 12) ------------------------------------

namespace cuadmm
{
namespace io
{
namespace detail
{

#ifdef CUADMM_HAVE_MATIO

std::vector<std::string> mat_variable_names(const std::string &path)
{
    MatFile file(path, MAT_ACC_RDONLY);
    std::vector<std::string> names;
    while (matvar_t *v = Mat_VarReadNextInfo(file.get()))
    {
        if (v->name)
            names.push_back(v->name);
        Mat_VarFree(v);
    }
    return names;
}

IoFormat detect_mat_format(const std::string &path)
{
    const std::vector<std::string> names = mat_variable_names(path);
    auto has = [&](const char *n)
    {
        for (const std::string &s : names)
            if (s == n)
                return true;
        return false;
    };
    if (has("prob"))
        return IoFormat::Mosek;
    if (has("blk") && has("At") && has("C") && has("b"))
        return IoFormat::Sdpt3;
    if (has("K") && (has("A") || has("At")))
        return IoFormat::Sedumi;
    std::string list;
    for (const std::string &s : names)
        list += (list.empty() ? "" : ", ") + s;
    throw std::invalid_argument(path + ": the variables (" + list + ") match no known convention (MOSEK: prob; SDPT3: blk, At, C, b; SeDuMi: K with A or At); use io.format");
}

#else

std::vector<std::string> mat_variable_names(const std::string &path)
{
    throw std::invalid_argument(path + ": reading .mat files needs matio, which is not compiled into this build (CUADMM_WITH_MATIO)");
}

IoFormat detect_mat_format(const std::string &path)
{
    throw std::invalid_argument(path + ": reading .mat files needs matio, which is not compiled into this build (CUADMM_WITH_MATIO)");
}

#endif

} // namespace detail
} // namespace io
} // namespace cuadmm
