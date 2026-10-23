/*

    mat_common.h

    Helpers over matio for the .mat readers and writers (SeDuMi, SDPT3, MOSEK prob; plan Section 12): RAII file and
    variable handles, numeric access independent of the stored class (MATLAB stores integer-valued doubles
    compactly; matio converts on read, but the class may also be an integer class), sparse matrices, cells,
    structs, strings, and the construction of variables for writing. Only compiled with CUADMM_HAVE_MATIO.

*/

#ifndef CUADMM_IO_MAT_COMMON_H
#define CUADMM_IO_MAT_COMMON_H

#ifdef CUADMM_HAVE_MATIO

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <matio.h>

namespace cuadmm
{
namespace io
{
namespace detail
{

class MatFile
{
  public:
    MatFile(const std::string &path, int mode); // MAT_ACC_RDONLY / MAT_ACC_RDWR; throws when the file cannot be opened
    static MatFile create(const std::string &path, bool v73);
    ~MatFile();
    MatFile(const MatFile &) = delete;
    MatFile &operator=(const MatFile &) = delete;
    MatFile(MatFile &&o) noexcept : mat_(o.mat_), path_(std::move(o.path_)) { o.mat_ = nullptr; }
    mat_t *get() const { return mat_; }
    const std::string &path() const { return path_; }
    std::string version() const; // "5", "7.3", "4"

  private:
    MatFile() = default;
    mat_t *mat_ = nullptr;
    std::string path_;
};

/// @brief Owning matvar_t pointer.
class MatVar
{
  public:
    MatVar() = default;
    explicit MatVar(matvar_t *v) : v_(v) {}
    ~MatVar()
    {
        if (v_)
            Mat_VarFree(v_);
    }
    MatVar(const MatVar &) = delete;
    MatVar &operator=(const MatVar &) = delete;
    MatVar(MatVar &&o) noexcept : v_(o.v_) { o.v_ = nullptr; }
    MatVar &operator=(MatVar &&o) noexcept
    {
        if (this != &o)
        {
            if (v_)
                Mat_VarFree(v_);
            v_ = o.v_;
            o.v_ = nullptr;
        }
        return *this;
    }
    matvar_t *get() const { return v_; }
    matvar_t *release()
    {
        matvar_t *v = v_;
        v_ = nullptr;
        return v;
    }
    explicit operator bool() const { return v_ != nullptr; }

  private:
    matvar_t *v_ = nullptr;
};

/// @brief Reads a top-level variable with its data; null when absent.
MatVar read_variable(const MatFile &file, const char *name);

std::size_t numel(const matvar_t *v);
bool is_empty(const matvar_t *v); // null, class empty, or no elements
std::string class_name(const matvar_t *v);
std::string dims_string(const matvar_t *v);

/// @brief The field of a struct (first element); null when absent. `what` names the struct in messages.
const matvar_t *struct_field(const matvar_t *s, const char *name, const std::string &what);
/// @brief Cell element i (column-major linear index); throws when out of range.
const matvar_t *cell_at(const matvar_t *c, std::size_t i, const std::string &what);

/// @brief A numeric (dense or sparse, real) array as doubles in column-major order; rejects complex and non-numeric.
std::vector<double> numeric_values(const matvar_t *v, const std::string &what);
/// @brief Numeric values that must be integers.
std::vector<std::int64_t> integer_values(const matvar_t *v, const std::string &what);
double scalar_value(const matvar_t *v, const std::string &what);
std::string string_value(const matvar_t *v, const std::string &what);

/// @brief A real sparse matrix (MATLAB CSC) with 64-bit indices and double values.
struct SparseMatrix
{
    std::int64_t rows = 0, cols = 0;
    std::vector<std::int64_t> colptr, rowind;
    std::vector<double> val;
};
SparseMatrix sparse_values(const matvar_t *v, const std::string &what);

/// @brief Calls f(row, col, value) for every nonzero of a dense or sparse real matrix (column by column).
void for_each_nonzero(const matvar_t *v, const std::string &what, const std::function<void(std::int64_t, std::int64_t, double)> &f);
/// @brief The number of nonzeros a for_each_nonzero visit would report at most (sparse: nnz; dense: numel).
std::size_t nonzero_capacity(const matvar_t *v);

// construction for writing (the returned variables are owned by the caller until written or attached)
matvar_t *make_double_matrix(const char *name, std::int64_t rows, std::int64_t cols, const double *data);
matvar_t *make_string(const char *name, const std::string &s);
matvar_t *make_sparse(const char *name, const SparseMatrix &m);
matvar_t *make_cell(const char *name, std::int64_t rows, std::int64_t cols, std::vector<matvar_t *> &cells); // takes the cells
matvar_t *make_struct(const char *name, std::vector<std::pair<std::string, matvar_t *>> &fields);        // takes the fields
/// @brief Writes and frees the variable (zlib-compressed); throws on failure.
void write_variable(const MatFile &file, matvar_t *var);

} // namespace detail
} // namespace io
} // namespace cuadmm

#endif // CUADMM_HAVE_MATIO
#endif // CUADMM_IO_MAT_COMMON_H
