/*

    linsys/normal_system.h

    NormalSystem (plan Section 11.3): the system of the y-step, (A A^T + eps I) y = r, described by the engine's
    scaled, row-normalized A^T in CSC on the host (32-bit indices, as the engine stores them in Milestone 1) and the
    regularization eps (1e-15 today). The lower triangle of A A^T + eps I is assembled on demand with cholmod_aat and
    cached; backends that work on the device may assemble it there instead (NormalOperator is matrix-free).

    The data pointers are borrowed: the caller keeps the arrays alive while the NormalSystem is used.

*/

#ifndef CUADMM_LINSYS_NORMAL_SYSTEM_H
#define CUADMM_LINSYS_NORMAL_SYSTEM_H

#include <cstdint>
#include <memory>
#include <vector>

namespace cuadmm
{
namespace linsys
{

/// A symmetric matrix by its lower triangle in CSC (column-major, row indices increasing within a column).
struct HostCscLower
{
    int n = 0;
    std::vector<int> colptr; // n + 1
    std::vector<int> rowind; // nnz
    std::vector<double> val; // nnz
    std::int64_t nnz() const { return colptr.empty() ? 0 : colptr.back(); }
};

class NormalSystem
{
public:
    /// A^T in CSC: n rows (vec_len) and m columns (constraints); nnz entries. eps: the regularization added to the
    /// diagonal. The arrays are borrowed.
    NormalSystem(int m, int n, const int *at_colptr, const int *at_rowind, const double *at_val, double eps = 1e-15);
    ~NormalSystem();
    NormalSystem(const NormalSystem &) = delete;
    NormalSystem &operator=(const NormalSystem &) = delete;

    int m() const { return m_; }
    int n() const { return n_; }
    double eps() const { return eps_; }
    const int *at_colptr() const { return at_colptr_; }
    const int *at_rowind() const { return at_rowind_; }
    const double *at_val() const { return at_val_; }
    std::int64_t nnz_at() const { return at_colptr_[m_]; }

    /// The lower triangle of A A^T + eps I (assembled on the first call with cholmod_aat, cached; throws on an
    /// allocation failure).
    const HostCscLower &lower() const;

    /// Applies y -> (A A^T + eps I) y on the host (reference for the tests; O(nnz(A))).
    void apply_host(const double *y, double *out) const;

private:
    int m_, n_;
    const int *at_colptr_;
    const int *at_rowind_;
    const double *at_val_;
    double eps_;
    mutable std::unique_ptr<HostCscLower> lower_;
};

} // namespace linsys
} // namespace cuadmm

#endif // CUADMM_LINSYS_NORMAL_SYSTEM_H
