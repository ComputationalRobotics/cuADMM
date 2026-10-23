/*

    linsys/normal_system.cc

    NormalSystem::lower() assembles the lower triangle of A A^T + eps I on the host with CHOLMOD, the way the ported
    engine's wrapper (cuadmm_legacy/cholesky_cpu.h) does: cholmod_aat on A in CSC (which is the transpose of the
    A^T in CSC the system holds), then cholmod_add with eps * I so that every diagonal entry exists, then the lower
    triangle (stype -1). The result is normalized so that every column starts with its diagonal entry and the row
    indices increase: the dense backend scatters it and the cuDSS backend transposes it to the CSR lower triangle.

    Also here: relative_residual_host() and the default NormalEquationSolver::update() of normal_equation_solver.h.

*/

#include "linsys/normal_system.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cholmod.h>

#include "core/check.h"
#include "linsys/normal_equation_solver.h"

namespace cuadmm
{
namespace linsys
{

namespace
{

// cholmod_common with the lifetime of a scope
struct CholmodScope
{
    cholmod_common cc;
    CholmodScope() { cholmod_start(&cc); }
    ~CholmodScope() { cholmod_finish(&cc); }
    CholmodScope(const CholmodScope &) = delete;
    CholmodScope &operator=(const CholmodScope &) = delete;
};

// a cholmod_sparse freed at the end of the scope
struct SparseHolder
{
    cholmod_sparse *p = nullptr;
    cholmod_common *cc = nullptr;
    SparseHolder(cholmod_sparse *s, cholmod_common *c) : p(s), cc(c) {}
    ~SparseHolder()
    {
        if (p != nullptr)
            cholmod_free_sparse(&p, cc);
    }
    SparseHolder(const SparseHolder &) = delete;
    SparseHolder &operator=(const SparseHolder &) = delete;
};

} // namespace

NormalSystem::NormalSystem(int m, int n, const int *at_colptr, const int *at_rowind, const double *at_val, double eps)
    : m_(m), n_(n), at_colptr_(at_colptr), at_rowind_(at_rowind), at_val_(at_val), eps_(eps)
{
    if (m < 0 || n < 0)
        throw std::invalid_argument("NormalSystem: negative dimensions (m = " + std::to_string(m) + ", n = " + std::to_string(n) + ")");
    if (at_colptr == nullptr)
        throw std::invalid_argument("NormalSystem: the column pointers of A^T are null");
    if (at_colptr[0] != 0)
        throw std::invalid_argument("NormalSystem: the column pointers of A^T must start at 0");
    if (at_colptr[m] > 0 && (at_rowind == nullptr || at_val == nullptr))
        throw std::invalid_argument("NormalSystem: the row indices or values of A^T are null");
    if (!(eps >= 0.0))
        throw std::invalid_argument("NormalSystem: eps must be nonnegative and finite");
}

NormalSystem::~NormalSystem() = default;

const HostCscLower &NormalSystem::lower() const
{
    if (lower_)
        return *lower_;

    const std::int64_t nnz = nnz_at();
    CholmodScope scope;
    cholmod_common *cc = &scope.cc;
    cc->print = 0; // errors are reported by the exceptions below

    // A^T in CSC (n x m), as given; sorted when the row indices increase in every column
    bool sorted = true;
    for (int j = 0; j < m_ && sorted; j++)
        for (int k = at_colptr_[j] + 1; k < at_colptr_[j + 1]; k++)
            if (at_rowind_[k] <= at_rowind_[k - 1])
            {
                sorted = false;
                break;
            }
    SparseHolder At(cholmod_allocate_sparse(std::size_t(n_), std::size_t(m_), std::size_t(nnz), sorted ? 1 : 0, 1, 0,
                                            CHOLMOD_DOUBLE + CHOLMOD_REAL, cc),
                    cc);
    if (At.p == nullptr)
        throw std::runtime_error("NormalSystem::lower: cholmod_allocate_sparse failed for A^T (" + std::to_string(nnz) +
                                 " nonzeros): out of memory");
    std::memcpy(At.p->p, at_colptr_, sizeof(int) * std::size_t(m_ + 1));
    if (nnz > 0)
    {
        std::memcpy(At.p->i, at_rowind_, sizeof(int) * std::size_t(nnz));
        std::memcpy(At.p->x, at_val_, sizeof(double) * std::size_t(nnz));
    }

    // A (m x n) = (A^T)^T, then A A^T with cholmod_aat (numerical, diagonal kept), as cholesky_cpu.h does on the
    // engine's CSC of A
    SparseHolder A(cholmod_transpose(At.p, 1, cc), cc);
    if (A.p == nullptr)
        throw std::runtime_error("NormalSystem::lower: cholmod_transpose failed (CHOLMOD status " + std::to_string(cc->status) + ")");
    SparseHolder AAt(cholmod_aat(A.p, nullptr, 0, 1, cc), cc);
    if (AAt.p == nullptr)
        throw std::runtime_error("NormalSystem::lower: cholmod_aat failed (CHOLMOD status " + std::to_string(cc->status) + ")");

    // + eps I: every diagonal entry present, columns sorted
    SparseHolder I(cholmod_speye(std::size_t(m_), std::size_t(m_), CHOLMOD_DOUBLE + CHOLMOD_REAL, cc), cc);
    if (I.p == nullptr)
        throw std::runtime_error("NormalSystem::lower: cholmod_speye failed");
    double alpha = 1.0, beta = eps_;
    SparseHolder S(cholmod_add(AAt.p, I.p, &alpha, &beta, 1, 1, cc), cc);
    if (S.p == nullptr)
        throw std::runtime_error("NormalSystem::lower: cholmod_add failed (CHOLMOD status " + std::to_string(cc->status) + ")");
    // the lower triangle (stype -1)
    SparseHolder L(cholmod_copy(S.p, -1, 1, cc), cc);
    if (L.p == nullptr)
        throw std::runtime_error("NormalSystem::lower: cholmod_copy failed (CHOLMOD status " + std::to_string(cc->status) + ")");

    // normalize: the diagonal first in every column, row indices increasing; the rare missing diagonal (eps = 0 and an
    // empty row, if cholmod_add dropped the zero) is inserted as an explicit zero
    const int *Lp = static_cast<const int *>(L.p->p);
    const int *Li = static_cast<const int *>(L.p->i);
    const double *Lx = static_cast<const double *>(L.p->x);
    auto out = std::make_unique<HostCscLower>();
    out->n = m_;
    out->colptr.assign(std::size_t(m_) + 1, 0);
    out->rowind.reserve(std::size_t(Lp[m_]) + std::size_t(m_));
    out->val.reserve(std::size_t(Lp[m_]) + std::size_t(m_));
    std::vector<std::pair<int, double>> column;
    for (int j = 0; j < m_; j++)
    {
        column.clear();
        bool has_diagonal = false;
        for (int k = Lp[j]; k < Lp[j + 1]; k++)
        {
            if (Li[k] < j)
                continue; // not in the lower triangle (cannot happen with stype -1; defensive)
            has_diagonal = has_diagonal || Li[k] == j;
            column.emplace_back(Li[k], Lx[k]);
        }
        if (!has_diagonal)
            column.emplace_back(j, 0.0);
        std::sort(column.begin(), column.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) { return a.first < b.first; });
        for (const auto &e : column)
        {
            out->rowind.push_back(e.first);
            out->val.push_back(e.second);
        }
        out->colptr[std::size_t(j) + 1] = int(out->rowind.size());
    }
    lower_ = std::move(out);
    return *lower_;
}

void NormalSystem::apply_host(const double *y, double *out) const
{
    // v = A^T y (n), then out = A v + eps y
    std::vector<double> v(std::size_t(n_), 0.0);
    for (int j = 0; j < m_; j++)
    {
        const double yj = y[j];
        if (yj == 0.0)
            continue;
        for (int k = at_colptr_[j]; k < at_colptr_[j + 1]; k++)
            v[std::size_t(at_rowind_[k])] += at_val_[k] * yj;
    }
    for (int j = 0; j < m_; j++)
    {
        double s = eps_ * y[j];
        for (int k = at_colptr_[j]; k < at_colptr_[j + 1]; k++)
            s += at_val_[k] * v[std::size_t(at_rowind_[k])];
        out[j] = s;
    }
}

void NormalEquationSolver::update(const NormalSystem &)
{
    throw std::logic_error("NormalEquationSolver::update is not supported by " + std::string(name()) +
                           " (the system of the y-step does not depend on sigma; plan Section 11.3)");
}

double relative_residual_host(const NormalSystem &sys, const double *d_rhs, const double *d_y, cudaStream_t stream)
{
    const std::size_t m = std::size_t(sys.m());
    std::vector<double> rhs(m), y(m), r(m);
    if (m > 0)
    {
        CUADMM_CHECK_CUDA(cudaMemcpyAsync(rhs.data(), d_rhs, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
        CUADMM_CHECK_CUDA(cudaMemcpyAsync(y.data(), d_y, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
    }
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(stream));
    sys.apply_host(y.data(), r.data());
    double num = 0.0, den = 0.0;
    for (std::size_t j = 0; j < m; j++)
    {
        num += (r[j] - rhs[j]) * (r[j] - rhs[j]);
        den += rhs[j] * rhs[j];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

} // namespace linsys
} // namespace cuadmm
