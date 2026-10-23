/*

    problem/presolve.cc

*/

#include "problem/presolve.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <cholmod.h>

namespace cuadmm
{
namespace problem
{

// ------------------------------------------------------------------------------------------------------ RowMap

RowMap RowMap::identity(int m)
{
    RowMap map;
    map.m_full = map.m_reduced = m;
    map.kept.resize(static_cast<std::size_t>(m));
    std::iota(map.kept.begin(), map.kept.end(), 0);
    map.to_reduced = map.kept;
    return map;
}

RowMap RowMap::from_removed(int m, const std::vector<int> &removed_rows)
{
    RowMap map;
    map.m_full = m;
    std::vector<char> is_removed(std::size_t(m), 0);
    for (int r : removed_rows)
    {
        if (r < 0 || r >= m)
            throw std::invalid_argument("RowMap::from_removed: row " + std::to_string(r) + " out of range");
        is_removed[std::size_t(r)] = 1;
    }
    map.to_reduced.assign(std::size_t(m), -1);
    for (int i = 0; i < m; i++)
    {
        if (is_removed[std::size_t(i)])
            map.removed.push_back(i);
        else
        {
            map.to_reduced[std::size_t(i)] = int(map.kept.size());
            map.kept.push_back(i);
        }
    }
    map.m_reduced = int(map.kept.size());
    return map;
}

void RowMap::expand(const double *y_reduced, double *y_full) const
{
    std::fill(y_full, y_full + m_full, 0.0);
    for (int k = 0; k < m_reduced; k++)
        y_full[kept[std::size_t(k)]] = y_reduced[k];
}

std::vector<double> RowMap::expand(const std::vector<double> &y_reduced) const
{
    if (int(y_reduced.size()) != m_reduced)
        throw std::invalid_argument("RowMap::expand: the reduced vector has " + std::to_string(y_reduced.size()) + " entries, the map " +
                                    std::to_string(m_reduced));
    std::vector<double> y(static_cast<std::size_t>(m_full));
    expand(y_reduced.data(), y.data());
    return y;
}

void RowMap::restrict(const double *v_full, double *v_reduced) const
{
    for (int k = 0; k < m_reduced; k++)
        v_reduced[k] = v_full[kept[std::size_t(k)]];
}

std::vector<double> RowMap::restrict(const std::vector<double> &v_full) const
{
    if (int(v_full.size()) != m_full)
        throw std::invalid_argument("RowMap::restrict: the full vector has " + std::to_string(v_full.size()) + " entries, the map " +
                                    std::to_string(m_full));
    std::vector<double> v(static_cast<std::size_t>(m_reduced));
    restrict(v_full.data(), v.data());
    return v;
}

// ---------------------------------------------------------------------------------------------------- presolve

namespace
{

struct CholmodScope
{
    cholmod_common cc;
    CholmodScope()
    {
        cholmod_start(&cc);
        cc.print = 0; // failures are handled, not printed (the "not positive definite" message of level 1)
        cc.supernodal = CHOLMOD_SIMPLICIAL; // the simplicial LDL^T: pivots D_jj, as cholesky_cpu.h's fallback
        cc.final_asis = false;
        cc.final_super = false;
        cc.final_ll = false;
    }
    ~CholmodScope() { cholmod_finish(&cc); }
    CholmodScope(const CholmodScope &) = delete;
    CholmodScope &operator=(const CholmodScope &) = delete;
};

template <typename T, int (*Free)(T **, cholmod_common *)>
struct Holder
{
    T *p = nullptr;
    cholmod_common *cc = nullptr;
    Holder() = default;
    Holder(T *q, cholmod_common *c) : p(q), cc(c) {}
    ~Holder() { reset(); }
    void reset(T *q = nullptr)
    {
        if (p != nullptr)
            Free(&p, cc);
        p = q;
    }
    Holder(const Holder &) = delete;
    Holder &operator=(const Holder &) = delete;
};
using SparseHolder = Holder<cholmod_sparse, cholmod_free_sparse>;
using FactorHolder = Holder<cholmod_factor, cholmod_free_factor>;
using DenseHolder = Holder<cholmod_dense, cholmod_free_dense>;

// the kept columns of A^T (n x m CSC) as a cholmod_sparse (n x |kept|)
cholmod_sparse *kept_columns(int n, const int *colptr, const int *rowind, const double *val, const std::vector<int> &kept, cholmod_common *cc)
{
    std::size_t nnz = 0;
    for (int j : kept)
        nnz += std::size_t(colptr[j + 1] - colptr[j]);
    cholmod_sparse *S = cholmod_allocate_sparse(std::size_t(n), kept.size(), nnz, 1, 1, 0, CHOLMOD_DOUBLE + CHOLMOD_REAL, cc);
    if (S == nullptr)
        throw std::runtime_error("presolve: cholmod_allocate_sparse failed (" + std::to_string(nnz) + " nonzeros)");
    int *Sp = static_cast<int *>(S->p);
    int *Si = static_cast<int *>(S->i);
    double *Sx = static_cast<double *>(S->x);
    std::size_t pos = 0;
    bool sorted = true;
    for (std::size_t c = 0; c < kept.size(); c++)
    {
        const int j = kept[c];
        Sp[c] = int(pos);
        for (int k = colptr[j]; k < colptr[j + 1]; k++)
        {
            if (k > colptr[j] && rowind[k] <= rowind[k - 1])
                sorted = false;
            Si[pos] = rowind[k];
            Sx[pos] = val[k];
            pos++;
        }
    }
    Sp[kept.size()] = int(pos);
    S->sorted = sorted ? 1 : 0;
    return S;
}

// S = A_K A_K^T + eps I (stype 1, upper) from A_K^T (n x |K|)
cholmod_sparse *normal_matrix(cholmod_sparse *AtK, double eps, cholmod_common *cc)
{
    SparseHolder A(cholmod_transpose(AtK, 1, cc), cc);
    if (A.p == nullptr)
        throw std::runtime_error("presolve: cholmod_transpose failed");
    SparseHolder AAt(cholmod_aat(A.p, nullptr, 0, 1, cc), cc);
    if (AAt.p == nullptr)
        throw std::runtime_error("presolve: cholmod_aat failed");
    SparseHolder I(cholmod_speye(AtK->ncol, AtK->ncol, CHOLMOD_DOUBLE + CHOLMOD_REAL, cc), cc);
    double alpha = 1.0, beta = eps;
    SparseHolder Sum(cholmod_add(AAt.p, I.p, &alpha, &beta, 1, 1, cc), cc);
    if (Sum.p == nullptr)
        throw std::runtime_error("presolve: cholmod_add failed");
    cholmod_sparse *S = cholmod_copy(Sum.p, 1, 1, cc);
    if (S == nullptr)
        throw std::runtime_error("presolve: cholmod_copy failed");
    return S;
}

struct PivotScan
{
    double min_pivot = 0.0, max_pivot = 0.0, max_abs = 0.0;
    int tiny = 0, nonpositive = 0;
    std::vector<int> tiny_rows; // the kept-index rows with tiny pivots (and the zero-pivot row of a failed factorization)
    bool clean = true;          // the factorization completed with no tiny pivot
};

// factors S (simplicial LDL^T) and scans the pivots; L->minor < n means that CHOLMOD stopped at a zero pivot
PivotScan factor_and_scan(cholmod_sparse *S, cholmod_factor **L_out, double tiny_rel, cholmod_common *cc)
{
    cc->status = CHOLMOD_OK;
    cholmod_factor *L = cholmod_analyze(S, cc);
    if (L == nullptr)
        throw std::runtime_error("presolve: cholmod_analyze failed (status " + std::to_string(cc->status) + ")");
    cholmod_factorize(S, L, cc);
    if (cc->status != CHOLMOD_OK && cc->status != CHOLMOD_NOT_POSDEF)
    {
        cholmod_free_factor(&L, cc);
        throw std::runtime_error("presolve: cholmod_factorize failed (status " + std::to_string(cc->status) + ")");
    }
    if (L->is_super || L->is_ll)
    {
        cholmod_free_factor(&L, cc);
        throw std::runtime_error("presolve: the factor is not the simplicial LDL^T");
    }
    const int n = int(L->n);
    const int minor = cc->status == CHOLMOD_NOT_POSDEF ? int(L->minor) : n; // pivots j < minor are valid
    const int *Lp = static_cast<const int *>(L->p);
    const double *Lx = static_cast<const double *>(L->x);
    const int *perm = static_cast<const int *>(L->Perm);
    PivotScan scan;
    if (minor > 0)
    {
        scan.min_pivot = scan.max_pivot = Lx[Lp[0]];
        for (int j = 0; j < minor; j++)
        {
            const double d = Lx[Lp[j]];
            scan.min_pivot = std::min(scan.min_pivot, d);
            scan.max_pivot = std::max(scan.max_pivot, d);
            scan.max_abs = std::max(scan.max_abs, std::abs(d));
            if (d <= 0.0)
                scan.nonpositive++;
        }
        for (int j = 0; j < minor; j++)
            if (std::abs(Lx[Lp[j]]) < tiny_rel * scan.max_abs)
            {
                scan.tiny++;
                scan.tiny_rows.push_back(perm[j]);
            }
    }
    if (minor < n)
    {
        // the zero pivot: its row is dependent on the rows eliminated before it (exactly, with eps = 0)
        scan.tiny_rows.push_back(perm[minor]);
        scan.clean = false;
    }
    if (scan.tiny > 0)
        scan.clean = false;
    cc->status = CHOLMOD_OK;
    *L_out = L;
    return scan;
}

std::string format(double v)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2e", v);
    return buf;
}

} // namespace

PresolveResult presolve(int m, int n, const int *at_colptr, const int *at_rowind, const double *at_val, const double *b,
                        const PresolveOptions &options)
{
    if (m < 0 || n < 0 || at_colptr == nullptr || b == nullptr || (at_colptr[m] > 0 && (at_rowind == nullptr || at_val == nullptr)))
        throw std::invalid_argument("presolve: malformed input");
    if (!(options.eps >= 0.0))
        throw std::invalid_argument("presolve: eps must be nonnegative");
    for (int j = 0; j < m; j++)
        for (int k = at_colptr[j]; k < at_colptr[j + 1]; k++)
            if (at_rowind[k] < 0 || at_rowind[k] >= n)
                throw std::invalid_argument("presolve: row index out of range in column " + std::to_string(j));

    PresolveResult result;
    result.m = m;
    result.n = n;

    auto untouched = [&](const std::string &message) {
        result.at_colptr.assign(at_colptr, at_colptr + m + 1);
        result.at_rowind.assign(at_rowind, at_rowind + at_colptr[m]);
        result.at_val.assign(at_val, at_val + at_colptr[m]);
        result.b.assign(b, b + m);
        result.map = RowMap::identity(m);
        result.removed_rows = 0;
        result.message = message;
        return result;
    };
    auto fail = [&](const std::string &why) {
        if (!options.relax_on_failure)
            throw std::runtime_error("presolve: " + why);
        result.relaxed = true;
        return untouched("presolve: " + why + ": all " + std::to_string(m) + " rows kept (relaxed)");
    };

    if (m == 0)
        return untouched("presolve: no constraints");

    CholmodScope scope;
    cholmod_common *cc = &scope.cc;

    double b_inf = 0.0;
    for (int i = 0; i < m; i++)
        b_inf = std::max(b_inf, std::abs(b[i]));
    const double b_scale = std::max(1.0, b_inf);

    std::vector<char> dependent(std::size_t(m), 0), protect(std::size_t(m), 0);
    std::vector<int> kept;
    SparseHolder AtK(nullptr, cc);
    FactorHolder L(nullptr, cc);
    bool first = true;
    int rounds = 0;
    for (int outer = 0; outer < options.max_rounds; outer++)
    {
        // step 1: detection rounds on the kept rows until the factorization is clean
        bool clean = false;
        while (rounds < options.max_rounds)
        {
            kept.clear();
            for (int i = 0; i < m; i++)
                if (!dependent[std::size_t(i)])
                    kept.push_back(i);
            AtK.reset(kept_columns(n, at_colptr, at_rowind, at_val, kept, cc));
            SparseHolder S(normal_matrix(AtK.p, options.eps, cc), cc);
            cholmod_factor *Lraw = nullptr;
            const PivotScan scan = factor_and_scan(S.p, &Lraw, options.tiny_pivot_rel, cc);
            L.reset(Lraw);
            rounds++;
            if (first)
            {
                result.min_pivot = scan.min_pivot;
                result.max_pivot = scan.max_pivot;
                result.tiny_pivots = scan.tiny;
                result.nonpositive_pivots = scan.nonpositive;
                first = false;
            }
            int newly = 0;
            for (int kidx : scan.tiny_rows)
            {
                const int row = kept[std::size_t(kidx)];
                if (!protect[std::size_t(row)] && !dependent[std::size_t(row)])
                {
                    dependent[std::size_t(row)] = 1;
                    newly++;
                }
            }
            if (scan.clean || newly == 0)
            {
                clean = true; // clean, or only protected rows have tiny pivots (regularized by eps, as today)
                break;
            }
        }
        if (!clean)
            return fail("the dependent-row detection did not settle in " + std::to_string(options.max_rounds) + " factorizations");

        std::vector<int> dep_rows;
        for (int i = 0; i < m; i++)
            if (dependent[std::size_t(i)])
                dep_rows.push_back(i);
        result.dependent_rows = int(dep_rows.size());
        result.rounds = rounds;
        if (dep_rows.empty())
        {
            result.consistent = true;
            return untouched("presolve: all " + std::to_string(m) + " rows independent (pivots in [" + format(result.min_pivot) + ", " +
                             format(result.max_pivot) + "], " + std::to_string(rounds) + " factorization(s))");
        }

        // step 2 and 3: Z = S^{-1} A_K A_D^T in batches; span residual and the implied right-hand side per dependent row
        const int mk = int(kept.size());
        SparseHolder AK(cholmod_transpose(AtK.p, 1, cc), cc); // A_K (mk x n) in CSC: column r = the entries of variable r
        if (AK.p == nullptr)
            throw std::runtime_error("presolve: cholmod_transpose failed");
        const int *AKp = static_cast<const int *>(AK.p->p);
        const int *AKi = static_cast<const int *>(AK.p->i);
        const double *AKx = static_cast<const double *>(AK.p->x);
        const int *AtKp = static_cast<const int *>(AtK.p->p);
        const int *AtKi = static_cast<const int *>(AtK.p->i);
        const double *AtKx = static_cast<const double *>(AtK.p->x);
        std::vector<double> bK(static_cast<std::size_t>(mk));
        for (int k = 0; k < mk; k++)
            bK[std::size_t(k)] = b[kept[std::size_t(k)]];

        const int batch = 64;
        double span_residual = 0.0, consistency_residual = 0.0;
        std::vector<int> reinstate;
        std::vector<double> t(static_cast<std::size_t>(n));
        for (std::size_t start = 0; start < dep_rows.size(); start += batch)
        {
            const int cols = int(std::min<std::size_t>(batch, dep_rows.size() - start));
            DenseHolder R(cholmod_zeros(std::size_t(mk), std::size_t(cols), CHOLMOD_REAL, cc), cc);
            if (R.p == nullptr)
                throw std::runtime_error("presolve: cholmod_zeros failed");
            double *Rx = static_cast<double *>(R.p->x);
            // R(:, c) = A_K a_i
            for (int c = 0; c < cols; c++)
            {
                const int i = dep_rows[start + std::size_t(c)];
                double *col = Rx + std::size_t(c) * std::size_t(mk);
                for (int k = at_colptr[i]; k < at_colptr[i + 1]; k++)
                {
                    const int r = at_rowind[k];
                    const double v = at_val[k];
                    for (int q = AKp[r]; q < AKp[r + 1]; q++)
                        col[AKi[q]] += v * AKx[q];
                }
            }
            DenseHolder Z(cholmod_solve(CHOLMOD_A, L.p, R.p, cc), cc);
            if (Z.p == nullptr)
                throw std::runtime_error("presolve: cholmod_solve failed (status " + std::to_string(cc->status) + ")");
            const double *Zx = static_cast<const double *>(Z.p->x);
            for (int c = 0; c < cols; c++)
            {
                const int i = dep_rows[start + std::size_t(c)];
                const double *z = Zx + std::size_t(c) * std::size_t(mk);
                // t = A_K^T z (n), then the residual against a_i
                std::fill(t.begin(), t.end(), 0.0);
                for (int k = 0; k < mk; k++)
                {
                    const double zk = z[k];
                    if (zk == 0.0)
                        continue;
                    for (int q = AtKp[k]; q < AtKp[k + 1]; q++)
                        t[std::size_t(AtKi[q])] += AtKx[q] * zk;
                }
                double a_norm2 = 0.0;
                for (int k = at_colptr[i]; k < at_colptr[i + 1]; k++)
                {
                    const double v = at_val[k];
                    a_norm2 += v * v;
                    t[std::size_t(at_rowind[k])] -= v;
                }
                double res2 = 0.0;
                for (double tv : t)
                    res2 += tv * tv;
                const double rel = a_norm2 > 0.0 ? std::sqrt(res2 / a_norm2) : std::sqrt(res2);
                if (rel > options.span_tol)
                    reinstate.push_back(i);
                else
                {
                    span_residual = std::max(span_residual, rel);
                    double implied = 0.0;
                    for (int k = 0; k < mk; k++)
                        implied += z[k] * bK[std::size_t(k)];
                    consistency_residual = std::max(consistency_residual, std::abs(b[i] - implied) / b_scale);
                }
            }
        }
        if (!reinstate.empty())
        {
            // nearly dependent rows: kept, protected from the pivot test, and the detection restarts
            for (int i : reinstate)
            {
                dependent[std::size_t(i)] = 0;
                protect[std::size_t(i)] = 1;
            }
            result.reinstated_rows += int(reinstate.size());
            continue;
        }
        result.span_residual = span_residual;
        result.consistency_residual = consistency_residual;
        result.consistent = consistency_residual <= options.consistency_tol;
        if (!result.consistent)
            return fail(std::to_string(dep_rows.size()) + " of " + std::to_string(m) + " rows dependent but b is inconsistent with them (residual " +
                        format(consistency_residual) + " > " + format(options.consistency_tol) + ")");

        // the reduced problem
        result.map = RowMap::from_removed(m, dep_rows);
        result.removed_rows = int(dep_rows.size());
        result.at_colptr.assign(std::size_t(mk) + 1, 0);
        result.at_rowind.clear();
        result.at_val.clear();
        result.b.assign(std::size_t(mk), 0.0);
        for (int k = 0; k < mk; k++)
        {
            const int j = kept[std::size_t(k)];
            result.at_rowind.insert(result.at_rowind.end(), at_rowind + at_colptr[j], at_rowind + at_colptr[j + 1]);
            result.at_val.insert(result.at_val.end(), at_val + at_colptr[j], at_val + at_colptr[j + 1]);
            result.at_colptr[std::size_t(k) + 1] = int(result.at_rowind.size());
            result.b[std::size_t(k)] = b[j];
        }
        result.m = mk;
        result.message = "presolve: " + std::to_string(dep_rows.size()) + " of " + std::to_string(m) + " rows dependent and removed (span residual " +
                         format(span_residual) + ", consistency residual " + format(consistency_residual) + ", " + std::to_string(rounds) +
                         " factorization(s)" + (result.reinstated_rows > 0 ? ", " + std::to_string(result.reinstated_rows) + " reinstated" : "") + ")";
        return result;
    }
    return fail("the dependent-row detection did not settle in " + std::to_string(options.max_rounds) + " rounds");
}

} // namespace problem
} // namespace cuadmm
