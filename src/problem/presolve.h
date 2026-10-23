/*

    problem/presolve.h

    Presolve for dependent constraint rows (plan Section 11.7). Moment relaxations often carry constraints that are
    linear combinations of others (the SPOT data); today a tiny regularization (eps = 1e-15) plus the simplicial LDL^T
    factorization makes them work, fragilely (B9). The presolve detects the dependent rows from the pivots of the
    CHOLMOD factorization of A A^T + eps I (simplicial LDL^T, as cuadmm_legacy/cholesky_cpu.h), checks that they
    are reproduced by the kept rows and that b is consistent with them, removes them, and provides the RowMap that
    expands a reduced y (zeros on the removed rows) at the end. Pure host code; off during the bitwise phases.

    Method (presolve()):
      1. Factor S = A_K A_K^T + eps I for the kept rows K (all rows at first) with the simplicial LDL^T. Pivots
         |D_jj| < tiny_pivot_rel max |D| and, when CHOLMOD stops at an exactly zero pivot (eps = 0), the row at which
         it stopped, mark their rows (through the fill-reducing permutation) as dependent. Repeat on the kept rows
         until the factorization is clean (at most max_rounds rounds).
      2. With the final factor, solve S Z = A_K A_D^T for the dependent rows D. Row i of D is reproduced by the kept
         rows when ||a_i - A_K^T z_i|| <= span_tol ||a_i||; a row that is not (a nearly dependent row whose pivot was
         tiny) is reinstated as a kept row, protected from step 1, and the detection restarts.
      3. Consistency of b: the implied right-hand side of a removed row is z_i^T b_K; the rows are removed when
         max_i |b_i - z_i^T b_K| <= consistency_tol max(1, ||b||_inf), otherwise the problem is inconsistent (no
         x satisfies both rows) and the presolve relaxes: all rows are kept and the result says so (the ADMM then
         runs on the original problem with the regularization, as today).

    Cost: a few CHOLMOD factorizations of the kept system and one multiple-right-hand-side solve with as many columns
    as dependent rows (batched; the explicit residual A_K^T Z costs nnz(A_K) per dependent row).

*/

#ifndef CUADMM_PROBLEM_PRESOLVE_H
#define CUADMM_PROBLEM_PRESOLVE_H

#include <cstddef>
#include <string>
#include <vector>

namespace cuadmm
{
namespace problem
{

struct PresolveOptions
{
    double eps = 1e-15;             // regularization of A A^T + eps I in the detection factorization
    double tiny_pivot_rel = 1e-12;  // |D_jj| < tiny_pivot_rel * max |D| classifies the row as dependent
    double span_tol = 1e-8;         // a dependent row must be reproduced by the kept rows to this relative accuracy
    double consistency_tol = 1e-8;  // |b_i - z_i^T b_K| <= consistency_tol * max(1, ||b||_inf) for every removed row
    int max_rounds = 20;            // detection rounds (refactorizations) before the presolve gives up (relaxes)
    bool relax_on_failure = true;   // keep all rows when a check fails; false: throw std::runtime_error
};

/// The correspondence between the full and the reduced constraint rows.
struct RowMap
{
    int m_full = 0, m_reduced = 0;
    std::vector<int> kept;       // reduced index k -> full row kept[k] (increasing)
    std::vector<int> removed;    // the removed full rows (increasing)
    std::vector<int> to_reduced; // full row -> reduced index, -1 for a removed row

    static RowMap identity(int m);
    static RowMap from_removed(int m, const std::vector<int> &removed_rows);
    bool is_identity() const { return m_full == m_reduced; }
    /// y_full[kept[k]] = y_reduced[k]; zero on the removed rows.
    void expand(const double *y_reduced, double *y_full) const;
    std::vector<double> expand(const std::vector<double> &y_reduced) const;
    /// v_reduced[k] = v_full[kept[k]].
    void restrict(const double *v_full, double *v_reduced) const;
    std::vector<double> restrict(const std::vector<double> &v_full) const;
};

struct PresolveResult
{
    // the reduced problem: A^T (n x m) in CSC with 32-bit indices and b (size m); copies of the input when nothing
    // was removed
    int m = 0, n = 0;
    std::vector<int> at_colptr, at_rowind;
    std::vector<double> at_val, b;
    RowMap map;

    int dependent_rows = 0;            // rows the pivots classified as dependent (final detection round)
    int removed_rows = 0;              // rows removed in the end (0 when relaxed or nothing was dependent)
    int reinstated_rows = 0;           // rows the span check gave back (tiny pivot, not reproduced by the kept rows)
    bool consistent = true;            // the consistency check of b passed (true when nothing was removed)
    bool relaxed = false;              // the fallback: all rows kept because a check failed or the detection did not settle
    double span_residual = 0.0;        // max over the removed rows of ||a_i - A_K^T z_i|| / ||a_i||
    double consistency_residual = 0.0; // max over the removed rows of |b_i - z_i^T b_K| / max(1, ||b||_inf)
    double min_pivot = 0.0, max_pivot = 0.0; // pivots D_jj of the first factorization (all rows)
    int tiny_pivots = 0, nonpositive_pivots = 0;
    int rounds = 0;                    // factorizations performed
    std::string message;               // one line for the log

    bool changed() const { return removed_rows > 0; }
};

/// A^T in CSC (n rows, m columns, 32-bit indices), b of size m. Throws std::invalid_argument on malformed input and
/// std::runtime_error when CHOLMOD fails or (relax_on_failure = false) a check fails.
PresolveResult presolve(int m, int n, const int *at_colptr, const int *at_rowind, const double *at_val, const double *b,
                        const PresolveOptions &options = PresolveOptions());

} // namespace problem
} // namespace cuadmm

#endif // CUADMM_PROBLEM_PRESOLVE_H
