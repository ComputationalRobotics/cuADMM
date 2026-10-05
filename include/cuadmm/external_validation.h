/*

    external_validation.h

    Independent validation of an SDP iterate, used by solve() for validation-aware stopping.

    The metrics are recomputed on the CPU, in double precision (long double accumulation), from the ORIGINAL problem
    data given to SDPSolver::init() and the UNSCALED iterate. They use none of the solver's GPU residuals or
    eigendecompositions. The definitions are those of the standalone validator
    (experiments/2026-09-27_paper_replication_2xh200/validator/validate.cpp), plus the per-block X-cone criterion:

        primal residual     ||A(X) - b|| / (1 + ||b||)
        dual residual       ||A^T(y) + S - C|| / (1 + ||C||)
        relative gap        |<C,X> - <b,y>| / (1 + |<C,X>| + |<b,y>|)                          (eq. 27)
        X cone violation    max_b max(0, -lambda_min(X_b)) / (1 + ||X_b||_F)                    (success criterion)
        dual cone violation max_b max(0, -lambda_min(Z_b)) / (1 + ||C||),  Z = C - A^T y
        S cone violation    max_b max(0, -lambda_min(S_b)) / (1 + ||C||)                        (S as returned)
        (also the earlier X-cone measure max_b max(0, -lambda_min(X_b)) / (1 + ||b||))

    and the six error measures of the 7th DIMACS Challenge (Mittelmann, Math. Prog. 95 (2003); plato.asu.edu/dimacs/
    node3.html) for min <C,X> s.t. A(X) = b, X in K, with z = S:
        err1 = ||A(X) - b||_2 / (1 + ||b||_inf)          err2 = max(0, -lambda_min,K(X)) / (1 + ||b||_inf)
        err3 = ||A^T y + S - C||_K / (1 + ||C||_inf)     err4 = max(0, -lambda_min,K(S)) / (1 + ||C||_inf)
        err5 = (<C,X> - b^T y) / (1 + |<C,X>| + |b^T y|) err6 = <X,S> / (1 + |<C,X>| + |b^T y|)
    ||.||_K is the DIMACS norm (sum over PSD blocks of Frobenius norms + 2-norm of the linear part), ||.||_inf the
    largest absolute component (DIMACS's "||.||_1"; for C taken over matrix entries: the svec sqrt(2) is undone),
    lambda_min,K the minimum over PSD-block eigenvalues and linear components. The block kinds used for these norms
    are those of the original data (set_dimacs_kinds(); by default 's' blocks are PSD and 'l'/'u' blocks linear).
    Also the variant with z = C - A^T y (err4_z, err6_z; its err3 is 0).

    Eigenvalues come from LAPACK dsyevd on each PSD block. For 'l' blocks the smallest entry is used; 'u' blocks have
    no cone. An iterate is validated only if all its entries are finite and every criterion is <= its tolerance. The
    comparisons fail closed: a NaN metric never passes.

*/

#ifndef CUADMM_EXTERNAL_VALIDATION_H
#define CUADMM_EXTERNAL_VALIDATION_H

#include <limits>
#include <string>
#include <vector>

struct ValidationTolerances
{
    double primal = 1e-4;
    double dual = 1e-4;
    double gap = 1e-4;
    double cone = 1e-4;      // X cone violation (per-block Frobenius normalization)
    double dual_cone = 1e-4; // dual-cone violation of Z = C - A^T y
    double s_cone = 1e-4;    // cone violation of the returned S

    // the same tolerance for every criterion
    static ValidationTolerances all(double tol)
    {
        ValidationTolerances t;
        t.primal = t.dual = t.gap = t.cone = t.dual_cone = t.s_cone = tol;
        return t;
    }
};

struct ValidationResult
{
    bool finite = false;                  // all entries of X, y, S are finite
    double primal_res = std::numeric_limits<double>::quiet_NaN();
    double dual_res = std::numeric_limits<double>::quiet_NaN();
    double relgap = std::numeric_limits<double>::quiet_NaN();
    double kkt_full = std::numeric_limits<double>::quiet_NaN(); // max(primal_res, dual_res, relgap)
    double pobj = std::numeric_limits<double>::quiet_NaN();
    double dobj = std::numeric_limits<double>::quiet_NaN();
    double X_cone_violation = std::numeric_limits<double>::quiet_NaN();       // per-block Frobenius normalization
    double X_cone_violation_normb = std::numeric_limits<double>::quiet_NaN(); // normalized by 1 + ||b||
    double dual_cone_violation = std::numeric_limits<double>::quiet_NaN();
    double S_cone_violation = std::numeric_limits<double>::quiet_NaN();
    double min_eig_X = std::numeric_limits<double>::quiet_NaN();
    double min_eig_Z = std::numeric_limits<double>::quiet_NaN();
    double min_eig_S = std::numeric_limits<double>::quiet_NaN();
    // DIMACS error measures (see above); dimacs[0] = err1 ... dimacs[5] = err6
    double dimacs[6] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    double max_abs_dimacs = std::numeric_limits<double>::quiet_NaN(); // max_i |err_i| (NaN if any is not finite)
    bool err6_defined = false;                                         // err2 == err4 == 0 (DIMACS defines err6 then)
    double err4_z = std::numeric_limits<double>::quiet_NaN();          // variant z = C - A^T y
    double err6_z = std::numeric_limits<double>::quiet_NaN();
    int X_worst_block = -1;
    int X_negative_blocks = 0;
    double certificate_lower_bound_conditional = std::numeric_limits<double>::quiet_NaN();
    bool validated = false;
    std::string failed; // comma-separated names of the failed criteria ("" when validated)

    // max of the six practical criteria; +inf when an entry or a metric is not finite (used to rank iterates)
    double merit() const;
    // strict stage: a validated (practical) snapshot is ranked by max_abs_dimacs, before every non-validated one
    bool better_than(const ValidationResult &o) const;
};

class ExternalValidator
{
public:
    // At: CSC of A^T (vec_len x con_num: column j holds constraint j, row ids are svec indices), as given to
    // SDPSolver::init(); b and C: sparse (indices, values); blocks in the order of the svec vector.
    ExternalValidator(
        int vec_len, int con_num,
        const int *At_col_ptrs, const int *At_row_ids, const double *At_vals,
        int b_nnz, const int *b_indices, const double *b_vals,
        int C_nnz, const int *C_indices, const double *C_vals,
        const std::vector<char> &blk_types, const std::vector<int> &blk_sizes);

    // Metrics of the unscaled iterate (X: vec_len, y: con_num, S: vec_len). threads: number of CPU threads for the
    // per-block eigendecompositions.
    ValidationResult evaluate(const double *X, const double *y, const double *S, const ValidationTolerances &tol,
                              int threads = 1) const;

    // block kinds of the original data for the DIMACS norms ('s': PSD, 'l': linear), one per block
    void set_dimacs_kinds(const std::vector<char> &kinds);

    int vec_len() const { return vec_len_; }
    int con_num() const { return con_num_; }

private:
    int vec_len_, con_num_;
    std::vector<int> At_col_ptrs_, At_row_ids_;
    std::vector<double> At_vals_, b_, C_;
    std::vector<char> blk_types_, dimacs_kinds_;
    std::vector<int> blk_sizes_;
    std::vector<long> blk_offsets_;
    double norm_b_, norm_C_, norm_b_inf_, norm_C_inf_;
};

#endif // CUADMM_EXTERNAL_VALIDATION_H
