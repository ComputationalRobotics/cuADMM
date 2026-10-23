/*

    result.h

    SolveResult: what Solver::solve() returns (plan Section 6.3): the best and the final iterate in original units,
    the residuals, cone violations, certificate quantities, stop reason, timings, the penalty policy and its
    parameters, the choices made at setup, and the per-iteration history.

*/

#ifndef CUADMM_RESULT_H
#define CUADMM_RESULT_H

#include <cstdint>
#include <string>
#include <vector>

namespace cuadmm
{

struct Iterate
{
    std::vector<double> X, y, S; // original units
    int iteration = 0;           // the iteration that produced it (0: the starting point)
    double eta_p = 0.0, eta_d = 0.0, eta_g = 0.0, pobj = 0.0, dobj = 0.0; // the solver's own measures of it
};

enum class Status { Converged, MaxIter, TimeLimit, NonFinite, Validated, StrictValidated };
const char *name_of(Status s);

/// @brief One row of the per-iteration history (plan Section 7.5: post-step residuals paired with the sigma of the
/// next iteration, as today).
struct HistoryRow
{
    int iteration;
    double time_s, pobj, dobj, eta_p, eta_d, eta_g, sigma, tau;
    bool sgs; // the iteration used the sGS step
};

struct SigmaChange
{
    int iteration;
    double old_sigma, new_sigma;
    std::string rule;
    double eta_p, eta_d, eta_g, feasratio;
    int prim_win, dual_win;
};

struct SolveResult
{
    Status status = Status::MaxIter;
    std::string stop_reason;       // "internal_kkt", "max_iter", "time_limit", "non_finite", "externally_validated", "strict_validated"
    bool converged = false;        // the stopping rule was met (internal, or external validation)
    bool returned_best_iterate = false; // `best` is not the final iterate
    Iterate best;                  // the returned iterate
    Iterate final;                 // the last iterate of the run (same as best when returned_best_iterate is false)
    double eta_p = 0.0, eta_d = 0.0, eta_g = 0.0; // of the returned iterate (original units)
    double pobj = 0.0, dobj = 0.0;                // of the returned iterate, in the problem's sense with the offset
    double cone_violation_X = 0.0, cone_violation_S = 0.0; // errPSD_X, errPSD_S
    int iterations = 0;
    long y_solves = 0;
    double setup_time_s = 0.0, solve_time_s = 0.0, total_time_s = 0.0;
    double validation_time_s = 0.0;
    std::size_t peak_gpu_memory_bytes = 0;
    // penalty policy
    std::string penalty_policy;
    std::string penalty_parameters_json; // {"factor": ..., ...}
    double sigma0 = 0.0, sigma_final = 0.0;                // scaled units
    double sigma0_original = 0.0, sigma_final_original = 0.0;
    int sigma_changes = 0;
    int sigma_changes_after_1e4 = 0;
    double sigma_total_variation_log2 = 0.0; // sum_k |log2(sigma_{k+1} / sigma_k)|
    int freeze_iteration = -1;
    std::string freeze_reason;
    std::vector<SigmaChange> sigma_log;
    // scheme
    std::string scheme;
    int sgs_iterations = 0, admm_iterations = 0, switch_iteration = -1;
    // certificate (original units): per block lambda_min / min entry / max |entry| of C - A^T y, and c_b
    std::vector<double> dual_block_min, dual_cone_coef;
    double certified_lower_bound = 0.0; // with certificate.trace_bound (NaN if none)
    double certified_lower_bound_scaled = 0.0; // with certificate.trace_bound_scale (NaN if none)
    bool certificate_conditional_on_free_bounds = false; // B24: a free block's bound is in force (valid only if |X_u| <= R_u)
    std::vector<int> certificate_missing_bounds;           // blocks without a bound (negative entry of certificate.block_bounds)
    std::string certificate_note;                          // "unavailable: ..." when a bound is missing
    // external validation
    bool externally_validated = false, practical_validated = false, strict_validated = false;
    int first_validated_iteration = -1, first_practical_iteration = -1, first_strict_iteration = -1;
    int validations = 0;
    double best_max_abs_dimacs = 0.0; // of the best validated snapshot (NaN if none)
    std::string validation_history_csv; // the CSV lines of the validations (empty without validation)
    // setup choices
    std::string linsys_backend; // "cholmod", "cholmod_dense_gpu"
    bool aat_supernodal = false;
    double aat_nnz_L = 0.0;
    int aat_tiny_pivots = 0, aat_nonpositive_pivots = 0;
    int lobpcg_calls = 0, lobpcg_fallbacks = 0, composite_fallbacks = 0, jacobi_not_converged = 0;
    // history
    std::vector<HistoryRow> history;
};

} // namespace cuadmm

#endif // CUADMM_RESULT_H
