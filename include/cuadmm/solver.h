/*

    solver.h

    Main solver header, works for any sizes of matrices.
    Solves an SDP problem with sGS-ADMM (Algorithm 1 of arXiv 2406.05846) for the iterations before
    switch_admm and plain two-block ADMM afterwards; switch_admm = 0 (the default) is plain ADMM throughout.

*/

#ifndef CUADMM_SOLVER_H
#define CUADMM_SOLVER_H

#include "cuadmm/check.h"
#include "cuadmm/io.h"
#include "cuadmm/kernels.h"
#include "cuadmm/memory.h"
#include "cuadmm/utils.h"
#include "cuadmm/cholesky_cpu.h"
#include "cuadmm/cublas.h"
#include "cuadmm/cusparse.h"
#include "cuadmm/cusolver.h"
#include "cuadmm/matrix_sizes.h"
#include "cuadmm/block_structure.h"
#include "cuadmm/external_validation.h"
#include "cuda.h"
#include <functional>
#include <limits>
#include <memory>
#include <string>

// Safety margins applied to the Lanczos estimate of ||A||_2 before a composite projection:
// the polynomial filters overflow if the scaled spectrum exceeds ~1.01 (FP32) / ~1.02 (FP16).
#define COMPOSITE_FP32_SCALE_MARGIN 1.02 // COMPOSITE_FP32 and COMPOSITE_FP32_EMULATED
#define COMPOSITE_FP16_SCALE_MARGIN 1.05

/// @brief Method for the PSD projection of large matrices.
enum ProjectionMethod
{
    EIG_FP64, // EIG_FP32,
    COMPOSITE_FP32,
    COMPOSITE_FP32_EMULATED,
    COMPOSITE_FP16
};

inline const char *get_projection_method_name(ProjectionMethod method, bool dash = true)
{
    if (dash)
        switch (method)
        {
        case ProjectionMethod::EIG_FP64:
            return "EIG_FP64 ---------------";
        case ProjectionMethod::COMPOSITE_FP32:
            return "COMPOSITE_FP32 ---------";
        case ProjectionMethod::COMPOSITE_FP32_EMULATED:
            return "COMPOSITE_FP32_EMULATED ";
        case ProjectionMethod::COMPOSITE_FP16:
            return "COMPOSITE_FP16 ---------";
        default:
            return "UNKNOWN ----------------";
        }
    else
        switch (method)
        {
        case ProjectionMethod::EIG_FP64:
            return "EIG_FP64";
        case ProjectionMethod::COMPOSITE_FP32:
            return "COMPOSITE_FP32";
        case ProjectionMethod::COMPOSITE_FP32_EMULATED:
            return "COMPOSITE_FP32_EMULATED";
        case ProjectionMethod::COMPOSITE_FP16:
            return "COMPOSITE_FP16";
        default:
            return "UNKNOWN";
        }
}

/// @brief How a large matrix is projected in the LOBPCG phase (decided by the periodic rank analysis).
enum LargeProjectionMode
{
    FULL,            // exact EVD with cuSOLVER
    LOBPCG_POSITIVE, // few positive eigenvalues: P = V max(D, 0) V^T with the largest eigenpairs of A
    LOBPCG_NEGATIVE  // few negative eigenvalues: P = A + V max(D, 0) V^T with the largest eigenpairs of -A
};

/// @brief Name of the algorithm run by SDPSolver::solve for these parameters: the sGS-ADMM step (Algorithm 1 of
/// arXiv 2406.05846) is used for iter < switch_admm, the plain two-block ADMM step afterwards (iter = 1, 2, ...).
inline std::string admm_mode_name(int switch_admm, int max_iter)
{
    if (switch_admm <= 1)
        return "plain ADMM";
    if (switch_admm > max_iter)
        return "pure sGS-ADMM (Algorithm 1)";
    // a mode of the public code, not Algorithm 1 of the paper (and not its kNN warm start)
    return "hybrid: sGS-ADMM for iterations 1-" + std::to_string(switch_admm - 1) + ", then plain ADMM";
}

/// @brief How solve() updates the penalty parameter sigma (set SDPSolver::sigma_policy before solve()).
/// - LEGACY_ADAPTIVE (the default, unchanged behaviour of the public code): in the sGS phase sigma is multiplied or
///   divided by sigscale when the primal or the dual residual "wins" (sig_update_threshold / stage_1 / stage_2); in the
///   plain ADMM phase it follows a fixed schedule with factor 2 and the Monitor1 "chasing" correction after iteration
///   5000. Kept for reproducibility of earlier results.
/// - FIXED: sigma_k = sigma_0 for every iteration, as in Algorithm 1 of arXiv 2406.05846 (a fixed sigma > 0). None of
///   the rules above runs, and solve() throws std::logic_error if sigma ever differs from sigma_0.
/// - FIXED_SGS_LEGACY_ADMM (for the hybrid sGS -> plain ADMM mode): sigma = sigma_0 during the sGS phase (enforced as
///   with FIXED), then, from the first plain-ADMM iteration, exactly the LEGACY_ADAPTIVE plain-ADMM rules (schedule
///   with factor 2 and Monitor1) as they run in a plain-ADMM solve started at that point: the schedule and the Monitor1
///   gate count the plain-ADMM iterations k = 1, 2, ... (not the global iteration), and the primal/dual "win" counters
///   start from 0 at the switch. With no sGS phase (switch_admm <= 1) it is identical to LEGACY_ADAPTIVE.
enum class SigmaPolicy
{
    LEGACY_ADAPTIVE,
    FIXED,
    FIXED_SGS_LEGACY_ADMM
};

inline const char *sigma_policy_name(SigmaPolicy policy)
{
    switch (policy)
    {
    case SigmaPolicy::FIXED:
        return "fixed";
    case SigmaPolicy::FIXED_SGS_LEGACY_ADMM:
        return "fixed_sgs_then_legacy_admm";
    default:
        return "legacy_adaptive";
    }
}

// Parses "fixed", "legacy_adaptive" or "fixed_sgs_then_legacy_admm"; returns false for any other string.
inline bool parse_sigma_policy(const std::string &name, SigmaPolicy &policy)
{
    if (name == "fixed")
        policy = SigmaPolicy::FIXED;
    else if (name == "legacy_adaptive")
        policy = SigmaPolicy::LEGACY_ADAPTIVE;
    else if (name == "fixed_sgs_then_legacy_admm")
        policy = SigmaPolicy::FIXED_SGS_LEGACY_ADMM;
    else
        return false;
    return true;
}

/// @brief One external validation of the iterate during solve() (see ValidationConfig).
struct ValidationRecord
{
    int iter = 0;              // completed iterations when the snapshot was taken
    double time_s = 0.0;       // seconds since the start of the iterations of solve(), before this validation
    double sigma = 0.0;        // sigma at the snapshot
    double internal_kkt = 0.0; // the solver's own max(errRp, errRd, relgap) at the snapshot
    // the solver's own residuals and objectives of the snapshot (restored with a best-iterate return)
    double internal_errRp = 0.0, internal_errRd = 0.0, internal_relgap = 0.0, internal_pobj = 0.0, internal_dobj = 0.0;
    // why the snapshot was taken, by precedence: "max_iter", "time_limit", "internal_crossing", "checkpoint",
    // "interval"; a requested checkpoint is flagged by `checkpoint` whatever the trigger
    std::string trigger;
    bool checkpoint = false;
    double validation_s = 0.0; // seconds for the snapshot (device-to-host copy) and the validation
    double callback_s = 0.0;   // seconds in ValidationConfig::on_validation (e.g. checkpoint files), not in validation_s
    ValidationResult result;   // external metrics of the unscaled snapshot
};

/// @brief Validation-aware stopping of solve(). With interval > 0, solve() snapshots the iterate every `interval`
/// completed iterations, at every downward crossing of stop_tol by its internal KKT residual, at the
/// checkpoint_iterations and at termination. It validates each snapshot with ExternalValidator on the original
/// problem data, and stops only when a snapshot passes (stop_reason "externally_validated"). The internal residual
/// alone no longer stops the solve. The starting point is validated too when it already satisfies the internal
/// tolerance (warm start) or when the solve ends before the first iteration. The trajectory is never restarted, and
/// the validation time (including the final validation) is part of the solve time. If the solve ends at max_iter, at
/// the time limit or on non-finite residuals (stop_reason "non_finite"), the best externally evaluated iterate
/// (smallest merit) is returned. With interval = 0 (the default), solve() behaves as before and stops on its internal
/// KKT residual.
struct ValidationConfig
{
    int interval = 0;                       // external validation every `interval` iterations (0: off)
    ValidationTolerances tol;               // success criteria (1e-4 each by default)
    std::vector<int> checkpoint_iterations; // extra snapshots (recorded and passed to on_validation)
    // strict stage (0: off). With strict_dimacs_tol > 0, the first snapshot that passes the criteria above is recorded
    // (first_practical_*) but the solve continues until a validated snapshot also has max_abs_dimacs <= this value
    // (stop_reason "strict_validated") or a safety limit; the returned iterate is then the validated snapshot with the
    // smallest max_abs_dimacs (or, if none validated, the smallest merit)
    double strict_dimacs_tol = 0.0;
    int threads = 1;                        // CPU threads for the per-block eigendecompositions
    // called after every validation with the record and the unscaled iterate (X, y, S), e.g. to save checkpoints;
    // an exception thrown by it is reported and counted (SDPSolver::callback_failures), and the solve continues
    std::function<void(const ValidationRecord &, const std::vector<double> &, const std::vector<double> &,
                       const std::vector<double> &)>
        on_validation;
};

/// @brief Main solver class for the SDP problem.
/// Uses sGS-ADMM and/or plain two-block ADMM (see solve()) to solve the problem:
///      min(X) <C, X>  s.t. A(X) = b,       X >= 0,
/// of dual:
///   max(y, S) <b, y>  s.t. At(y) + S = C,  S >= 0
class SDPSolver
{
public:
    /* Problem data */
    DeviceSparseMatrixDoubleCSC At_csc; // |
    DeviceSparseMatrixDoubleCSR At_csr; // | constraint matrix
    DeviceSparseMatrixDoubleCSR A_csr;  // |
    DeviceSparseVector<double> b;       // constraint vector
    DeviceSparseVector<double> C;       // cost matrix
    DeviceDenseVector<double> blk;      // block sizes
    DeviceDenseVector<double> X;        // primal variable
    DeviceDenseVector<double> S;        // dual variable 2
    DeviceDenseVector<double> y;        // dual variable 1

    /* Hyperparameters */
    double sig;  // Lagrangian penalty sigma
    int vec_len; // length of X in vector form
    int con_num; // number of constraints (length of y)

    /* Scaling */
    DeviceDenseVector<double> normA; // norm of A
    DeviceSparseVector<double> borg;
    DeviceSparseVector<double> Corg;

    /* KKT residuals */
    DeviceDenseVector<double> Aty;   // A^T * y
    DeviceDenseVector<double> Rp;    // primal residual, b - AX
    DeviceDenseVector<double> SmC;   // S - C
    DeviceDenseVector<double> Rd;    // dual residual, A^T * y - C + S
    DeviceDenseVector<double> Rporg; // original primal residual (size con_num)
    DeviceDenseVector<double> Rdorg; // original dual residual (size vec_len)
    double norm_borg;                // 1 + original norm of b
    double norm_Corg;                // 1 + original norm of C
    double bscale;                   // scale for b
    double Cscale;                   // scale for C
    double objscale;                 // scale for objective function
    double errRp;                    // primal feasibility violation (eta_p)
    double errRd;                    // dual feasibility violation (eta_d)
    double maxfeas;                  // maximum feasibility violation (max(eta_p, eta_d))
    double pobj;                     // primal objective (<C,X>)
    double dobj;                     // dual objective (<b, y>)
    double relgap;                   // normalized duality gap (eta_g)
    double errPSD_X;                 // cone violation of the returned X over all 's' and 'l' blocks, set at termination
    double errPSD_S;                 // cone violation of the returned S over all 's' and 'l' blocks, set at termination
    bool converged;                  // whether the last solve() met the stopping tolerance
    bool returned_best_iterate;      // whether the last solve() returned the best iterate instead of the last one

    // buffers for cuSPARSE matrix-vector and inner products
    size_t SpMV_Aty_buffer_size;               // |
    DeviceDenseVector<double> SpMV_Aty_buffer; // |
    size_t SpMV_AX_buffer_size;                // |
    DeviceDenseVector<double> SpMV_AX_buffer;  // |- buffer sizes & buffers
    size_t SpVV_CtX_buffer_size;               // |  for cuSPARSE SpMV
    DeviceDenseVector<double> SpVV_CtX_buffer; // |
    size_t SpVV_bty_buffer_size;               // |
    DeviceDenseVector<double> SpVV_bty_buffer; // |

    /* Cholesky decomposition on CPU */
    CholeskySolverCPU cpu_AAt_solver;    // solver for AAt * x = b
    DeviceDenseVector<int> perm;         // permutation in the L factor of AAt
    DeviceDenseVector<int> perm_inv;     // inverse of perm
    DeviceDenseVector<double> rhsy;      // right-hand side of AAt * x = b
    DeviceDenseVector<double> rhsy_perm; // permuted rhsy
    DeviceDenseVector<double> y_perm;    // y after permutation
    bool AAt_dense_gpu;                  // y-step solves with the dense L D L^T factor on the GPU (see init)
    DeviceDenseVector<double> AAt_L;     // |- dense unit lower triangular L (column-major, con_num x con_num)
    DeviceDenseVector<double> AAt_D;     // |  and diagonal D of the factor of cpu_AAt_solver, if AAt_dense_gpu
    DeviceDenseVector<double> Rd1;
    size_t CSCtoCSR_At2A_buffer_size;               // cached call to buffersize_cusparse
    DeviceDenseVector<double> CSCtoCSR_At2A_buffer; // buffer for CSC to CSR

    /* Sparse vector <-> sparse matrix mapping */
    std::vector<BlockInfo> blk_info; // type, size and offsets of every block, in input order
    std::vector<int> psd_blk_sizes; // sizes of the matrices (without muliplicity)
    std::vector<int> psd_blk_nums;  // number of matrices of each size
    MatrixSizes sizes;
    DeviceDenseVector<int> map_B;  // |
    DeviceDenseVector<int> map_M1; // |- maps for vectorization of matrices
    DeviceDenseVector<int> map_M2; // |    (cached from get_maps())
    DeviceDenseVector<double> Xinput;

    /* Medium matrices decomposition */
    DeviceDenseVector<double> medium_mat;
    DeviceDenseVector<double> medium_W;
    DeviceDenseVector<int> medium_info;
    DeviceDenseVector<double> medium_mat_tmp;
    DeviceDenseVector<double> medium_mat_P;

    std::vector<size_t> eig_medium_buffer_size;     // one GPU buffer size per unique medium size
    DeviceDenseVector<double> eig_medium_buffer;    // one GPU buffer per unique medium size
    std::vector<size_t> cpu_eig_medium_buffer_size; // one CPU buffer size per unique medium size
    HostDenseVector<double> cpu_eig_medium_buffer;  // one CPU buffer per unique medium size
    std::vector<DeviceStream> eig_medium_stream_arr;
    std::vector<DeviceSolverDnHandle> cusolverH_eig_medium_arr; // one handle per stream
    // The medium size groups with at least medium_batch_min_count matrices are decomposed with one batched EVD call
    // (cusolverDnXsyevBatched, CUDA >= 12.8) instead of one cusolverDnXsyevd call per matrix on the streams, which is
    // bound by kernel launches and host synchronizations (pendulum: 26 -> 3 ms per iteration).
    int medium_batch_min_count = 2;                  // set before init(); 0 disables the batched EVD
    std::vector<char> medium_group_batched;          // per medium size group
    DeviceSolverDnHandle cusolverH_eig_medium_batch; // on the legacy default stream
    size_t medium_batch_buffer_size = 0;             // |- workspaces of the batched EVD (shared by the groups,
    size_t medium_batch_cpu_buffer_size = 0;         // |  which are decomposed one after the other)
    DeviceDenseVector<double> medium_batch_buffer;   // |
    HostDenseVector<double> medium_batch_cpu_buffer; // |
    std::vector<DeviceBlasHandle> cublasH_eig_medium_arr;       // one handle per stream

    /* Large matrix decomposition */
    DeviceDenseVector<double> large_mat;
    DeviceDenseVector<double> large_W;
    DeviceDenseVector<int> large_info;
    int eig_stream_num_per_gpu;               // number of streams per GPU
    DeviceSolverDnHandle cusolverH_eig_large; // single handle for large matrices
    SingleEigParameter eig_param_single;
    std::vector<size_t> eig_large_buffer_size;     // one GPU buffer size per unique large size
    DeviceDenseVector<double> eig_large_buffer;    // one GPU buffer per unique large size
    std::vector<size_t> cpu_eig_large_buffer_size; // one CPU buffer size per unique large size
    HostDenseVector<double> cpu_eig_large_buffer;  // one CPU buffer per unique large size
    DeviceDenseVector<float> float_proj_workspace; // workspace for projection of large matrices
    DeviceDenseVector<__half> half_proj_workspace; // workspace for projection of large matrices

    ProjectionMethod current_proj_method; // projection currently used
    ProjectionMethod initial_proj_method; // projection initially used (low precision)
    ProjectionMethod final_proj_method;   // projection used in the end (high precision)
    bool switched_proj_method;            // whether the projection method has been switched
    int switched_proj_method_iter;        // iteration at which the projection method was switched
    DeviceBlasHandle cublasH_composite_proj;
    DeviceSolverDnHandle cusolverH_composite_proj;

    bool use_lobpcg;                                 // whether to use LOBPCG
    DeviceDenseVector<int> positive_ranks;           // positive ranks of large matrices
    DeviceDenseVector<int> negative_ranks;           // negative ranks of large matrices
    HostDenseVector<int> cpu_positive_ranks;         // positive ranks of large matrices (CPU copy)
    HostDenseVector<int> cpu_negative_ranks;         // negative ranks of large matrices (CPU copy)
    std::vector<DeviceDenseVector<double>> lobpcg_W; // eigenvalues of large matrices computed with LOBPCG
    std::vector<DeviceDenseVector<double>> lobpcg_P; // eigenvectors of large matrices computed with LOBPCG
    std::vector<int> lobpcg_k_alloc;                 // number of columns allocated in lobpcg_W/lobpcg_P per large matrix
    DeviceDenseVector<double> lobpcg_W_relu;         // large_W after ReLU
    DeviceBlasHandle cublasH_eig_large;
    DeviceBlasHandle cublasH_eig_large_update;
    std::vector<int> large_mode;                     // LargeProjectionMode of each large matrix
    std::vector<int> large_k;                        // number of LOBPCG eigenpairs of each large matrix
    std::vector<double> large_rank_tol;              // rank tolerance of each large matrix (from the last rank analysis)
    double stop_tol;                                 // stopping tolerance of the current solve()
    int lobpcg_calls;                                // number of LOBPCG projections in the last solve()
    int lobpcg_fallbacks;                            // number of LOBPCG results rejected by the certificate
    int composite_fallbacks;                         // number of non-finite composite projections in the last solve()
    int jacobi_not_converged;                        // number of batched Jacobi EVDs that did not reach their tolerance

    /* Small matrices eigen decomposition (batched Jacobi)  */
    DeviceDenseVector<double> small_mat;
    DeviceDenseVector<double> small_W;
    DeviceDenseVector<int> small_info;
    BatchEigParameter eig_param_batch;
    DeviceSolverDnHandle cusolverH_eig_small;
    std::vector<size_t> eig_small_buffer_size;
    DeviceDenseVector<double> eig_small_buffer;
    /* Projection on PSD cones */
    DeviceDenseVector<double> large_mat_tmp;
    DeviceDenseVector<double> small_mat_tmp;
    DeviceDenseVector<double> large_mat_P;
    DeviceDenseVector<double> small_mat_P;

    /* Other */
    std::vector<DeviceStream> stream_flex;
    DeviceSparseHandle cusparseH; // main cuSPARSE handle
    DeviceBlasHandle cublasH;     // main cuBLAS handle

    /* Rescale and update sigma */
    int prim_win;
    int dual_win;
    double bscale2;
    double Cscale2;
    double ratioconst;
    double feasratio;
    double sigmax;
    double sigmin;
    double sigscale;

    /* Info */
    int info_iter_num;                   // iteration number
    std::vector<double> info_pobj_arr;   // |
    std::vector<double> info_dobj_arr;   // |
    std::vector<double> info_errRp_arr;  // |
    std::vector<double> info_errRd_arr;  // |- arrays to log info
    std::vector<double> info_relgap_arr; // |
    std::vector<double> info_sig_arr;    // |
    std::vector<double> info_bscale_arr; // |
    std::vector<double> info_Cscale_arr; // |
    std::vector<double> info_tau_arr;    // | step length tau of each iteration's X update

    /* Time */
    cudaEvent_t start;
    cudaEvent_t stop;
    double total_time; // count time in seconds (from the start of init() to the end of solve())
    double init_time;  // seconds spent in init()
    double solve_time; // seconds spent in the iterations of the last solve() (without the final post-processing)
    double time_limit; // solve() stops after this many seconds of iterations, like at max_iter (default: none)
    bool time_limit_reached;          // whether the last solve() stopped because of time_limit
    SigmaPolicy sigma_policy;         // sigma update rule of solve() (default LEGACY_ADAPTIVE), see SigmaPolicy
    double sigma0;                    // sigma at the start of the last solve()
    int sigma_changes;                // iterations of the last solve() in which sigma changed (0 with FIXED)
    int admm_phase_iterations;        // iterations of the last solve() that used the plain ADMM step (0 for pure sGS)
    long y_solves = 0;                // y linear solves (A A^T systems) of the last solve(): one per plain-ADMM iteration
    /// One change of sigma by the adaptive rules, with the quantities the rule used.
    struct SigmaUpdate
    {
        int iter;                 // iteration whose update changed sigma (1-based)
        double old_sigma, new_sigma;
        std::string rule;         // "admm_schedule", "monitor1" or "sgs_rule"
        double errRp, errRd, relgap, feasratio; // residuals of the iterate the rule saw (internal, scaled problem)
        int prim_win, dual_win;   // win counters before the rule reset them
    };
    std::vector<SigmaUpdate> sigma_log; // every sigma change of the last solve()
    // validation-aware stopping with a strict stage (ValidationConfig::strict_dimacs_tol)
    bool practical_validated = false;    // a snapshot passed the practical criteria
    int first_practical_iteration = -1;
    double first_practical_time_s = -1.0; // solve clock after that validation
    bool strict_validated = false;        // a validated snapshot also had max_abs_dimacs <= strict_dimacs_tol
    int first_strict_iteration = -1;
    double first_strict_time_s = -1.0;
    std::vector<char> dimacs_kinds;       // block kinds of the original data for the DIMACS norms (empty: from blk types)
    /* Phases of the last solve() (the hybrid mode: sGS-ADMM for iter < switch_admm, then plain ADMM) */
    int sgs_phase_iterations = 0;          // iterations that used the sGS step
    int switch_iteration = -1;             // first iteration that used the plain ADMM step (-1: none)
    double sgs_phase_time_s = 0.0;         // seconds of the solve clock before the first plain-ADMM step (the whole
                                           // solve if there was none); includes the validations of the sGS iterates
    double admm_phase_time_s = 0.0;        // solve_time - sgs_phase_time_s if a plain-ADMM step ran, else 0
    double sgs_phase_validation_s = 0.0;   // validation seconds within the sGS phase (included in sgs_phase_time_s)
    double admm_phase_validation_s = 0.0;  // validation seconds within the plain-ADMM phase
    double kkt_at_switch = -1.0;           // internal max(errRp, errRd, relgap) of the iterate handed to plain ADMM
    double sigma_at_switch = -1.0;         // sigma of that iterate (the first plain-ADMM step uses it)
    double tau_last_sgs = -1.0;            // tau of the last sGS step before the switch (-1: none)
    int sigma_changes_sgs_phase = 0;       // sigma changes made in sGS iterations
    int sigma_changes_admm_phase = 0;      // sigma changes made in plain-ADMM iterations

    /* Validation-aware stopping (see ValidationConfig), results of the last solve() */
    ValidationConfig validation;
    std::string stop_reason;                     // "internal_kkt", "externally_validated", "max_iter", "time_limit",
                                                 // "non_finite" (validation mode only; legacy mode throws)
    bool externally_validated_converged = false; // a snapshot passed the external validation
    bool internal_converged_at_return = false;   // internal KKT < stop_tol at the returned iterate
    int first_validated_iteration = -1;          // iteration of the first validated snapshot (-1: none)
    int first_internal_below_1e_3 = -1;          // first completed iteration with internal KKT < 1e-3 (-1: none)
    int first_internal_below_tol = -1;           // first completed iteration with internal KKT < stop_tol (-1: none)
    std::vector<ValidationRecord> validation_history;
    ValidationRecord best_validation;            // best externally evaluated iterate (smallest merit)
    bool returned_best_external = false;         // the returned iterate is best_validation, not the last iterate
    double validation_time_s = 0.0;              // seconds spent in snapshots and external validations
    double callback_time_s = 0.0;                // seconds spent in ValidationConfig::on_validation
    int callback_failures = 0;                   // exceptions thrown by ValidationConfig::on_validation
    size_t peak_gpu_mem_bytes = 0;               // largest (total - free) device memory seen at the start of the
                                                 // iterations, at the validations and at termination
    // host copies of the original problem data (made by init() outside its timing) for the external validation;
    // released once the ExternalValidator, which keeps its own copy, is built by the first validating solve()
    std::vector<int> host_At_col_ptrs, host_At_row_ids, host_b_idx, host_C_idx, host_blk_sizes;
    std::vector<double> host_At_vals, host_b_vals, host_C_vals;
    std::vector<char> host_blk_types;
    std::unique_ptr<ExternalValidator> external_validator;
    // snapshot of the current iterate, external validation, best-iterate bookkeeping (see ValidationConfig)
    ValidationRecord validate_current_iterate(int iter, double time_s, const std::string &trigger, bool checkpoint);
    std::vector<double> info_time_arr; // seconds since the start of the iterations, after each iteration

    /* sGS-ADMM */
    double tau;
    int switch_admm; // the iteration at which to switch to standard ADMM
    int sig_update_threshold;
    int sig_update_stage_1;
    int sig_update_stage_2;
    double sgs_KKT;
    double best_KKT;                  // |
    double best_errRp;                // |
    double best_errRd;                // |
    double best_relgap;               // |- save the best variables, KKT residuals
    double best_pobj;                 // |  and objectives to use in the end
    double best_dobj;                 // |
    DeviceDenseVector<double> X_best; // |
    DeviceDenseVector<double> y_best; // |
    DeviceDenseVector<double> S_best; // |

    SDPSolver() : AAt_dense_gpu(false), start(nullptr), stop(nullptr), time_limit(std::numeric_limits<double>::infinity()),
                  time_limit_reached(false), sigma_policy(SigmaPolicy::LEGACY_ADAPTIVE), sigma0(0.0), sigma_changes(0),
                  admm_phase_iterations(0) {}
    ~SDPSolver();

    /// @brief Initializes the SDP solver.
    /// @param eig_stream_num_per_gpu Number of eigenvalue streams per GPU.
    /// @param vec_len Length of the vector.
    /// @param con_num Number of constraints.
    /// @param cpu_At_csc_col_ptrs Column pointers of the constraint matrix in CSC format.
    /// @param cpu_At_csc_row_ids Row indices of the constraint matrix in CSC format.
    /// @param cpu_At_csc_vals Values of the constraint matrix in CSC format.
    /// @param At_nnz Number of non-zero entries in the constraint matrix.
    /// @param cpu_b_indices Indices of the constraint vector.
    /// @param cpu_b_vals Values of the constraint vector.
    /// @param b_nnz Number of non-zero entries in the constraint vector.
    /// @param cpu_C_indices Indices of the cost matrix.
    /// @param cpu_C_vals Values of the cost matrix.
    /// @param C_nnz Number of non-zero entries in the cost matrix.
    /// @param cpu_blk_types Block types (s, u, ...).
    /// @param cpu_blk_sizes Block sizes.
    /// @param mat_num Number of blocks.
    /// @param initial_proj_method Projection method to use initially (default: COMPOSITE_FP16).
    /// @param final_proj_method Projection method to use in the end (default: EIG_FP64).
    /// @param cpu_X_vals Initial values for X (optional, default: zero vector).
    /// @param cpu_y_vals Initial values for y (optional, default: zero vector).
    /// @param cpu_S_vals Initial values for S (optional, default: zero vector).
    /// @param sig Initial value for sigma (optional, default: `2e2`).
    /// @param use_lobpcg Whether to use LOBPCG for large matrices (default: true).
    void init(
        int eig_stream_num_per_gpu,
        // core data
        int vec_len, int con_num,
        int *cpu_At_csc_col_ptrs, int *cpu_At_csc_row_ids, double *cpu_At_csc_vals, int At_nnz,
        int *cpu_b_indices, double *cpu_b_vals, int b_nnz,
        int *cpu_C_indices, double *cpu_C_vals, int C_nnz,
        char *cpu_blk_types,
        int *cpu_blk_sizes, int mat_num,
        ProjectionMethod initial_proj_method = ProjectionMethod::COMPOSITE_FP16,
        ProjectionMethod final_proj_method = ProjectionMethod::EIG_FP64,
        double *cpu_X_vals = nullptr, // |
        double *cpu_y_vals = nullptr, // |- values for warm start
        double *cpu_S_vals = nullptr, // |
        double sig = 1.0,
        bool use_lobpcg = true);

    /// @brief Solves the SDP problem: sGS-ADMM steps (the update of Algorithm 1 of arXiv 2406.05846, tau = 1.95,
    /// reduced to max(1.618, tau / 1.1) at iterations whose incoming dual residual is below stop_tol) for
    /// iter < switch_admm, plain two-block ADMM steps (tau = 1.618) afterwards, see admm_mode_name(). With
    /// 1 < switch_admm <= max_iter this is the hybrid mode: the plain-ADMM phase continues from the same X, y, S,
    /// sigma and workspaces (nothing is reset, rescaled or rebuilt at the switch); see the phase members.
    /// @param max_iter Maximum number of iterations.
    /// @param stop_tol Stopping tolerance for KKT residual.
    /// @param sig_update_threshold |- sigma update of the sGS phase only: sigma is multiplied or divided by sigscale
    /// @param sig_update_stage_1   |  every sig_update_stage_1 iterations up to sig_update_threshold, every
    /// @param sig_update_stage_2   |  sig_update_stage_2 iterations afterwards (the ADMM phase uses its own schedule
    ///                             |  with factor 2)
    /// @param switch_admm The iteration at which to switch from sGS-ADMM to standard ADMM (0: plain ADMM throughout,
    ///                    > max_iter: sGS-ADMM throughout).
    /// @param sigscale Factor of the sigma updates of the sGS phase (with sigma_policy LEGACY_ADAPTIVE only; use
    ///                 sigma_policy = SigmaPolicy::FIXED, not sigscale = 1, for a fixed sigma).
    /// @param if_first Boolean flag to indicate if this is the first call to solve. For the second call, we assume that new X, y, S, sig are passed, but that they are unscaled.
    void solve(
        int max_iter, double stop_tol,
        int sig_update_threshold = 500,
        int sig_update_stage_1 = 50,
        int sig_update_stage_2 = 100,
        int switch_admm = 0,
        int switch_proj_max_iter = 5000,
        double switch_proj_tol = 1e-2,
        double sigscale = 2,
        bool if_first = true);

    // Synchronizes the three streams of GPU0.
    void synchronize_gpu0_streams();

    // Solves the permuted y-step system L D L^T y_perm = rhsy_perm (P (eps I + AA^T) P^T = L D L^T): with CHOLMOD on
    // the CPU, or with two dense triangular solves on the GPU if AAt_dense_gpu.
    void solve_AAt_perm();

    /* Certificate of suboptimality (paper eq. 29-31), computed at termination for the returned y */
    std::vector<double> dual_block_min; // per block, original units: lambda_min((C - A^T y)_b) for 's', min entry for 'l', max |entry| for 'u'
    std::vector<double> dual_cone_coef; // per block, original units: c_b with <(C - A^T y)_b, X_b> >= R_b c_b (see cone_measures)

    // Cone measures of the scaled svec vector v, per block in input order (overwrites the projection buffers):
    // - block_min: smallest eigenvalue ('s'), smallest entry ('l'), largest absolute entry ('u');
    // - bound_coef: c_b such that <v_b, X_b> >= R_b c_b for every X_b with X_b PSD and tr(X_b) <= R_b ('s'),
    //   0 <= X_b <= R_b entrywise ('l') or |X_b| <= R_b entrywise ('u'), i.e. min(0, lambda_min),
    //   sum_i min(0, v_i) and -sum_i |v_i| respectively.
    void cone_measures(DeviceDenseVector<double> &v, std::vector<double> &block_min, std::vector<double> &bound_coef);

    // Certified lower bound on the optimal value (paper eq. 29c-30), valid if an optimal X satisfies the bounds
    // of cone_measures with R_b = trace_bounds[b] (one bound per block, in input order, original units):
    //     <b, y> + sum_b R_b * dual_cone_coef[b].
    // For the moment matrix of order kappa of a probability measure on the box |x_i| <= R (n variables), tr(M) <= sum_{j <= kappa}
    // C(n+j-1, j) R^{2j} <= s(n, kappa) max(1, R)^{2 kappa}: the bound s(n, kappa) R^2 of the paper (Appendix C,
    // Theorem 2) holds for R = 1 only. For a localizing matrix of g, multiply by max |g| over the box.
    double certified_lower_bound(const std::vector<double> &trace_bounds) const;
    // Same with the same bound R for every block.
    double certified_lower_bound(double trace_bound) const;

    /* Projection of the large matrices, idx is the flat index of the j-th matrix of size index i */

    // Exact projection with the cuSOLVER EVD: large_mat_P <- Pi(large_mat), large_mat <- eigenvectors.
    // If analyze_rank, the ranks are recomputed and the LOBPCG mode / warmstart of the matrix refreshed.
    // If !multiply, the final GEMM P = (Q max(W, 0)) Q^T is left to the caller (large_mat_tmp holds Q max(W, 0)).
    void large_full_eig_project(int i, int j, int idx, bool analyze_rank, bool multiply = true);

    // Projection with warm-started LOBPCG (large_mode[idx] must be LOBPCG_POSITIVE or LOBPCG_NEGATIVE).
    // Returns false if LOBPCG did not converge or if its k-th Ritz value is above the rank tolerance
    // (an eigenvalue of the wanted sign may be missing); large_mat is then unchanged.
    bool large_lobpcg_project(int i, int j, int idx);

    // Projection with the composite polynomial filter of the current projection method, after scaling by
    // COMPOSITE_*_SCALE_MARGIN times the Lanczos estimate of ||A||_2.
    // Returns false if the result is not finite; large_mat is then unchanged.
    bool large_composite_project(int i, int j, int idx);
};

#endif // CUADMM_SOLVER_H