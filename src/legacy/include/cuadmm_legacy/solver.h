/*

    solver.h

    SDPSolver: the ported engine of cuADMM 1.x behind the facade cuadmm::Solver. It holds the problem data, the
    scaling, the iterate and the data of the y-step; solve() runs plain ADMM on it (the driver of the tests).

*/

#ifndef CUADMM_SOLVER_H
#define CUADMM_SOLVER_H

#include "cuadmm_legacy/check.h"
#include "cuadmm_legacy/io.h"
#include "cuadmm_legacy/kernels.h"
#include "cuadmm_legacy/memory.h"
#include "cuadmm_legacy/utils.h"
#include "cuadmm_legacy/cholesky_cpu.h"
#include "cuadmm/options.h"
#include "linsys/normal_equation_solver.h"
#include <memory>
#include "cuadmm_legacy/cublas.h"
#include "cuadmm_legacy/cusparse.h"
#include "cuadmm_legacy/cusolver.h"
#include "cuadmm_legacy/matrix_sizes.h"
#include "cuadmm_legacy/block_structure.h"
#include "cuadmm_legacy/external_validation.h"
#include "cuda.h"
#include <functional>
#include <limits>
#include <memory>
#include <string>

// Largest PSD block the exact EVD handles (B21): on CUDA 12.9 the cusolverDnXsyevd workspace query fails above it
#define CUADMM_MAX_EVD_BLOCK_SIZE 32768

/// @brief Method for the PSD projection of large matrices (the FP32 and FP32-emulated composite filters of main are
/// not ported: plan decision Q3).
enum ProjectionMethod
{
    EIG_FP64,
    COMPOSITE_FP16,
    COMPOSITE_FP32
};

inline const char *get_projection_method_name(ProjectionMethod method, bool dash = true)
{
    if (dash)
        switch (method)
        {
        case ProjectionMethod::EIG_FP64:
            return "EIG_FP64 ---------------";
        case ProjectionMethod::COMPOSITE_FP16:
            return "COMPOSITE_FP16 ---------";
        case ProjectionMethod::COMPOSITE_FP32:
            return "COMPOSITE_FP32 ---------";
        default:
            return "UNKNOWN ----------------";
        }
    else
        switch (method)
        {
        case ProjectionMethod::EIG_FP64:
            return "EIG_FP64";
        case ProjectionMethod::COMPOSITE_FP16:
            return "COMPOSITE_FP16";
        case ProjectionMethod::COMPOSITE_FP32:
            return "COMPOSITE_FP32";
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
    double time_interval = 0.0;             // and every this many seconds of net solve time (0: off)
    std::vector<double> target_tols;        // extra validations at the first downward crossing of each tolerance by the internal residual
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
/// Uses plain two-block ADMM (see solve()) to solve the problem:
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

    /* The normal equations (plan Section 11, Phase 7): the backend behind the interface; CHOLMOD of main by default */
    std::unique_ptr<cuadmm::linsys::NormalEquationSolver> linsys; // set by init() (CholmodSolver); the facade may replace it
    cuadmm::linsys::FactorInfo linsys_factor;                   // the factorization statistics of the backend in use
    double linsys_nnz_L = 0.0;                                   // nnz(L) of the CHOLMOD factor (0 for the other backends)
    double linsys_regularization = 1e-15;                        // eps of A A^T + eps I (linsys.regularization); set before init()
    std::vector<double> host_At_vals_scaled;                     // the normalized A^T values on the host (the backends' input)
    DeviceDenseVector<double> rhsy;      // right-hand side of AAt * x = b
    DeviceDenseVector<double> Rd1;
    size_t CSCtoCSR_At2A_buffer_size;               // cached call to buffersize_cusparse
    DeviceDenseVector<double> CSCtoCSR_At2A_buffer; // buffer for CSC to CSR

    /* Sparse vector <-> sparse matrix mapping */
    std::vector<BlockInfo> blk_info; // type, size and offsets of every block, in input order
    std::vector<int> psd_blk_sizes; // sizes of the matrices (without muliplicity)
    std::vector<int> psd_blk_nums;  // number of matrices of each size
    MatrixSizes sizes;
    DeviceDenseVector<double> Xinput;

    /* Medium matrices decomposition */
    // The medium size groups with at least medium_batch_min_count matrices are decomposed with one batched EVD call
    // (cusolverDnXsyevBatched, CUDA >= 12.8) instead of one cusolverDnXsyevd call per matrix on the streams, which is
    // bound by kernel launches and host synchronizations (pendulum: 26 -> 3 ms per iteration).
    int medium_batch_min_count = 2;                  // set before init(); 0 disables the batched EVD

    /* Large matrix decomposition */
    int eig_stream_num_per_gpu;               // number of streams per GPU

    ProjectionMethod current_proj_method; // projection currently used
    ProjectionMethod initial_proj_method; // projection initially used (low precision)
    ProjectionMethod final_proj_method;   // projection used in the end (high precision)
    bool switched_proj_method;            // whether the projection method has been switched
    int switched_proj_method_iter;        // iteration at which the projection method was switched

    bool use_lobpcg;                                 // whether to use LOBPCG
    double lobpcg_tol = 1e-5;                        // LOBPCG convergence tolerance (psd.lowrank.tol; absolute, B13)
    int lobpcg_maxit = 20;                           // LOBPCG iterations (psd.lowrank.max_iter)
    double lobpcg_ratio = 0.05;                      // low-rank threshold as a fraction of n; sizes the warm start (psd.lowrank.ratio)
    double lobpcg_relaxation = 1.2;                  // eigenpairs requested: ceil(relaxation * rank) (psd.lowrank.relaxation)
    unsigned long long lanczos_seed = 0;             // seed of the Lanczos start vectors (options.seed; per block: mixed with the block index, B8)
    DeviceDenseVector<int> positive_ranks;           // positive ranks of large matrices
    DeviceDenseVector<int> negative_ranks;           // negative ranks of large matrices
    HostDenseVector<int> cpu_positive_ranks;         // positive ranks of large matrices (CPU copy)
    HostDenseVector<int> cpu_negative_ranks;         // negative ranks of large matrices (CPU copy)
    std::vector<int> large_mode;                     // LargeProjectionMode of each large matrix
    std::vector<int> large_k;                        // number of LOBPCG eigenpairs of each large matrix
    std::vector<double> large_rank_tol;              // rank tolerance of each large matrix (from the last rank analysis)
    double stop_tol;                                 // stopping tolerance of the current solve()
    int lobpcg_calls;                                // number of LOBPCG projections in the last solve()
    int lobpcg_fallbacks;                            // number of LOBPCG results rejected by the certificate
    int composite_fallbacks;                         // number of non-finite composite projections in the last solve()
    int composite_rejections = 0;                    // composite results that failed the finiteness / norm test (fallback or kept)
    int composite_kept = 0;                          // of those, the ones used anyway (psd.composite.on_reject keep)
    int jacobi_not_converged;                        // number of batched Jacobi EVDs that did not reach their tolerance

    /* Small matrices eigen decomposition (batched Jacobi)  */
    /* Projection on PSD cones */

    /* Other */
    std::vector<DeviceStream> stream_flex;
    DeviceSparseHandle cusparseH; // main cuSPARSE handle
    DeviceBlasHandle cublasH;     // main cuBLAS handle

    /* sigma: the penalty options of solve() (the policy and its parameters; sigma_0 is sig) */
    cuadmm::PenaltyOptions penalty;

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
    double sigma0;                    // sigma at the start of the last solve()
    int sigma_changes;                // iterations of the last solve() in which sigma changed (0 with the fixed policy)
    int iterations = 0;               // iterations of the last solve()
    long y_solves = 0;                // y linear solves (A A^T systems) of the last solve(): one per iteration
    /// One change of sigma by the adaptive rules, with the quantities the rule used.
    struct SigmaUpdate
    {
        int iter;                 // iteration whose update changed sigma (1-based)
        double old_sigma, new_sigma;
        std::string rule;         // "admm_schedule", or "freeze_freeze_iteration" at the freeze
        double errRp, errRd, relgap, feasratio; // residuals of the iterate the rule saw (internal, original units)
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
    double cone_step_time_s = 0.0;               // seconds spent in the cone steps (A^T y, the input, the projections) of the last solve()
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
    // (validate_current_iterate moved to cuadmm::termination::ValidatedRule in Phase 4)
    std::vector<double> info_time_arr; // seconds since the start of the iterations, after each iteration

    /* Step length and the best iterate */
    double tau;
    double best_KKT;                  // |
    double best_errRp;                // |
    double best_errRd;                // |
    double best_relgap;               // |- save the best variables, KKT residuals
    double best_pobj;                 // |  and objectives to use in the end
    double best_dobj;                 // |
    DeviceDenseVector<double> X_best; // |
    DeviceDenseVector<double> y_best; // |
    DeviceDenseVector<double> S_best; // |

    SDPSolver() : start(nullptr), stop(nullptr), time_limit(std::numeric_limits<double>::infinity()),
                  time_limit_reached(false), sigma0(0.0), sigma_changes(0) {}
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
    /// @param sig Initial value for sigma (optional, default: 1.0; the CLI and the MEX default to 1e2).
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

    /// @brief Runs plain ADMM (tau = 1.618) on the engine's data with the penalty options `penalty` (sigma_0 = sig):
    /// the driver of the tests (the front-ends use cuadmm::Solver). The PSD cone has the settings of the options, with
    /// the warm-up on every large block when init() chose the composite filter, until eta < switch_proj_tol or
    /// iteration switch_proj_max_iter.
    /// @param max_iter Maximum number of iterations.
    /// @param stop_tol Stopping tolerance for KKT residual.
    /// @param if_first false for a second solve: X, y, S and sig were set again, in original units.
    void solve(int max_iter, double stop_tol, int switch_proj_max_iter = 5000, double switch_proj_tol = 1e-2, bool if_first = true);

    // Synchronizes the three streams of GPU0.
    void synchronize_gpu0_streams();

    // The system of the y-step as the backends see it (the normalized A^T on the host, eps I): for a replacement backend.
    cuadmm::linsys::NormalSystem normal_system() const;

    /* Certificate of suboptimality (paper eq. 29-31), computed at termination for the returned y */
    std::vector<double> dual_block_min; // per block, original units: lambda_min((C - A^T y)_b) for 's', min entry for 'l', max |entry| for 'u'
    std::vector<double> dual_cone_coef; // per block, original units: c_b with <(C - A^T y)_b, X_b> >= R_b c_b (see cone_measures)

    // Cone measures of the scaled svec vector v, per block in input order (overwrites the projection buffers):
    // - block_min: smallest eigenvalue ('s'), smallest entry ('l'), largest absolute entry ('u');
    // - bound_coef: c_b such that <v_b, X_b> >= R_b c_b for every X_b with X_b PSD and tr(X_b) <= R_b ('s'),
    //   0 <= X_b <= R_b entrywise ('l') or |X_b| <= R_b entrywise ('u'), i.e. min(0, lambda_min),
    //   sum_i min(0, v_i) and -sum_i |v_i| respectively.

    // Certified lower bound on the optimal value (paper eq. 29c-30), valid if an optimal X satisfies the bounds
    // of cone_measures with R_b = trace_bounds[b] (one bound per block, in input order, original units):
    //     <b, y> + sum_b R_b * dual_cone_coef[b].
    // For the moment matrix of order kappa of a probability measure on the box |x_i| <= R (n variables), tr(M) <= sum_{j <= kappa}
    // C(n+j-1, j) R^{2j} <= s(n, kappa) max(1, R)^{2 kappa}: the bound s(n, kappa) R^2 of the paper (Appendix C,
    // Theorem 2) holds for R = 1 only. For a localizing matrix of g, multiply by max |g| over the box.
    double certified_lower_bound(const std::vector<double> &trace_bounds) const;
    // Same with the same bound R for every block.
    double certified_lower_bound(double trace_bound) const;

    // The projections of the large matrices are the projector classes of src/cone/psd/large/ (Phase 6a).
};

#endif // CUADMM_SOLVER_H