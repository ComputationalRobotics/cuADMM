/*

    solver.h

    Main solver header, works for any sizes of matrices.
    Uses the sGS-ADMM algorithm to solve an SDP problem.

*/

#ifndef CUADMM_SOLVER_H
#define CUADMM_SOLVER_H

#include "cuadmm/check.h"
#include "cuadmm/eig_cpu.h"
#include "cuadmm/io.h"
#include "cuadmm/kernels.h"
#include "cuadmm/memory.h"
#include "cuadmm/utils.h"
#include "cuadmm/cholesky_cpu.h"
#include "cuadmm/cublas.h"
#include "cuadmm/cusparse.h"
#include "cuadmm/cusolver.h"
#include "cuadmm/matrix_sizes.h"
#include "cuda.h"

/// @brief Method for the PSD projection of large matrices.
enum ProjectionMethod {
    EIG_FP64,// EIG_FP32,
    COMPOSITE_FP32, COMPOSITE_FP32_EMULATED,
    COMPOSITE_FP16
};

inline const char* get_projection_method_name(ProjectionMethod method) {
    switch (method) {
        case ProjectionMethod::EIG_FP64:                return "EIG_FP64 ---------------";
        case ProjectionMethod::COMPOSITE_FP32:          return "COMPOSITE_FP32 ---------";
        case ProjectionMethod::COMPOSITE_FP32_EMULATED: return "COMPOSITE_FP32_EMULATED ";
        case ProjectionMethod::COMPOSITE_FP16:          return "COMPOSITE_FP16 ---------";
        default:                                        return "UNKNOWN ----------------";
    }
}

/// @brief Main solver class for the SDP problem.
/// Uses the sGS-ADMM algorithm to solve the problem:
///      min(X) <C, X>  s.t. A(X) = b,       X >= 0,
/// of dual:
///   max(y, S) <b, y>  s.t. At(y) + S = C,  S >= 0
class SDPSolver {
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
        double sig;   // Lagrangian penalty sigma
        int vec_len;  // length of X in vector form
        int con_num;  // number of constraints (length of y)

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
        double norm_borg; // original norm of b
        double norm_Corg; // original norm of C
        double bscale;    // scale for b
        double Cscale;    // scale for C
        double objscale;  // scale for objective function
        double errRp;
        double errRd;
        double maxfeas;
        double pobj;      // primal objective (<C,X>)
        double dobj;      // dual objective (<b, y>)
        double relgap;    // eta_g, the normalized duality gap
        // buffers for cuSPARSE matrix-vector and inner products
        size_t SpMV_Aty_buffer_size;                // |
        DeviceDenseVector<double> SpMV_Aty_buffer;  // |
        size_t SpMV_AX_buffer_size;                 // |
        DeviceDenseVector<double> SpMV_AX_buffer;   // |- buffer sizes & buffers
        size_t SpVV_CtX_buffer_size;                // |  for cuSPARSE SpMV
        DeviceDenseVector<double> SpVV_CtX_buffer;  // |
        size_t SpVV_bty_buffer_size;                // |
        DeviceDenseVector<double> SpVV_bty_buffer;  // |

        /* Cholesky decomposition on CPU */
        CholeskySolverCPU cpu_AAt_solver;    // solver for AAt * x = b
        DeviceDenseVector<int> perm;         // permutation in the L factor of AAt
        DeviceDenseVector<int> perm_inv;     // inverse of perm
        DeviceDenseVector<double> rhsy;      // right-hand side of AAt * x = b
        DeviceDenseVector<double> rhsy_perm; // permuted rhsy
        DeviceDenseVector<double> y_perm;    // y after permutation
        DeviceDenseVector<double> Rd1;
        size_t CSCtoCSR_At2A_buffer_size;    // cached call to buffersize_cusparse
        DeviceDenseVector<double> CSCtoCSR_At2A_buffer; // buffer for CSC to CSR

        /* Sparse vector <-> sparse matrix mapping */
        std::vector<int> psd_blk_sizes; // sizes of the matrices (without muliplicity)
        std::vector<int> psd_blk_nums;  // number of matrices of each size
        MatrixSizes sizes;
        DeviceDenseVector<int> map_B;  // |
        DeviceDenseVector<int> map_M1; // |- maps for vectorization of matrices
        DeviceDenseVector<int> map_M2; // |    (cached from get_maps())

        /* Large matrix decomposition */
        DeviceDenseVector<double> Xinput;
        DeviceDenseVector<double> large_mat;
        DeviceDenseVector<double> large_W;
        DeviceDenseVector<int> large_info;
        int eig_stream_num_per_gpu;    // number of streams per GPU
        std::vector<DeviceStream> eig_stream_arr;
        std::vector<DeviceSolverDnHandle> cusolverH_eig_large_arr; // one handle per stream
        SingleEigParameter eig_param_single;
        std::vector<size_t> eig_large_buffer_size;                // one GPU buffer size per unique large size
        DeviceDenseVector<double> eig_large_buffer;  // one GPU buffer per unique large size
        std::vector<size_t> cpu_eig_large_buffer_size;             // one CPU buffer size per unique large size
        HostDenseVector<double> cpu_eig_large_buffer; // one CPU buffer per unique large size
        DeviceDenseVector<float> float_proj_workspace; // workspace for projection of large matrices
        DeviceDenseVector<__half> half_proj_workspace; // workspace for projection of large matrices

        ProjectionMethod current_proj_method; // projection currently used
        ProjectionMethod initial_proj_method; // projection initially used (low precision)
        ProjectionMethod final_proj_method; // projection used in the end (high precision)
        bool switched_proj_method; // whether the projection method has been switched
        DeviceBlasHandle cublasH_proj;
        DeviceSolverDnHandle cusolverH_proj;

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
        int info_iter_num;  // iteration number
        std::vector<double> info_pobj_arr;   // |
        std::vector<double> info_dobj_arr;   // |
        std::vector<double> info_errRp_arr;  // |
        std::vector<double> info_errRd_arr;  // |- arrays to log info
        std::vector<double> info_relgap_arr; // |
        std::vector<double> info_sig_arr;    // |
        std::vector<double> info_bscale_arr; // |
        std::vector<double> info_Cscale_arr; // |

        /* Time */
        cudaEvent_t start;
        cudaEvent_t stop;
        double total_time; // count time in seconds

        /* sGS-ADMM */
        double tau;
        int switch_admm; // the iteration at which to switch to standard ADMM
        int sig_update_threshold;
        int sig_update_stage_1;
        int sig_update_stage_2;
        double sgs_KKT;
        double best_KKT;                  // |
        DeviceDenseVector<double> X_best; // |- save the best variables and KKT residual
        DeviceDenseVector<double> y_best; // |  to use in the end
        DeviceDenseVector<double> S_best; // |

        SDPSolver() {}

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
        void init(
            int eig_stream_num_per_gpu,
            // core data
            int vec_len, int con_num,
            int* cpu_At_csc_col_ptrs, int* cpu_At_csc_row_ids, double* cpu_At_csc_vals, int At_nnz,
            int* cpu_b_indices, double* cpu_b_vals, int b_nnz,
            int* cpu_C_indices, double* cpu_C_vals, int C_nnz,
            char* cpu_blk_types,
            int* cpu_blk_sizes, int mat_num,
            ProjectionMethod initial_proj_method = ProjectionMethod::COMPOSITE_FP16,
            ProjectionMethod final_proj_method = ProjectionMethod::EIG_FP64,
            double* cpu_X_vals = nullptr, // |
            double* cpu_y_vals = nullptr, // |- values for warm start
            double* cpu_S_vals = nullptr, // |
            double sig = 1.0
        );

        /// @brief Solves the SDP problem using the sGS-ADMM algorithm.
        /// @param max_iter Maximum number of iterations.
        /// @param stop_tol Stopping tolerance for KKT residual.
        /// @param sig_update_threshold
        /// @param sig_update_stage_1
        /// @param sig_update_stage_2
        /// @param switch_admm The iteration at which to switch from sGS-ADMM to standard ADMM.
        /// @param sigscale 
        /// @param if_first Boolean flag to indicate if this is the first call to solve. For the second call, we assume that new X, y, S, sig are passed, but that they are unscaled.
        void solve(
            int max_iter, double stop_tol,
            int sig_update_threshold = 500,
            int sig_update_stage_1 = 50,
            int sig_update_stage_2 = 100,
            int switch_admm = 0,
            int switch_proj_iter = 5000,
            double switch_proj_tol = 1e-2,
            double sigscale = 2,
            bool if_first = true
        );

        // Synchronizes the three streams of GPU0.
        void synchronize_gpu0_streams();
};

#endif // CUADMM_SOLVER_H