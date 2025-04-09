/*

    solver.h

    Main solver header.
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

// Main solver class for the SDP problem.
// Uses the sGS-ADMM algorithm to solve the problem:
//      min(X) <C, X>  s.t. A(X) = b,       X >= 0,
// of dual:
//   max(y, S) <b, y>  s.t. At(y) + S = C,  S >= 0
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
        DeviceDenseVector<double> Aty;
        DeviceDenseVector<double> Rp;
        DeviceDenseVector<double> SmC;
        DeviceDenseVector<double> Rd;
        DeviceDenseVector<double> Rporg;
        DeviceDenseVector<double> Rdorg;
        double norm_borg;
        double norm_Corg;
        double bscale;
        double Cscale;
        double objscale;
        double errRp;
        double errRd;
        double maxfeas;
        double pobj;
        double dobj;
        double relgap;
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

        /* GPU and CPU eigen decomposition + X, S computation */
        bool if_gpu_eig_mom; // if true, use multiple GPUs for eig decomposition
        DeviceDenseVector<double> Xold;
        DeviceDenseVector<double> Xb;
        
        /* Sparse vector <-> sparse matrix mapping */
        int LARGE;        // size of moment matrices
        int SMALL;        // size of localization matrices
        int mom_mat_num;  // number of moment matrices
        int loc_mat_num;  // number of localization matrices
        DeviceDenseVector<int> map_B;  // |
        DeviceDenseVector<int> map_M1; // |- maps for vectorization of matrices
        DeviceDenseVector<int> map_M2; // |    (cached from get_maps())
        
        /* Moment matrix decomposition on multiple GPU */
        std::vector<DeviceDenseVector<double>> mom_mat_arr;
        std::vector<DeviceDenseVector<double>> mom_W_arr;
        std::vector<DeviceDenseVector<int>> mom_info_arr;
        int device_num_requested;      // number of GPUs requested by the suer
        int eig_stream_num_per_gpu;    // number of streams per GPU
        std::vector<int> mom_mat_num_col_ptrs_arr;
        std::vector<std::vector<DeviceStream>> eig_stream_arr;
        std::vector<std::vector<DeviceSolverDnHandle>> cusolverH_eig_mom_arr;
        SingleEigParameter eig_param_single;
        size_t eig_mom_buffer_size;                                 // | GPU eig dec.
        std::vector<DeviceDenseVector<double>> eig_mom_buffer_arr;  // | (size and buffer)
        
        /* Moment matrix eigen decomposition: multi-core CPU */
        size_t cpu_eig_mom_buffer_size;                              // | CPU eig dec.
        std::vector<HostDenseVector<double>> cpu_eig_mom_buffer_arr; // | (size and buffer)
        HostDenseVector<double> cpu_mom_mat;
        HostDenseVector<double> cpu_mom_W;
        HostDenseVector<ptrdiff_t> cpu_mom_info;
        int cpu_eig_thread_num;
        std::vector<int> cpu_eig_col_ptrs_arr;
        int cpu_eig_mom_lwork;
        int cpu_eig_mom_lwork2;
        HostDenseVector<double> cpu_eig_mom_workspace;
        HostDenseVector<ptrdiff_t> cpu_eig_mom_workspace_2;
        DeviceDenseVector<int> loc_info;
        BatchEigParameter eig_param_batch;
        DeviceSolverDnHandle cusolverH_eig_loc;
        size_t eig_loc_buffer_size;
        DeviceDenseVector<double> eig_loc_buffer;
        /* Projection on PSD cones */
        DeviceDenseVector<double> mom_mat_tmp;
        DeviceDenseVector<double> loc_mat_tmp;
        DeviceDenseVector<double> mom_mat_P;
        DeviceDenseVector<double> loc_mat_P;

        /* Other */
        std::vector<std::vector<DeviceStream>> stream_flex_arr;
        std::vector<DeviceSparseHandle> cusparseH_flex_arr;
        std::vector<DeviceBlasHandle> cublasH_flex_arr;
        DeviceSparseHandle cusparseH;
        DeviceBlasHandle cublasH;

        /* Rescale and update sig */
        int prim_win;
        int dual_win;
        int rescale;
        double normy;
        double normAty;
        double normX;
        double normS;
        double normyS;
        double bscale2;
        double Cscale2;
        double ratioconst;
        double feasratio;
        double sigmax;
        double sigmin;
        double sigscale;

        /* Info */
        int info_iter_num;
        std::vector<double> info_pobj_arr;
        std::vector<double> info_dobj_arr;
        std::vector<double> info_errRp_arr;
        std::vector<double> info_errRd_arr;
        std::vector<double> info_relgap_arr;
        std::vector<double> info_sig_arr;
        std::vector<double> info_bscale_arr;
        std::vector<double> info_Cscale_arr;

        /* Time */
        cudaEvent_t start;
        cudaEvent_t stop;
        double total_time; // count time in seconds

        /* sGS-ADMM */
        double tau;
        DeviceDenseVector<double> Xproj;
        DeviceDenseVector<double> Xdiff;
        int switch_admm;
        int eig_rank;
        int begin_low_rank_proj;
        DeviceDenseVector<int> mom_W_rank_mask;
        DeviceDenseVector<int> loc_W_rank_mask;
        int sig_update_threshold;
        int sig_update_stage_1;
        int sig_update_stage_2;
        double sgs_KKT;
        double best_KKT;
        DeviceDenseVector<double> X_best;
        DeviceDenseVector<double> y_best;
        DeviceDenseVector<double> S_best;

        SDPSolver() {}

        // Initializes an SDPSolver.
        void init(
            // eig
            bool if_gpu_eig_mom,
            // do moment matrix eigen decomposition on GPU
            int device_num_requested,
            int eig_stream_num_per_gpu,
            // do moment matrix eigen decomposition on CPU
            int cpu_eig_thread_num,

            // core data
            int vec_len, int con_num,
            int* cpu_At_csc_col_ptrs, int* cpu_At_csc_row_ids, double* cpu_At_csc_vals, int At_nnz,
            int* cpu_b_indices, double* cpu_b_vals, int b_nnz,
            int* cpu_C_indices, double* cpu_C_vals, int C_nnz,
            int* cpu_blk_vals, int mat_num,
            double* cpu_X_vals = nullptr, // |
            double* cpu_y_vals = nullptr, // |- values for warm start
            double* cpu_S_vals = nullptr, // |
            double sig = 2e2
        );

        // Solves the SDP problem using the sGS-ADMM algorithm.
        //
        // Args:
        // - max_iter: maximum number of iterations
        // - stop_tol: stopping tolerance for KKT residual
        // - sig_update_threshold:
        // - sig_update_stage_1:
        // - sig_update_stage_2:
        // - switch_admm:
        // - sigscale:
        // - if_first: if this is the first call to solve()
        void solve(
            int max_iter, double stop_tol,
            int sig_update_threshold = 500,
            int sig_update_stage_1 = 50,
            int sig_update_stage_2 = 100,
            int switch_admm = (int) 1.1e4,
            double sigscale = 1.05,
            bool if_first = true
        );

        // Synchronizes the three streams of GPU0.
        void synchronize_gpu0_streams();
};

#endif // CUADMM_SOLVER_H