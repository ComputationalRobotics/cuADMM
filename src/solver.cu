/*

    solver.cu

    Main solver, works for any sizes of matrices.
    Uses the sGS-ADMM algorithm to solve an SDP problem.

*/

#include "cuadmm/solver.h"
#include "cuadmm/kernels.h"
#include "cuadmm/projection.h"

#include <algorithm>
#include <stdio.h>

#define SIG_UPDATE_THRESHOLD 500
#define SIG_UPDATE_STAGE_1 50
#define SIG_UPDATE_STAGE_2 100
#define SIG_SCALE 1.05

void SDPSolver::synchronize_gpu0_streams() {
    CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[0].stream) );
    CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[1].stream) );
    CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[2].stream) );
}

void SDPSolver::init(
    int eig_stream_num_per_gpu,
    int cpu_eig_thread_num,
    int vec_len, int con_num,
    int* cpu_At_csc_col_ptrs, int* cpu_At_csc_row_ids, double* cpu_At_csc_vals, int At_nnz,
    int* cpu_b_indices, double* cpu_b_vals, int b_nnz,
    int* cpu_C_indices, double* cpu_C_vals, int C_nnz,
    char* cpu_blk_types, int* cpu_blk_sizes,
    int mat_num,
    double* cpu_X_vals,
    double* cpu_y_vals,
    double* cpu_S_vals,
    double sig
) {
    // start record time
    this->total_time = 0.0;
    cudaEventCreate(&this->start);
    cudaEventCreate(&this->stop);
    cudaEventRecord(this->start);

    // prepare streams for copy data
    /*
    we create three flexible streams per GPU, corresponding to copy mom_mat, mom_W, mom_info
    they can also be used to parallelize kernel launches and cuda toolkit calls
    */
    this->stream_flex = std::vector<DeviceStream>(3);
    for (int stream_id = 0; stream_id < 3; stream_id++) {
        this->stream_flex[stream_id].set_gpu_id(GPU0);
        this->stream_flex[stream_id].activate();
    }

    // create handles for cuSPARSE and cuBLAS
    this->cusparseH.set_gpu_id(GPU0);
    this->cusparseH.activate();
    this->cublasH.set_gpu_id(GPU0);
    this->cublasH.activate();

    /* Initialize the A matrix */
    this->vec_len = vec_len;
    this->con_num = con_num;
    this->At_csc.allocate(GPU0, vec_len, con_num, At_nnz);
    this->At_csr.allocate(GPU0, vec_len, con_num, At_nnz);
    this->A_csr.allocate(GPU0, con_num, vec_len, At_nnz);
    // first stream for col_ptrs
    CHECK_CUDA( cudaMemcpyAsync(this->At_csc.col_ptrs, cpu_At_csc_col_ptrs, sizeof(int) * (con_num + 1), H2D, this->stream_flex[0].stream) );
    // second stream for row_ids
    CHECK_CUDA( cudaMemcpyAsync(this->At_csc.row_ids, cpu_At_csc_row_ids, sizeof(int) * At_nnz, H2D, this->stream_flex[1].stream) );
    // third stream for vals
    CHECK_CUDA( cudaMemcpyAsync(this->At_csc.vals, cpu_At_csc_vals, sizeof(double) * At_nnz, H2D, this->stream_flex[2].stream) );
    // wait for the streams to finish
    this->synchronize_gpu0_streams();

    // compute the norm of A
    this->normA.allocate(GPU0, con_num);
    get_normA(this->At_csc, this->normA);

    /* convert the At matrix from CSC to CSR format */
    this->CSCtoCSR_At2A_buffer_size = CSC_to_CSR_get_buffersize_cusparse(this->cusparseH, this->At_csc, this->At_csr);
    this->CSCtoCSR_At2A_buffer.allocate(GPU0, CSCtoCSR_At2A_buffer_size, true);
    CSC_to_CSR_cusparse(this->cusparseH, this->At_csc, this->At_csr, this->CSCtoCSR_At2A_buffer);
    CHECK_CUDA( cudaMemcpyAsync(this->A_csr.row_ptrs ,this->At_csc.col_ptrs, sizeof(int) * (con_num + 1), D2D, this->stream_flex[0].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->A_csr.col_ids ,this->At_csc.row_ids, sizeof(int) * At_nnz, D2D, this->stream_flex[1].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->A_csr.vals ,this->At_csc.vals, sizeof(double) * At_nnz, D2D, this->stream_flex[2].stream) );

    /* Initialize the AAt solver on CPU */
    this->cpu_AAt_solver.get_A(
        this->At_csr.row_ptrs, this->At_csr.col_ids, this->At_csr.vals,
        this->At_csr.col_size, this->At_csr.row_size, this->At_csr.nnz,
        true, 1e-15
    );
    this->cpu_AAt_solver.factorize();
    // retrieve permutation of the L factor
    this->perm.allocate(GPU0, con_num);
    CHECK_CUDA( cudaMemcpyAsync(this->perm.vals, this->cpu_AAt_solver.chol_fac_L->Perm, sizeof(int) * con_num, H2D, this->stream_flex[0].stream) );
    // allocate memory of right-hand side vector
    this->rhsy.allocate(GPU0, con_num);
    this->rhsy_perm.allocate(GPU0, con_num);
    this->y_perm.allocate(GPU0, con_num);
    // compute inverse permutation
    std::vector<int> perm_tmp(con_num, 0);
    std::vector<int> perm_inv_tmp;
    memcpy(perm_tmp.data(), this->cpu_AAt_solver.chol_fac_L->Perm, sizeof(int) * con_num);
    this->perm_inv.allocate(GPU0, con_num);
    get_inverse_permutation(perm_inv_tmp, perm_tmp);
    CHECK_CUDA( cudaMemcpyAsync(this->perm_inv.vals, perm_inv_tmp.data(), sizeof(int) * con_num, H2D, this->stream_flex[1].stream) );

    /* Initialize b, C, X, y, S, sig on GPU */
    this->b.allocate(GPU0, con_num, b_nnz);
    this->C.allocate(GPU0, vec_len, C_nnz);
    this->X.allocate(GPU0, vec_len);
    this->y.allocate(GPU0, con_num);
    this->S.allocate(GPU0, vec_len);
    CHECK_CUDA( cudaMemcpyAsync(this->b.indices, cpu_b_indices, sizeof(int) * b_nnz, H2D, this->stream_flex[0].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->b.vals, cpu_b_vals, sizeof(double) * b_nnz, H2D, this->stream_flex[1].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->C.indices, cpu_C_indices, sizeof(int) * C_nnz, H2D, this->stream_flex[2].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->C.vals, cpu_C_vals, sizeof(double) * C_nnz, H2D, this->stream_flex[0].stream) );

    // copy X, y, and S from CPU to GPU
    // if the input is nullptr (no warm start), we will set them to 0
    if (cpu_X_vals != nullptr) {
        // copy
        CHECK_CUDA( cudaMemcpyAsync(this->X.vals, cpu_X_vals, sizeof(double) * vec_len, H2D, this->stream_flex[1].stream) );
    } else {
        // set to 0
        CHECK_CUDA( cudaMemsetAsync(this->X.vals, 0, sizeof(double) * vec_len, this->stream_flex[1].stream) );
    }
    if (cpu_y_vals != nullptr) {
        CHECK_CUDA( cudaMemcpyAsync(this->y.vals, cpu_y_vals, sizeof(double) * con_num, H2D, this->stream_flex[2].stream) );
    } else {
        CHECK_CUDA( cudaMemsetAsync(this->y.vals, 0, sizeof(double) * con_num, this->stream_flex[2].stream) );
    }
    if (cpu_S_vals != nullptr) {
        CHECK_CUDA( cudaMemcpyAsync(this->S.vals, cpu_S_vals, sizeof(double) * vec_len, H2D, this->stream_flex[0].stream) );
    } else {
        CHECK_CUDA( cudaMemsetAsync(this->S.vals, 0, sizeof(double) * vec_len, this->stream_flex[0].stream) );
    }
    this->sig = sig;

    /* Initialize blk and maps */
    // copy blk values and analyze it to retrieve the block sizes and numbers
    HostDenseVector<int> host_blk_sizes(mat_num);
    memcpy(host_blk_sizes.vals, cpu_blk_sizes, sizeof(int) * mat_num);
    analyze_blk(cpu_blk_types, host_blk_sizes, this->psd_blk_sizes, this->psd_blk_nums);
    this->sizes.init(this->psd_blk_sizes, this->psd_blk_nums);

    /* Compute the maps for vectorization of matrices */
    // compute on CPU
    std::vector<int> map_B_tmp;  // |
    std::vector<int> map_M1_tmp; // |- CPU version
    std::vector<int> map_M2_tmp; // |
    get_maps(cpu_blk_types, host_blk_sizes, this->vec_len, map_B_tmp, map_M1_tmp, map_M2_tmp, this->sizes);

    // copy to GPU
    this->map_B.allocate(GPU0, vec_len);  // |
    this->map_M1.allocate(GPU0, vec_len); // |- GPU version
    this->map_M2.allocate(GPU0, vec_len); // |
    CHECK_CUDA( cudaMemcpyAsync(this->map_B.vals, map_B_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex[0].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->map_M1.vals, map_M1_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex[1].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->map_M2.vals, map_M2_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex[2].stream) );

    /* Scale (A is already scaled) */
    // move b and C to GPU
    this->borg.allocate(GPU0, this->con_num, this->b.nnz);
    this->Corg.allocate(GPU0, this->vec_len, this->C.nnz);
    CHECK_CUDA( cudaMemcpyAsync(this->borg.indices, this->b.indices, sizeof(int) * this->b.nnz, D2D, this->stream_flex[0].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->borg.vals, this->b.vals, sizeof(double) * this->b.nnz, D2D, this->stream_flex[1].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->Corg.indices, this->C.indices, sizeof(int) * this->C.nnz, D2D, this->stream_flex[2].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->Corg.vals, this->C.vals, sizeof(double) * this->C.nnz, D2D, this->stream_flex[0].stream) );
    this->synchronize_gpu0_streams();
    // compute the norms of b and C
    this->norm_borg = 1 + this->borg.get_norm(this->cublasH);
    this->norm_Corg = 1 + this->Corg.get_norm(this->cublasH);

    // scale b and C by normA
    sparse_vector_div_dense_vector(this->b, this->normA);
    dense_vector_mul_dense_vector(this->y, this->normA);
    // divide b, C, X, y, and S by the corresponding norms
    this->bscale = 1 + this->b.get_norm(this->cublasH);
    this->Cscale = 1 + this->C.get_norm(this->cublasH);
    this->objscale = this->bscale * this->Cscale;
    sparse_vector_div_scalar(this->b, this->bscale);
    sparse_vector_div_scalar(this->C, this->Cscale);
    dense_vector_div_scalar(this->X, this->bscale);
    dense_vector_div_scalar(this->S, this->Cscale);
    dense_vector_div_scalar(this->y, this->Cscale);

    /* Initialize KKT residuals */
    // simple allocations
    this->Aty.allocate(GPU0, this->vec_len);
    this->Rp.allocate(GPU0, this->con_num);
    this->SmC.allocate(GPU0, this->vec_len);
    this->Rd.allocate(GPU0, this->vec_len);
    this->Rporg.allocate(GPU0, this->con_num);
    this->Rdorg.allocate(GPU0, this->vec_len);

    // retrieve buffer sizes and allocate
    this->SpMV_Aty_buffer_size = SpMV_get_buffersize_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0);
    this->SpMV_Aty_buffer.allocate(GPU0, this->SpMV_Aty_buffer_size, true);
    SpMV_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0, this->SpMV_Aty_buffer);
    this->SpMV_AX_buffer_size = SpMV_get_buffersize_cusparse(this->cusparseH, this->A_csr, this->X, this->Rp, -1.0, 0.0);
    this->SpMV_AX_buffer.allocate(GPU0, this->SpMV_AX_buffer_size, true);
    SpMV_cusparse(this->cusparseH, this->A_csr, this->X, this->Rp, -1.0, 0.0, this->SpMV_AX_buffer);

    //
    axpby_cusparse(this->cusparseH, this->b, this->Rp, 1.0, 1.0);
    CHECK_CUDA( cudaMemcpy(this->SmC.vals, this->S.vals, sizeof(double) * this->vec_len, D2D) );
    axpby_cusparse(this->cusparseH, this->C, this->SmC, -1.0, 1.0);
    dense_vector_add_dense_vector(this->Rd, this->Aty, this->SmC);
    dense_vector_mul_dense_vector_mul_scalar(this->Rporg, this->normA, this->Rp, this->bscale);
    dense_vector_mul_scalar(this->Rdorg, this->Rd, this->Cscale);

    // compute initial residuals
    this->errRp = this->Rporg.get_norm(this->cublasH) / this->norm_borg;
    this->errRd = this->Rdorg.get_norm(this->cublasH) / this->norm_Corg;
    this->maxfeas = max(this->errRp, this->errRd);
    this->SpVV_CtX_buffer_size = SparseVV_get_buffersize_cusparse(this->cusparseH, this->C, this->X);
    this->SpVV_CtX_buffer.allocate(GPU0, this->SpVV_CtX_buffer_size, true);
    this->pobj = SparseVV_cusparse(this->cusparseH, this->C, this->X, this->SpVV_CtX_buffer) * this->objscale;
    this->SpVV_bty_buffer_size = SparseVV_get_buffersize_cusparse(this->cusparseH, this->b, this->y);
    this->SpVV_bty_buffer.allocate(GPU0, this->SpVV_bty_buffer_size, true);
    this->dobj = SparseVV_cusparse(this->cusparseH, this->b, this->y, this->SpVV_bty_buffer) * this->objscale;
    this->relgap = abs(this->pobj - this->dobj) / (1 + abs(this->pobj) + abs(this->dobj));

    /* Eigen decomposition for large matrices */
    // allocate GPU0 memory for large matrices
    this->large_mat.allocate(GPU0, this->sizes.total_large_mat_size);
    this->large_W.allocate(GPU0, this->sizes.sum_large_mat_size);
    this->large_info.allocate(GPU0, this->sizes.large_mat_num);

    // if the decomposition is on GPU, use cuSOLVER (cf cusolver.h)
    this->eig_stream_num_per_gpu = eig_stream_num_per_gpu;

    // streams and handles for eigen decomposition
    this->eig_stream_arr = std::vector<DeviceStream>(this->eig_stream_num_per_gpu);
    this->cusolverH_eig_large_arr = std::vector<DeviceSolverDnHandle>(this->eig_stream_num_per_gpu);
    for (int stream_id = 0; stream_id < this->eig_stream_num_per_gpu; stream_id++) {
        // ininitialize and activate the streams and handles
        this->eig_stream_arr[stream_id].set_gpu_id(GPU0);
        this->eig_stream_arr[stream_id].activate();
        this->cusolverH_eig_large_arr[stream_id].set_gpu_id(GPU0);
        this->cusolverH_eig_large_arr[stream_id].activate(this->eig_stream_arr[stream_id]);
    }
    
    // compute the buffer sizes of the large matrices eig decomposition
    this->eig_large_buffer_size.assign(this->sizes.large_mat_sizes.size(), 0);
    // this->eig_large_buffer.reserve(this->sizes.large_mat_sizes.size());
    this->cpu_eig_large_buffer_size.assign(this->sizes.large_mat_sizes.size(), 0);
    // this->cpu_eig_large_buffer.reserve(this->sizes.large_mat_sizes.size());

    // TODO: comment out this part
    this->sizes.large_buffer_start_indices.push_back(0);
    this->sizes.large_cpu_buffer_start_indices.push_back(0);
    int total_eig_large_buffer_size = 0;
    int total_cpu_eig_large_buffer_size = 0;
    for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
        single_eig_get_buffersize_cusolver(
            this->cusolverH_eig_large_arr[i % this->eig_stream_num_per_gpu], eig_param_single, this->large_mat, this->large_W,
            this->sizes.large_mat_sizes[i],
            &this->eig_large_buffer_size[i],
            &this->cpu_eig_large_buffer_size[i],
            this->sizes.large_mat_offset(i, 0), this->sizes.large_W_offset(i, 0)
        ); // buffer size per large matrix of a given size

        // we need to multiply the buffer size by the number of matrices of this size
        total_eig_large_buffer_size += this->eig_large_buffer_size[i] * this->sizes.large_mat_nums[i];
        total_cpu_eig_large_buffer_size += this->cpu_eig_large_buffer_size[i] * this->sizes.large_mat_nums[i];

        this->sizes.large_buffer_start_indices.push_back(
            this->sizes.large_buffer_start_indices[i] + this->sizes.large_mat_nums[i] * this->eig_large_buffer_size[i]
        );
        this->sizes.large_cpu_buffer_start_indices.push_back(
            this->sizes.large_cpu_buffer_start_indices[i] + this->sizes.large_mat_nums[i] * this->cpu_eig_large_buffer_size[i]
        );
    }

    // allocate memory for the two buffers, host and device
    this->eig_large_buffer.allocate(GPU0, total_eig_large_buffer_size, true);
    this->cpu_eig_large_buffer.allocate(total_cpu_eig_large_buffer_size, true);

    /* Eigenvalue decomposition for small matrices */
    this->cusolverH_eig_small.set_gpu_id(GPU0);
    this->cusolverH_eig_small.activate();
    this->small_mat.allocate(GPU0, this->sizes.total_small_mat_size);
    this->small_W.allocate(GPU0, this->sizes.sum_small_mat_size);
    this->small_info.allocate(GPU0, this->sizes.small_mat_num);
    this->eig_small_buffer_size.reserve(this->sizes.small_mat_sizes.size());

    this->sizes.small_buffer_start_indices.push_back(0);
    for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++) {
        this->eig_small_buffer_size.push_back(
            batch_eig_get_buffersize_cusolver(
                this->cusolverH_eig_small, this->eig_param_batch,
                this->small_mat, this->small_W,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->sizes.small_mat_offset(i), this->sizes.small_W_offset(i)
            )
        );

        this->sizes.small_buffer_start_indices.push_back(
            this->sizes.small_buffer_start_indices[i] + this->eig_small_buffer_size[i]
        );
    }
    
    CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[0].stream) );
    // we do not need to multiply the buffer size by the number of matrices,
    // since it is already done in the function
    this->eig_small_buffer.allocate(GPU0, this->sizes.small_buffer_start_indices.back(), true);

    /* For the computation of y, X, S */
    this->large_mat_tmp.allocate(GPU0, this->sizes.total_large_mat_size);
    this->small_mat_tmp.allocate(GPU0, this->sizes.total_small_mat_size);
    this->large_mat_P.allocate(GPU0, this->sizes.total_large_mat_size); // TODO: remove this
    this->small_mat_P.allocate(GPU0, this->sizes.total_small_mat_size);
    this->Rd1.allocate(GPU0, this->vec_len);
    this->Xb.allocate(GPU0, this->vec_len);

    /* others */
    this->prim_win = 0;
    this->dual_win = 0;
    this->ratioconst = 1e0;
    this->sigmax = 1e3;
    this->sigmin = 1e-3;

    /* Main elements for the sGS-ADMM algorithm */
    this->Xproj.allocate(GPU0, this->vec_len);
    this->Xdiff.allocate(GPU0, this->vec_len);
    this->switch_admm = (int) 5e4;
    this->sig_update_threshold = SIG_UPDATE_THRESHOLD;
    this->sig_update_stage_1 = SIG_UPDATE_STAGE_1;
    this->sig_update_stage_2 = SIG_UPDATE_STAGE_2;
    this->sigscale = SIG_SCALE;
    this->X_best.allocate(GPU0, this->vec_len);
    this->y_best.allocate(GPU0, this->con_num);
    this->S_best.allocate(GPU0, this->vec_len);

    return;
}

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
// - if_first: if this is the first call to solve() (optional)
void SDPSolver::solve(
    int max_iter, double stop_tol,
    int sig_update_threshold,
    int sig_update_stage_1,
    int sig_update_stage_2,
    int switch_admm,
    double sigscale,
    bool if_first
) {
    // save parameters
    this->sig_update_threshold = sig_update_threshold;
    this->sig_update_stage_1 = sig_update_stage_1;
    this->sig_update_stage_2 = sig_update_stage_2;
    this->switch_admm = switch_admm;
    this->sigscale = sigscale;

    // declare variables
    bool breakyes = false;   // for breaking out of the loop
    std::string final_msg;   // output message

    this->info_iter_num = 0; // iteration number

    /* Start the solver */
    printf("\n -------------------------------------------------------------------------------");
    printf("\n                                    cuADMM");
    printf("\n -------------------------------------------------------------------------------");
    printf("\n norm of C = %2.1e, norm of b = %2.1e\n", norm_Corg, norm_borg);
    float milliseconds;
    float seconds;

    if (!if_first) {
        // we suppose that for the second call, new X, y, S, sig are passed, but they are unscaled

        // scale X, y, S
        dense_vector_mul_dense_vector(this->y, this->normA);
        dense_vector_div_scalar(this->X, this->bscale);
        dense_vector_div_scalar(this->S, this->Cscale);
        dense_vector_div_scalar(this->y, this->Cscale);

        // SmC <-- S
        CHECK_CUDA( cudaMemcpy(this->SmC.vals, this->S.vals, sizeof(double) * this->vec_len, D2D) );
        // hence Smc = S

        // SmC <-- -1.0 * C + 1.0 * SmC
        axpby_cusparse(this->cusparseH, this->C, this->SmC, -1.0, 1.0);
        // hence SmC = S - C

        // Rp <-- -1.0 * A * X + 0.0 * Rp
        SpMV_cusparse(this->cusparseH, this->A_csr, this->X, this->Rp, -1.0, 0.0, this->SpMV_AX_buffer);
        // hence Rp = - A X

        // Rp <-- 1.0 * b + 1.0 * Rp
        axpby_cusparse(this->cusparseH, this->b, this->Rp, 1.0, 1.0);
        // hence Rp = b - A X
    }

    std::cout << std::endl << "  it. | p infeas d infeas | primal obj.   dual obj. rel. gap |  time |   sigma | " << std::endl;
    std::cout << " -------------------------------------------------------------------------------" << std::endl;

    // for each iteration of the main solver
    for (int iter = 1; iter <= max_iter + 1; iter++) {
        /*
            Step 0: Check if terminal conditions hold and log information
        */
        if (max(this->maxfeas, this->relgap) < stop_tol ) {
            // stop if the stopping criterion is met
            breakyes = true;
            final_msg = "Solver ended: converged.";
        }
        if (iter > max_iter) {
            // stop if the maximum number of iterations is reached
            breakyes = true;
            final_msg = "Solver ended: maximum iteration reached";
        }
        if (
            true || // TODO: remove
            ( breakyes == true ) ||
            ( (iter <= 200) && ((iter % 50) == 1) ) ||
            ( (iter > 200) && ((iter % 100) == 1) )
        ) {
            // print the iteration number and the residuals
            cudaEventRecord(this->stop);
            cudaEventSynchronize(this->stop);
            cudaEventElapsedTime(&milliseconds, this->start, this->stop);
            seconds = milliseconds / 1000;
            printf(
                " %4d | %3.2e %3.2e | %- 5.4e %- 5.4e %3.2e | %5.1f | %2.1e |",
                iter-1, this->errRp, this->errRd, this->pobj, this->dobj, this->relgap, seconds, this->sig
            );
            std::cout << std::endl;
        }
        if (breakyes > 0) {
            // print the final message
            printf("\n -------------------------------------------------------------------------------\n\n");
            std::cout << final_msg << std::endl;
            printf(
                "\n primal infeasibility = %2.1e \n dual   infeasibility = %2.1e \n relative gap         = %2.1e",
                this->errRp, this->errRd, this->relgap
            );
            printf(
                "\n primal objective = %- 9.8e \n dual   objective = %- 9.8e",
                this->pobj, this->dobj
            );
            printf(
                "\n\n time per iteration = %2.4fs \n total time         = %2.1fs",
                seconds/iter, seconds
            );
            printf("\n -------------------------------------------------------------------------------\n\n");

            cudaEventRecord(this->stop);
            cudaEventSynchronize(this->stop);
            cudaEventElapsedTime(&milliseconds, this->start, this->stop);
            this->total_time = milliseconds / 1000;
        }

        /*
            Step 1: Compute
                        r_s^{k+1/2} = 1/sigma b - A(X/sigma + S^k - C)
                                             and
                               y^{k+1/2} = (AA^T)^{-1} r_s^{k+1/2}
        */

        /* r_s^{k+1/2} = b/sigma - A(X/sigma + S - C) */
        // rhsy <-- -1.0 * A * SmC + 0.0 * rhsy
        SpMV_cusparse(this->cusparseH, this->A_csr, this->SmC, this->rhsy, -1.0, 0.0, this->SpMV_AX_buffer);
        // hence rhsy = - A S

        // rhsy <-- 1/sig * Rp + rhsy
        axpy_cublas(this->cublasH, this->Rp, this->rhsy, 1/this->sig);
        // hence rhsy = 1/sig * Rp - A S

        /* y^{k+1/2} = (AA^T)^{-1} r_s^{k+1/2} */
        // y <-- linsys(rhsy)
        perform_permutation(this->rhsy_perm, this->rhsy, this->perm_inv);
        CHECK_CUDA( cudaDeviceSynchronize() );
        CHECK_CUDA( cudaMemcpyAsync(
            this->cpu_AAt_solver.chol_dn_rhs->x, this->rhsy_perm.vals,
            sizeof(double) * this->con_num, D2H, this->stream_flex[0].stream
        ) );
        CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[0].stream) );
        this->cpu_AAt_solver.solve();
        CHECK_CUDA( cudaMemcpyAsync(
            this->y_perm.vals, this->cpu_AAt_solver.chol_dn_res->x,
            sizeof(double) * this->con_num, H2D, this->stream_flex[0].stream
        ) );
        CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[0].stream) );
        perform_permutation(this->y, this->y_perm, this->perm);
        // hence y = (AA^T)^{-1} r_s^{k+1/2}


        /*
            Step 2: Compute the optimization variables :

                    X_b^{k+1} = X^k + sigma(A^T y^{k+1/2} - C)
                                         and
                    S^{k+1} = 1/sigma (Pi(X_b^{k+1}) - X_b^{k+1})
        */

        /* Compute X^{k+1} */
        // Aty <-- 1.0 * At * y + 0.0 * Aty
        SpMV_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0, this->SpMV_Aty_buffer);
        // hence Aty = A^T y^{k+1/2}

        // Rd1 <-- Aty
        CHECK_CUDA( cudaMemcpy(this->Rd1.vals, this->Aty.vals, sizeof(double) * this->vec_len, D2D) );
        // Rd1 <-- (-1.0) * C + 1.0 * Rd1
        axpby_cusparse(this->cusparseH, this->C, this->Rd1, -1.0, 1.0);
        // hence Rd1 = A^T y^{k+1/2} - C

        double norm_rhsy = this->rhsy.get_norm(this->cublasH);
        double norm_y = this->y.get_norm(this->cublasH);

        // Xb <-- X + sig * Rd1
        dense_vector_plus_dense_vector_mul_scalar(this->Xb, this->X, this->Rd1, this->sig);
        // hence Xb = X^k + sig * (A^T y^{k+1/2} - C) = X^{k+1}


        /* Compute Pi(X^{k+1}) (this is long) */

        // first, we convert Xb back to matrices (large and small)
        vector_to_matrices(this->Xb, this->large_mat, this->small_mat, this->map_B, this->map_M1, this->map_M2);

        // we perform the GPU decomposition of large matrices
        // for each large matrix on this GPU, compute the eig decomposition
        // int stream_id;
        // int counter = 0; // serves as a stream id and as an info offset

        // TODO: move the cuBLAS handle creation outside the loop
        cublasHandle_t cublasH;
        CHECK_CUBLAS( cublasCreate(&cublasH) );
        CHECK_CUBLAS( cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH) );

        for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
            for (int j = 0; j < this->sizes.large_mat_nums[i]; j++) {
                // stream_id = counter % this->eig_stream_num_per_gpu;

                // simply calls the cuSOLVER wrapper
                // single_eig_cusolver(
                //     this->cusolverH_eig_large_arr[stream_id], eig_param_single,
                //     this->large_mat, this->large_W,
                //     this->eig_large_buffer, this->cpu_eig_large_buffer, this->large_info,
                //     this->sizes.large_mat_sizes[i],
                //     this->eig_large_buffer_size[i], this->cpu_eig_large_buffer_size[i],
                //     this->sizes.large_mat_offset(i, j), this->sizes.large_W_offset(i, j),
                //     this->sizes.large_buffer_offset(i, j, this->eig_large_buffer_size),
                //     this->sizes.large_cpu_buffer_offset(i, j, this->eig_large_buffer_size),
                //     counter
                // );

                // use the custom routine
                projection_TF16(
                    cublasH,
                    this->large_mat,
                    this->sizes.large_mat_sizes[i],
                    this->sizes.large_mat_offset(i, j)
                );

                // counter++;
            }
        }
        cublasDestroy(cublasH);
        // for each stream, synchronize
        for (int stream_id = 0; stream_id < this->eig_stream_num_per_gpu; stream_id++) {
            CHECK_CUDA( cudaStreamSynchronize(this->eig_stream_arr[stream_id].stream) );
        }

        // we perform an ADMM switch
        if (breakyes) {
            if (iter > this->switch_admm) {
                CHECK_CUDA( cudaMemcpyAsync(this->X.vals, this->X_best.vals, sizeof(double) * this->vec_len, D2D, this->stream_flex[0].stream) );
                CHECK_CUDA( cudaMemcpyAsync(this->y.vals, this->y_best.vals, sizeof(double) * this->con_num, D2D, this->stream_flex[1].stream) );
                CHECK_CUDA( cudaMemcpyAsync(this->S.vals, this->S_best.vals, sizeof(double) * this->vec_len, D2D, this->stream_flex[2].stream) );
                this->synchronize_gpu0_streams();
                printf("best max KKT residual after switch  = %2.1e \n", this->best_KKT);
            }
            break;
        }

        // we call cuSOLVER for the batch eig decomposition of small matrices
        int info_offset = 0;
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++) {
            batch_eig_cusolver(
                this->cusolverH_eig_small, this->eig_param_batch,
                this->small_mat, this->small_W,
                this->eig_small_buffer, this->small_info,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->eig_small_buffer_size[i],
                this->sizes.small_mat_offset(i), this->sizes.small_W_offset(i),
                this->sizes.small_buffer_offset(i, this->eig_small_buffer_size),
                info_offset
            );
            info_offset += this->sizes.small_mat_nums[i];
        }

        // max_dense_vector_zero(this->large_W);
        max_dense_vector_zero(this->small_W);

        // int stream_id;
        // multiply the large matrices by their eigenvalues
        // for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
        //     // stream_id = i % this->eig_stream_num_per_gpu;
        //     dense_matrix_mul_diag_batch(
        //         large_mat_tmp, this->large_mat, this->large_W,
        //         this->sizes.large_mat_sizes[i], this->sizes.large_mat_nums[i],
        //         this->sizes.large_mat_offset(i, 0), this->sizes.large_W_offset(i, 0)//,
        //         // this->eig_stream_arr[stream_id].stream
        //     );
        // }

        // TODO: use multiple streams
        // multiply the small matrices by their eigenvalues
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++) {
            // stream_id = (this->sizes.large_mat_sizes.size() + i) % this->eig_stream_num_per_gpu;
            dense_matrix_mul_diag_batch(
                small_mat_tmp, this->small_mat, this->small_W,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->sizes.small_mat_offset(i), this->sizes.small_W_offset(i)//,
                // this->eig_stream_arr[stream_id].stream
            );
        }

        // synchronize the streams
        // for (int stream_id = 0; stream_id < this->eig_stream_num_per_gpu; stream_id++) {
        //     CHECK_CUDA( cudaStreamSynchronize(this->eig_stream_arr[stream_id].stream) );
        // }


        // for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
        //     dense_matrix_mul_trans_batch(
        //         this->cublasH,
        //         this->large_mat_P, this->large_mat_tmp, this->large_mat,
        //         this->sizes.large_mat_sizes[i], this->sizes.large_mat_nums[i],
        //         this->sizes.large_mat_offset(i, 0)
        //     );
        // }

        // TODO: use multiple cuBLAS handles
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++) {
            dense_matrix_mul_trans_batch(
                this->cublasH,
                this->small_mat_P, this->small_mat_tmp, this->small_mat,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->sizes.small_mat_offset(i)
            );
        }

        // we copy Xb to Xproj since free variables are not modified
        // TODO: only copy the free variables
        CHECK_CUDA( cudaMemcpy(this->Xproj.vals, this->Xb.vals, sizeof(double) * this->vec_len, D2D) );

        // convert the matrices back to vectorized format
        // matrices_to_vector(this->Xproj, this->large_mat_P, this->small_mat_P, this->map_B, this->map_M1, this->map_M2);
        matrices_to_vector(this->Xproj, this->large_mat, this->small_mat_P, this->map_B, this->map_M1, this->map_M2);

        /* Finish the computation of S^{k+1} */

        // Xdiff <-- 1.0 * Xproj + (-1.0) * X
        dense_vector_add_dense_vector(this->Xdiff, this->Xproj, this->X, 1.0, -1.0);
        // hence Xdiff = Pi(X^{k+1}) - X^k

        // S <-- 1/sig * Xdiff + (-1.0) * Rd1
        dense_vector_add_dense_vector(this->S, this->Xdiff, this->Rd1, 1/this->sig, -1.0);
        // hence S = 1/sig * (Pi(X^{k+1}) - X^k) - (A^T y^{k+1/2} - C)
        // which is S = 1/sig * (Pi(X^{k+1}) - X^{k+1})



        /*
            Step 3: Compute:
                        r_s^{k+1} = 1/sigma b - A(X^k/sigma + S^{k+1} - C)
                                              and
                                y^{k+1} = (AA^T)^{-1} r_s^{k+1}
        */

        /* Compute r_s^{k+1} */

        // SmC <-- S
        CHECK_CUDA( cudaMemcpy(this->SmC.vals, this->S.vals, sizeof(double) * this->vec_len, D2D) );
        // SmC <-- -1.0 * C + 1.0 * SmC
        axpby_cusparse(this->cusparseH, this->C, this->SmC, -1.0, 1.0);
        // hence SmC = S^{k+1} - C


        /* Compute y^{k+1} */
        // If the number of iterations goes large but sGS-ADMM still fail to converge,
        // switch to ordinary ADMM
        if (iter == this->switch_admm) {
            std::cout << " switching to normal ADMM!" << std::endl;
            this->sig_update_stage_2 = this->sig_update_stage_2 / 2;
            this->sigscale = this->sigscale * 1.23;
            this->sgs_KKT = max(this->maxfeas, this->relgap);
            this->best_KKT = this->sgs_KKT;
            CHECK_CUDA( cudaMemcpyAsync(this->X_best.vals, this->X.vals, sizeof(double) * this->vec_len, D2D, this->stream_flex[0].stream) );
            CHECK_CUDA( cudaMemcpyAsync(this->y_best.vals, this->y.vals, sizeof(double) * this->con_num, D2D, this->stream_flex[1].stream) );
            CHECK_CUDA( cudaMemcpyAsync(this->S_best.vals, this->S.vals, sizeof(double) * this->vec_len, D2D, this->stream_flex[2].stream) );
        }

        // when before the switch, perform the special sGS-ADMM step
        if (iter < this->switch_admm) {
            // rhsy <-- -1.0 * A * SmC + 0.0 * rhsy
            SpMV_cusparse(this->cusparseH, this->A_csr, this->SmC, this->rhsy, -1.0, 0.0, this->SpMV_AX_buffer);
            // hence rhsy = - A(S - C)

            // rhsy <-- 1/sig * Rp + rhsy
            axpy_cublas(this->cublasH, this->Rp, this->rhsy, 1/this->sig);
            // hence rhsy = 1/sigma Rp - A(S - C) = 1/sigma (b - A(X^k)) - A(S - C)
            // hence rhsy = 1/sigma b - A(X^k /sigma + S^{k+1} - C)

            // y <-- linsys(rhsy)
            perform_permutation(this->rhsy_perm, this->rhsy, this->perm_inv);
            CHECK_CUDA( cudaDeviceSynchronize() );
            CHECK_CUDA( cudaMemcpyAsync(
                this->cpu_AAt_solver.chol_dn_rhs->x, this->rhsy_perm.vals,
                sizeof(double) * this->con_num, D2H, this->stream_flex[0].stream
            ) );
            CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[0].stream) );
            this->cpu_AAt_solver.solve();
            CHECK_CUDA( cudaMemcpyAsync(
                this->y_perm.vals, this->cpu_AAt_solver.chol_dn_res->x,
                sizeof(double) * this->con_num, H2D, this->stream_flex[0].stream
            ) );
            CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[0].stream) );
            perform_permutation(this->y, this->y_perm, this->perm);
            // hence y = (AA^T)^{-1} r_s^{k+1}

            // Aty <-- 1.0 * At * y + 0.0 * Aty
            SpMV_cusparse(this->cusparseH, this->At_csr, this->y, this->Aty, 1.0, 0.0, this->SpMV_Aty_buffer);
            // hence Aty = A^T y^{k+1}

            // Rd1 <-- Aty
            CHECK_CUDA( cudaMemcpy(this->Rd1.vals, this->Aty.vals, sizeof(double) * this->vec_len, D2D) );
            // Rd1 <-- (-1.0) * C + 1.0 * Rd1
            axpby_cusparse(this->cusparseH, this->C, this->Rd1, -1.0, 1.0);
            // hence Rd1 = A^T y^{k+1} - C
        }

        // when after the switch, use values computed in previous steps
        if (iter > this->switch_admm) {
            // if the current KKT residual is smaller than the best one so far,
            // update the best solution so far
            if (this->best_KKT > max(this->maxfeas, this->relgap)) {
                CHECK_CUDA( cudaMemcpyAsync(this->X_best.vals, this->X.vals, sizeof(double) * this->vec_len, D2D, this->stream_flex[0].stream) );
                CHECK_CUDA( cudaMemcpyAsync(this->y_best.vals, this->y.vals, sizeof(double) * this->con_num, D2D, this->stream_flex[1].stream) );
                CHECK_CUDA( cudaMemcpyAsync(this->S_best.vals, this->S.vals, sizeof(double) * this->vec_len, D2D, this->stream_flex[2].stream) );
                this->best_KKT = max(this->maxfeas, this->relgap);
            }
        }


        /* Step 4: Compute X^{k+1} = X^k + tau * sigma (S^{k+1} + A^T y^{k+1} - C) */
        // Rd <-- 1.0 * Rd1 + 1.0 * S
        dense_vector_add_dense_vector(this->Rd, this->Rd1, this->S, 1.0, 1.0);
        if (iter < this->switch_admm) {
            this->tau = 1.95;
        } else {
            this->tau = 1.618; // (1 + sqrt(5)) / 2
        }
        if (this->errRd < stop_tol) {
            this->tau = max(1.618, this->tau / 1.1);
        }
        // hence Rd = Rd1 + S = A^T y^{k+1} - C + S

        // X <-- X + (tau * sig) * Rd
        dense_vector_add_dense_vector(this->X, this->Rd, 1.0, this->tau * this->sig);
        // hence X = X^k + (tau * sig) * (A^T y^{k+1} - C + S)

        /* Step "5": Compute KKT residuals, update parameters */

        // Rp <-- -1.0 * A * X + 0.0 * Rp
        SpMV_cusparse(this->cusparseH, this->A_csr, this->X, this->Rp, -1.0, 0.0, this->SpMV_AX_buffer);
        // hence Rp = - A X

        // Rp <-- 1.0 * b + 1.0 * Rp
        axpby_cusparse(this->cusparseH, this->b, this->Rp, 1.0, 1.0);
        // hence Rp = b - A X

        /* Update errors and compute residuals */
        dense_vector_mul_dense_vector_mul_scalar(this->Rporg, this->normA, this->Rp, this->bscale);
        this->errRp = this->Rporg.get_norm(this->cublasH) / this->norm_borg;
        this->pobj = SparseVV_cusparse(this->cusparseH, this->C, this->X, this->SpVV_CtX_buffer) * this->objscale;
        dense_vector_mul_scalar(this->Rdorg, this->Rd, this->Cscale);
        this->errRd = this->Rdorg.get_norm(this->cublasH) / this->norm_Corg;
        this->dobj = SparseVV_cusparse(this->cusparseH, this->b, this->y, this->SpVV_bty_buffer) * this->objscale;
        this->maxfeas = max(this->errRp, this->errRd);
        this->relgap = abs(this->pobj - this->dobj) / (1 + abs(this->pobj) + abs(this->dobj));
        this->feasratio = this->ratioconst * this->errRp / this->errRd;
        if (this->feasratio < 1) {
            this->prim_win += 1;
        } else {
            this->dual_win += 1;
        }

        /* Update sigma */
        if (
            ( (iter <= this->sig_update_threshold) && ((iter % this->sig_update_stage_1) == 1) ) ||
            ( (iter > this->sig_update_threshold) && ((iter % this->sig_update_stage_2) == 1) )
        ) {
            if (this->prim_win > 1.2 * this->dual_win) {
                this->prim_win = 0;
                this->sig = min(this->sigmax, this->sig * this->sigscale);
            } else if (this->dual_win > 1.2 * this->prim_win) {
                this->dual_win = 0;
                this->sig = max(this->sigmin, this->sig / this->sigscale);
            }
        }

        /* Add info */
        this->info_pobj_arr.push_back(this->pobj);
        this->info_dobj_arr.push_back(this->dobj);
        this->info_errRp_arr.push_back(this->errRp);
        this->info_errRd_arr.push_back(this->errRd);
        this->info_relgap_arr.push_back(this->relgap);
        this->info_sig_arr.push_back(this->sig);
        this->info_bscale_arr.push_back(this->bscale);
        this->info_Cscale_arr.push_back(this->Cscale);
        this->info_iter_num++;
    }

    // recover the original solution by unscaling
    dense_vector_mul_scalar(this->X, this->bscale);
    dense_vector_div_dense_vector_mul_scalar(this->y, this->normA, this->Cscale);
    dense_vector_mul_scalar(this->S, this->Cscale);

    // free the memory
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->stop);

    return;
}