/*

    solver.cu

    Main solver.
    Uses the sGS-ADMM algorithm to solve an SDP problem.

*/

#include "cuadmm/solver.h"
#include "cuadmm/kernels.h"

void synchronize_gpu0_streams(SDPSolver &solver) {
    CHECK_CUDA( cudaStreamSynchronize(solver.stream_flex_arr[GPU0][0].stream) );
    CHECK_CUDA( cudaStreamSynchronize(solver.stream_flex_arr[GPU0][1].stream) );
    CHECK_CUDA( cudaStreamSynchronize(solver.stream_flex_arr[GPU0][2].stream) );
}

void SDPSolver::init(
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
    double* cpu_X_vals, // |
    double* cpu_y_vals, // |- values for warm start
    double* cpu_S_vals, // |
    double sig
) {
    // start record time
    this->total_time = 0.0;
    cudaEventCreate(&this->start);
    cudaEventCreate(&this->stop);
    cudaEventRecord(this->start);

    // prepare streams for copy data
    int device_num_detected = check_gpus();
    this->device_num_requested = device_num_requested;
    this->if_gpu_eig_mom = if_gpu_eig_mom;

    /*
    we create three flexible streams per GPU, corresponding to copy mom_mat, mom_W, mom_info
    they can also be used to parallelize kernel launches and cuda toolkit calls
    */
    this->stream_flex_arr = std::vector<std::vector<DeviceStream>>(
        this->device_num_requested, std::vector<DeviceStream>(3)
    );
    for (int stream_id = 0; stream_id < 3; stream_id++) {
        this->stream_flex_arr[GPU0][stream_id].set_gpu_id(GPU0);
        this->stream_flex_arr[GPU0][stream_id].activate();
    }
    if (this->if_gpu_eig_mom) {
        // check if the requested GPU number is less than the detected ones
        // if so, we will use the detected ones
        if (device_num_detected < this->device_num_requested) {
            std::cerr << "To use multiple GPUs to calculate mom eig, make sure detected GPU number is no less than the requested ones!" << std::endl;
            std::cerr << "Device number detected: " << device_num_detected << std::endl;
            std::cerr << "Device number requested: " << device_num_requested << std::endl;
            std::exit(1);
        }
        // create the streams for the other GPUs (non-GPU0)
        for (int gpu_id = 1; gpu_id < this->device_num_requested; gpu_id++) {
            for (int stream_id = 0; stream_id < 3; stream_id++) {
                this->stream_flex_arr[gpu_id][stream_id].set_gpu_id(gpu_id);
                this->stream_flex_arr[gpu_id][stream_id].activate();
            }
        }
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
    CHECK_CUDA( cudaMemcpyAsync(this->At_csc.col_ptrs, cpu_At_csc_col_ptrs, sizeof(int) * (con_num + 1), H2D, this->stream_flex_arr[GPU0][0].stream) );
    // second stream for row_ids
    CHECK_CUDA( cudaMemcpyAsync(this->At_csc.row_ids, cpu_At_csc_row_ids, sizeof(int) * At_nnz, H2D, this->stream_flex_arr[GPU0][1].stream) );
    // third stream for vals
    CHECK_CUDA( cudaMemcpyAsync(this->At_csc.vals, cpu_At_csc_vals, sizeof(double) * At_nnz, H2D, this->stream_flex_arr[GPU0][2].stream) );
    // wait for the streams to finish
    synchronize_gpu0_streams(*this);

    // compute the norm of A
    this->normA.allocate(GPU0, con_num);
    get_normA(this->At_csc, this->normA);

    /* convert the A matrix from CSC to CSR format */
    this->CSCtoCSR_At2A_buffer_size = CSC_to_CSR_get_buffersize_cusparse(this->cusparseH, this->At_csc, this->At_csr);
    this->CSCtoCSR_At2A_buffer.allocate(GPU0, CSCtoCSR_At2A_buffer_size, true);
    CSC_to_CSR_cusparse(this->cusparseH, this->At_csc, this->At_csr, this->CSCtoCSR_At2A_buffer);
    CHECK_CUDA( cudaMemcpyAsync(this->A_csr.row_ptrs ,this->At_csc.col_ptrs, sizeof(int) * (con_num + 1), D2D, this->stream_flex_arr[GPU0][0].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->A_csr.col_ids ,this->At_csc.row_ids, sizeof(int) * At_nnz, D2D, this->stream_flex_arr[GPU0][1].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->A_csr.vals ,this->At_csc.vals, sizeof(double) * At_nnz, D2D, this->stream_flex_arr[GPU0][2].stream) );

    /* Initialize the AAt solver on CPU */
    this->cpu_AAt_solver.get_A(
        this->At_csr.row_ptrs, this->At_csr.col_ids, this->At_csr.vals,
        this->At_csr.col_size, this->At_csr.row_size, this->At_csr.nnz,
        true
    );
    this->cpu_AAt_solver.factorize();
    // retrieve permutation of the L factor
    this->perm.allocate(GPU0, con_num);
    CHECK_CUDA( cudaMemcpyAsync(this->perm.vals, this->cpu_AAt_solver.chol_fac_L->Perm, sizeof(int) * con_num, H2D, this->stream_flex_arr[GPU0][0].stream) );
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
    CHECK_CUDA( cudaMemcpyAsync(this->perm_inv.vals, perm_inv_tmp.data(), sizeof(int) * con_num, H2D, this->stream_flex_arr[GPU0][1].stream) );

    /* Initialize b, C, X, y, S, sig on GPU */
    this->b.allocate(GPU0, con_num, b_nnz);
    this->C.allocate(GPU0, vec_len, C_nnz);
    this->X.allocate(GPU0, vec_len);
    this->y.allocate(GPU0, con_num);
    this->S.allocate(GPU0, vec_len);
    CHECK_CUDA( cudaMemcpyAsync(this->b.indices, cpu_b_indices, sizeof(int) * b_nnz, H2D, this->stream_flex_arr[GPU0][0].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->b.vals, cpu_b_vals, sizeof(double) * b_nnz, H2D, this->stream_flex_arr[GPU0][1].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->C.indices, cpu_C_indices, sizeof(int) * C_nnz, H2D, this->stream_flex_arr[GPU0][2].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->C.vals, cpu_C_vals, sizeof(double) * C_nnz, H2D, this->stream_flex_arr[GPU0][0].stream) );

    // copy X, y, and S from CPU to GPU
    // if the input is nullptr (no warm start), we will set them to 0
    if (cpu_X_vals != nullptr) {
        CHECK_CUDA( cudaMemcpyAsync(this->X.vals, cpu_X_vals, sizeof(double) * vec_len, H2D, this->stream_flex_arr[GPU0][1].stream) );
    } else {
        CHECK_CUDA( cudaMemsetAsync(this->X.vals, 0, sizeof(double) * vec_len, this->stream_flex_arr[GPU0][1].stream) );
    }
    if (cpu_y_vals != nullptr) {
        CHECK_CUDA( cudaMemcpyAsync(this->y.vals, cpu_y_vals, sizeof(double) * con_num, H2D, this->stream_flex_arr[GPU0][2].stream) );
    } else {
        CHECK_CUDA( cudaMemsetAsync(this->y.vals, 0, sizeof(double) * con_num, this->stream_flex_arr[GPU0][2].stream) );
    }
    if (cpu_S_vals != nullptr) {
        CHECK_CUDA( cudaMemcpyAsync(this->S.vals, cpu_S_vals, sizeof(double) * vec_len, H2D, this->stream_flex_arr[GPU0][0].stream) );
    } else {
        CHECK_CUDA( cudaMemsetAsync(this->S.vals, 0, sizeof(double) * vec_len, this->stream_flex_arr[GPU0][0].stream) );
    }
    this->sig = sig;

    /* Initialize blk and maps */
    // copy blk values and analyze it to retrieve the block sizes and numbers
    HostDenseVector<int> cpu_blk(mat_num);
    memcpy(cpu_blk.vals, cpu_blk_vals, sizeof(int) * mat_num);
    analyze_blk(cpu_blk, &this->LARGE, &this->SMALL, &this->mom_mat_num, &this->loc_mat_num);
    std::vector<int> map_B_tmp;
    std::vector<int> map_M1_tmp;
    std::vector<int> map_M2_tmp;

    // get the maps for vectorization of matrices
    get_maps(cpu_blk, this->LARGE, this->SMALL, this->vec_len, map_B_tmp, map_M1_tmp, map_M2_tmp);
    this->map_B.allocate(GPU0, vec_len);
    this->map_M1.allocate(GPU0, vec_len);
    this->map_M2.allocate(GPU0, vec_len);
    CHECK_CUDA( cudaMemcpyAsync(this->map_B.vals, map_B_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex_arr[GPU0][0].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->map_M1.vals, map_M1_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex_arr[GPU0][1].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->map_M2.vals, map_M2_tmp.data(), sizeof(int) * vec_len, H2D, this->stream_flex_arr[GPU0][2].stream) );

    /* Scale (A is already scaled) */
    // move b and C to GPU
    this->borg.allocate(GPU0, this->con_num, this->b.nnz);
    this->Corg.allocate(GPU0, this->vec_len, this->C.nnz);
    CHECK_CUDA( cudaMemcpyAsync(this->borg.indices, this->b.indices, sizeof(int) * this->b.nnz, D2D, this->stream_flex_arr[GPU0][0].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->borg.vals, this->b.vals, sizeof(double) * this->b.nnz, D2D, this->stream_flex_arr[GPU0][1].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->Corg.indices, this->C.indices, sizeof(int) * this->C.nnz, D2D, this->stream_flex_arr[GPU0][2].stream) );
    CHECK_CUDA( cudaMemcpyAsync(this->Corg.vals, this->C.vals, sizeof(double) * this->C.nnz, D2D, this->stream_flex_arr[GPU0][0].stream) );
    synchronize_gpu0_streams(*this);
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

    /* Eigen decomposition for moment matrices */
    // allocate GPU0 memory for moment matrices
    this->mom_mat_arr = std::vector<DeviceDenseVector<double>>(this->device_num_requested);
    this->mom_W_arr = std::vector<DeviceDenseVector<double>>(this->device_num_requested);
    this->mom_info_arr = std::vector<DeviceDenseVector<int>>(this->device_num_requested);
    this->mom_mat_arr[GPU0].allocate(GPU0, this->mom_mat_num * this->LARGE * this->LARGE);
    this->mom_W_arr[GPU0].allocate(GPU0, this->mom_mat_num * this->LARGE);
    this->mom_info_arr[GPU0].allocate(GPU0, this->mom_mat_num);

    // if the decomposition is on GPU, use cuSOLVER (cf cusolver.h)
    if (this->if_gpu_eig_mom) {
        this->eig_stream_num_per_gpu = eig_stream_num_per_gpu;

        // compute the number of moment matrices for each GPU
        int base_nb_per_gpu = std::floor(static_cast<double>( this->mom_mat_num ) / this->device_num_requested);
        std::vector<int> mom_per_gpu(this->device_num_requested, 0); // given a GPU, how many moment matrices it will compute
        for (int gpu_id = 0; gpu_id < this->device_num_requested - 1; gpu_id++) {
            mom_per_gpu[gpu_id] = base_nb_per_gpu; // start by assigning the base number
        }
        // the last GPU takes the rest (for now)
        mom_per_gpu[this->device_num_requested - 1] = this->mom_mat_num - (this->device_num_requested - 1) * base_nb_per_gpu;
        if (this->device_num_requested > 2) { // if there are 1 or 2 GPUs, the distribution alread is optimal
            int i = 0;
            while (
                ( mom_per_gpu[this->device_num_requested - 1] - mom_per_gpu[i] >= 2 ) &&
                ( i < this->device_num_requested - 1 )
            ) {
                // balance the number of moment matrices to have the best possible distribution
                mom_per_gpu[i] += 1;
                mom_per_gpu[this->device_num_requested - 1] -= 1;
                i++;
            }
        }
        // for each GPU, gives the index of the first moment matrix it will compute
        this->mom_mat_num_col_ptrs_arr = std::vector<int>(this->device_num_requested + 1, 0);
        int sum = 0;
        for (int gpu_id = 0; gpu_id < this->device_num_requested; gpu_id++) {
            sum += mom_per_gpu[gpu_id];
            this->mom_mat_num_col_ptrs_arr[gpu_id + 1] = sum;
        }

        // streams and handles for eigen decomposition
        this->eig_stream_arr = std::vector<std::vector<DeviceStream>>(
            this->device_num_requested, std::vector<DeviceStream>(this->eig_stream_num_per_gpu)
        );
        this->cusolverH_eig_mom_arr = std::vector<std::vector<DeviceSolverDnHandle>>(
            this->device_num_requested, std::vector<DeviceSolverDnHandle>(this->eig_stream_num_per_gpu)
        );
        for (int gpu_id = 0; gpu_id < this->device_num_requested; gpu_id++) {
            for (int stream_id = 0; stream_id < this->eig_stream_num_per_gpu; stream_id++) {
                // ininitialize and activate the streams and handles
                this->eig_stream_arr[gpu_id][stream_id].set_gpu_id(gpu_id);
                this->eig_stream_arr[gpu_id][stream_id].activate();
                this->cusolverH_eig_mom_arr[gpu_id][stream_id].set_gpu_id(gpu_id);
                this->cusolverH_eig_mom_arr[gpu_id][stream_id].activate(this->eig_stream_arr[gpu_id][stream_id]);
            }
        }

        // allocate memory for the moment matrices eig decomposition
        int mom_mat_num_this_gpu;
        for (int gpu_id = 1; gpu_id < this->device_num_requested; gpu_id++) {
            mom_mat_num_this_gpu = this->mom_mat_num_col_ptrs_arr[gpu_id + 1] - this->mom_mat_num_col_ptrs_arr[gpu_id];
            this->mom_mat_arr[gpu_id].allocate(gpu_id, mom_mat_num_this_gpu * this->LARGE * this->LARGE);
            this->mom_W_arr[gpu_id].allocate(gpu_id, mom_mat_num_this_gpu * this->LARGE);
            this->mom_info_arr[gpu_id].allocate(gpu_id, mom_mat_num_this_gpu);
        }
        // compute the buffer sizes of the moment matrices eig decomposition
        this->eig_mom_buffer_arr = std::vector<DeviceDenseVector<double>>(this->device_num_requested); // one buffer per GPU
        this->cpu_eig_mom_buffer_arr = std::vector<HostDenseVector<double>>(this->device_num_requested); // also one buffer per GPU (this is the host buffer)
        single_eig_get_buffersize_cusolver(
            this->cusolverH_eig_mom_arr[GPU0][0], eig_param_single, this->mom_mat_arr[0], mom_W_arr[0],
            this->LARGE, &this->eig_mom_buffer_size, &this->cpu_eig_mom_buffer_size
        ); // buffer size per moment matrix
        for (int gpu_id = 0; gpu_id < this->device_num_requested; gpu_id++) {
            mom_mat_num_this_gpu = this->mom_mat_num_col_ptrs_arr[gpu_id + 1] - this->mom_mat_num_col_ptrs_arr[gpu_id];
            // allocate memory for the two buffers, host and device
            this->eig_mom_buffer_arr[gpu_id].allocate(gpu_id, this->eig_mom_buffer_size * mom_mat_num_this_gpu, true);
            if (this->cpu_eig_mom_buffer_size > 0) {
                this->cpu_eig_mom_buffer_arr[gpu_id].allocate(this->cpu_eig_mom_buffer_size * mom_mat_num_this_gpu, true);
            }
        }
    } else { // if the decomposition is on CPU, use CHOLMOD (cf cholesky_cpu.h)
        // allocate memory for the moment matrices, eigenvalues, and info
        this->cpu_mom_mat.allocate(this->mom_mat_num * this->LARGE * this->LARGE);
        this->cpu_mom_W.allocate(this->mom_mat_num * this->LARGE);
        this->cpu_mom_info.allocate(this->mom_mat_num);
        this->cpu_eig_thread_num = cpu_eig_thread_num;

        // same logic as for GPU, we will distribute the moment matrices to the threads
        // base number
        int base_nb_per_thread = std::floor(static_cast<double>( this->mom_mat_num ) / this->cpu_eig_thread_num);
        std::vector<int> mom_per_thread(this->cpu_eig_thread_num, 0);
        for (int thread_id = 0; thread_id < this->cpu_eig_thread_num - 1; thread_id++) {
            mom_per_thread[thread_id] = base_nb_per_thread;
        }
        // last takes the rest
        mom_per_thread[this->cpu_eig_thread_num - 1] = this->mom_mat_num - (this->cpu_eig_thread_num - 1) * base_nb_per_thread;
        // if there are more than 2 threads, we balance the number of moment matrices
        if (this->cpu_eig_thread_num > 2) {
            int i = 0;
            while (
                ( mom_per_thread[this->cpu_eig_thread_num - 1] - mom_per_thread[i] >= 2 ) &&
                ( i < this->cpu_eig_thread_num - 1 )
            ) {
                mom_per_thread[i] += 1;
                mom_per_thread[this->cpu_eig_thread_num - 1] -= 1;
                i++;
            }
        }
        // we compute the range of moment matrices for each thread
        this->cpu_eig_col_ptrs_arr = std::vector<int>(this->cpu_eig_thread_num + 1);
        int sum = 0;
        for (int thread_id = 0; thread_id < this->cpu_eig_thread_num; thread_id++) {
            sum += mom_per_thread[thread_id];
            this->cpu_eig_col_ptrs_arr[thread_id + 1] = sum;
        }

        // compute the buffer size for the moment matrices eig decomposition
        // note that in this case, CHOLMOD explicitely gives us the formula
        // and we don't have to call an auxiliary function as in cuSOLVER
        this->cpu_eig_mom_lwork = 1 + 6 * this->LARGE + 2 * this->LARGE * this->LARGE;
        this->cpu_eig_mom_lwork2 = 2 * (3 + 5 * this->LARGE);
        this->cpu_eig_mom_workspace.allocate(this->cpu_eig_mom_lwork * this->mom_mat_num);
        this->cpu_eig_mom_workspace_2.allocate(this->cpu_eig_mom_lwork2 * this->mom_mat_num);
    }
}