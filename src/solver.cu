/*

    solver.cu

    Main solver, works for any sizes of matrices.
    Uses the sGS-ADMM algorithm to solve an SDP problem.

*/

#include "cuadmm/solver.h"
#include "cuadmm/kernels.h"
#include "cuadmm/rank.h"
#include "cuadmm/matrix_sizes.h"

#include "psd_projection/composite_FP32.h"
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
#include "psd_projection/composite_FP32_emulated.h"
#endif
#include "psd_projection/composite_FP16.h"
#include "psd_projection/lobpcg.h"
#include "psd_projection/utils.h"

#include <algorithm>
#include <stdio.h>
#include <limits>

#define LOBPCG_MAXIT 100
#define LOBPCG_TOL 1e-8
#define LOBPCG_WARMSTART true

void SDPSolver::synchronize_gpu0_streams() {
    CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[0].stream) );
    CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[1].stream) );
    CHECK_CUDA( cudaStreamSynchronize(this->stream_flex[2].stream) );
}

void SDPSolver::init(
    int eig_stream_num_per_gpu,
    int vec_len, int con_num,
    int* cpu_At_csc_col_ptrs, int* cpu_At_csc_row_ids, double* cpu_At_csc_vals, int At_nnz,
    int* cpu_b_indices, double* cpu_b_vals, int b_nnz,
    int* cpu_C_indices, double* cpu_C_vals, int C_nnz,
    char* cpu_blk_types, int* cpu_blk_sizes,
    int mat_num,
    ProjectionMethod initial_proj_method,
    ProjectionMethod final_proj_method,
    double* cpu_X_vals,
    double* cpu_y_vals,
    double* cpu_S_vals,
    double sig,
    bool use_lobpcg
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

    this->eig_stream_num_per_gpu = eig_stream_num_per_gpu;

    this->initial_proj_method = initial_proj_method;
    this->final_proj_method = final_proj_method;
    this->current_proj_method = initial_proj_method;
    this->switched_proj_method = false;
    #if !(defined(CUDA_VERSION) && (CUDA_VERSION >= 12090))
    if (initial_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED || final_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED) {
        fprintf(stderr, "ERROR: the projection method 'COMPOSITE_FP32_EMULATED' was selected, but is not supported. BF16x9 emulation requires CUDA 12.9 or higher.\n");
        exit(EXIT_FAILURE);
    }
    #endif

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

    std::cout << std::endl << " ||C|| = " << norm_Corg << ", ||b|| = " << norm_borg << std::endl;

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

    /* Eigen decomposition for medium matrices */
    this->medium_mat.allocate(GPU0, this->sizes.total_medium_mat_size);
    this->medium_W.allocate(GPU0, this->sizes.sum_medium_mat_size);
    this->medium_info.allocate(GPU0, this->sizes.medium_mat_num);

    // streams and handles for eigen decomposition
    this->eig_medium_stream_arr = std::vector<DeviceStream>(this->eig_stream_num_per_gpu);
    this->cusolverH_eig_medium_arr = std::vector<DeviceSolverDnHandle>(this->eig_stream_num_per_gpu);
    this->cublasH_eig_medium_arr = std::vector<DeviceBlasHandle>(this->eig_stream_num_per_gpu);
    for (int stream_id = 0; stream_id < this->eig_stream_num_per_gpu; stream_id++) {
        // ininitialize and activate the streams and handles
        this->eig_medium_stream_arr[stream_id].set_gpu_id(GPU0);
        this->eig_medium_stream_arr[stream_id].activate();
        this->cusolverH_eig_medium_arr[stream_id].set_gpu_id(GPU0);
        this->cusolverH_eig_medium_arr[stream_id].activate(this->eig_medium_stream_arr[stream_id]);
        this->cublasH_eig_medium_arr[stream_id].set_gpu_id(GPU0);
        this->cublasH_eig_medium_arr[stream_id].activate(this->eig_medium_stream_arr[stream_id]);
    }

    // compute the buffer sizes of the medium matrices eig decomposition
    this->eig_medium_buffer_size.assign(this->sizes.medium_mat_sizes.size(), 0);
    this->cpu_eig_medium_buffer_size.assign(this->sizes.medium_mat_sizes.size(), 0);

    this->sizes.medium_buffer_start_indices.push_back(0);
    this->sizes.medium_cpu_buffer_start_indices.push_back(0);
    int total_eig_medium_buffer_size = 0;
    int total_cpu_eig_medium_buffer_size = 0;
    for (int i = 0; i < this->sizes.medium_mat_sizes.size(); i++) {
        single_eig_get_buffersize_cusolver(
            this->cusolverH_eig_medium_arr[i % this->eig_stream_num_per_gpu], eig_param_single, this->medium_mat, this->large_W,
            this->sizes.medium_mat_sizes[i],
            &this->eig_medium_buffer_size[i],
            &this->cpu_eig_medium_buffer_size[i],
            this->sizes.medium_mat_offset(i, 0), this->sizes.large_W_offset(i, 0)
        ); // buffer size per medium matrix of a given size

        // we need to multiply the buffer size by the number of matrices of this size
        total_eig_medium_buffer_size += this->eig_medium_buffer_size[i] * this->sizes.medium_mat_nums[i];
        total_cpu_eig_medium_buffer_size += this->cpu_eig_medium_buffer_size[i] * this->sizes.medium_mat_nums[i];

        this->sizes.medium_buffer_start_indices.push_back(
            this->sizes.medium_buffer_start_indices[i] + this->sizes.medium_mat_nums[i] * this->eig_medium_buffer_size[i]
        );
        this->sizes.medium_cpu_buffer_start_indices.push_back(
            this->sizes.medium_cpu_buffer_start_indices[i] + this->sizes.medium_mat_nums[i] * this->cpu_eig_medium_buffer_size[i]
        );
    }

    // allocate memory for the two buffers, host and device
    if (total_eig_medium_buffer_size != 0)
        this->eig_medium_buffer.allocate(GPU0, total_eig_medium_buffer_size, true);
    if (total_cpu_eig_medium_buffer_size != 0)
        this->cpu_eig_medium_buffer.allocate(total_cpu_eig_medium_buffer_size, true);

    /* Eigen decomposition for large matrices */
    // allocate GPU0 memory for large matrices
    this->large_mat.allocate(GPU0, this->sizes.total_large_mat_size);
    this->large_W.allocate(GPU0, this->sizes.sum_large_mat_size);
    this->large_info.allocate(GPU0, this->sizes.large_mat_num);

    this->cusolverH_eig_large.set_gpu_id(GPU0);
    this->cusolverH_eig_large.activate();

    // compute the buffer sizes of the large matrices eig decomposition
    this->eig_large_buffer_size.assign(this->sizes.large_mat_sizes.size(), 0);
    this->cpu_eig_large_buffer_size.assign(this->sizes.large_mat_sizes.size(), 0);

    this->sizes.large_buffer_start_indices.push_back(0);
    this->sizes.large_cpu_buffer_start_indices.push_back(0);
    int total_eig_large_buffer_size = 0;
    int total_cpu_eig_large_buffer_size = 0;
    int counter = 0;
    for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
        // we only do this if cuSOLVER is used at some point
        if (this->initial_proj_method == ProjectionMethod::EIG_FP64 || this->final_proj_method == ProjectionMethod::EIG_FP64) {
            single_eig_get_buffersize_cusolver(
                this->cusolverH_eig_large, eig_param_single, this->large_mat, this->large_W,
                this->sizes.large_mat_sizes[i],
                &this->eig_large_buffer_size[counter],
                &this->cpu_eig_large_buffer_size[counter],
                this->sizes.large_mat_offset(i, 0), this->sizes.large_W_offset(i, 0)
            ); // buffer size per large matrix of a given size

            // we need to multiply the buffer size by the number of matrices of this size
            total_eig_large_buffer_size += this->eig_large_buffer_size[counter] * this->sizes.large_mat_nums[i];
            total_cpu_eig_large_buffer_size += this->cpu_eig_large_buffer_size[counter] * this->sizes.large_mat_nums[i];

            this->sizes.large_buffer_start_indices.push_back(
                this->sizes.large_buffer_start_indices[counter] + this->sizes.large_mat_nums[i] * this->eig_large_buffer_size[counter]
            );
            this->sizes.large_cpu_buffer_start_indices.push_back(
                this->sizes.large_cpu_buffer_start_indices[counter] + this->sizes.large_mat_nums[i] * this->cpu_eig_large_buffer_size[counter]
            );
            counter++;
        }
    }

    // allocate memory for the two buffers, host and device
    if (total_eig_large_buffer_size != 0)
        this->eig_large_buffer.allocate(GPU0, total_eig_large_buffer_size, true);
    if (total_cpu_eig_large_buffer_size != 0)
        this->cpu_eig_large_buffer.allocate(total_cpu_eig_large_buffer_size, true);

    if (
        this->sizes.large_mat_sizes.size() > 0 && (
           this->initial_proj_method == ProjectionMethod::COMPOSITE_FP32 
        || this->initial_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED 
        || this->initial_proj_method == ProjectionMethod::COMPOSITE_FP16

        || this->final_proj_method == ProjectionMethod::COMPOSITE_FP32 
        || this->final_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED 
        || this->final_proj_method == ProjectionMethod::COMPOSITE_FP16
    )) {
        // create a workspace for the composite projection
        int largest_size = *std::max_element(this->sizes.large_mat_sizes.begin(), this->sizes.large_mat_sizes.end());
        size_t nn = largest_size * largest_size;
        int stride = nn % 4 == 0 ? nn : nn + (4 - nn % 4); // we need to ensure proper memory alignment

        this->float_proj_workspace.allocate(GPU0, 3 * stride);
        if (this->initial_proj_method == ProjectionMethod::COMPOSITE_FP16 || this->final_proj_method == ProjectionMethod::COMPOSITE_FP16) // if FP16, we need a second workspace
            this->half_proj_workspace.allocate(GPU0, 3 * stride);

        // create a cuBLAS handle
        this->cublasH_composite_proj.set_gpu_id(GPU0);
        this->cublasH_composite_proj.activate();
        CHECK_CUBLAS( cublasSetMathMode(this->cublasH_composite_proj.cublas_handle, CUBLAS_TENSOR_OP_MATH) );
        #if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
        if (this->initial_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED || this->final_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED) {
            CHECK_CUBLAS(cublasSetEmulationStrategy(this->cublasH_composite_proj.cublas_handle, CUBLAS_EMULATION_STRATEGY_EAGER));
        }
        #endif

        // create a cuSOLVER handle
        this->cusolverH_composite_proj.set_gpu_id(GPU0);
        this->cusolverH_composite_proj.activate();
    }

    this->cublasH_eig_large.set_gpu_id(GPU0);
    this->cublasH_eig_large.activate();
    this->cusolverH_eig_large.activate();

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
    if (this->sizes.medium_mat_num > 0) {
        this->medium_mat_tmp.allocate(GPU0, this->sizes.total_medium_mat_size);
        this->medium_mat_P.allocate(GPU0, this->sizes.total_medium_mat_size);
    }
    if (this->sizes.large_mat_num > 0 && (this->initial_proj_method == ProjectionMethod::EIG_FP64 || this->final_proj_method == ProjectionMethod::EIG_FP64)) {
        this->large_mat_tmp.allocate(GPU0, this->sizes.total_large_mat_size);
        this->large_mat_P.allocate(GPU0, this->sizes.total_large_mat_size);
    }
    this->small_mat_tmp.allocate(GPU0, this->sizes.total_small_mat_size);
    this->small_mat_P.allocate(GPU0, this->sizes.total_small_mat_size);
    this->Rd1.allocate(GPU0, this->vec_len);
    this->Xinput.allocate(GPU0, this->vec_len);

    /* Application of LOBPCG to large eigenvalues */
    this->use_lobpcg = use_lobpcg;
    if (use_lobpcg) {
        if (final_proj_method != ProjectionMethod::EIG_FP64) {
            std::cout << " ERROR: when 'use_lobpcg' is enabled, the final projection method must be 'EIG_FP64'." << std::endl;
            exit(EXIT_FAILURE);
        }

        this->positive_ranks.allocate(GPU0, this->sizes.large_mat_num);
        this->negative_ranks.allocate(GPU0, this->sizes.large_mat_num);
        this->cpu_positive_ranks.allocate(this->sizes.large_mat_num);
        this->cpu_negative_ranks.allocate(this->sizes.large_mat_num);

        this->lobpcg_W.allocate(GPU0, 1.5 * 0.05 * this->sizes.sum_large_mat_size); // since k <= 0.05*n
        this->lobpcg_P.allocate(GPU0, 1.5 * 0.05 * this->sizes.total_large_mat_size);
    }

    /* others */
    this->prim_win = 0;
    this->dual_win = 0;
    this->ratioconst = 1e0;
    this->sigmax = 1e6;
    this->sigmin = 1e-6;

    /* Main elements for the sGS-ADMM algorithm */
    this->X_best.allocate(GPU0, this->vec_len);
    this->y_best.allocate(GPU0, this->con_num);
    this->S_best.allocate(GPU0, this->vec_len);

    return;
}

void SDPSolver::solve(
    int max_iter, double stop_tol,
    int sig_update_threshold,
    int sig_update_stage_1,
    int sig_update_stage_2,
    int switch_admm,
    int switch_proj_max_iter,
    double switch_proj_tol,
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

    double minus_one = -1.0;

    std::cout << std::endl;
    std::cout << "Problem parameters:" << std::endl;
    std::cout << "              solver max iter: " << max_iter << std::endl;
    std::cout << "       KKT stopping tolerance: " << stop_tol << std::endl;
    std::cout << "       sigma update threshold: " << sig_update_threshold << std::endl;
    std::cout << "         sigma update stage 1: " << sig_update_stage_1 << std::endl;
    std::cout << "         sigma update stage 2: " << sig_update_stage_2 << std::endl;
    std::cout << "                  switch admm: " << switch_admm << std::endl;
    std::cout << "         switch proj max iter: " << switch_proj_max_iter << std::endl;
    std::cout << "              switch proj tol: " << switch_proj_tol << std::endl;
    std::cout << "                     sigscale: " << sigscale << std::endl;
    std::cout << "                initial sigma: " << this->sig << std::endl;
    std::cout << "                   use LOBPCG: " << (this->use_lobpcg ? "true" : "false") << std::endl;
    std::cout << "    initial projection method: " << get_projection_method_name(this->initial_proj_method, false) << std::endl;
    std::cout << "      final projection method: " << get_projection_method_name(this->final_proj_method, false) << std::endl;
    std::cout << "              LOBPCG max iter: " << LOBPCG_MAXIT << std::endl;
    std::cout << "             LOBPCG tolerance: " << LOBPCG_TOL << std::endl;
    std::cout << "             LOBPCG warmstart: " << (LOBPCG_WARMSTART ? "true" : "false") << std::endl;
    std::cout << "          medium matrix limit: " << MEDIUM_MAT_LIMIT << std::endl;

    /* Start the solver */
    std::cout << std::endl;
    std::cout << " ------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                   cuADMM" << std::endl;
    std::cout << " ------------------------------------------------------------------------------" << std::endl;
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

    std::cout << "  it. | p infeas d infeas | primal obj.   dual obj. rel. gap |  time |   sigma " << std::endl;
    std::cout << " ------------------------------------------------------------------------------" << std::endl;
    std::cout << " --------------- Starting with projection method " << get_projection_method_name(this->current_proj_method) << "------" << std::endl;

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
                " %4d | %3.2e %3.2e | %- 5.4e %- 5.4e %3.2e | %5.1f | %2.1e ",
                iter-1, this->errRp, this->errRd, this->pobj, this->dobj, this->relgap, seconds, this->sig
            );
            std::cout << std::endl;
        }
        if (breakyes > 0) {
            // print the final message
            printf(" ------------------------------------------------------------------------------\n\n");
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
            break;
        }

        // check if the conditions to switch the projection method are met
        if (
            !this->switched_proj_method
            && (iter > switch_proj_max_iter || max(this->maxfeas, this->relgap) < switch_proj_tol)
            && iter > 1
        ) {
            // switch the projection method
            if (this->final_proj_method != this->current_proj_method)
                std::cout << " ---------------- Switching projection method to " << get_projection_method_name(this->final_proj_method) << "------" << std::endl;
            this->current_proj_method = this->final_proj_method;
            this->switched_proj_method = true;
            this->switched_proj_method_iter = iter;
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

        // Xinput <-- -(Rd1 + 1/sig * X)
        dense_vector_plus_dense_vector_mul_scalar(this->Xinput, this->Rd1, this->X, 1.0/this->sig);
        dense_vector_negate(this->Xinput);

        /* Compute Pi(X^{k+1}) (this is long) */

        // first, we convert Xinput back to matrices
        vector_to_matrices(this->Xinput, this->large_mat, this->medium_mat, this->small_mat, this->map_B, this->map_M1, this->map_M2);
        CHECK_CUDA( cudaDeviceSynchronize() ); 

        /* 
            Step 3.1. PSD projection of the large matrices
            - if before the switch, use the initial projection method
            - if after the switch
                - if low rank, use LOBPCG
                - else, use cuSOLVER
        */

        // project with the composite method
        if (
            this->current_proj_method == ProjectionMethod::COMPOSITE_FP32 ||
            this->current_proj_method == ProjectionMethod::COMPOSITE_FP16 ||
            this->current_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED
        ) {
            for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
                for (int j = 0; j < this->sizes.large_mat_nums[i]; j++) {

                    if (this->current_proj_method == ProjectionMethod::COMPOSITE_FP32)
                        composite_FP32_auto_scale(
                            this->cublasH_composite_proj.cublas_handle,
                            this->cusolverH_composite_proj.cusolver_dn_handle,
                            this->large_mat.vals + this->sizes.large_mat_offset(i, j),
                            this->sizes.large_mat_sizes[i],
                            this->float_proj_workspace.vals
                        );
                    else if (this->current_proj_method == ProjectionMethod::COMPOSITE_FP16) {
                        composite_FP16_auto_scale(
                            this->cublasH_composite_proj.cublas_handle,
                            this->cusolverH_composite_proj.cusolver_dn_handle,
                            this->large_mat.vals + this->sizes.large_mat_offset(i, j),
                            this->sizes.large_mat_sizes[i],
                            this->float_proj_workspace.vals,
                            this->half_proj_workspace.vals
                        );
                    }
                    #if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
                    else if (this->current_proj_method == ProjectionMethod::COMPOSITE_FP32_EMULATED) {
                        composite_FP32_emulated_auto_scale(
                            this->cublasH_composite_proj.cublas_handle,
                            this->cusolverH_composite_proj.cusolver_dn_handle,
                            this->large_mat.vals + this->sizes.large_mat_offset(i, j),
                            this->sizes.large_mat_sizes[i],
                            this->float_proj_workspace.vals
                        );
                    }
                    #endif
                }
            }
        }

        // project using cuSOLVER
        int all_counter = 0; // serves as an info offset
        int icounter = 0;
        int number_low_rank_matrices = 0;
        if (this->current_proj_method == ProjectionMethod::EIG_FP64) { // cuSOLVER version
            for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {

                for (int j = 0; j < this->sizes.large_mat_nums[i]; j++) {
                    int n = this->sizes.large_mat_sizes[i];

                    bool is_lobpcg_phase = this->use_lobpcg && this->switched_proj_method;

                    // we apply cuSOLVER if:
                    // - we are not in the LOBPCG phase
                    // - or if we are every 100 iterations in the LOBPCG phase
                    bool apply_cusolver = !is_lobpcg_phase || (iter - this->switched_proj_method_iter) % 100 == 0;

                    // if we don't do LOBPCG at this step,
                    // or if we are every 100 iterations
                    if (apply_cusolver) {
                        // compute the EVD using cuSOLVER
                        single_eig_cusolver(
                            this->cusolverH_eig_large, eig_param_single,
                            this->large_mat, this->large_W,
                            this->eig_large_buffer, this->cpu_eig_large_buffer, this->large_info,
                            this->sizes.large_mat_sizes[i],
                            this->eig_large_buffer_size[icounter], this->cpu_eig_large_buffer_size[icounter],
                            this->sizes.large_mat_offset(i, j), this->sizes.large_W_offset(i, j),
                            this->sizes.large_buffer_offset(icounter, j, this->eig_large_buffer_size),
                            this->sizes.large_cpu_buffer_offset(icounter, j, this->eig_large_buffer_size),
                            all_counter
                        );
                    }

                    // if we are in the LOBPCG phase
                    if (is_lobpcg_phase) {
                        double *eigenvalues = this->lobpcg_W.vals + (int)(this->sizes.large_W_offset(i, j) * 1.5 * 0.05);
                        double *eigenvectors = this->lobpcg_P.vals + (int)(this->sizes.large_mat_offset(i, j) * 1.5 * 0.05);

                        // every 100 iterations, we computed the EVD with the full matrix to compute the ranks
                        if (apply_cusolver) {
                            // compute the rank tolerance
                            double largest_eigenvalue;
                            CHECK_CUDA( cudaMemcpy(
                                &largest_eigenvalue,
                                this->large_W.vals + this->sizes.large_W_offset(i, j) + n - 1,
                                sizeof(double), D2H
                            ) );
                            double rank_tol = std::numeric_limits<double>::epsilon() * n * largest_eigenvalue;

                            // compute the ranks
                            compute_ranks(
                                this->large_W.vals + this->sizes.large_W_offset(i, j),
                                n,
                                this->positive_ranks.vals + all_counter,
                                this->negative_ranks.vals + all_counter,
                                rank_tol
                            );

                            // copy ranks to CPU
                            CHECK_CUDA(cudaMemcpy(this->cpu_positive_ranks.vals + all_counter, this->positive_ranks.vals, sizeof(int), D2H));
                            CHECK_CUDA(cudaMemcpy(this->cpu_negative_ranks.vals + all_counter, this->negative_ranks.vals, sizeof(int), D2H));

                            // copy largest eigenpairs to use as a warmstart for LOBPCG
                            // TODO: take both positive and negative eigenpairs instead of top k
                            if (cpu_positive_ranks.vals[all_counter] < 0.05 * n && cpu_positive_ranks.vals[all_counter] > 0) {
                                number_low_rank_matrices++;
                                int k = 1.5 * cpu_positive_ranks.vals[all_counter];

                                // copy to eigenvalues and reverse them
                                reverse_vector(this->large_W.vals + this->sizes.large_W_offset(i, j) + n - k, eigenvalues, k);

                                // copy to eigenvectors and reverse the columns (vectors)
                                reverse_columns(this->large_mat.vals + this->sizes.large_mat_offset(i, j) + (n - k) * n, eigenvectors, n, k);
                            } else if (cpu_negative_ranks.vals[all_counter] < 0.05 * n && cpu_negative_ranks.vals[all_counter] > 0) {
                                number_low_rank_matrices++;
                                int k = 1.5 * cpu_negative_ranks.vals[all_counter];

                                // copy to eigenvalues and reverse them
                                CHECK_CUDA(cudaMemcpy(
                                    eigenvalues, this->large_W.vals + this->sizes.large_W_offset(i, j), sizeof(double) * k, D2D
                                ));

                                // copy to eigenvectors and reverse the columns (vectors)
                                CHECK_CUDA(cudaMemcpy(
                                    eigenvectors, 
                                    this->large_mat.vals + this->sizes.large_mat_offset(i, j), 
                                    sizeof(double) * k*n, D2D
                                ));
                            }
                        }
                        // otherwise, we compute the EVD using LOBPCG
                        else {
                            // if the matrix is positive low rank
                            if (cpu_positive_ranks.vals[all_counter] < 0.05 * this->sizes.large_mat_sizes[i] && cpu_positive_ranks.vals[all_counter] > 0) {
                                int k = 1.5 * cpu_positive_ranks.vals[all_counter];

                                // compute the largest eigenpairs
                                lobpcg(
                                    this->cublasH_eig_large.cublas_handle, this->cusolverH_eig_large.cusolver_dn_handle,
                                    this->large_mat.vals + this->sizes.large_mat_offset(i, j),
                                    eigenvectors, eigenvalues,
                                    n, k, LOBPCG_WARMSTART, LOBPCG_MAXIT, LOBPCG_TOL, false
                                );

                                // set the matrix to zero
                                CHECK_CUDA(cudaMemset(
                                    this->large_mat.vals + this->sizes.large_mat_offset(i, j), 0, sizeof(double) * n * n
                                ));
                            }
                            // if the matrix is negative low rank
                            else if (cpu_negative_ranks.vals[all_counter] < 0.05 * this->sizes.large_mat_sizes[i] && cpu_negative_ranks.vals[all_counter] > 0) {
                                int k = 1.5 * cpu_negative_ranks.vals[all_counter];

                                // change the matrix sign to reuse LOBPCG code
                                CHECK_CUBLAS(cublasDscal(
                                    this->cublasH_eig_large.cublas_handle, n*n, &minus_one, 
                                    this->large_mat.vals + this->sizes.large_mat_offset(i, j), 1
                                ));

                                // multiply the warmstart values by -1 since we will use -A
                                CHECK_CUBLAS(cublasDscal(this->cublasH_eig_large.cublas_handle, k, &minus_one, eigenvalues, 1));
                                CHECK_CUBLAS(cublasDscal(this->cublasH_eig_large.cublas_handle, n*(n - k), &minus_one, eigenvectors, 1));

                                // compute the largest eigenpairs
                                lobpcg(
                                    this->cublasH_eig_large.cublas_handle, this->cusolverH_eig_large.cusolver_dn_handle,
                                    this->large_mat.vals + this->sizes.large_mat_offset(i, j),
                                    eigenvectors, eigenvalues,
                                    n, k, LOBPCG_WARMSTART, LOBPCG_MAXIT, LOBPCG_TOL, false
                                );
                                // note: the eigenvalues are already negated since we use -A

                                // negate eigenvalues back
                                CHECK_CUBLAS(cublasDscal(this->cublasH_eig_large.cublas_handle, k, &minus_one, eigenvalues, 1));

                                // put the matrix to zero
                                CHECK_CUDA(cudaMemset(
                                    this->large_mat.vals + this->sizes.large_mat_offset(i, j), 0, sizeof(double) * n * n
                                ));
                            }
                            // otherwise, we don't use LOBPCG
                            else {
                                single_eig_cusolver(
                                    this->cusolverH_eig_large, eig_param_single,
                                    this->large_mat, this->large_W,
                                    this->eig_large_buffer, this->cpu_eig_large_buffer, this->large_info,
                                    this->sizes.large_mat_sizes[i],
                                    this->eig_large_buffer_size[icounter], this->cpu_eig_large_buffer_size[icounter],
                                    this->sizes.large_mat_offset(i, j), this->sizes.large_W_offset(i, j),
                                    this->sizes.large_buffer_offset(icounter, j, this->eig_large_buffer_size),
                                    this->sizes.large_cpu_buffer_offset(icounter, j, this->eig_large_buffer_size),
                                    all_counter
                                );
                            }
                        }
                    }

                    all_counter++;
                }

                icounter++;
            }
            if (this->use_lobpcg && this->switched_proj_method && (iter - this->switched_proj_method_iter) % 100 == 0)
                std::cout << " ------------------ Number of low rank matrices = " << std::setw(3) << number_low_rank_matrices << " / " << std::setw(3) << this->sizes.large_mat_num << " ------------------" << std::endl;
        }

        if (this->sizes.large_mat_num > 0 || this->current_proj_method == ProjectionMethod::EIG_FP64) {
            max_dense_vector_zero(this->large_W);
        }

        // multiply the large matrices by their eigenvalues
        // TODO: only do it for matrices without LOBPCG
        if (this->current_proj_method == ProjectionMethod::EIG_FP64) {
            for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
                dense_matrix_mul_diag_batch(
                    this->large_mat_tmp, this->large_mat, this->large_W,
                    this->sizes.large_mat_sizes[i], this->sizes.large_mat_nums[i],
                    this->sizes.large_mat_offset(i, 0), this->sizes.large_W_offset(i, 0)
                );
            }
        }

        // multiply the large matrices by their eigenvectors
        // TODO: only do it for the matrices without LOBPCG
        for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
            if (this->current_proj_method == ProjectionMethod::EIG_FP64) {
                dense_matrix_mul_trans_batch(
                    this->cublasH,
                    this->large_mat_P, this->large_mat_tmp, this->large_mat,
                    this->sizes.large_mat_sizes[i], this->sizes.large_mat_nums[i],
                    this->sizes.large_mat_offset(i, 0)
                );
            } else {
                // copy large_mat to large_mat_P
                CHECK_CUDA( cudaMemcpyAsync(
                    this->large_mat_P.vals + this->sizes.large_mat_offset(i, 0),
                    this->large_mat.vals + this->sizes.large_mat_offset(i, 0),
                    sizeof(double) * this->sizes.large_mat_sizes[i] * this->sizes.large_mat_sizes[i] * this->sizes.large_mat_nums[i],
                    D2D
                ) );
            }
        }

        // add the eigenvalues computed with LOBPCG back to the matrices
        // TODO: try to do this the same way as other matrices,
        // by storing eigenpairs in the standard buffers (this->large_W and this->large_mat)
        all_counter = 0;
        if (this->use_lobpcg && this->switched_proj_method && (iter - this->switched_proj_method_iter) % 100 != 0) {
            for (int i = 0; i < this->sizes.large_mat_sizes.size(); i++) {
                int n = this->sizes.large_mat_sizes[i];
                for (int j = 0; j < this->sizes.large_mat_nums[i]; j++) {
                    
                    // if the matrix is positive or negative low rank
                    if (
                        (cpu_positive_ranks.vals[all_counter] < 0.05 * this->sizes.large_mat_sizes[i] && cpu_positive_ranks.vals[all_counter] > 0)
                        || (cpu_negative_ranks.vals[all_counter] < 0.05 * this->sizes.large_mat_sizes[i] && cpu_negative_ranks.vals[all_counter] > 0)
                    ) {
                        int k;
                        if ((cpu_positive_ranks.vals[all_counter] < 0.05 * this->sizes.large_mat_sizes[i] && cpu_positive_ranks.vals[all_counter] > 0)) {
                            k = 1.5 * cpu_positive_ranks.vals[all_counter];
                        } else {
                            k = 1.5 * cpu_negative_ranks.vals[all_counter];
                        }
                        int n = this->sizes.large_mat_sizes[i];

                        double *eigenvalues = this->lobpcg_W.vals + (int)(this->sizes.large_W_offset(i, j) * 1.5 * 0.05);
                        double *eigenvectors = this->lobpcg_P.vals + (int)(this->sizes.large_mat_offset(i, j) * 1.5 * 0.05);

                        // TODO: preallocate
                        DeviceDenseVector<double> relu_eigenvalues;
                        relu_eigenvalues.allocate(GPU0, k);
                        CHECK_CUDA( cudaMemcpy(
                            relu_eigenvalues.vals, eigenvalues, sizeof(double) * k, D2D
                        ) );

                        // add back only the positive eigenvalues
                        max_dense_vector_zero(relu_eigenvalues.vals, k);

                        cublasSetPointerMode(this->cublasH_eig_large.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
                        for (int l = 0; l < k; l++) {
                            // X <- X + \lambda_i * v_i v_i^T
                            double *v_i = eigenvectors + l * n;
                            double *m_lambda_i = relu_eigenvalues.vals + l;
                            CHECK_CUBLAS( cublasDger(this->cublasH_eig_large.cublas_handle, n, n, m_lambda_i, v_i, 1, v_i, 1, this->large_mat_P.vals + this->sizes.large_mat_offset(i, j), n) );
                        }
                        cublasSetPointerMode(this->cublasH_eig_large.cublas_handle, CUBLAS_POINTER_MODE_HOST);
                    }

                    all_counter++;
                }
            }
        }
        

        /* Step 3.2. Projection of the medium matrices */
        // TODO: project medium matrices


        /* Step 3.3. Projection of the small matrices */
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

        if (this->sizes.small_mat_num > 0) {
            max_dense_vector_zero(this->small_W);
        }

        // multiply the small matrices by their eigenvalues
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++) {
            dense_matrix_mul_diag_batch(
                this->small_mat_tmp, this->small_mat, this->small_W,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->sizes.small_mat_offset(i), this->sizes.small_W_offset(i)
            );
        }

        // multiply the small matrices by their eigenvectors
        for (int i = 0; i < this->sizes.small_mat_sizes.size(); i++) {
            dense_matrix_mul_trans_batch(
                this->cublasH,
                this->small_mat_P, this->small_mat_tmp, this->small_mat,
                this->sizes.small_mat_sizes[i], this->sizes.small_mat_nums[i],
                this->sizes.small_mat_offset(i)
            );
        }

        /* Step 3.4. Compute S */

        // put S to zero (the projection of free variables)
        CHECK_CUDA( cudaMemsetAsync(this->S.vals, 0, sizeof(double) * this->vec_len) );

        // convert the matrices back to vectorized format
        matrices_to_vector(this->S, this->large_mat_P, this->medium_mat_P, this->small_mat_P, this->map_B, this->map_M1, this->map_M2);


        if (breakyes) {
            if (iter > this->switch_admm) {
                CHECK_CUDA( cudaMemcpyAsync(this->X.vals, this->X_best.vals, sizeof(double) * this->vec_len, D2D, this->stream_flex[0].stream) );
                CHECK_CUDA( cudaMemcpyAsync(this->y.vals, this->y_best.vals, sizeof(double) * this->con_num, D2D, this->stream_flex[1].stream) );
                CHECK_CUDA( cudaMemcpyAsync(this->S.vals, this->S_best.vals, sizeof(double) * this->vec_len, D2D, this->stream_flex[2].stream) );
                this->synchronize_gpu0_streams();
                printf("Best max KKT residual after switch  = %2.1e \n", this->best_KKT);
            }
            break;
        }

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
            std::cout << " -------------------------- Switching to normal ADMM --------------------------" << std::endl;
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
        // hence Rd = Rd1 + S = A^T y^{k+1} - C + S

        // update tau
        if (iter < this->switch_admm) {
            this->tau = 1.95;
        } else {
            this->tau = 1.618; // (1 + sqrt(5)) / 2
        }
        if (this->errRd < stop_tol) {
            this->tau = max(1.618, this->tau / 1.1);
        }

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
        if (iter < this->switch_admm) {
            // sGS-ADMM update rule
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
        } else {
            // standard ADMM update rule
            if (
                (               iter <=  200 && (iter %   10) == 1) ||
                (iter >  200 && iter <= 1000 && (iter %   25) == 1) ||
                (iter > 1000 && iter <= 5000 && (iter %   50) == 1) ||
                (iter > 5000                 && (iter % 1000) == 1)
            ) {
                if (this->prim_win > 1.35 * this->dual_win) {
                    this->prim_win = 0;
                    this->sig = min(this->sigmax, this->sig * this->sigscale);
                } else if (this->dual_win > 1.35 * this->prim_win) {
                    this->dual_win = 0;
                    this->sig = max(this->sigmin, this->sig / this->sigscale);
                }
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