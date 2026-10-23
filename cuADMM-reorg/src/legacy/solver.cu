/*

    solver.cu

    Main solver, works for any sizes of matrices.
    Uses the sGS-ADMM algorithm to solve an SDP problem.

*/

#include "cuadmm_legacy/solver.h"
#include "cuadmm_legacy/block_structure.h"
#include "cuadmm_legacy/kernels.h"
#include "cuadmm_legacy/rank.h"
#include "cuadmm_legacy/matrix_sizes.h"
#include "cuadmm_legacy/monitors.h"

#include "admm/admm_solver.h"
#include "report/observers.h"
#include "termination/best_iterate_tracker.h"
#include "termination/stopping_rule.h"

#include "cone/cone_product.h"
#include "cone/psd/psd_cone.h"
#include "core/stream_pool.h"
#include "core/workspace_arena.h"

#include <algorithm>
#include <stdio.h>
#include <limits>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>


SigmaPolicyParameters SDPSolver::sigma_policy_parameters() const
{
    SigmaPolicyParameters p;
    switch (this->sigma_policy)
    {
    case SigmaPolicy::FIXED:
        p.factor = 1.0;
        p.sigma_min = p.sigma_max = this->sig; // sigma_k = sigma_0
        p.guard = false;
        p.monitor1 = false;
        p.sgs_rule = "";
        break;
    case SigmaPolicy::LEGACY_ADAPTIVE: // 9bb4ba5
        p.factor = 2.0;
        p.sigma_min = 1e-6;
        p.sigma_max = 1e6;
        p.guard = false;
        p.monitor1 = true;
        p.sgs_rule = "legacy_sgs";
        break;
    case SigmaPolicy::RESIDUAL_BALANCING: // 2d8a755 with the factor as an option
        p.factor = this->penalty_factor;
        p.sigma_min = 1e-3;
        p.sigma_max = 1e3;
        p.guard = true;
        p.monitor1 = false;
        p.sgs_rule = "legacy_sgs";
        break;
    }
    if (this->sigma_policy != SigmaPolicy::FIXED)
    {
        if (this->penalty_sigma_min > 0)
            p.sigma_min = this->penalty_sigma_min;
        if (this->penalty_sigma_max > 0)
            p.sigma_max = this->penalty_sigma_max;
        if (!(p.sigma_min > 0) || !(p.sigma_max >= p.sigma_min) || !(p.factor > 0))
            throw std::invalid_argument("sigma policy parameters: need 0 < sigma_min <= sigma_max and factor > 0 (got " +
                                        std::to_string(p.sigma_min) + ", " + std::to_string(p.sigma_max) + ", " +
                                        std::to_string(p.factor) + ")");
    }
    p.win_ratio = 1.35;
    p.residual_weight = this->ratioconst;
    return p;
}

void SDPSolver::synchronize_gpu0_streams()
{
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[0].stream));
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[1].stream));
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[2].stream));
}

// The PSD tiers, their projections and their measures are the cone layer's (src/cone/psd/) since Phases 5-6b; the
// engine keeps the block structure, the per-block scalars of the rank analysis and the counters.

double SDPSolver::certified_lower_bound(const std::vector<double> &trace_bounds) const
{
    if (trace_bounds.size() != this->dual_cone_coef.size())
        throw std::invalid_argument(
            "certified_lower_bound: " + std::to_string(trace_bounds.size()) + " bounds given for " +
            std::to_string(this->dual_cone_coef.size()) + " blocks (call it after solve())");
    double bound = this->dobj;
    for (size_t b = 0; b < trace_bounds.size(); b++)
        bound += trace_bounds[b] * this->dual_cone_coef[b];
    return bound;
}

double SDPSolver::certified_lower_bound(double trace_bound) const
{
    return this->certified_lower_bound(std::vector<double>(this->dual_cone_coef.size(), trace_bound));
}

void SDPSolver::init(
    int eig_stream_num_per_gpu,
    int vec_len, int con_num,
    int *cpu_At_csc_col_ptrs, int *cpu_At_csc_row_ids, double *cpu_At_csc_vals, int At_nnz,
    int *cpu_b_indices, double *cpu_b_vals, int b_nnz,
    int *cpu_C_indices, double *cpu_C_vals, int C_nnz,
    char *cpu_blk_types, int *cpu_blk_sizes,
    int mat_num,
    ProjectionMethod initial_proj_method,
    ProjectionMethod final_proj_method,
    double *cpu_X_vals,
    double *cpu_y_vals,
    double *cpu_S_vals,
    double sig,
    bool use_lobpcg)
{
    // host copies of the original problem data, for the external validation of solve() (ValidationConfig); made
    // before the timing of init() starts
    this->host_At_col_ptrs.assign(cpu_At_csc_col_ptrs, cpu_At_csc_col_ptrs + con_num + 1);
    this->host_At_row_ids.assign(cpu_At_csc_row_ids, cpu_At_csc_row_ids + At_nnz);
    this->host_At_vals.assign(cpu_At_csc_vals, cpu_At_csc_vals + At_nnz);
    this->host_b_idx.assign(cpu_b_indices, cpu_b_indices + b_nnz);
    this->host_b_vals.assign(cpu_b_vals, cpu_b_vals + b_nnz);
    this->host_C_idx.assign(cpu_C_indices, cpu_C_indices + C_nnz);
    this->host_C_vals.assign(cpu_C_vals, cpu_C_vals + C_nnz);
    this->host_blk_types.assign(cpu_blk_types, cpu_blk_types + mat_num);
    this->host_blk_sizes.assign(cpu_blk_sizes, cpu_blk_sizes + mat_num);
    this->external_validator.reset();

    // start record time
    this->total_time = 0.0;
    if (this->start == nullptr)
        CHECK_CUDA(cudaEventCreate(&this->start));
    if (this->stop == nullptr)
        CHECK_CUDA(cudaEventCreate(&this->stop));
    cudaEventRecord(this->start);

    // prepare streams for copy data
    /*
    we create three flexible streams per GPU, corresponding to copy mom_mat, mom_W, mom_info
    they can also be used to parallelize kernel launches and cuda toolkit calls
    */
    this->stream_flex = std::vector<DeviceStream>(3);
    for (int stream_id = 0; stream_id < 3; stream_id++)
    {
        this->stream_flex[stream_id].set_gpu_id(GPU0);
        this->stream_flex[stream_id].activate();
    }

    // create handles for cuSPARSE and cuBLAS
    this->cusparseH.set_gpu_id(GPU0);
    this->cusparseH.activate();
    this->cublasH.set_gpu_id(GPU0);
    this->cublasH.activate();

    // B17: the medium-block EVDs are distributed over the streams with counter % eig_stream_num_per_gpu
    if (eig_stream_num_per_gpu < 1)
        throw std::invalid_argument("init(): eig_stream_num_per_gpu must be at least 1 (got " + std::to_string(eig_stream_num_per_gpu) + ")");
    this->eig_stream_num_per_gpu = eig_stream_num_per_gpu;

    this->initial_proj_method = initial_proj_method;
    this->final_proj_method = final_proj_method;
    this->current_proj_method = initial_proj_method;
    this->switched_proj_method = false;

    /* Initialize the A matrix */
    this->vec_len = vec_len;
    this->con_num = con_num;
    this->At_csc.allocate(GPU0, vec_len, con_num, At_nnz);
    this->At_csr.allocate(GPU0, vec_len, con_num, At_nnz);
    this->A_csr.allocate(GPU0, con_num, vec_len, At_nnz);
    // first stream for col_ptrs
    CHECK_CUDA(cudaMemcpyAsync(this->At_csc.col_ptrs, cpu_At_csc_col_ptrs, sizeof(int) * (con_num + 1), H2D, this->stream_flex[0].stream));
    // second stream for row_ids
    CHECK_CUDA(cudaMemcpyAsync(this->At_csc.row_ids, cpu_At_csc_row_ids, sizeof(int) * At_nnz, H2D, this->stream_flex[1].stream));
    // third stream for vals
    CHECK_CUDA(cudaMemcpyAsync(this->At_csc.vals, cpu_At_csc_vals, sizeof(double) * At_nnz, H2D, this->stream_flex[2].stream));
    // wait for the streams to finish
    this->synchronize_gpu0_streams();

    // compute the norm of A (get_normA rewrites At_csc.vals on the legacy default stream)
    this->normA.allocate(GPU0, con_num);
    get_normA(this->At_csc, this->normA);

    /* convert the At matrix from CSC to CSR format */
    this->CSCtoCSR_At2A_buffer_size = CSC_to_CSR_get_buffersize_cusparse(this->cusparseH, this->At_csc, this->At_csr);
    this->CSCtoCSR_At2A_buffer.allocate(GPU0, CSCtoCSR_At2A_buffer_size, true);
    CSC_to_CSR_cusparse(this->cusparseH, this->At_csc, this->At_csr, this->CSCtoCSR_At2A_buffer);
    // B2: the copies of At_csc into A_csr run on the non-blocking flex streams and must see the normalized values:
    // an event on the default stream orders them after get_normA (and after the CSC-to-CSR conversion)
    cudaEvent_t normA_done;
    CHECK_CUDA(cudaEventCreateWithFlags(&normA_done, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(normA_done, (cudaStream_t)0));
    for (int stream_id = 0; stream_id < 3; stream_id++)
        CHECK_CUDA(cudaStreamWaitEvent(this->stream_flex[stream_id].stream, normA_done, 0));
    CHECK_CUDA(cudaMemcpyAsync(this->A_csr.row_ptrs, this->At_csc.col_ptrs, sizeof(int) * (con_num + 1), D2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->A_csr.col_ids, this->At_csc.row_ids, sizeof(int) * At_nnz, D2D, this->stream_flex[1].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->A_csr.vals, this->At_csc.vals, sizeof(double) * At_nnz, D2D, this->stream_flex[2].stream));

    /* Initialize the AAt solver on CPU */
    const auto factorization_start = std::chrono::steady_clock::now();
    this->cpu_AAt_solver.get_A(
        this->At_csr.row_ptrs, this->At_csr.col_ids, this->At_csr.vals,
        this->At_csr.col_size, this->At_csr.row_size, this->At_csr.nnz,
        true, 1e-15);
    this->cpu_AAt_solver.factorize();
    printf("\n factorization of AA^T + 1e-15 I: %s, nnz(L) = %.3e, %.1f s\n",
           this->cpu_AAt_solver.supernodal_used ? "supernodal LL^T -> simplicial LDL^T" : "simplicial LDL^T",
           this->cpu_AAt_solver.cc.lnz,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - factorization_start).count());
    printf("   pivots in [%.2e, %.2e], %d tiny (< 1e-12 max) and %d nonpositive%s\n",
           this->cpu_AAt_solver.min_pivot, this->cpu_AAt_solver.max_pivot,
           this->cpu_AAt_solver.tiny_pivots, this->cpu_AAt_solver.nonpositive_pivots,
           this->cpu_AAt_solver.tiny_pivots > 0 ? ": A does not have full row rank (dependent constraints)" : "");
    // With a dense factor (few constraints, e.g. neosfrbr25: nnz(L) = m^2/2 = 1e8, 119 ms per CPU solve), the y-step
    // solves are two dense triangular solves on the GPU (4 ms), with the same factor. Sparse factors stay on the CPU:
    // cuSPARSE triangular solves were measured slower than CHOLMOD's there (0.1-1.3x).
    {
        size_t free_mem = 0, total_mem = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
        const double m = con_num;
        const cholmod_factor *F = this->cpu_AAt_solver.chol_fac_L;
        this->AAt_dense_gpu = con_num >= 2000 && m * m <= double(std::numeric_limits<int>::max()) &&
                              8.0 * m * m <= 0.25 * double(free_mem) &&
                              this->cpu_AAt_solver.cc.lnz > 0.3 * m * (m + 1) / 2 &&
                              !F->is_super && !F->is_ll;
        if (this->AAt_dense_gpu)
        {
            // simplicial LDL^T: column j holds D_jj first, then the entries L(i, j), i > j (Lnz[j] entries in all)
            const int *Lp = (const int *)F->p, *Li = (const int *)F->i, *Lnz = (const int *)F->nz;
            const double *Lx = (const double *)F->x;
            std::vector<double> L_dense(size_t(con_num) * con_num, 0.0), D(con_num);
            for (int j = 0; j < con_num; j++)
            {
                if (Lnz[j] < 1 || Li[Lp[j]] != j)
                    throw std::runtime_error("unexpected layout of the CHOLMOD factor: the diagonal is not first");
                D[j] = Lx[Lp[j]];
                L_dense[size_t(j) * con_num + j] = 1.0;
                for (int k = Lp[j] + 1; k < Lp[j] + Lnz[j]; k++)
                    L_dense[size_t(j) * con_num + Li[k]] = Lx[k];
            }
            this->AAt_L.allocate(GPU0, con_num * con_num);
            this->AAt_D.allocate(GPU0, con_num);
            CHECK_CUDA(cudaMemcpy(this->AAt_L.vals, L_dense.data(), sizeof(double) * L_dense.size(), H2D));
            CHECK_CUDA(cudaMemcpy(this->AAt_D.vals, D.data(), sizeof(double) * con_num, H2D));
        }
        if (this->AAt_dense_gpu)
            printf("   y-step solves: dense triangular solves on the GPU (%.2f GB)\n", 8.0 * m * m / 1e9);
        else
            printf("   y-step solves: CHOLMOD on the CPU\n");
    }
    // retrieve permutation of the L factor
    this->perm.allocate(GPU0, con_num);
    CHECK_CUDA(cudaMemcpyAsync(this->perm.vals, this->cpu_AAt_solver.chol_fac_L->Perm, sizeof(int) * con_num, H2D, this->stream_flex[0].stream));
    // allocate memory of right-hand side vector
    this->rhsy.allocate(GPU0, con_num);
    CHECK_CUDA(cudaMemset(this->rhsy.vals, 0, sizeof(double) * con_num)); // SpMV output, see Aty below
    this->rhsy_perm.allocate(GPU0, con_num);
    this->y_perm.allocate(GPU0, con_num);
    // compute inverse permutation
    std::vector<int> perm_tmp(con_num, 0);
    std::vector<int> perm_inv_tmp;
    memcpy(perm_tmp.data(), this->cpu_AAt_solver.chol_fac_L->Perm, sizeof(int) * con_num);
    this->perm_inv.allocate(GPU0, con_num);
    get_inverse_permutation(perm_inv_tmp, perm_tmp);
    CHECK_CUDA(cudaMemcpyAsync(this->perm_inv.vals, perm_inv_tmp.data(), sizeof(int) * con_num, H2D, this->stream_flex[1].stream));

    /* Initialize b, C, X, y, S, sig on GPU */
    this->b.allocate(GPU0, con_num, b_nnz);
    this->C.allocate(GPU0, vec_len, C_nnz);
    this->X.allocate(GPU0, vec_len);
    this->y.allocate(GPU0, con_num);
    this->S.allocate(GPU0, vec_len);
    CHECK_CUDA(cudaMemcpyAsync(this->b.indices, cpu_b_indices, sizeof(int) * b_nnz, H2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->b.vals, cpu_b_vals, sizeof(double) * b_nnz, H2D, this->stream_flex[1].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->C.indices, cpu_C_indices, sizeof(int) * C_nnz, H2D, this->stream_flex[2].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->C.vals, cpu_C_vals, sizeof(double) * C_nnz, H2D, this->stream_flex[0].stream));

    // copy X, y, and S from CPU to GPU
    // if the input is nullptr (no warm start), we will set them to 0
    if (cpu_X_vals != nullptr)
    {
        // copy from CPU to GPU
        CHECK_CUDA(cudaMemcpyAsync(this->X.vals, cpu_X_vals, sizeof(double) * vec_len, H2D, this->stream_flex[1].stream));
    }
    else
    {
        // set to 0
        CHECK_CUDA(cudaMemsetAsync(this->X.vals, 0, sizeof(double) * vec_len, this->stream_flex[1].stream));
    }
    if (cpu_y_vals != nullptr)
    {
        CHECK_CUDA(cudaMemcpyAsync(this->y.vals, cpu_y_vals, sizeof(double) * con_num, H2D, this->stream_flex[2].stream));
    }
    else
    {
        CHECK_CUDA(cudaMemsetAsync(this->y.vals, 0, sizeof(double) * con_num, this->stream_flex[2].stream));
    }
    if (cpu_S_vals != nullptr)
    {
        CHECK_CUDA(cudaMemcpyAsync(this->S.vals, cpu_S_vals, sizeof(double) * vec_len, H2D, this->stream_flex[0].stream));
    }
    else
    {
        CHECK_CUDA(cudaMemsetAsync(this->S.vals, 0, sizeof(double) * vec_len, this->stream_flex[0].stream));
    }
    this->sig = sig;

    /* Initialize blk and maps */
    // analyze the blk vector (validated, int-overflow checked): block sizes and numbers,
    // matrix layouts, and the maps for vectorization of matrices (computed on CPU)
    const BlockStructure blk_struct(cpu_blk_types, cpu_blk_sizes, mat_num);
    blk_struct.print();
    // get_maps-style maps have vec_len entries: the caller's vec_len must match the blocks
    if (blk_struct.vec_len != this->vec_len)
        throw std::invalid_argument("vec_len = " + std::to_string(this->vec_len) +
                                    " does not match the block sizes (expected " + std::to_string(blk_struct.vec_len) + ")");
    this->blk_info = blk_struct.blocks;
    this->psd_blk_sizes = blk_struct.psd_blk_sizes;
    this->psd_blk_nums = blk_struct.psd_blk_nums;
    this->sizes = blk_struct.sizes;
    // B21: the exact EVD (the fallback of every projection method) has a size limit on CUDA 12.9 (plan O5)
    if (this->sizes.large_mat_num > 0 && this->sizes.max_large_mat_size > CUADMM_MAX_EVD_BLOCK_SIZE)
        throw std::invalid_argument("PSD block of size " + std::to_string(this->sizes.max_large_mat_size) +
                                    " exceeds the exact-EVD limit of " + std::to_string(CUADMM_MAX_EVD_BLOCK_SIZE) +
                                    " (cusolverDnXsyevd workspace queries fail above it on CUDA 12.9)");
    // the svec maps of main are not needed: the PSD tiers of the cone layer convert their own blocks (Phase 6b)

    /* Scale (A is already scaled) */
    // move b and C to GPU
    this->borg.allocate(GPU0, this->con_num, this->b.nnz);
    this->Corg.allocate(GPU0, this->vec_len, this->C.nnz);
    CHECK_CUDA(cudaMemcpyAsync(this->borg.indices, this->b.indices, sizeof(int) * this->b.nnz, D2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->borg.vals, this->b.vals, sizeof(double) * this->b.nnz, D2D, this->stream_flex[1].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->Corg.indices, this->C.indices, sizeof(int) * this->C.nnz, D2D, this->stream_flex[2].stream));
    CHECK_CUDA(cudaMemcpyAsync(this->Corg.vals, this->C.vals, sizeof(double) * this->C.nnz, D2D, this->stream_flex[0].stream));
    this->synchronize_gpu0_streams();
    CHECK_CUDA(cudaEventDestroy(normA_done)); // every stream that waited on it has completed
    // compute the norms of b and C
    this->norm_borg = 1 + this->borg.get_norm(this->cublasH);
    this->norm_Corg = 1 + this->Corg.get_norm(this->cublasH);

    std::cout << std::endl
              << " ||C|| = " << norm_Corg << ", ||b|| = " << norm_borg << std::endl;

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
    // cusparseSpMV reads its output vector even with beta = 0: never let it read uninitialized memory (0 * NaN = NaN)
    CHECK_CUDA(cudaMemset(this->Aty.vals, 0, sizeof(double) * this->vec_len));
    CHECK_CUDA(cudaMemset(this->Rp.vals, 0, sizeof(double) * this->con_num));
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
    CHECK_CUDA(cudaMemcpy(this->SmC.vals, this->S.vals, sizeof(double) * this->vec_len, D2D));
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

    /* The PSD tiers (small, medium, large) live in the cone layer since Phase 6b (src/cone/psd/): their buffers,
       streams, handles and workspaces come from the WorkspaceArena and the StreamPool of the owner. The engine keeps
       the per-block scalars of the rank analysis and the counters. */
    this->large_mode = std::vector<int>(this->sizes.large_mat_num, LargeProjectionMode::FULL);
    this->large_k = std::vector<int>(this->sizes.large_mat_num, 0);
    this->large_rank_tol = std::vector<double>(this->sizes.large_mat_num, 0.0);
    this->stop_tol = 0.0; // set by solve()
    this->lobpcg_calls = 0;
    this->lobpcg_fallbacks = 0;
    this->composite_fallbacks = 0;
    this->jacobi_not_converged = 0;

    /* For the computation of y, X, S */
    this->Rd1.allocate(GPU0, this->vec_len);
    this->Xinput.allocate(GPU0, this->vec_len);

    /* Application of LOBPCG to large eigenvalues */
    this->use_lobpcg = use_lobpcg;
    if (use_lobpcg)
    {
        if (final_proj_method != ProjectionMethod::EIG_FP64)
        {
            throw std::invalid_argument("when 'use_lobpcg' is enabled, the final projection method must be 'EIG_FP64'.");
        }

        this->positive_ranks.allocate(GPU0, this->sizes.large_mat_num);
        this->negative_ranks.allocate(GPU0, this->sizes.large_mat_num);
        this->cpu_positive_ranks.allocate(this->sizes.large_mat_num);
        this->cpu_negative_ranks.allocate(this->sizes.large_mat_num);
    }

    /* others */
    this->prim_win = 0;
    this->dual_win = 0;
    this->ratioconst = 1e0;
    // sigmin / sigmax: set by solve() from the sigma policy (sigma_policy_parameters())

    /* Main elements for the sGS-ADMM algorithm */
    this->X_best.allocate(GPU0, this->vec_len);
    this->y_best.allocate(GPU0, this->con_num);
    this->S_best.allocate(GPU0, this->vec_len);
    this->best_KKT = std::numeric_limits<double>::infinity(); // no best iterate saved yet
    this->errPSD_X = 0.0;
    this->errPSD_S = 0.0;
    this->converged = false;
    this->returned_best_iterate = false;
    this->time_limit_reached = false;
    this->solve_time = 0.0;

    // time spent in init()
    float init_milliseconds;
    CHECK_CUDA(cudaEventRecord(this->stop));
    CHECK_CUDA(cudaEventSynchronize(this->stop));
    CHECK_CUDA(cudaEventElapsedTime(&init_milliseconds, this->start, this->stop));
    this->init_time = init_milliseconds / 1000;

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
    bool if_first)
{
    // Phase 4: the loop lives in cuadmm::admm::AdmmSolver on the components of the algorithm layer; this method keeps
    // the signature of main for the copied tests and builds the components from the engine's fields.
    using namespace cuadmm;
    this->stop_tol = stop_tol;
    this->sig_update_threshold = sig_update_threshold;
    this->sig_update_stage_1 = sig_update_stage_1;
    this->sig_update_stage_2 = sig_update_stage_2;
    this->switch_admm = switch_admm;
    this->sigscale = sigscale;
    const SigmaPolicyParameters params = this->sigma_policy_parameters(); // validates the bounds and the factor
    this->sigmin = params.sigma_min;
    this->sigmax = params.sigma_max;

    PenaltyOptions po;
    po.policy = this->sigma_policy == SigmaPolicy::FIXED              ? PenaltyPolicyKind::Fixed
                : this->sigma_policy == SigmaPolicy::LEGACY_ADAPTIVE ? PenaltyPolicyKind::LegacyAdaptive
                                                                     : PenaltyPolicyKind::ResidualBalancing;
    po.sigma0 = this->sig;
    po.factor = this->penalty_factor;
    po.sigma_min = this->penalty_sigma_min;
    po.sigma_max = this->penalty_sigma_max;
    po.sgs_update_threshold = sig_update_threshold;
    po.sgs_update_stage_1 = sig_update_stage_1;
    po.sgs_update_stage_2 = sig_update_stage_2;
    po.sgs_factor = sigscale;
    std::unique_ptr<admm::PenaltyPolicy> penalty = admm::make_penalty_policy(po);

    // the PSD cone with the schedule of the C++ API of main: the warm-up (if the initial method is the composite
    // filter) on every large block, and the low-rank analysis of every block from the switch (switch_proj_max_iter,
    // switch_proj_tol), B3
    cone::PsdConeConfig pc;
    pc.policy.warmup = this->initial_proj_method == ProjectionMethod::COMPOSITE_FP16;
    pc.policy.warmup_min_size = 1;
    pc.policy.switch_tol = switch_proj_tol;
    pc.policy.switch_iter = switch_proj_max_iter;
    pc.policy.lowrank_enabled = this->use_lobpcg;
    pc.policy.legacy_global_start = true;
    pc.rank.ratio = this->lobpcg_ratio;
    pc.rank.relaxation = this->lobpcg_relaxation;
    pc.lobpcg.max_iter = this->lobpcg_maxit;
    pc.lobpcg.tol = this->lobpcg_tol;
    pc.composite.seed = this->lanczos_seed;
    pc.medium_batch_min = this->medium_batch_min_count;
    // the C++ API of main reproduces main's arithmetic: the final products of Milestone 1 (DECISIONS 50); the facade's
    // defaults are the SYRK / scaled products of Phase 6b
    pc.gemm_compat = true;
    pc.batched_gemm_compat = true;
    core::StreamPoolConfig pool_config;
    pool_config.medium_streams = this->eig_stream_num_per_gpu;
    core::StreamPool pool(pool_config);
    core::WorkspaceArena arena;
    cone::ConeProduct cones(this->host_blk_types, this->host_blk_sizes, cone::ConeBuildContext{this->vec_len, this, &pc, &arena, &pool});
    arena.commit();
    admm::AdmmState state(*this, cones);
    termination::BestIterateTracker tracker(*this);
    std::unique_ptr<termination::StoppingRule> rule;
    if (this->validation.interval > 0)
        rule.reset(new termination::ValidatedRule(*this, stop_tol, this->time_limit));
    else
        rule.reset(new termination::InternalKktRule(stop_tol, this->time_limit));
    report::ObserverSet observers;
    observers.add(std::make_shared<report::ConsoleObserver>(1));
    observers.add(std::make_shared<report::LegacyInfoObserver>(*this));

    admm::RunConfig cfg;
    cfg.max_iter = max_iter;
    cfg.tol = stop_tol;
    cfg.switch_admm = switch_admm;
    cfg.tau_admm = 1.618;
    cfg.tau_sgs = 1.95;
    cfg.legacy_sgs_tau_reduction = this->legacy_sgs_tau_reduction;
    cfg.verbose = true;
    admm::AdmmSolver loop(state, *penalty, *rule, tracker, observers, cfg);
    loop.run(this->sig, !if_first);
}

void SDPSolver::solve_AAt_perm()
{
    if (this->AAt_dense_gpu)
    {
        // y_perm <- L^{-T} D^{-1} L^{-1} rhsy_perm (cuBLAS and the kernels run on the default stream)
        CHECK_CUDA(cudaMemcpy(this->y_perm.vals, this->rhsy_perm.vals, sizeof(double) * this->con_num, D2D));
        CHECK_CUBLAS(cublasDtrsv(
            this->cublasH.cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
            this->con_num, this->AAt_L.vals, this->con_num, this->y_perm.vals, 1));
        dense_vector_div_dense_vector_mul_scalar(this->y_perm, this->AAt_D, 1.0);
        CHECK_CUBLAS(cublasDtrsv(
            this->cublasH.cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
            this->con_num, this->AAt_L.vals, this->con_num, this->y_perm.vals, 1));
        return;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpyAsync(
        this->cpu_AAt_solver.chol_dn_rhs->x, this->rhsy_perm.vals,
        sizeof(double) * this->con_num, D2H, this->stream_flex[0].stream));
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[0].stream));
    this->cpu_AAt_solver.solve();
    CHECK_CUDA(cudaMemcpyAsync(
        this->y_perm.vals, this->cpu_AAt_solver.chol_dn_res->x,
        sizeof(double) * this->con_num, H2D, this->stream_flex[0].stream));
    CHECK_CUDA(cudaStreamSynchronize(this->stream_flex[0].stream));
}

SDPSolver::~SDPSolver()
{
    // the events live as long as the solver, so that solve() can be called again and nothing leaks if it throws
    if (this->start != nullptr)
        CHECK_CUDA_NOTHROW(cudaEventDestroy(this->start));
    if (this->stop != nullptr)
        CHECK_CUDA_NOTHROW(cudaEventDestroy(this->stop));
}