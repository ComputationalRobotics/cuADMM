/*

    cuadmm_MATLAB.cu

    This file is part of cuADMM. It defines MATLAB interface functions for the cuADMM library.

*/

#include <memory>

#include "mex.h"
#include "matrix.h"
#include "mat.h"

#include "cuadmm/check.h"
#include "cuadmm/io.h"
#include "cuadmm/solver.h"

void get_dnvec_from_matlab(
    const mxArray* mx_dnvec,
    int& cpu_dnvec_size, 
    std::vector<double>& cpu_dnvec_vals
) {
    // matlab should pass a column vector, so col_size should always be 1
    int col_size = static_cast<int>( mxGetN(mx_dnvec) );
    assert(col_size == 1);
    cpu_dnvec_size = static_cast<int>( mxGetM(mx_dnvec) );
    double* cpu_dnvec_vals_pointer = mxGetPr(mx_dnvec);
    cpu_dnvec_vals.clear();
    cpu_dnvec_vals.resize(cpu_dnvec_size, 0);
    memcpy(cpu_dnvec_vals.data(), cpu_dnvec_vals_pointer, sizeof(double) * cpu_dnvec_size);
    return;
}

void get_spvec_from_matlab(
    const mxArray* mx_spvec,
    int& cpu_spvec_size, int& cpu_spvec_nnz,
    std::vector<int>& cpu_spvec_indices, std::vector<double>& cpu_spvec_vals
) {
    // matlab should pass a column vector, so col_size should always be 1
    int col_size = static_cast<int>( mxGetN(mx_spvec) );
    assert(col_size == 1);
    size_t* col_ptrs_long = static_cast<size_t*>( mxGetJc(mx_spvec) );
    cpu_spvec_nnz = col_ptrs_long[col_size];
    cpu_spvec_size = static_cast<int>( mxGetM(mx_spvec) );

    size_t* cpu_spvec_indices_long = static_cast<size_t*>( mxGetIr(mx_spvec) );
    DeviceDenseVector<size_t> gpu_spvec_indices_long(GPU0, cpu_spvec_nnz);
    DeviceDenseVector<int> gpu_spvec_indices(GPU0, cpu_spvec_nnz);
    CHECK_CUDA( cudaMemcpy(gpu_spvec_indices_long.vals, cpu_spvec_indices_long, sizeof(size_t) * cpu_spvec_nnz, H2D) );
    long_int_to_int(gpu_spvec_indices, gpu_spvec_indices_long);
    cpu_spvec_indices.clear();
    cpu_spvec_indices.resize(cpu_spvec_nnz);
    CHECK_CUDA( cudaMemcpy(cpu_spvec_indices.data(), gpu_spvec_indices.vals, sizeof(int) * cpu_spvec_nnz, D2H) );

    double* cpu_spvec_vals_pointer = mxGetPr(mx_spvec);
    cpu_spvec_vals.clear();
    cpu_spvec_vals.resize(cpu_spvec_nnz);
    memcpy(cpu_spvec_vals.data(), cpu_spvec_vals_pointer, sizeof(double) * cpu_spvec_nnz);
    return;
}

void get_spmat_csc_from_matlab(
    const mxArray* mx_spmat_csc,
    int& cpu_spmat_csc_row_size, int& cpu_spmat_csc_col_size, int& cpu_spmat_csc_nnz,
    std::vector<int>& cpu_spmat_csc_col_ptrs, std::vector<int>& cpu_spmat_csc_row_ids, std::vector<double>& cpu_spmat_csc_vals
) {
    cpu_spmat_csc_row_size = static_cast<int>( mxGetM(mx_spmat_csc) );
    cpu_spmat_csc_col_size = static_cast<int>( mxGetN(mx_spmat_csc) );

    size_t* cpu_spmat_csc_col_ptrs_long = static_cast<size_t*>( mxGetJc(mx_spmat_csc) );
    cpu_spmat_csc_nnz = cpu_spmat_csc_col_ptrs_long[cpu_spmat_csc_col_size];
    DeviceDenseVector<size_t> gpu_spmat_csc_col_ptrs_long(GPU0, cpu_spmat_csc_col_size + 1);
    DeviceDenseVector<int> gpu_spmat_csc_col_ptrs(GPU0, cpu_spmat_csc_col_size + 1);
    CHECK_CUDA( cudaMemcpy(gpu_spmat_csc_col_ptrs_long.vals, cpu_spmat_csc_col_ptrs_long, sizeof(size_t) * (cpu_spmat_csc_col_size + 1), H2D) );
    long_int_to_int(gpu_spmat_csc_col_ptrs, gpu_spmat_csc_col_ptrs_long);
    cpu_spmat_csc_col_ptrs.clear();
    cpu_spmat_csc_col_ptrs.resize(cpu_spmat_csc_col_size + 1);
    CHECK_CUDA( cudaMemcpy(cpu_spmat_csc_col_ptrs.data(), gpu_spmat_csc_col_ptrs.vals, sizeof(int) * (cpu_spmat_csc_col_size + 1), D2H) );

    size_t* cpu_spmat_csc_row_ids_long = static_cast<size_t*>( mxGetIr(mx_spmat_csc) );
    DeviceDenseVector<size_t> gpu_spmat_csc_row_ids_long(GPU0, cpu_spmat_csc_nnz);
    DeviceDenseVector<int> gpu_spmat_csc_row_ids(GPU0, cpu_spmat_csc_nnz);
    CHECK_CUDA( cudaMemcpy(gpu_spmat_csc_row_ids_long.vals, cpu_spmat_csc_row_ids_long, sizeof(size_t) * cpu_spmat_csc_nnz, H2D) );
    long_int_to_int(gpu_spmat_csc_row_ids, gpu_spmat_csc_row_ids_long);
    cpu_spmat_csc_row_ids.clear();
    cpu_spmat_csc_row_ids.resize(cpu_spmat_csc_nnz);
    CHECK_CUDA( cudaMemcpy(cpu_spmat_csc_row_ids.data(), gpu_spmat_csc_row_ids.vals, sizeof(int) * cpu_spmat_csc_nnz, D2H) );

    double* cpu_spmat_csc_vals_pointer = mxGetPr(mx_spmat_csc);
    cpu_spmat_csc_vals.clear();
    cpu_spmat_csc_vals.resize(cpu_spmat_csc_nnz);
    memcpy(cpu_spmat_csc_vals.data(), cpu_spmat_csc_vals_pointer, sizeof(double) * cpu_spmat_csc_nnz);
    return;
}

// input order
class INPUT_ID_factory {
    public:
        // int device_num_requested;
        int eig_stream_num_per_gpu;
        int max_iter;
        int stop_tol;
        int At;
        int b;
        int C;
        int blk;
        int X;
        int y;
        int S;
        int sig;
        // int lam;
        int sig_update_threshold;
        int sig_update_stage_1;
        int sig_update_stage_2;
        int switch_admm;
        int sigscale;

        INPUT_ID_factory(int offset = 0) {
            // this->device_num_requested = offset + 0;
            this->eig_stream_num_per_gpu = offset + 0;
            this->max_iter = offset + 1;
            this->stop_tol = offset + 2;
            this->At = offset + 3;
            this->b = offset + 4;
            this->C = offset + 5;
            this->blk = offset + 6;
            this->X = offset + 7;
            this->y = offset + 8;
            this->S = offset + 9;
            this->sig = offset + 10;
            // this->lam = offset + 12;
            this->sig_update_threshold = offset + 11;
            this->sig_update_stage_1 = offset + 12;
            this->sig_update_stage_2 = offset + 13;
            this->switch_admm = offset + 14;
            this->sigscale = offset + 15;
        }
};

// output order
class OUTPUT_ID_factory {
    public:
        int X;
        int y;
        int S;
        int info;

        OUTPUT_ID_factory(int offset = 0) {
            this->X = offset + 0;
            this->y = offset + 1;
            this->S = offset + 2;
            this->info = offset + 3;
        }
};

const int info_size = 10;
class OUTPUT_INFO_RID_factory {
    public:
        int iter_num;
        int pobj_arr;
        int dobj_arr;
        int errRp_arr;
        int errRd_arr;
        int relgap_arr;
        int sig_arr;
        int bscale_arr;
        int Cscale_arr;
        int total_time;

        OUTPUT_INFO_RID_factory() {
            this->iter_num = 0;
            this->pobj_arr = 1;
            this->dobj_arr = 2;
            this->errRp_arr = 3;
            this->errRd_arr = 4;
            this->relgap_arr = 5;
            this->sig_arr = 6;
            this->bscale_arr = 7;
            this->Cscale_arr = 8;
            this->total_time = 9;
        }
};

void set_cell_array(
    mxArray*& mx_info, mxArray*& mx_dst_ptr, const std::vector<double>& src_arr, 
    int vec_num, int info_rid
) {
    double* dst_ptr = mxGetPr(mx_dst_ptr);
    memcpy(dst_ptr, src_arr.data(), sizeof(double) * vec_num);
    mxSetCell(mx_info, info_rid + info_size * 1, mx_dst_ptr);
    return;
}



void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    INPUT_ID_factory INPUT_ID(0);
    OUTPUT_ID_factory OUTPUT_ID(0);
    OUTPUT_INFO_RID_factory OUTPUT_INFO_RID;

    // -------------------------------------------------------
    // input:

    // eig_stream_num_per_gpu
    int eig_stream_num_per_gpu = static_cast<int>( mxGetScalar(prhs[INPUT_ID.eig_stream_num_per_gpu]) );

    // max_iter
    int max_iter = static_cast<int>( mxGetScalar(prhs[INPUT_ID.max_iter]) );

    // stop_tol
    double stop_tol = mxGetScalar(prhs[INPUT_ID.stop_tol]);

    // At
    int vec_len;
    int con_num;
    std::vector<int> cpu_At_csc_col_ptrs; 
    std::vector<int> cpu_At_csc_row_ids; 
    std::vector<double> cpu_At_csc_vals; 
    int At_nnz;
    get_spmat_csc_from_matlab(
        prhs[INPUT_ID.At],
        vec_len, con_num, At_nnz, cpu_At_csc_col_ptrs, cpu_At_csc_row_ids, cpu_At_csc_vals
    );
    
    
    // b
    std::vector<int> cpu_b_indices;
    std::vector<double> cpu_b_vals; 
    int b_nnz;
    int b_size;
    get_spvec_from_matlab(
        prhs[INPUT_ID.b],
        b_size, b_nnz, cpu_b_indices, cpu_b_vals
    );
    assert(b_size == con_num);
    
    // C
    std::vector<int> cpu_C_indices; 
    std::vector<double> cpu_C_vals; 
    int C_nnz;
    int C_size;
    get_spvec_from_matlab(
        prhs[INPUT_ID.C],
        C_size, C_nnz, cpu_C_indices, cpu_C_vals
    );
    assert(C_size == vec_len);

    // blk
    int mat_num;
    std::vector<double> cpu_blk_vals_double;
    get_dnvec_from_matlab(
        prhs[INPUT_ID.blk], 
        mat_num, cpu_blk_vals_double
    );
    std::vector<int> cpu_blk_vals(mat_num, 0);
    int vec_len_from_blk = 0;
    for (int i = 0; i < mat_num; i++) {
        cpu_blk_vals[i] = static_cast<int>( cpu_blk_vals_double[i] );
        vec_len_from_blk = vec_len_from_blk + cpu_blk_vals[i] * (cpu_blk_vals[i] + 1) / 2;
    }
    assert(vec_len_from_blk == vec_len);

    // X
    int X_size;
    std::vector<double> cpu_X_vals;
    get_dnvec_from_matlab(
        prhs[INPUT_ID.X], 
        X_size, cpu_X_vals
    );
    assert(X_size == vec_len);


    // y
    int y_size;
    std::vector<double> cpu_y_vals;
    get_dnvec_from_matlab(
        prhs[INPUT_ID.y],
        y_size, cpu_y_vals
    );
    assert(y_size = con_num);


    // S
    int S_size;
    std::vector<double> cpu_S_vals;
    get_dnvec_from_matlab(
        prhs[INPUT_ID.S], 
        S_size, cpu_S_vals
    );
    assert(S_size == vec_len);

    // sig
    double sig = mxGetScalar(prhs[INPUT_ID.sig]);

    // sig_update_threshold
    int sig_update_threshold;
    if (nlhs >= 12) {
        sig_update_threshold = static_cast<int>( mxGetScalar(prhs[INPUT_ID.sig_update_threshold]) );
    } else {
        sig_update_threshold = 500;
    }

    // sig_update_stage_1
    int sig_update_stage_1;
    if (nlhs >= 13) {
        sig_update_stage_1 = static_cast<int>( mxGetScalar(prhs[INPUT_ID.sig_update_stage_1]) );
    } else {
        sig_update_stage_1 = 50;
    }

    // sig_update_stage_2
    int sig_update_stage_2;
    if (nlhs >= 14) {
        sig_update_stage_2 = static_cast<int>( mxGetScalar(prhs[INPUT_ID.sig_update_stage_2]) );
    } else {
        sig_update_stage_2 = 100;
    }

    // switch_admm
    int switch_admm;
    if (nlhs >= 15) {
        switch_admm = static_cast<int>( mxGetScalar(prhs[INPUT_ID.switch_admm]) );
    } else {
        switch_admm = (int) 1.1e4;
    } 

    // sigscale
    double sigscale;
    if (nlhs >= 16) {
        sigscale = mxGetScalar(prhs[INPUT_ID.sigscale]);
    } else {
        sigscale = 1.0;
    }

    // -------------------------------------------------------

    // -------------------------------------------------------
    // start solver:

    // bool if_gpu_eig_mom = true;
    int cpu_eig_thread_num = -1;    // inactive parameter
    SDPSolver solver;
    solver.init(
        eig_stream_num_per_gpu, cpu_eig_thread_num,

        vec_len, con_num,
        cpu_At_csc_col_ptrs.data(), cpu_At_csc_row_ids.data(), cpu_At_csc_vals.data(), At_nnz,
        cpu_b_indices.data(), cpu_b_vals.data(), b_nnz,
        cpu_C_indices.data(), cpu_C_vals.data(), C_nnz,
        cpu_blk_vals.data(), mat_num,
        cpu_X_vals.data(), cpu_y_vals.data(), cpu_S_vals.data(), sig
    );
    solver.solve(
        max_iter, stop_tol,
        sig_update_threshold = sig_update_threshold,
        sig_update_stage_1 = sig_update_stage_1,
        sig_update_stage_2 = sig_update_stage_2,
        switch_admm = switch_admm,
        sigscale = sigscale
    );
    // -------------------------------------------------------

    // -------------------------------------------------------
    // output:

    // X
    mxArray* mx_X_out = mxCreateDoubleMatrix(vec_len, 1, mxREAL);
    double* X_out = mxGetPr(mx_X_out);
    CHECK_CUDA( cudaMemcpy(X_out, solver.X.vals, sizeof(double) * vec_len, D2H) );
    plhs[OUTPUT_ID.X] = mx_X_out;

    // y
    mxArray* mx_y_out = mxCreateDoubleMatrix(con_num, 1, mxREAL);
    double* y_out = mxGetPr(mx_y_out);
    CHECK_CUDA( cudaMemcpy(y_out, solver.y.vals, sizeof(double) * con_num, D2H) );
    plhs[OUTPUT_ID.y] = mx_y_out;

    // S
    mxArray* mx_S_out = mxCreateDoubleMatrix(vec_len, 1, mxREAL);
    double* S_out = mxGetPr(mx_S_out);
    CHECK_CUDA( cudaMemcpy(S_out, solver.S.vals, sizeof(double) * vec_len, D2H) );
    plhs[OUTPUT_ID.S] = mx_S_out;

    // info
    mxArray* mx_info_out = mxCreateCellMatrix(info_size, 2);
    // info_iter_num
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.iter_num + info_size * 0, mxCreateString("iter_num"));
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.iter_num + info_size * 1, mxCreateDoubleScalar((double)(solver.info_iter_num)));
    // info_pobj_arr
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.pobj_arr + info_size * 0, mxCreateString("pobj_arr"));
    mxArray* mx_info_pobj_arr_out = mxCreateDoubleMatrix(solver.info_iter_num, 1, mxREAL);
    set_cell_array(mx_info_out, mx_info_pobj_arr_out, solver.info_pobj_arr, solver.info_iter_num, OUTPUT_INFO_RID.pobj_arr);
    // info_dobj_arr
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.dobj_arr + info_size * 0, mxCreateString("dobj_arr"));
    mxArray* mx_info_dobj_arr_out = mxCreateDoubleMatrix(solver.info_iter_num, 1, mxREAL);
    set_cell_array(mx_info_out, mx_info_dobj_arr_out, solver.info_dobj_arr, solver.info_iter_num, OUTPUT_INFO_RID.dobj_arr);
    // info_errRp_arr
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.errRp_arr + info_size * 0, mxCreateString("errRp_arr"));
    mxArray* mx_info_errRp_arr_out = mxCreateDoubleMatrix(solver.info_iter_num, 1, mxREAL);
    set_cell_array(mx_info_out, mx_info_errRp_arr_out, solver.info_errRp_arr, solver.info_iter_num, OUTPUT_INFO_RID.errRp_arr);
    // info_errRd_arr
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.errRd_arr + info_size * 0, mxCreateString("errRd_arr"));
    mxArray* mx_info_errRd_arr_out = mxCreateDoubleMatrix(solver.info_iter_num, 1, mxREAL);
    set_cell_array(mx_info_out, mx_info_errRd_arr_out, solver.info_errRd_arr, solver.info_iter_num, OUTPUT_INFO_RID.errRd_arr);
    // info_relgap_arr
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.relgap_arr + info_size * 0, mxCreateString("relgap_arr"));
    mxArray* mx_info_relgap_arr_out = mxCreateDoubleMatrix(solver.info_iter_num, 1, mxREAL);
    set_cell_array(mx_info_out, mx_info_relgap_arr_out, solver.info_relgap_arr, solver.info_iter_num, OUTPUT_INFO_RID.relgap_arr);
    // info_sig_arr
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.sig_arr + info_size * 0, mxCreateString("sig_arr"));
    mxArray* mx_info_sig_arr_out = mxCreateDoubleMatrix(solver.info_iter_num, 1, mxREAL);
    set_cell_array(mx_info_out, mx_info_sig_arr_out, solver.info_sig_arr, solver.info_iter_num, OUTPUT_INFO_RID.sig_arr);
    // info_bscale_arr
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.bscale_arr + info_size * 0, mxCreateString("bscale_arr"));
    mxArray* mx_info_bscale_arr_out = mxCreateDoubleMatrix(solver.info_iter_num, 1, mxREAL);
    set_cell_array(mx_info_out, mx_info_bscale_arr_out, solver.info_bscale_arr, solver.info_iter_num, OUTPUT_INFO_RID.bscale_arr);
    // info_Cscale_arr
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.Cscale_arr + info_size * 0, mxCreateString("Cscale_arr"));
    mxArray* mx_info_Cscale_arr_out = mxCreateDoubleMatrix(solver.info_iter_num, 1, mxREAL);
    set_cell_array(mx_info_out, mx_info_Cscale_arr_out, solver.info_Cscale_arr, solver.info_iter_num, OUTPUT_INFO_RID.Cscale_arr);
    plhs[OUTPUT_ID.info] = mx_info_out;
    // info_total_time
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.total_time + info_size * 0, mxCreateString("total_time"));
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.total_time + info_size * 1, mxCreateDoubleScalar((double)(solver.total_time)));
    // -------------------------------------------------------

    // -------------------------------------------------------
    // debug
    
    // -------------------------------------------------------

    return;
}