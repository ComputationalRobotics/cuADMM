/*

    cuadmm_MATLAB.cu

    This file is part of cuADMM. It defines MATLAB interface functions for the cuADMM library.

    Usage:
        [X, y, S, info] = cuadmm_MATLAB(eig_stream_num_per_gpu, max_iter, stop_tol, At, b, C, blk_types, blk_sizes,
                                        X0, y0, S0, sig, sig_update_threshold, sig_update_stage_1,
                                        sig_update_stage_2, switch_admm, switch_proj_iter, switch_proj_tol, sigscale)
    At (vec_len x con_num), b (con_num x 1) and C (vec_len x 1) are sparse; blk_types is a char column vector of
    's' (PSD), 'l' (nonnegative) or 'u' (free) blocks and blk_sizes a column vector of the same length.
    X0, S0 (vec_len x 1) and y0 (con_num x 1) are dense column vectors (the starting point).
    The first 12 inputs are required. The last 7 are optional (defaults: 500, 50, 100, 0, 5000, 1e-2, 2.0): they can
    be omitted from the end, or passed as [] to keep their default. info is a 10x2 cell array of {name, value} pairs.
    switch_admm: sGS-ADMM (Algorithm 1 of arXiv 2406.05846) for iter < switch_admm, then plain two-block ADMM
    (0, the default: plain ADMM throughout). sig_update_* and sigscale act in the sGS phase only (sigscale = 1 keeps
    sigma fixed, as in Algorithm 1); the plain ADMM phase has its own schedule.

*/

#include <memory>
#include <stdexcept>
#include <string>

#include "mex.h"
#include "matrix.h"
#include "mat.h"

#include "cuadmm/check.h"
#include "cuadmm/io.h"
#include "cuadmm/solver.h"

// Invalid input from MATLAB, reported as cuADMM:invalidInput (other exceptions are reported as cuADMM:solverError)
class InputError : public std::runtime_error {
    public:
        explicit InputError(const std::string& msg) : std::runtime_error(msg) {}
};

// Throws an InputError unless arr is a real double column vector (or matrix), sparse or dense as requested,
// since the readers below access its data with mxGetPr/mxGetJc/mxGetIr without further checks
void check_double_input(const mxArray* arr, const char* name, bool sparse, bool column = true) {
    if (!mxIsDouble(arr) || mxIsComplex(arr) || mxIsSparse(arr) != sparse || (column && mxGetN(arr) != 1)) {
        throw InputError(std::string(name) + " must be a real " + (sparse ? "sparse" : "dense") + " double " +
                         (column ? "column vector" : "matrix") + ".");
    }
}

// An optional input is used if the caller passed it and it is not [] (pass [] to keep its default value)
bool has_optional_input(int nrhs, const mxArray* prhs[], int input_id) {
    return nrhs > input_id && !mxIsEmpty(prhs[input_id]);
}

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

void get_char_vec_from_matlab(
    const mxArray* mx_charvec,
    int& cpu_charvec_size, 
    std::vector<char>& cpu_charvec_vals
) {
    // matlab should pass a column vector, so col_size should always be 1
    int col_size = static_cast<int>( mxGetN(mx_charvec) );
    assert(col_size == 1);
    cpu_charvec_size = static_cast<int>( mxGetM(mx_charvec) );
    char* cpu_charvec_vals_pointer = mxArrayToString(mx_charvec);
    cpu_charvec_vals.clear();
    cpu_charvec_vals.resize(cpu_charvec_size, 0);
    memcpy(cpu_charvec_vals.data(), cpu_charvec_vals_pointer, sizeof(char) * cpu_charvec_size);
    mxFree(cpu_charvec_vals_pointer);
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
        int blk_types;
        int blk_vals;
        int X;
        int y;
        int S;
        int sig;
        int sig_update_threshold;
        int sig_update_stage_1;
        int sig_update_stage_2;
        int switch_admm;
        int switch_proj_iter;
        int switch_proj_tol;
        int sigscale;

        INPUT_ID_factory(int offset = 0) {
            this->eig_stream_num_per_gpu = offset + 0;
            this->max_iter = offset + 1;
            this->stop_tol = offset + 2;
            this->At = offset + 3;
            this->b = offset + 4;
            this->C = offset + 5;
            this->blk_types = offset + 6;
            this->blk_vals = offset + 7;
            this->X = offset + 8;
            this->y = offset + 9;
            this->S = offset + 10;
            this->sig = offset + 11;
            this->sig_update_threshold = offset + 12;
            this->sig_update_stage_1 = offset + 13;
            this->sig_update_stage_2 = offset + 14;
            this->switch_admm = offset + 15;
            this->switch_proj_iter = offset + 16;
            this->switch_proj_tol = offset + 17;
            this->sigscale = offset + 18;
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

// plhs only has room for max(nlhs, 1) outputs: hand the output to MATLAB only if it was requested
void set_output(int nlhs, mxArray* plhs[], int output_id, mxArray* mx_out) {
    if (output_id == 0 || output_id < nlhs)
        plhs[output_id] = mx_out;
    else
        mxDestroyArray(mx_out);
    return;
}


// Body of mexFunction; reports errors by throwing (InputError for invalid inputs).
void cuadmm_mex(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    INPUT_ID_factory INPUT_ID(0);
    OUTPUT_ID_factory OUTPUT_ID(0);
    OUTPUT_INFO_RID_factory OUTPUT_INFO_RID;

    // -------------------------------------------------------
    // input:

    // inputs up to sig are required, the ones after it are optional
    if (nrhs < INPUT_ID.sig + 1 || nrhs > INPUT_ID.sigscale + 1) {
        throw InputError("cuadmm_MATLAB expects between " + std::to_string(INPUT_ID.sig + 1) + " and " +
                         std::to_string(INPUT_ID.sigscale + 1) + " inputs, got " + std::to_string(nrhs) + ".");
    }
    if (nlhs > OUTPUT_ID.info + 1) {
        throw InputError("cuadmm_MATLAB returns at most " + std::to_string(OUTPUT_ID.info + 1) + " outputs.");
    }

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
    check_double_input(prhs[INPUT_ID.At], "At", true, false);
    get_spmat_csc_from_matlab(
        prhs[INPUT_ID.At],
        vec_len, con_num, At_nnz, cpu_At_csc_col_ptrs, cpu_At_csc_row_ids, cpu_At_csc_vals
    );
    
    // b
    std::vector<int> cpu_b_indices;
    std::vector<double> cpu_b_vals; 
    int b_nnz;
    int b_size;
    check_double_input(prhs[INPUT_ID.b], "b", true);
    get_spvec_from_matlab(
        prhs[INPUT_ID.b],
        b_size, b_nnz, cpu_b_indices, cpu_b_vals
    );
    if (b_size != con_num) {
        throw InputError("The length of b (" + std::to_string(b_size) + ") does not match the number of columns of At (" +
                         std::to_string(con_num) + ").");
    }
    
    // C
    std::vector<int> cpu_C_indices; 
    std::vector<double> cpu_C_vals; 
    int C_nnz;
    int C_size;
    check_double_input(prhs[INPUT_ID.C], "C", true);
    get_spvec_from_matlab(
        prhs[INPUT_ID.C],
        C_size, C_nnz, cpu_C_indices, cpu_C_vals
    );
    if (C_size != vec_len) {
        throw InputError("The length of C (" + std::to_string(C_size) + ") does not match the number of rows of At (" +
                         std::to_string(vec_len) + ").");
    }

    // blk
    // TODO: adapt for new signature
    int mat_num;
    std::vector<char> cpu_blk_types;
    if (!mxIsChar(prhs[INPUT_ID.blk_types]) || mxGetN(prhs[INPUT_ID.blk_types]) != 1) {
        throw InputError("blk_types must be a char column vector.");
    }
    get_char_vec_from_matlab(
        prhs[INPUT_ID.blk_types], 
        mat_num, cpu_blk_types
    );
    int blk_vals_size;
    std::vector<double> cpu_blk_vals_double;
    check_double_input(prhs[INPUT_ID.blk_vals], "blk_sizes", false);
    get_dnvec_from_matlab(
        prhs[INPUT_ID.blk_vals], 
        blk_vals_size, cpu_blk_vals_double
    );
    if (blk_vals_size != mat_num) {
        throw InputError("blk_types and blk_sizes must have the same length (got " + std::to_string(mat_num) + " and " +
                         std::to_string(blk_vals_size) + ").");
    }
    std::vector<int> cpu_blk_vals(mat_num, 0);
    int vec_len_from_blk = 0;
    for (int i = 0; i < mat_num; i++) {
        cpu_blk_vals[i] = static_cast<int>( cpu_blk_vals_double[i] );
        if (cpu_blk_types[i] == 's')
            vec_len_from_blk = vec_len_from_blk + cpu_blk_vals[i] * (cpu_blk_vals[i] + 1) / 2;
        else if (cpu_blk_types[i] == 'u' || cpu_blk_types[i] == 'l')
            vec_len_from_blk = vec_len_from_blk + cpu_blk_vals[i];
        else {
            throw InputError(std::string("The type of blk should be 's', 'l' or 'u', but got '") + cpu_blk_types[i] + "'.");
        }

    }
    if (vec_len_from_blk != vec_len) {
        throw InputError("The length of blk does not match the length of At. (blk length: " +
                         std::to_string(vec_len_from_blk) + ", At length: " + std::to_string(vec_len) + ")");
    }

    // X
    int X_size;
    std::vector<double> cpu_X_vals;
    check_double_input(prhs[INPUT_ID.X], "X0", false);
    get_dnvec_from_matlab(
        prhs[INPUT_ID.X], 
        X_size, cpu_X_vals
    );
    if (X_size != vec_len) {
        throw InputError("The length of X0 (" + std::to_string(X_size) + ") does not match the number of rows of At (" +
                         std::to_string(vec_len) + ").");
    }


    // y
    int y_size;
    std::vector<double> cpu_y_vals;
    check_double_input(prhs[INPUT_ID.y], "y0", false);
    get_dnvec_from_matlab(
        prhs[INPUT_ID.y],
        y_size, cpu_y_vals
    );
    if (y_size != con_num) {
        throw InputError("The length of y0 (" + std::to_string(y_size) + ") does not match the number of columns of At (" +
                         std::to_string(con_num) + ").");
    }


    // S
    int S_size;
    std::vector<double> cpu_S_vals;
    check_double_input(prhs[INPUT_ID.S], "S0", false);
    get_dnvec_from_matlab(
        prhs[INPUT_ID.S], 
        S_size, cpu_S_vals
    );
    if (S_size != vec_len) {
        throw InputError("The length of S0 (" + std::to_string(S_size) + ") does not match the number of rows of At (" +
                         std::to_string(vec_len) + ").");
    }

    // sig
    double sig = mxGetScalar(prhs[INPUT_ID.sig]);

    // sig_update_threshold
    int sig_update_threshold;
    if (has_optional_input(nrhs, prhs, INPUT_ID.sig_update_threshold)) {
        sig_update_threshold = static_cast<int>( mxGetScalar(prhs[INPUT_ID.sig_update_threshold]) );
    } else {
        sig_update_threshold = 500;
    }

    // sig_update_stage_1
    int sig_update_stage_1;
    if (has_optional_input(nrhs, prhs, INPUT_ID.sig_update_stage_1)) {
        sig_update_stage_1 = static_cast<int>( mxGetScalar(prhs[INPUT_ID.sig_update_stage_1]) );
    } else {
        sig_update_stage_1 = 50;
    }

    // sig_update_stage_2
    int sig_update_stage_2;
    if (has_optional_input(nrhs, prhs, INPUT_ID.sig_update_stage_2)) {
        sig_update_stage_2 = static_cast<int>( mxGetScalar(prhs[INPUT_ID.sig_update_stage_2]) );
    } else {
        sig_update_stage_2 = 100;
    }

    // switch_admm
    int switch_admm;
    if (has_optional_input(nrhs, prhs, INPUT_ID.switch_admm)) {
        switch_admm = static_cast<int>( mxGetScalar(prhs[INPUT_ID.switch_admm]) );
    } else {
        switch_admm = 0;
    }

    // switch_proj_iter
    int switch_proj_iter;
    if (has_optional_input(nrhs, prhs, INPUT_ID.switch_proj_iter)) {
        switch_proj_iter = static_cast<int>( mxGetScalar(prhs[INPUT_ID.switch_proj_iter]) );
    } else {
        switch_proj_iter = (int) 5000;
    }

    // switch_proj_tol
    double switch_proj_tol;
    if (has_optional_input(nrhs, prhs, INPUT_ID.switch_proj_tol)) {
        switch_proj_tol = mxGetScalar(prhs[INPUT_ID.switch_proj_tol]);
    } else {
        switch_proj_tol = (double) 1e-2;
    }

    // sigscale
    double sigscale;
    if (has_optional_input(nrhs, prhs, INPUT_ID.sigscale)) {
        sigscale = mxGetScalar(prhs[INPUT_ID.sigscale]);
    } else {
        sigscale = 2.0;
    }

    // -------------------------------------------------------

    // -------------------------------------------------------
    // start solver:

    SDPSolver solver;
    solver.init(
        eig_stream_num_per_gpu,
        vec_len, con_num,
        cpu_At_csc_col_ptrs.data(), cpu_At_csc_row_ids.data(), cpu_At_csc_vals.data(), At_nnz,
        cpu_b_indices.data(), cpu_b_vals.data(), b_nnz,
        cpu_C_indices.data(), cpu_C_vals.data(), C_nnz,
        cpu_blk_types.data(),
        cpu_blk_vals.data(),
        mat_num,
        ProjectionMethod::EIG_FP64,
        ProjectionMethod::EIG_FP64,
        cpu_X_vals.data(), cpu_y_vals.data(), cpu_S_vals.data(), sig
    );
    solver.solve(
        max_iter, stop_tol,
        sig_update_threshold, // sig_update_threshold
        sig_update_stage_1,   // sig_update_stage_1
        sig_update_stage_2,   // sig_update_stage_2
        switch_admm,          // switch_admm
        switch_proj_iter,     // switch_proj_max_iter
        switch_proj_tol,      // switch_proj_tol
        sigscale              // sigscale
    );
    // -------------------------------------------------------

    // -------------------------------------------------------
    // output:

    // X
    mxArray* mx_X_out = mxCreateDoubleMatrix(vec_len, 1, mxREAL);
    double* X_out = mxGetPr(mx_X_out);
    CHECK_CUDA( cudaMemcpy(X_out, solver.X.vals, sizeof(double) * vec_len, D2H) );
    set_output(nlhs, plhs, OUTPUT_ID.X, mx_X_out);

    // y
    mxArray* mx_y_out = mxCreateDoubleMatrix(con_num, 1, mxREAL);
    double* y_out = mxGetPr(mx_y_out);
    CHECK_CUDA( cudaMemcpy(y_out, solver.y.vals, sizeof(double) * con_num, D2H) );
    set_output(nlhs, plhs, OUTPUT_ID.y, mx_y_out);

    // S
    mxArray* mx_S_out = mxCreateDoubleMatrix(vec_len, 1, mxREAL);
    double* S_out = mxGetPr(mx_S_out);
    CHECK_CUDA( cudaMemcpy(S_out, solver.S.vals, sizeof(double) * vec_len, D2H) );
    set_output(nlhs, plhs, OUTPUT_ID.S, mx_S_out);

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
    // info_total_time
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.total_time + info_size * 0, mxCreateString("total_time"));
    mxSetCell(mx_info_out, OUTPUT_INFO_RID.total_time + info_size * 1, mxCreateDoubleScalar((double)(solver.total_time)));
    set_output(nlhs, plhs, OUTPUT_ID.info, mx_info_out);
    // -------------------------------------------------------

    // -------------------------------------------------------
    // debug
    
    // -------------------------------------------------------

    return;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // mexErrMsgIdAndTxt does not return and does not unwind the C++ stack (it jumps back to MATLAB), so it must only be
    // called once every C++ object owning host or GPU memory is gone. All of them (the SDPSolver included) live in
    // cuadmm_mex and are destroyed while the exception propagates out of it; the message is copied into a local buffer
    // so that the exception object itself is destroyed at the end of its handler, before the error is raised.
    const char* err_id = "cuADMM:solverError";
    char err_msg[1024];
    try {
        cuadmm_mex(nlhs, plhs, nrhs, prhs);
        return;
    } catch (const InputError& e) {
        err_id = "cuADMM:invalidInput";
        snprintf(err_msg, sizeof(err_msg), "%s", e.what());
    } catch (const std::exception& e) {
        snprintf(err_msg, sizeof(err_msg), "%s", e.what());
    } catch (...) {
        snprintf(err_msg, sizeof(err_msg), "unknown C++ exception");
    }
    mexErrMsgIdAndTxt(err_id, "%s", err_msg);
}