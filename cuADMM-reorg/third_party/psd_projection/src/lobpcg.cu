#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

#include "psd_projection/lobpcg.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

// norms[j] = ||R(:, j)||_2 for an n x m column-major matrix R (one thread per column, no shared memory)
__global__ void column_norms_kernel(const double* R, double* norms, const int n, const int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m) {
        const double* col = R + (size_t)j * n;
        double s = 0.0;
        for (int i = 0; i < n; i++)
            s += col[i] * col[i];
        norms[j] = sqrt(s);
    }
}

static void column_norms(const double* R, double* norms, const int n, const int m) {
    const int threads = 64;
    column_norms_kernel<<<(m + threads - 1) / threads, threads>>>(R, norms, n, m);
    CHECK_CUDA(cudaGetLastError());
}

// Writes a sentinel (-1) to devInfo before a cuSOLVER call: a call that fails without writing devInfo
// (e.g. an invalid argument or an internal allocation failure) is then detected by check_dev_info.
static void reset_dev_info(int* devInfo) {
    CHECK_CUDA(cudaMemset(devInfo, 0xFF, sizeof(int)));
}

// Returns true if the cuSOLVER devInfo is 0, otherwise prints a warning and returns false.
static bool check_dev_info(const int* devInfo, const char* what) {
    int hInfo = 0;
    CHECK_CUDA(cudaMemcpy(&hInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (hInfo != 0) {
        std::fprintf(stderr, "LOBPCG: %s failed: devInfo = %d\n", what, hInfo);
        return false;
    }
    return true;
}

void lobpcg(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, // n x n, device pointer
    double* V,       // n x m, device pointer (output eigenvectors)
    double* D,       // m x m, device pointer (output eigenvalues, diagonal)
    const int n,
    const int m ,      // number of eigenpairs
    const bool warmstart,
    const int maxiter, // maximum iterations
    const double tol,   // convergence tolerance
    const bool verbose,
    LobpcgInfo* info,
    const double conv_threshold,
    const int min_checked
) {
    assert(m > 0);
    assert(n > 0);
    assert(3*m <= n);

    /* Allocations */
    // allocate the device memory
    double *X_k, *X_k_tmp, *Lam_k, *Lam_k_tmp, *T, *Delta_X_k, *T_tmp, *R_k;
    double *XRD, *Lam_all, *XRD_tmp, *T_XRD, *T_tmp_XRD;
    CHECK_CUDA(cudaMalloc(&X_k,            n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&X_k_tmp,        n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_k,              m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_k_tmp,          m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Delta_X_k,      n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_tmp,          n * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T,              m * m * sizeof(double)));
    
    CHECK_CUDA(cudaMalloc(&XRD,          n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Lam_all,          3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&XRD_tmp,      n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_XRD,      3*m * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T_tmp_XRD,    n * 3*m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&R_k,          n * 3*m * sizeof(double)));

    double *R_norms; // per-column residual norms ||A x_j - lambda_j x_j||_2
    CHECK_CUDA(cudaMalloc(&R_norms, m * sizeof(double)));
    // NaN until copied back, so that a failed copy cannot look like convergence
    std::vector<double> h_R_norms(m, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> h_Lam_k(m, std::numeric_limits<double>::quiet_NaN());
    double max_res = std::numeric_limits<double>::infinity(); // residual of the current (X_k, Lam_k)
    bool converged = false;
    bool failed = false; // a cuSOLVER call reported devInfo != 0
    int nb_updates = 0;  // number of Rayleigh-Ritz updates performed

    // useful constants
    const double one = 1.0;
    const double zero = 0.0;
    const double half = 0.5;
    const double neg1 = -1.0;

    // workspace for eigenvalue decomposition
    int lwork_eig;
    double *d_work_eig;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                                m, T, m, Lam_k, &lwork_eig));
    CHECK_CUDA(cudaMalloc(&d_work_eig, lwork_eig * sizeof(double)));

    // workspace for eigenvalue decomposition of XRD
    int lwork_eig_XRD;
    double *d_work_eig_XRD;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                                3*m, T_XRD, 3*m, Lam_all, &lwork_eig_XRD));
    CHECK_CUDA(cudaMalloc(&d_work_eig_XRD, lwork_eig_XRD * sizeof(double)));

    int *devInfo;
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

    // workspace for QR decomposition of XRD
    int lwork_xrd;
    double *d_work_xrd, *tau_xrd;
    CHECK_CUDA(cudaMalloc(&tau_xrd, 3*m * sizeof(double)));

    int lwork_geqrf = 0, lwork_orgqr = 0;
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, n, 3*m, XRD, n, &lwork_geqrf));
    CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, n, 3*m, 3*m, XRD, n, tau_xrd, &lwork_orgqr));
    lwork_xrd = std::max(lwork_geqrf, lwork_orgqr);

    CHECK_CUDA(cudaMalloc(&d_work_xrd, lwork_xrd * sizeof(double)));

    /* Initialization of X_k */
    if (warmstart) {
        CHECK_CUDA(cudaMemcpy(X_k, V, n * m * sizeof(double), D2D));
        CHECK_CUDA(cudaMemcpy(Lam_k, D, m * sizeof(double), D2D));
        // note: we assume vectors in V are orthonormal
    } else {
        // workspace for QR decomposition of X_k
        double *d_work, *tau;
        int lwork;
        CHECK_CUDA(cudaMalloc(&tau, m * sizeof(double)));
        int lwork_orgqr_x = 0;
        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, n, m, X_k, n, &lwork));
        CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, n, m, m, X_k, n, tau, &lwork_orgqr_x));
        lwork = std::max(lwork, lwork_orgqr_x);
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));

        fill_random(X_k, n * m, 0);

        // compute QR factorization (X_k overwritten with R, tau contains Householder scalars)
        reset_dev_info(devInfo);
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, n, m, X_k, n, tau, d_work, lwork, devInfo));
        if (!check_dev_info(devInfo, "initial GEQRF"))
            failed = true;

        // generate Q from the result (X_k overwritten with Q)
        reset_dev_info(devInfo);
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, n, m, m, X_k, n, tau, d_work, lwork, devInfo));
        if (!check_dev_info(devInfo, "initial ORGQR"))
            failed = true;

        CHECK_CUDA(cudaFree(d_work));
        CHECK_CUDA(cudaFree(tau));
    }

    /* Compute new X_k using T */
    // T = Q^T * A * Q
    // T_tmp = Q^T * A
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, n,
                             &one, X_k, n, A, n,
                             &zero, T_tmp, m));
    // T = T_tmp * Q
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, n,
                             &one, T_tmp, m, X_k, n,
                             &zero, T, m));
    // copy T to T_tmp
    CHECK_CUBLAS(cublasDcopy(cublasH, m * m, T, 1, T_tmp, 1));

    // T = 0.5 * (T + T^T)
    CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, m,
                             &half, T, m,
                             &half, T_tmp, m,
                             T, m));

    // compute eigenvalues and eigenvectors of T
    // both are in increasing order
    reset_dev_info(devInfo);
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    m, T, m, Lam_k_tmp, d_work_eig, lwork_eig, devInfo));
    if (!check_dev_info(devInfo, "initial SYEVD"))
        failed = true;
    // reverse Lam_k_tmp to get Lam_k in decreasing order
    reverse_vector(Lam_k_tmp, Lam_k, m);
    // reverse the columns of T accordingly, so that column j of X_k matches Lam_k[j]
    reverse_columns(T, T_tmp, m, m);

    // X_k = Q * T
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m,
                             &one, X_k, n, T_tmp, m,
                             &zero, Delta_X_k, n));
    CHECK_CUDA(cudaMemcpy(X_k, Delta_X_k, n * m * sizeof(double), D2D));

    // Delta_X_k = X_k
    CHECK_CUBLAS(cublasDcopy_v2(cublasH, n * m, X_k, 1, Delta_X_k, 1));

    for (int iter = 1; !failed; iter++) {
        // R_k = A * X_k - X_k * Lam_k
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n,
                                 &one, A, n, X_k, n,
                                 &zero, R_k, n));
        // scale X_k by Lam_k
        CHECK_CUBLAS(cublasDdgmm(
            cublasH,
            CUBLAS_SIDE_RIGHT, // scale columns
            n,                 // number of rows
            m,                 // number of columns
            X_k, n,            // input matrix
            Lam_k, 1,         // vector (stride 1)
            X_k_tmp, n       // output matrix
        ));
        // substract it from R_k
        CHECK_CUBLAS(cublasDaxpy(cublasH, n * m, &neg1, X_k_tmp, 1, R_k, 1));

        // largest residual norm among the first min_checked Ritz pairs and the pairs with value > conv_threshold
        // (Ritz values are lower bounds: a pair below the threshold may still be an unconverged direction)
        column_norms(R_k, R_norms, n, m);
        CHECK_CUDA(cudaMemcpy(h_R_norms.data(), R_norms, m * sizeof(double), D2H));
        CHECK_CUDA(cudaMemcpy(h_Lam_k.data(), Lam_k, m * sizeof(double), D2H));
        max_res = 0.0;
        for (int j = 0; j < m; j++) {
            if (j < min_checked || h_Lam_k[j] > conv_threshold)
                max_res = std::max(max_res, h_R_norms[j]); // NaN residuals are caught below
            if (std::isnan(h_R_norms[j]) || std::isnan(h_Lam_k[j]))
                max_res = std::numeric_limits<double>::quiet_NaN();
        }

        if (verbose) {
            std::cout << "LOBPCG iter: " << iter << " max_j ||r_j|| = " << max_res << std::endl;
        }

        // if the largest checked residual is less than tol, break; at least one Rayleigh-Ritz update is
        // required: the (orthonormalized) residual block brings into the search subspace any direction that
        // couples to the start subspace, e.g. a new eigenvector hidden behind a pair below conv_threshold
        if (max_res < tol && nb_updates > 0) {
            converged = true;
            if (verbose) {
                std::cout << "Converged: max_j ||r_j|| < tol" << std::endl;
            }
            break;
        }
        if (iter > maxiter || std::isnan(max_res))
            break;

        // concatenate X_k, R_k, and Delta_X_k into XRD
        CHECK_CUDA(cudaMemcpy(XRD            ,       X_k, n * m * sizeof(double), D2D));
        CHECK_CUDA(cudaMemcpy(XRD +     n * m,       R_k, n * m * sizeof(double), D2D));
        CHECK_CUDA(cudaMemcpy(XRD + 2 * n * m, Delta_X_k, n * m * sizeof(double), D2D));

        // compute QR factorization of XRD
        reset_dev_info(devInfo);
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, n, 3*m, XRD, n, tau_xrd, d_work_xrd, lwork_xrd, devInfo));
        if (!check_dev_info(devInfo, "QR/GEQRF")) {
            failed = true;
            break;
        }
        reset_dev_info(devInfo);
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, n, 3*m, 3*m, XRD, n, tau_xrd, d_work_xrd, lwork_xrd, devInfo));
        if (!check_dev_info(devInfo, "QR/ORGQR")) {
            failed = true;
            break;
        }

        // T = Q^T * A * Q
        // T_tmp = Q^T * A
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 3*m, n, n,
                                &one, XRD, n, A, n,
                                &zero, T_tmp_XRD, 3*m));
        // T = T_tmp * Q
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 3*m, 3*m, n,
                                &one, T_tmp_XRD, 3*m, XRD, n,
                                &zero, T_XRD, 3*m));
        // copy T to T_tmp
        CHECK_CUBLAS(cublasDcopy(cublasH, 3*m * 3*m, T_XRD, 1, T_tmp_XRD, 1));

        // T = 0.5 * (T + T^T)
        CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 3*m, 3*m,
                                &half, T_XRD, 3*m,
                                &half, T_tmp_XRD, 3*m,
                                T_XRD, 3*m));

        // compute eigenvalues and eigenvectors of T
        // both are in increasing order
        reset_dev_info(devInfo);
        CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                        3*m, T_XRD, 3*m, Lam_all, d_work_eig_XRD, lwork_eig_XRD, devInfo));
        if (!check_dev_info(devInfo, "SYEVD")) {
            failed = true; // X_k and Lam_k still hold the last (checked) iterate
            break;
        }
        // reverse columns of T_XRD
        reverse_columns(T_XRD, T_tmp_XRD, 3*m, 3*m);

        // XRD_tmp = Q * T
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, 3*m, 3*m,
                                &one, XRD, n, T_tmp_XRD, 3*m,
                                &zero, XRD_tmp, n));

        // Delta_X_k = - X_k
        CHECK_CUBLAS(cublasDcopy(cublasH, n * m, X_k, 1, Delta_X_k, 1));
        CHECK_CUBLAS(cublasDscal(cublasH, n * m, &neg1, Delta_X_k, 1));

        // X_k = XRD_tmp(1:m)
        extract_columns(XRD_tmp, X_k, n, m);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Delta = X_kp1 - X_k
        CHECK_CUBLAS(cublasDaxpy(cublasH, n * m, &one, X_k, 1, Delta_X_k, 1));
        
        // Lam_k = Lam_all(2m:3m)
        CHECK_CUBLAS(cublasDcopy(cublasH, m, Lam_all + 2*m, 1, Lam_k_tmp, 1));
        reverse_vector(Lam_k_tmp, Lam_k, m);
        nb_updates++;
    }

    if (info != nullptr) {
        info->iterations = nb_updates;
        info->residual_norm = max_res;
        info->converged = converged && !failed;
    }

    /* Copy results to output */
    // V = X_k
    CHECK_CUBLAS(cublasDcopy(cublasH, n * m, X_k, 1, V, 1));
    // D = Lam_k
    CHECK_CUBLAS(cublasDcopy(cublasH, m, Lam_k, 1, D, 1));


    // Free device memory
    CHECK_CUDA(cudaFree(X_k));
    CHECK_CUDA(cudaFree(X_k_tmp));
    CHECK_CUDA(cudaFree(Lam_k));
    CHECK_CUDA(cudaFree(Lam_k_tmp));
    CHECK_CUDA(cudaFree(d_work_xrd));
    CHECK_CUDA(cudaFree(tau_xrd));
    CHECK_CUDA(cudaFree(T_tmp));
    CHECK_CUDA(cudaFree(T_tmp_XRD));
    CHECK_CUDA(cudaFree(d_work_eig));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(T));
    CHECK_CUDA(cudaFree(Delta_X_k));
    CHECK_CUDA(cudaFree(R_k));
    
    CHECK_CUDA(cudaFree(XRD));
    CHECK_CUDA(cudaFree(Lam_all));
    CHECK_CUDA(cudaFree(T_XRD));
    CHECK_CUDA(cudaFree(XRD_tmp));
    CHECK_CUDA(cudaFree(d_work_eig_XRD));
    CHECK_CUDA(cudaFree(R_norms));

    if (failed && info == nullptr)
        throw std::runtime_error("lobpcg: a cuSOLVER factorization failed (devInfo != 0)");
}