#include "cuda.h"

// this file is compatible with CUDA 12.9 and later
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "psd_projection/composite_FP32_emulated.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

void composite_FP32_emulated(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    float* workspace
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3;
    if (workspace == nullptr) {
        CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );
    } else {
        A = workspace;
        A2 = workspace + (size_t)nn;     // offsets in size_t: 2 * nn overflows an int for n >= 32768
        A3 = workspace + 2 * (size_t)nn;
    }

    // useful constants
    const float half       =  0.5f;
    const float minus_half = -0.5f;
    const float one        =  1.0f;
    const float one_n_half =  1.5f;
    const float zero       =  0.0f;

    /* Convert the initial matrix*/
    convert_double_to_float(mat, A, nn);

    /* Coefficients */
    // const std::vector<std::vector<float>> coeff = {
    //     { 8.509885302586273, -25.264304190830892,  18.753567899739625 },
    //     { 4.249573478922877,   -3.154976488114228,   0.585884782491327 },
    //     { 4.225122190777846,   -3.138044435084575,   0.583953455129916 },
    //     { 4.124838686994395,   -3.068332452805990,   0.576002953645695 },
    //     { 3.758010335802897,   -2.809273892403287,   0.546484206587685 },
    //     { 2.856177541291611,   -2.134056233175483,   0.470110769180275 },
    //     { 2.020600415776305,   -1.403721150466785,   0.390673896852026 },
    //     { 1.875875100481076,   -1.250971990481385,   0.375097212342072 },
    //     { 1.875,   -1.25,   0.375},
    //     { 1.875,   -1.25,   0.375},
    // }; const size_t smoothing_steps = 0; // best: 10 minimax original

    // const std::vector<std::vector<float>> coeff = {
    //     { 8.5117053694,  -25.2637545356,   18.7518511505 },
    //     { 4.2514746568,   -3.1551482052,    0.5855654848 },
    //     { 4.2314443096,   -3.1432483391,    0.5844187862 },
    //     { 4.1462871213,   -3.0853187659,    0.5781140029 },
    //     { 3.8679345846,   -2.8863505270,    0.5573798771 },
    //     { 3.0735744409,   -2.2984793859,    0.4942218088 },
    //     { 2.1692233704,   -1.5420827375,    0.4146319529 },
    //     { 2.0078578610,   -1.3793846146,    0.3989298303 },
    //     { 2.0029525899,   -1.3743625171,    0.3982429919 },
    //     { 1.8780193554,   -1.2544181003,    0.3764365891 },
    // }; const size_t smoothing_steps = 0; // 10 minimax refined

    const std::vector<std::vector<float>> coeff = {
        { 8.3119043343,  -23.0739115930,  16.4664144722 },
        { 4.1439360087,   -2.9176674704,   0.5246212487 },
        { 4.0257813209,   -2.9025002398,   0.5334261214 },
        { 3.5118574347,   -2.5740236523,   0.5050097282 },
        { 2.4398158400,   -1.7586675341,   0.4191290613 },
        { 1.9779835097,   -1.3337358510,   0.3772169049 },
        { 1.9559726949,   -1.3091355170,   0.3746734515 },
        { 1.9282822454,   -1.2823649693,   0.3704626545 },
        { 1.9220135179,   -1.2812524618,   0.3707011753 },
        { 1.8942192942,   -1.2613293407,   0.3676616051 }
    }; const size_t smoothing_steps = 0; // 10 polar express refined

    float scale_factor = 1.001f;

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        float a = coeff[i][0];
        float b = coeff[i][1];
        float c = coeff[i][2];

        if (i < 8) {
            a /= scale_factor;
            b /= scale_factor * scale_factor * scale_factor;
            c /= scale_factor * scale_factor * scale_factor * scale_factor 
                * scale_factor;
        }

        /* Compute the powers of A*/
        // A2 = A * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A, CUDA_R_32F, n,
            &zero,
            A2, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &zero,
            A3, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        // A = c * A3 * A2 + A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &c,
            A3, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &one,
            A, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Smoothing function */
    for (int i = 0; i < smoothing_steps; i++) {
        // A2 = A * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A, CUDA_R_32F, n,
            &zero,
            A2, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &zero,
            A3, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        /* Symmetrize A3 */
        symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

        /* Compute A = 1.5 * A - 0.5 * A3 */
        // A = 1.5 * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
        // A = -0.5 * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    // A = I + A
    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    // W = A_origin * A
    convert_double_to_float(mat, A2, nn);
    CHECK_CUBLAS( cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        A2, CUDA_R_32F, n,
        A, CUDA_R_32F, n,
        &zero,
        A3, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    if (workspace == nullptr) {
        CHECK_CUDA( cudaFree(A) );
        CHECK_CUDA( cudaFree(A2) );
        CHECK_CUDA( cudaFree(A3) );
    }
}

void composite_FP32_emulated_auto_scale(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int n,
    float* workspace
) {
    size_t nn = n * n;
    
    // Use the Lanczos method to approximate the two-norm of the matrix
    double lo, up;
    approximate_two_norm(
        cublasH, solverH, mat, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, mat, 1) );

    // project the matrix using the composite_FP32 function
    composite_FP32_emulated(cublasH, mat, n, workspace);

    // rescale the result back to the original scale
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale,  mat, 1) );
}

void composite_FP32_emulated_auto_scale_deflate(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int n,
    const size_t k,
    const double tol,
    const double ortho_tol,
    const bool verbose
) {
    size_t nn = n * n;
    
    /* Step 1: compute the largest eigenpairs of the matrix */
    size_t r;
    double *eigenvalues, *eigenvectors;
    CHECK_CUDA( cudaMalloc(&eigenvalues,      k * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&eigenvectors, n * k * sizeof(double)) );

    double _ = compute_eigenpairs(
        cublasH, solverH, mat, n, k, &r, eigenvalues, eigenvectors, false, 0, tol, ortho_tol
    );

    std::vector<double> eigenvalues_host(r);
    CHECK_CUDA( cudaMemcpy(eigenvalues_host.data(), eigenvalues, r * sizeof(double), D2H) );

    /* Step 2: remove the largest eigenvalues from the matrix */
    for (int i = 0; i < r; i++) {
        // X <- X - \lambda_i * v_i v_i^T
        double lambda = -eigenvalues_host[i];
        double *v_i = eigenvectors + i * n;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, mat, n) );
    }

    /* Step 3: scale the deflated matrix */
    double up = compute_eigenpairs(
        cublasH, solverH, mat, n, 0, nullptr, nullptr, nullptr, true, 100, 1e-10, 1e-5, verbose
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? 1.1 * up + 1e-5 : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, mat, 1) );

    /* Step 4: project the matrix using the composite_FP32 function */
    composite_FP32_emulated(cublasH, mat, n);

    /* Step 5: rescale the matrix back and add the deflated eigenvalues back */
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, mat, 1) );

    for (int i = 0; i < r; i++) {
        // X <- X + \lambda_i * v_i v_i^T
        double lambda = eigenvalues_host[i];
        if (lambda > 0.0) { // only add positive eigenvalues
            double *v_i = eigenvectors + i * n;
            CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, mat, n) );
        }
    }

    /* Free device memory */
    CHECK_CUDA( cudaFree(eigenvalues) );
    CHECK_CUDA( cudaFree(eigenvectors) );
}

#endif