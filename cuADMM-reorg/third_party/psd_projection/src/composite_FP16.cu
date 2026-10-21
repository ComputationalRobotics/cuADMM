#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "psd_projection/composite_FP16.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/lobpcg.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"

void composite_FP16(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    float* float_workspace,
    __half* half_workspace
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    // offsets in size_t: 2 * stride overflows an int for n >= 32768
    size_t stride = nn % 4 == 0 ? (size_t)nn : (size_t)nn + (4 - nn % 4);
    float *A, *A2, *A3;
    if (float_workspace == nullptr) {
        CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );
    } else {
        A = float_workspace;
        A2 = float_workspace + stride;
        A3 = float_workspace + 2 * stride;
    }

    __half *hA, *hA2, *hA3;
    if (half_workspace == nullptr) {
        CHECK_CUDA(cudaMalloc(&hA,  nn * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&hA2, nn * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&hA3, nn * sizeof(__half)));
    } else {
        hA = half_workspace;
        hA2 = half_workspace + stride;
        hA3 = half_workspace + 2 * stride;
    }

    // useful constants
    const float half       =  0.5f;
    const float one        =  1.0f;
    const float zero       =  0.0f;

    convert_double_to_float(mat, A, nn);

    /* Coefficients */
    // std::vector<std::vector<float>> coeff = {
    //     {8.4724206924, -24.5001735687, 17.7268180847},
    //     {4.2052841187, -3.0549299717, 0.5567536354},
    //     {4.0443077087, -2.9473149776, 0.5449726582},
    //     {3.5078327656, -2.5842490196, 0.5067413449},
    //     {2.5075511932, -1.8485442400, 0.4358045161}
    // };
    // std::vector<std::vector<float>> coeff = { 
    //     { 8.3885353390, -23.7796270883, 16.8664591580 }, 
    //     { 4.1636476423, -2.9650849331, 0.5297319805 }, 
    //     { 4.0042650581, -2.8606348801, 0.5185227850 }, 
    //     { 3.4731017481, -2.5082466382, 0.4821470022 }, 
    //     { 2.4827239537, -1.7941788274, 0.4146530436 }, 
    // };

    // const std::vector<std::vector<float>> coeff = { 
    //     { 8.3937001154, -23.7945582332, 16.8758390904 }, 
    //     { 4.1803895500, -2.9788012917, 0.5318143742 }, 
    //     { 4.0578478573, -2.9013956514, 0.5233571836 }, 
    //     { 3.6289664769, -2.6254124593, 0.4963343458 }, 
    //     { 2.7619020904, -1.9865006927, 0.4388497859 }, 
    //     { 2.0674922563, -1.4317903208, 0.3876934521 }, 
    //     { 1.8438914749, -1.1872290786, 0.3433825749 }, 
    // }; // current best

    // std::vector<std::vector<float>> coeff = {
    //     { 8.3864641622, -24.8594799076, 18.4448273259 },
    //     { 4.1414199835, -3.0779218910, 0.5748581003 },
    //     { 3.9226309693, -2.9248155905, 0.5574020970 },
    //     { 3.2540457594, -2.4403169649, 0.5023343504 },
    //     { 2.2512376183, -1.6283766019, 0.4120702252 },
    //     { 1.8700160370, -1.2526309162, 0.3727910154 },
    //     { 1.8564365206, -1.2376247369, 0.3712872262 },
    // };

    const std::vector<std::vector<float>> coeff = {
        { 8.2885332412,  -22.5927099246, 15.8201383114 },
        { 4.1666196466,   -2.9679004036,  0.5307623217 },
        { 4.0611848147,   -2.9698947955,  0.5492133813 },
        { 3.6678301399,   -2.7561018955,  0.5421513305 },
        { 2.7632556383,   -2.0607754898,  0.4695405857 },
        { 2.0527445797,   -1.4345145882,  0.4070669182 },
        { 1.8804816691,   -1.2583997294,  0.3779501813 }
    }; // polar express with refinement

    float scale_factor = 1.01f;

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        float a = coeff[i][0];
        float b = coeff[i][1];
        float c = coeff[i][2];

        a /= scale_factor;
        b /= scale_factor * scale_factor * scale_factor;
        c /= scale_factor * scale_factor * scale_factor * scale_factor * scale_factor;

        /* Compute the powers of A*/
        // A2 = A * A
        convert_float_to_half4(A, hA, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA, CUDA_R_16F, n,
            &zero,
            A2,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A3 = A2 * A
        convert_float_to_half4(A2, hA2, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA2, CUDA_R_16F, n,
            &zero,
            A3,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        /* Symmetrize A3, A5 */
        // symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace
        // symmetrizeFloat(cublasH, A5, n, A2); // we use A2 as a workspace

        // A = c * A3 * A2 + A
        convert_float_to_half4(A3, hA3, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &c,
            hA2, CUDA_R_16F, n,
            hA3, CUDA_R_16F, n,
            &one,
            A,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    // /* Smoothing function */
    // for (int i = 0; i < 3; i++) {
    //     // A2 = A * A
    //     convert_float_to_half4(A, hA, nn);
    //     CHECK_CUBLAS(cublasGemmEx(
    //         cublasH,
    //         CUBLAS_OP_N, CUBLAS_OP_N,
    //         n, n, n,
    //         &one,
    //         hA, CUDA_R_16F, n,
    //         hA, CUDA_R_16F, n,
    //         &zero,
    //         A2,    CUDA_R_32F, n,
    //         CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    //     // A3 = A2 * A
    //     convert_float_to_half4(A2, hA2, nn);
    //     CHECK_CUBLAS(cublasGemmEx(
    //         cublasH,
    //         CUBLAS_OP_N, CUBLAS_OP_N,
    //         n, n, n,
    //         &one,
    //         hA, CUDA_R_16F, n,
    //         hA2, CUDA_R_16F, n,
    //         &zero,
    //         A3,    CUDA_R_32F, n,
    //         CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    //     /* Symmetrize A3 */
    //     symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    //     /* Compute A = 1.5 * A - 0.5 * A3 */
    //     // A = 1.5 * A
    //     CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
    //     // A = -0.5 * A3 + A
    //     CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );

    //     /* Symmetrize A */
    //     symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    // }

    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    convert_double_to_float(mat, A2, nn);
    convert_float_to_half4(A, hA, nn);
    convert_float_to_half4(A2, hA2, nn);
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        hA2, CUDA_R_16F, n,
        hA, CUDA_R_16F, n,
        &zero,
        A3,    CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    if (float_workspace == nullptr) {
        CHECK_CUDA( cudaFree(A) );
        CHECK_CUDA( cudaFree(A2) );
        CHECK_CUDA( cudaFree(A3) );
    }
    if (half_workspace == nullptr) {
        CHECK_CUDA( cudaFree(hA) );
        CHECK_CUDA( cudaFree(hA2) );
        CHECK_CUDA( cudaFree(hA3) );
    }
}

void composite_FP16_auto_scale(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int n,
    float* float_workspace,
    __half* half_workspace
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

    // project the matrix using the composite_FP16 function
    composite_FP16(cublasH, mat, n, float_workspace, half_workspace);

    // rescale the result back to the original scale
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale,  mat, 1) );
}

void composite_FP16_auto_scale_deflate(
    cublasHandle_t cublasH,
    cusolverDnHandle_t solverH,
    double* mat,
    const int n,
    float* float_workspace,
    __half* half_workspace,
    const size_t k,
    const int maxiter,
    const double tol,
    const bool verbose
) {
    size_t nn = n * n;
    double minus_one = -1.0;

    // TODO: use a workspace for the eigenvalues and eigenvectors
    double *eigenvalues_max, *eigenvectors_max;
    double *eigenvalues_min, *eigenvectors_min;
    CHECK_CUDA( cudaMalloc(&eigenvalues_max,      k * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&eigenvectors_max, n * k * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&eigenvalues_min,      k * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&eigenvectors_min, n * k * sizeof(double)) );

    /* Step 1: compute the largest eigenpairs of the matrix */
    lobpcg(
        cublasH, solverH, mat, eigenvectors_max, eigenvalues_max, n, k, false, maxiter, tol, verbose
    );
    // negate the eigenvalues
    CHECK_CUBLAS(cublasDscal(cublasH, k, &minus_one, eigenvalues_max, 1));

    /* Step 2: remove the largest eigenvalues from the matrix */
    cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE);
    for (int i = 0; i < k; i++) {
        // X <- X - \lambda_i * v_i v_i^T
        double *v_i = eigenvectors_max + i * n;
        double *m_lambda_i = eigenvalues_max + i;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, m_lambda_i, v_i, 1, v_i, 1, mat, n) );
    }
    cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST);

    /* Step 1bis: compute the lowest eigenpairs of the matrix */
    // change the matrix sign to reuse LOBPCG code
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &minus_one, mat, 1));
    lobpcg(
        cublasH, solverH, mat, eigenvectors_min, eigenvalues_min, n, k, false, maxiter, tol, verbose
    );
    // note: the min eigenvalues are already negated since we used -A

    /* Step 2bis: remove the lowest eigenvalues from the matrix */
    // restore the matrix sign
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &minus_one, mat, 1));
    // remove them
    cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE);
    for (int i = 0; i < k; i++) {
        // X <- X - \lambda_i * v_i v_i^T
        double *v_i = eigenvectors_min + i * n;
        double *m_lambda_i = eigenvalues_min + i;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, m_lambda_i, v_i, 1, v_i, 1, mat, n) );
    }
    cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST);

    /* Step 3: scale the deflated matrix */
    double lo, up;
    approximate_two_norm(
        cublasH, solverH, mat, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? 1.1 * up + 1e-5 : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, mat, 1) );

    /* Step 4: project the matrix using the composite_FP16 function */
    composite_FP16(cublasH, mat, n, float_workspace, half_workspace);

    /* Step 5: rescale the matrix back and add the deflated eigenvalues back */
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, mat, 1) );

    // add only positive eigenvalues back
    CHECK_CUBLAS( cublasDscal(cublasH, k, &minus_one, eigenvalues_max, 1) );
    CHECK_CUBLAS( cublasDscal(cublasH, k, &minus_one, eigenvalues_min, 1) );
    max_dense_vector_zero(eigenvalues_max, k);
    max_dense_vector_zero(eigenvalues_min, k);

    cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_DEVICE);
    for (int i = 0; i < k; i++) {
        // X <- X + \lambda_i * v_i v_i^T
        double *m_lambda_i = eigenvalues_max + i;
        double *v_i = eigenvectors_max + i * n;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, m_lambda_i, v_i, 1, v_i, 1, mat, n) );
    }
    for (int i = 0; i < k; i++) {
        // X <- X + \lambda_i * v_i v_i^T
        double *m_lambda_i = eigenvalues_min + i;
        double *v_i = eigenvectors_min + i * n;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, m_lambda_i, v_i, 1, v_i, 1, mat, n) );
    }
    cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST);

    /* Free device memory */
    CHECK_CUDA( cudaFree(eigenvalues_max) );
    CHECK_CUDA( cudaFree(eigenvectors_max) );
    CHECK_CUDA( cudaFree(eigenvalues_min) );
    CHECK_CUDA( cudaFree(eigenvectors_min) );
}