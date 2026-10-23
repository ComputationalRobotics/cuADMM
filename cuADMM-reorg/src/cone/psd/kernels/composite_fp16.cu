/*

    cone/psd/kernels/composite_fp16.cu

    composite_FP16 of the frozen psd_projection submodule (b566590), moved in unchanged apart from the checks, the
    namespace, the coefficient table taken from the header and the removal of the commented-out variants.

*/

#include "cone/psd/kernels/composite_fp16.h"

#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "cone/psd/kernels/precision.h"
#include "core/check.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

void composite_fp16(
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
        CUADMM_CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
        CUADMM_CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
        CUADMM_CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );
    } else {
        A = float_workspace;
        A2 = float_workspace + stride;
        A3 = float_workspace + 2 * stride;
    }

    __half *hA, *hA2, *hA3;
    if (half_workspace == nullptr) {
        CUADMM_CHECK_CUDA(cudaMalloc(&hA,  nn * sizeof(__half)));
        CUADMM_CHECK_CUDA(cudaMalloc(&hA2, nn * sizeof(__half)));
        CUADMM_CHECK_CUDA(cudaMalloc(&hA3, nn * sizeof(__half)));
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

    /* Coefficients: kCompositeFp16 (the table of the header) */
    const float scale_factor = kCompositeDamping; // B11: the damping divides the input of every step, as coded

    /* Approximation of the step function */
    for (int i = 0; i < kCompositeSteps; i++) {
        float a = kCompositeFp16[i][0];
        float b = kCompositeFp16[i][1];
        float c = kCompositeFp16[i][2];

        a /= scale_factor;
        b /= scale_factor * scale_factor * scale_factor;
        c /= scale_factor * scale_factor * scale_factor * scale_factor * scale_factor;

        /* Compute the powers of A*/
        // A2 = A * A
        convert_float_to_half4(A, hA, nn);
        CUADMM_CHECK_CUBLAS(cublasGemmEx(
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
        CUADMM_CHECK_CUBLAS(cublasGemmEx(
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
        CUADMM_CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CUADMM_CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        /* Symmetrize A3, A5 */
        // symmetrize_float(cublasH, A3, n, A2); // we use A2 as a workspace
        // symmetrize_float(cublasH, A5, n, A2); // we use A2 as a workspace

        // A = c * A3 * A2 + A
        convert_float_to_half4(A3, hA3, nn);
        CUADMM_CHECK_CUBLAS(cublasGemmEx(
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
        symmetrize_float(cublasH, A, n, A2); // we use A2 as a workspace
    }

    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CUADMM_CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    convert_double_to_float(mat, A2, nn);
    convert_float_to_half4(A, hA, nn);
    convert_float_to_half4(A2, hA2, nn);
    CUADMM_CHECK_CUBLAS(cublasGemmEx(
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
    symmetrize_float(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    if (float_workspace == nullptr) {
        CUADMM_CHECK_CUDA( cudaFree(A) );
        CUADMM_CHECK_CUDA( cudaFree(A2) );
        CUADMM_CHECK_CUDA( cudaFree(A3) );
    }
    if (half_workspace == nullptr) {
        CUADMM_CHECK_CUDA( cudaFree(hA) );
        CUADMM_CHECK_CUDA( cudaFree(hA2) );
        CUADMM_CHECK_CUDA( cudaFree(hA3) );
    }
}


} // namespace psd
} // namespace cone
} // namespace cuadmm
