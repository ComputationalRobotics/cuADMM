/*

    cone/psd/kernels/composite_fp32.cu

    composite_FP32 of the psd_projection repository (b566590), moved in unchanged apart from the checks, the
    namespace, the coefficient table taken from the header and the removal of the commented-out variants and of the
    (unused, zero-step) Newton-Schulz smoothing loop.

*/

#include "cone/psd/kernels/composite_fp32.h"

#include <cuda_runtime.h>

#include "cone/psd/kernels/precision.h"
#include "core/check.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

void composite_fp32(cublasHandle_t cublasH, double *mat, const int n, float *float_workspace)
{
    const int nn = n * n;

    // offsets in size_t: 2 * stride overflows an int for n >= 32768
    const size_t stride = nn % 4 == 0 ? (size_t)nn : (size_t)nn + (4 - nn % 4);
    float *A, *A2, *A3;
    if (float_workspace == nullptr)
    {
        CUADMM_CHECK_CUDA(cudaMalloc(&A, nn * sizeof(float)));
        CUADMM_CHECK_CUDA(cudaMalloc(&A2, nn * sizeof(float)));
        CUADMM_CHECK_CUDA(cudaMalloc(&A3, nn * sizeof(float)));
    }
    else
    {
        A = float_workspace;
        A2 = float_workspace + stride;
        A3 = float_workspace + 2 * stride;
    }

    const float half = 0.5f;
    const float one = 1.0f;
    const float zero = 0.0f;

    convert_double_to_float(mat, A, nn);

    /* Approximation of the step function: ten odd quintic steps, 3 GEMMs each */
    for (int i = 0; i < kCompositeFp32Steps; i++)
    {
        float a = kCompositeFp32[i][0];
        float b = kCompositeFp32[i][1];
        float c = kCompositeFp32[i][2];
        if (i < kCompositeFp32DampedSteps)
        {
            const float d = kCompositeFp32Damping;
            a /= d;
            b /= d * d * d;
            c /= d * d * d * d * d;
        }

        // A2 = A * A
        CUADMM_CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n));
        // A3 = A2 * A
        CUADMM_CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n));
        // A = a * A + b * A3
        CUADMM_CHECK_CUBLAS(cublasSscal(cublasH, nn, &a, A, 1));
        CUADMM_CHECK_CUBLAS(cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1));
        // A = c * A3 * A2 + A
        CUADMM_CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &c, A3, n, A2, n, &one, A, n));
        // symmetrize (A2 as the workspace)
        symmetrize_float(cublasH, A, n, A2);
    }

    // A = (I + A) / 2
    add_identity(cublasH, A, n);
    CUADMM_CHECK_CUBLAS(cublasSscal(cublasH, nn, &half, A, 1));

    /* Multiply the original matrix by A: W = A_origin * A (the 31st GEMM) */
    convert_double_to_float(mat, A2, nn);
    CUADMM_CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n));
    symmetrize_float(cublasH, A3, n, A2);

    convert_float_to_double(A3, mat, nn);

    if (float_workspace == nullptr)
    {
        CUADMM_CHECK_CUDA(cudaFree(A));
        CUADMM_CHECK_CUDA(cudaFree(A2));
        CUADMM_CHECK_CUDA(cudaFree(A3));
    }
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
