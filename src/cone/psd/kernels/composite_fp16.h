/*

    cone/psd/kernels/composite_fp16.h

    The FP16 composite polynomial filter of the warm-up projection (plan Section 10.6; arXiv composite filter paper,
    Table 2): on the input scaled into [-1, 1] it evaluates seven odd cubic-quintic steps X_t = f_t(X_{t-1} / d_t),
    f_t(X) = a_t X + b_t X^3 + c_t X^5, with the damping d_t = 1.01 applied to the INPUT of every step as coded
    (the paper describes the damping on the output: B11), then P = (X_0 (I + X_7)) / 2, symmetrized. The three
    GEMMs of each step run on FP16 Tensor Cores with FP32 accumulation (cublasGemmEx, CUBLAS_COMPUTE_32F_FAST_16F
    set in every call; the handle keeps CUBLAS_DEFAULT_MATH: B1); the updates run in FP32.

    Coefficients (kCompositeFp16): the refined half-precision coefficients of the paper's Table 2 ("polar express
    with refinement"), identical to the frozen submodule b566590 (Remez initialization, gradient refinement); the
    other candidate tables of the submodule's source are not kept (plan Section 10.6, item 2).

*/

#ifndef CUADMM_CONE_PSD_KERNELS_COMPOSITE_FP16_H
#define CUADMM_CONE_PSD_KERNELS_COMPOSITE_FP16_H

#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace cuadmm
{
namespace cone
{
namespace psd
{

constexpr int kCompositeSteps = 7;
constexpr float kCompositeFp16[kCompositeSteps][3] = {
    {8.2885332412f, -22.5927099246f, 15.8201383114f},
    {4.1666196466f, -2.9679004036f, 0.5307623217f},
    {4.0611848147f, -2.9698947955f, 0.5492133813f},
    {3.6678301399f, -2.7561018955f, 0.5421513305f},
    {2.7632556383f, -2.0607754898f, 0.4695405857f},
    {2.0527445797f, -1.4345145882f, 0.4070669182f},
    {1.8804816691f, -1.2583997294f, 0.3779501813f}};
constexpr float kCompositeDamping = 1.01f;

/// mat (n x n, FP64, column-major, scaled so that its spectrum lies in [-1, 1]) <- the filtered projection; the
/// workspaces hold 3 n^2 floats and 3 n^2 halves (n^2 rounded up to a multiple of 4), allocated per call when null.
void composite_fp16(cublasHandle_t cublasH, double *mat, int n, float *float_workspace = nullptr, __half *half_workspace = nullptr);

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_KERNELS_COMPOSITE_FP16_H
