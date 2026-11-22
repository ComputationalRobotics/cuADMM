/*

    cone/psd/kernels/composite_fp32.h

    The FP32 composite polynomial filter (siopt-rebuttal branch): composite_FP32 of the psd_projection repository
    (b566590), the ten odd quintic steps of the paper's Table 1 ("polar express with refinement", 31 GEMMs) on the
    input scaled into [-1, 1], with the damping d = 1.001 applied to the INPUT of the first eight steps as coded
    (the same convention as composite_fp16: B11), then P = (X_0 (I + X_10)) / 2, symmetrized. Every GEMM is a
    native cublasSgemm under the handle's default math mode (no TF32, no BF16x9 emulation: Hopper has none), so the
    arithmetic is the "native FP32" of the paper's H100 tables.

*/

#ifndef CUADMM_CONE_PSD_KERNELS_COMPOSITE_FP32_H
#define CUADMM_CONE_PSD_KERNELS_COMPOSITE_FP32_H

#include <cublas_v2.h>

namespace cuadmm
{
namespace cone
{
namespace psd
{

constexpr int kCompositeFp32Steps = 10;
constexpr int kCompositeFp32DampedSteps = 8; // the damping applies to the first eight steps (psd_projection b566590)
constexpr float kCompositeFp32[kCompositeFp32Steps][3] = {
    {8.3119043343f, -23.0739115930f, 16.4664144722f},
    {4.1439360087f, -2.9176674704f, 0.5246212487f},
    {4.0257813209f, -2.9025002398f, 0.5334261214f},
    {3.5118574347f, -2.5740236523f, 0.5050097282f},
    {2.4398158400f, -1.7586675341f, 0.4191290613f},
    {1.9779835097f, -1.3337358510f, 0.3772169049f},
    {1.9559726949f, -1.3091355170f, 0.3746734515f},
    {1.9282822454f, -1.2823649693f, 0.3704626545f},
    {1.9220135179f, -1.2812524618f, 0.3707011753f},
    {1.8942192942f, -1.2613293407f, 0.3676616051f}};
constexpr float kCompositeFp32Damping = 1.001f;

/// mat (n x n, FP64, column-major, scaled so that its spectrum lies in [-1, 1]) <- the filtered projection; the
/// workspace holds 3 n^2 floats (n^2 rounded up to a multiple of 4, as the FP16 filter's), allocated per call when
/// null.
void composite_fp32(cublasHandle_t cublasH, double *mat, int n, float *float_workspace = nullptr);

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_KERNELS_COMPOSITE_FP32_H
