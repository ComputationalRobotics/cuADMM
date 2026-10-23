/*

    cone/psd/kernels/precision.h

    Helpers of the large-block projectors (plan Section 10.6, B10: one copy of each): FP64 <-> FP32 <-> FP16
    conversions, symmetrization in FP32, identity addition, column reversal and extraction, and the deterministic
    start vector of the Lanczos estimate and of LOBPCG (B8: a counter-based generator, seeded by the caller).
    Moved in from the psd_projection repository (b566590) under the CUADMM checks.

*/

#ifndef CUADMM_CONE_PSD_KERNELS_PRECISION_H
#define CUADMM_CONE_PSD_KERNELS_PRECISION_H

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace cuadmm
{
namespace cone
{
namespace psd
{

/// M <- (M + M^T) / 2 for an n x n column-major FP32 matrix; workspace: n x n floats.
void symmetrize_float(cublasHandle_t cublasH, float *M, int n, float *workspace);
void convert_double_to_float(const double *d_in, float *d_out, int n, int threads_per_block = 1024);
void convert_float_to_double(const float *d_in, double *d_out, int n, int threads_per_block = 1024);
/// mat <- mat + I (FP32, n x n column-major).
void add_identity(cublasHandle_t cublasH, float *mat, int n);
/// FP32 -> FP16 by groups of four (N rounded up to a multiple of 4: the buffers are allocated with that slack).
void convert_float_to_half4(const float *dA, __half *dB, std::size_t N);
/// out[i] = in[m - 1 - i].
void reverse_vector(const double *in, double *out, int m);
/// The m columns of an n x m column-major matrix in reverse order.
void reverse_columns(const double *in, double *out, int n, int m);
/// The first m columns of an n x k matrix (k >= m), column-major.
void extract_columns(const double *in, double *out, int n, int m);
/// v[i] = a value in (0, 1] from a counter-based generator (splitmix64 of seed and i): the same vector for the same
/// seed on every GPU and run (plan B8).
void fill_deterministic(double *v, int n, unsigned long long seed, int threads_per_block = 1024);

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_KERNELS_PRECISION_H
