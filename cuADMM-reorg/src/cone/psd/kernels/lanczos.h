/*

    cone/psd/kernels/lanczos.h

    The Lanczos estimate of ||A||_2 of the composite filter (plan Section 10.6): max_iter steps on A^T A from a
    deterministic start vector (B8), lo = sqrt(theta), up = sqrt(theta + ||A^T A y - theta y||) for the largest Ritz
    pair (not a certified bound: the caller applies the safety margin). Moved in from the frozen psd_projection
    submodule (b566590); the WIP eigenpair routines of the submodule stay there (plan Section 10.6, item 1).

*/

#ifndef CUADMM_CONE_PSD_KERNELS_LANCZOS_H
#define CUADMM_CONE_PSD_KERNELS_LANCZOS_H

#include <cstddef>

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace cuadmm
{
namespace cone
{
namespace psd
{

void approximate_two_norm(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, const double *A, std::size_t n, double *lo, double *up,
                          unsigned long long seed, std::size_t max_iter = 20, double tol = 1e-10,
                          double *workspace = nullptr); // lanczos_workspace_doubles(n, max_iter) doubles; null: allocated for this call

/// The workspace of approximate_two_norm(n, max_iter) in doubles.
std::size_t lanczos_workspace_doubles(cusolverDnHandle_t cusolverH, std::size_t n, std::size_t max_iter);

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_KERNELS_LANCZOS_H
