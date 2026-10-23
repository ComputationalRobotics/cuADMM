/*

    cone/psd/large/projectors/composite_filter.cu

    SDPSolver::large_composite_project of Milestone 1 (with the deterministic Lanczos start of Phase 6a) on the
    tier's regions: the FP32/FP16 buffers and the Lanczos workspace are carved from W, the input is not copied
    (a rejected block is unpacked again), the result is packed from M.

*/

#include "cone/psd/large/projectors/composite_filter.h"

#include <cmath>
#include <cstddef>

#include <cuda_fp16.h>

#include "cone/psd/kernels/composite_fp16.h"
#include "cone/psd/kernels/lanczos.h"
#include "cone/psd/large/large_tier.h"
#include "core/check.h"
#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

CompositeFilter::CompositeFilter(const CompositeFilterConfig &config, LargeTier &tier) : config_(config), tier_(tier) {}

unsigned long long CompositeFilter::block_seed(unsigned long long seed, int idx)
{
    return seed ^ (0x9E3779B97F4A7C15ULL * static_cast<unsigned long long>(idx + 1));
}

bool CompositeFilter::project(const LargeBlockJob &job, const double *z, double *s)
{
    SDPSolver &e = tier_.engine();
    const int n = job.n;
    const int nn = n * n; // fits in an int since n <= 46340
    const core::ExecContext ctx = tier_.ctx();
    cublasHandle_t cublasH = ctx.cublas;
    double *A = tier_.M();

    // the buffers in W: FP32 (3 stride floats), FP16 (3 stride halves), then the Lanczos workspace
    const std::size_t stride = static_cast<std::size_t>(nn) % 4 == 0 ? nn : nn + (4 - static_cast<std::size_t>(nn) % 4);
    unsigned char *base = reinterpret_cast<unsigned char *>(tier_.W());
    float *fws = reinterpret_cast<float *>(base);
    __half *hws = reinterpret_cast<__half *>(base + 3 * stride * sizeof(float));
    const std::size_t front = 3 * stride * sizeof(float) + 3 * stride * sizeof(__half);
    double *lanczos_ws = reinterpret_cast<double *>(base + (front + 255) / 256 * 256);

    double norm_in;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A, 1, &norm_in));

    // scale the spectrum into [-1, 1]: the Lanczos estimate 'up' is not a certified upper bound on
    // ||A||_2, and the filter overflows if ||A||/scale > ~1.02 (FP16), hence the margin
    double lo, up;
    approximate_two_norm(cublasH, ctx.cusolver, A, n, &lo, &up, block_seed(config_.seed, job.state.idx), config_.lanczos_steps, 1e-10, lanczos_ws);
    const double scale = up > 0.0 ? config_.margin * up : 1.0;
    const double inv_scale = 1.0 / scale;
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &inv_scale, A, 1));

    composite_fp16(cublasH, A, n, fws, hws);

    // rescale the result back to the original scale
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &scale, A, 1));

    // reject a non-finite result (overflow of the filter, or non-finite input) and, B22, a result whose norm exceeds
    // that of the input beyond the slack: the projection is nonexpansive with Pi(0) = 0, so ||P||_F <= ||Z||_F, and a
    // finite but wrong result from an underestimated scale fails this free necessary condition
    double norm;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A, 1, &norm));
    if (!std::isfinite(norm) || !std::isfinite(norm_in) || norm > (1.0 + config_.norm_slack) * norm_in)
    {
        tier_.unpack(job.state.idx, z); // the input again, for the exact fallback
        e.composite_fallbacks++;
        return false;
    }

    tier_.pack(job.state.idx, A, s, nullptr);
    return true;
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
