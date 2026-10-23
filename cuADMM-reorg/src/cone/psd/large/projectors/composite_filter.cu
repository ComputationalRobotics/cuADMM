/*

    cone/psd/large/projectors/composite_filter.cu

    SDPSolver::large_composite_project of Milestone 1 (with the deterministic Lanczos start of Phase 6a step 1),
    moved without a change of its calls; the macros COMPOSITE_FP16_SCALE_MARGIN and COMPOSITE_NORM_SLACK and the
    Lanczos step count are the configuration now.

*/

#include "cone/psd/large/projectors/composite_filter.h"

#include <cmath>

#include "cone/psd/kernels/composite_fp16.h"
#include "cone/psd/kernels/lanczos.h"
#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

CompositeFilter::CompositeFilter(const CompositeFilterConfig &config, SDPSolver &engine) : config_(config), engine_(engine) {}

unsigned long long CompositeFilter::block_seed(unsigned long long seed, int idx)
{
    return seed ^ (0x9E3779B97F4A7C15ULL * static_cast<unsigned long long>(idx + 1));
}

bool CompositeFilter::project(const LargeBlockJob &job)
{
    SDPSolver &e = engine_;
    const int n = job.n;
    const int mat_offset = job.mat_offset;
    const int nn = n * n; // fits in an int since n <= 46340
    cublasHandle_t cublasH = e.cublasH_composite_proj.cublas_handle;
    double *A = e.large_mat.vals + mat_offset;
    double *A_copy = e.large_mat_tmp.vals + mat_offset; // large_mat_tmp is free here

    // keep a copy of the input to restore it if the polynomial filter overflows or fails the norm test
    CHECK_CUDA(cudaMemcpy(A_copy, A, sizeof(double) * nn, D2D));
    double norm_in;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_copy, 1, &norm_in));

    // scale the spectrum into [-1, 1]: the Lanczos estimate 'up' is not a certified upper bound on
    // ||A||_2, and the filter overflows if ||A||/scale > ~1.02 (FP16), hence the margin
    double lo, up;
    approximate_two_norm(cublasH, e.cusolverH_composite_proj.cusolver_dn_handle, A, n, &lo, &up,
                         block_seed(config_.seed, job.state.idx), config_.lanczos_steps);
    const double scale = up > 0.0 ? config_.margin * up : 1.0;
    const double inv_scale = 1.0 / scale;
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &inv_scale, A, 1));

    composite_fp16(cublasH, A, n, e.float_proj_workspace.vals, e.half_proj_workspace.vals);

    // rescale the result back to the original scale
    CHECK_CUBLAS(cublasDscal(cublasH, nn, &scale, A, 1));

    // reject a non-finite result (overflow of the filter, or non-finite input) and, B22, a result whose norm exceeds
    // that of the input beyond the slack: the projection is nonexpansive with Pi(0) = 0, so ||P||_F <= ||Z||_F, and a
    // finite but wrong result from an underestimated scale fails this free necessary condition
    double norm;
    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A, 1, &norm));
    if (!std::isfinite(norm) || !std::isfinite(norm_in) || norm > (1.0 + config_.norm_slack) * norm_in)
    {
        CHECK_CUDA(cudaMemcpy(A, A_copy, sizeof(double) * nn, D2D));
        e.composite_fallbacks++;
        return false;
    }

    // copy large_mat to large_mat_P
    CHECK_CUDA(cudaMemcpy(e.large_mat_P.vals + mat_offset, A, sizeof(double) * nn, D2D));

    return true;
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
