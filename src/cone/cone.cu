/*

    cone/cone.cu

    The default primal projection by Moreau: Pi_K(z) = z + Pi_{K*}(-z), on the cone's blocks.

*/

#include "cone/cone.h"

#include <stdexcept>

#include "core/check.h"

namespace cuadmm
{
namespace cone
{

namespace
{
__global__ void negate_kernel(const double *z, double *out, std::int64_t n)
{
    const std::int64_t i = blockIdx.x * (std::int64_t)blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = -z[i];
}
__global__ void add_kernel(const double *z, double *x, std::int64_t n)
{
    const std::int64_t i = blockIdx.x * (std::int64_t)blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += z[i];
}
} // namespace

void Cone::project_primal(core::DeviceSpan<const double> z, core::DeviceSpan<double> x, core::ExecContext &ctx)
{
    // Moreau on the whole vector (the cone writes only its own entries of x; the others are restored below)
    if (z.size != x.size)
        throw std::invalid_argument("Cone::project_primal: length mismatch");
    core::DeviceBuffer<double> neg(z.size);
    const int block = 256;
    const std::int64_t grid = (static_cast<std::int64_t>(z.size) + block - 1) / block;
    negate_kernel<<<static_cast<unsigned>(grid), block, 0, ctx.stream>>>(z.data, neg.data(), z.size);
    CUADMM_CHECK_LAUNCH();
    // x_b = Pi_{K*}(-z)_b on the cone's blocks
    project_dual(neg.span().as_const(), x, ctx);
    // x_b += z_b on the cone's blocks only
    for (const BlockSlice &b : blocks())
    {
        const std::int64_t g = (b.length + block - 1) / block;
        add_kernel<<<static_cast<unsigned>(g), block, 0, ctx.stream>>>(z.data + b.offset, x.data + b.offset, b.length);
        CUADMM_CHECK_LAUNCH();
    }
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
}

} // namespace cone
} // namespace cuadmm
