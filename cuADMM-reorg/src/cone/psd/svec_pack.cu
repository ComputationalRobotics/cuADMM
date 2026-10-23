/*

    cone/psd/svec_pack.cu

*/

#include "cone/psd/svec_pack.h"

#include "core/check.h"
#include "cuadmm_legacy/kernels.h"      // SQRT2, SQRT2INV
#include "cuadmm_legacy/matrix_sizes.h" // MatrixSizeCategory

namespace cuadmm
{
namespace cone
{

namespace
{
__global__ void psd_unpack_kernel(const double *z, double *large, double *medium, double *small, const int *map_B, const int *map_M1,
                                  const int *map_M2, int vec_len)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= vec_len)
        return;
    const int b = map_B[idx], m1 = map_M1[idx], m2 = map_M2[idx];
    const bool if_diag = (m1 == m2);
    if (b == MatrixSizeCategory::LARGE)
    {
        large[m1] = z[idx] * (SQRT2INV + int(if_diag) * (1 - SQRT2INV));
        large[m2] = large[m1];
    }
    else if (b == MatrixSizeCategory::MEDIUM)
    {
        medium[m1] = z[idx] * (SQRT2INV + int(if_diag) * (1 - SQRT2INV));
        medium[m2] = medium[m1];
    }
    else if (b == MatrixSizeCategory::SMALL)
    {
        small[m1] = z[idx] * (SQRT2INV + int(if_diag) * (1 - SQRT2INV));
        small[m2] = small[m1];
    }
}

__global__ void psd_pack_kernel(double *s, const double *large, const double *medium, const double *small, const int *map_B,
                                const int *map_M1, const int *map_M2, int vec_len)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= vec_len)
        return;
    const int b = map_B[idx], m1 = map_M1[idx], m2 = map_M2[idx];
    const bool if_diag = (m1 == m2);
    if (b == MatrixSizeCategory::LARGE)
        s[idx] = large[m1] * (SQRT2 + int(if_diag) * (1 - SQRT2));
    else if (b == MatrixSizeCategory::MEDIUM)
        s[idx] = medium[m1] * (SQRT2 + int(if_diag) * (1 - SQRT2));
    else if (b == MatrixSizeCategory::SMALL)
        s[idx] = small[m1] * (SQRT2 + int(if_diag) * (1 - SQRT2));
}
} // namespace

void psd_unpack(const double *z, double *large, double *medium, double *small, const int *map_B, const int *map_M1, const int *map_M2,
                int vec_len, cudaStream_t stream)
{
    if (vec_len <= 0)
        return;
    const int block = 1024, grid = (vec_len + block - 1) / block;
    psd_unpack_kernel<<<grid, block, 0, stream>>>(z, large, medium, small, map_B, map_M1, map_M2, vec_len);
    CUADMM_CHECK_LAUNCH();
}

void psd_pack(double *s, const double *large, const double *medium, const double *small, const int *map_B, const int *map_M1,
              const int *map_M2, int vec_len, cudaStream_t stream)
{
    if (vec_len <= 0)
        return;
    const int block = 1024, grid = (vec_len + block - 1) / block;
    psd_pack_kernel<<<grid, block, 0, stream>>>(s, large, medium, small, map_B, map_M1, map_M2, vec_len);
    CUADMM_CHECK_LAUNCH();
}

} // namespace cone
} // namespace cuadmm
