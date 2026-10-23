/*

    cone/psd/svec_pack.h

    svec <-> dense conversions of the PSD blocks on the engine's category maps (plan Section 10; the kernels of
    kernels/vec_mat_conversion.cu of main restricted to the PSD entries: the free and nonnegative entries belong to
    their own cones). Upper triangle, column by column, off-diagonal entries times sqrt 2 in the svec.

*/

#ifndef CUADMM_CONE_PSD_SVEC_PACK_H
#define CUADMM_CONE_PSD_SVEC_PACK_H

#include <cuda_runtime_api.h>

namespace cuadmm
{
namespace cone
{

/// Dense matrices of the PSD blocks from the svec vector z (entries of the other cones ignored).
void psd_unpack(const double *z, double *large, double *medium, double *small, const int *map_B, const int *map_M1, const int *map_M2,
                int vec_len, cudaStream_t stream = nullptr);
/// The svec entries of the PSD blocks of s from the dense result matrices (the other entries of s untouched).
void psd_pack(double *s, const double *large, const double *medium, const double *small, const int *map_B, const int *map_M1,
              const int *map_M2, int vec_len, cudaStream_t stream = nullptr);

} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_SVEC_PACK_H
