/*

    cone/psd/eig_info.h

    check_eig_info: copies cuSOLVER info values to the host and throws cuadmm::Error-compatible std::runtime_error
    when an eigendecomposition failed (check_eig_info of main, shared by the tiers of the PSD cone). For the
    batched Jacobi EVD, info = n + 1 means "tolerance not reached" and is counted, not thrown (B22 note of the
    plan: the results are still accurate to machine precision).

*/

#ifndef CUADMM_CONE_PSD_EIG_INFO_H
#define CUADMM_CONE_PSD_EIG_INFO_H

namespace cuadmm
{
namespace cone
{
namespace psd
{

/// Returns the number of Jacobi decompositions that did not reach their tolerance (jacobi_n >= 0), 0 otherwise.
int check_eig_info(const int *d_info, int count, const char *what, int jacobi_n = -1);

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_EIG_INFO_H
