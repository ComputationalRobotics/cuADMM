/*

    cone/psd/eig_info.cu

*/

#include "cone/psd/eig_info.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/check.h"

namespace cuadmm
{
namespace cone
{
namespace psd
{

int check_eig_info(const int *d_info, const int count, const char *what, const int jacobi_n)
{
    if (count == 0)
        return 0;
    std::vector<int> info(count);
    CUADMM_CHECK_CUDA(cudaMemcpy(info.data(), d_info, sizeof(int) * count, cudaMemcpyDeviceToHost));
    int not_converged = 0;
    for (int l = 0; l < count; l++)
    {
        if (info[l] == 0)
            continue;
        if (jacobi_n >= 0 && info[l] == jacobi_n + 1)
        {
            not_converged++;
            continue;
        }
        throw std::runtime_error(std::string(what) + " failed for matrix " + std::to_string(l) + ": info = " + std::to_string(info[l]));
    }
    return not_converged;
}

} // namespace psd
} // namespace cone
} // namespace cuadmm
