/*

    core/gpu_context.cu

*/

#include "core/gpu_context.h"

#include "core/check.h"

namespace cuadmm
{
namespace core
{

GpuContext::GpuContext(int device)
{
    int count = 0;
    CUADMM_CHECK_CUDA(cudaGetDeviceCount(&count));
    if (device < 0 || device >= count)
        throw Error("gpu.device " + std::to_string(device) + " does not exist (" + std::to_string(count) + " device(s))", __FILE__, __LINE__);
    CUADMM_CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUADMM_CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    info_.device = device;
    info_.name = prop.name;
    info_.multiprocessors = prop.multiProcessorCount;
    info_.compute_major = prop.major;
    info_.compute_minor = prop.minor;
    info_.total_memory = prop.totalGlobalMem;
    CUADMM_CHECK_CUDA(cudaDriverGetVersion(&info_.driver_version));
    CUADMM_CHECK_CUDA(cudaRuntimeGetVersion(&info_.runtime_version));

    // the main stream role runs on the legacy default stream (as today); the math mode is explicit (B1)
    main_.stream = nullptr;
    CUADMM_CHECK_CUBLAS(cublasCreate_v2(&main_.cublas));
    CUADMM_CHECK_CUBLAS(cublasSetMathMode(main_.cublas, CUBLAS_DEFAULT_MATH));
    CUADMM_CHECK_CUSOLVER(cusolverDnCreate(&main_.cusolver));
    CUADMM_CHECK_CUSPARSE(cusparseCreate(&main_.cusparse));
}

GpuContext::~GpuContext()
{
    if (main_.cusparse)
        CUADMM_CHECK_CUSPARSE_NOTHROW(cusparseDestroy(main_.cusparse));
    if (main_.cusolver)
        CUADMM_CHECK_CUSOLVER_NOTHROW(cusolverDnDestroy(main_.cusolver));
    if (main_.cublas)
        CUADMM_CHECK_CUBLAS_NOTHROW(cublasDestroy_v2(main_.cublas));
}

std::size_t GpuContext::free_memory() const
{
    std::size_t free_mem = 0, total_mem = 0;
    CUADMM_CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

std::size_t GpuContext::used_memory() const
{
    std::size_t free_mem = 0, total_mem = 0;
    CUADMM_CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    return total_mem - free_mem;
}

void GpuContext::synchronize() const
{
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace core
} // namespace cuadmm
