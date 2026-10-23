/*

    core/gpu_context.h

    GpuContext (plan Section 14): the device, its properties, and the library handles of the main stream role with
    their math modes set explicitly. ExecContext bundles the stream and handles a component runs on (the stream pool
    and the workspace arena come with Phase 6b; until then the main context runs on the legacy default stream, as
    today, and the ported engine keeps its own medium-tier streams).

*/

#ifndef CUADMM_CORE_GPU_CONTEXT_H
#define CUADMM_CORE_GPU_CONTEXT_H

#include <cstddef>
#include <memory>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cusparse.h>

namespace cuadmm
{
namespace core
{

/// @brief What a component needs to run: a stream and the handles bound to it.
struct ExecContext
{
    cudaStream_t stream = nullptr; // nullptr: the legacy default stream
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    cusparseHandle_t cusparse = nullptr;
};

struct DeviceInfo
{
    int device = 0;
    std::string name;
    int multiprocessors = 0;
    int compute_major = 0, compute_minor = 0;
    std::size_t total_memory = 0;
    int driver_version = 0, runtime_version = 0;
};

class GpuContext
{
public:
    explicit GpuContext(int device = 0); // cudaSetDevice, properties, the main handles (CUBLAS_DEFAULT_MATH)
    ~GpuContext();
    GpuContext(const GpuContext &) = delete;
    GpuContext &operator=(const GpuContext &) = delete;

    const DeviceInfo &info() const { return info_; }
    int device() const { return info_.device; }
    ExecContext main() const { return main_; }
    cublasHandle_t cublas() const { return main_.cublas; }
    cusolverDnHandle_t cusolver() const { return main_.cusolver; }
    cusparseHandle_t cusparse() const { return main_.cusparse; }

    std::size_t free_memory() const;  // bytes free on the device now
    std::size_t used_memory() const;  // total - free
    void synchronize() const;

private:
    DeviceInfo info_;
    ExecContext main_;
};

} // namespace core
} // namespace cuadmm

#endif // CUADMM_CORE_GPU_CONTEXT_H
