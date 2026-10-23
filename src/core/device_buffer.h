/*

    core/device_buffer.h

    Owning and non-owning device memory (plan Section 14): DeviceBuffer<T> (RAII, std::size_t sizes), DeviceSpan<T>
    (a view), PinnedHostBuffer<T>, and the copies between host and device. No raw cudaMalloc outside core.

*/

#ifndef CUADMM_CORE_DEVICE_BUFFER_H
#define CUADMM_CORE_DEVICE_BUFFER_H

#include <cstddef>
#include <cstring>
#include <vector>

#include "core/check.h"

namespace cuadmm
{
namespace core
{

/// @brief Non-owning view of device memory.
template <typename T>
struct DeviceSpan
{
    T *data = nullptr;
    std::size_t size = 0;

    DeviceSpan() = default;
    DeviceSpan(T *d, std::size_t n) : data(d), size(n) {}
    DeviceSpan<const T> as_const() const { return DeviceSpan<const T>(data, size); }
    DeviceSpan<T> subspan(std::size_t offset, std::size_t n) const { return DeviceSpan<T>(data + offset, n); }
    bool empty() const { return size == 0; }
};

/// @brief Owning device memory.
template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { allocate(n); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&o) noexcept : data_(o.data_), size_(o.size_)
    {
        o.data_ = nullptr;
        o.size_ = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept
    {
        if (this != &o)
        {
            release();
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    // (Re)allocates n elements; the content is undefined afterwards.
    void allocate(std::size_t n)
    {
        release();
        if (n > 0)
            CUADMM_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&data_), sizeof(T) * n));
        size_ = n;
    }
    void release()
    {
        if (data_ != nullptr)
        {
            CUADMM_CHECK_CUDA_NOTHROW(cudaFree(data_));
            data_ = nullptr;
        }
        size_ = 0;
    }
    void zero(cudaStream_t stream = nullptr)
    {
        if (size_ > 0)
            CUADMM_CHECK_CUDA(cudaMemsetAsync(data_, 0, sizeof(T) * size_, stream));
    }

    T *data() { return data_; }
    const T *data() const { return data_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    DeviceSpan<T> span() { return DeviceSpan<T>(data_, size_); }
    DeviceSpan<const T> span() const { return DeviceSpan<const T>(data_, size_); }

private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
};

/// @brief Page-locked host memory (asynchronous copies).
template <typename T>
class PinnedHostBuffer
{
public:
    PinnedHostBuffer() = default;
    explicit PinnedHostBuffer(std::size_t n) { allocate(n); }
    ~PinnedHostBuffer() { release(); }
    PinnedHostBuffer(const PinnedHostBuffer &) = delete;
    PinnedHostBuffer &operator=(const PinnedHostBuffer &) = delete;

    void allocate(std::size_t n)
    {
        release();
        if (n > 0)
            CUADMM_CHECK_CUDA(cudaMallocHost(reinterpret_cast<void **>(&data_), sizeof(T) * n));
        size_ = n;
    }
    void release()
    {
        if (data_ != nullptr)
        {
            CUADMM_CHECK_CUDA_NOTHROW(cudaFreeHost(data_));
            data_ = nullptr;
        }
        size_ = 0;
    }
    T *data() { return data_; }
    const T *data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
};

/// @brief Synchronous copies.
template <typename T>
std::vector<T> to_host(DeviceSpan<const T> v)
{
    std::vector<T> h(v.size);
    if (v.size > 0)
        CUADMM_CHECK_CUDA(cudaMemcpy(h.data(), v.data, sizeof(T) * v.size, cudaMemcpyDeviceToHost));
    return h;
}
template <typename T>
std::vector<T> to_host(const T *d, std::size_t n)
{
    return to_host(DeviceSpan<const T>(d, n));
}
template <typename T>
void to_device(DeviceSpan<T> dst, const std::vector<T> &src)
{
    if (src.size() != dst.size)
        throw std::invalid_argument("to_device: size mismatch");
    if (!src.empty())
        CUADMM_CHECK_CUDA(cudaMemcpy(dst.data, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice));
}
template <typename T>
void copy_device(DeviceSpan<T> dst, DeviceSpan<const T> src, cudaStream_t stream = nullptr)
{
    if (src.size != dst.size)
        throw std::invalid_argument("copy_device: size mismatch");
    if (src.size > 0)
        CUADMM_CHECK_CUDA(cudaMemcpyAsync(dst.data, src.data, sizeof(T) * src.size, cudaMemcpyDeviceToDevice, stream));
}

} // namespace core
} // namespace cuadmm

#endif // CUADMM_CORE_DEVICE_BUFFER_H
