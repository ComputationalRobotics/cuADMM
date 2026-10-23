/*

    core/stream_pool.cu

*/

#include "core/stream_pool.h"

#include <stdexcept>
#include <string>

#include "core/check.h"

namespace cuadmm
{
namespace core
{

const char *stream_role_name(StreamRole role)
{
    switch (role)
    {
    case StreamRole::Main:
        return "main";
    case StreamRole::Small:
        return "small";
    case StreamRole::Medium:
        return "medium";
    case StreamRole::Large:
        return "large";
    case StreamRole::Copy:
        return "copy";
    }
    return "?";
}

StreamPool::Slot StreamPool::make_slot(bool own_stream)
{
    Slot s;
    if (own_stream)
    {
        CUADMM_CHECK_CUDA(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        s.owns_stream = true;
    }
    CUADMM_CHECK_CUBLAS(cublasCreate(&s.cublas));
    CUADMM_CHECK_CUBLAS(cublasSetStream(s.cublas, s.stream));
    CUADMM_CHECK_CUBLAS(cublasSetMathMode(s.cublas, CUBLAS_DEFAULT_MATH)); // B1: explicit
    CUADMM_CHECK_CUSOLVER(cusolverDnCreate(&s.cusolver));
    CUADMM_CHECK_CUSOLVER(cusolverDnSetStream(s.cusolver, s.stream));
    CUADMM_CHECK_CUDA(cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming));
    return s;
}

void StreamPool::destroy_slot(Slot &s)
{
    if (s.event != nullptr)
        CUADMM_CHECK_CUDA_NOTHROW(cudaEventDestroy(s.event));
    if (s.cusolver != nullptr)
        CUADMM_CHECK_CUSOLVER_NOTHROW(cusolverDnDestroy(s.cusolver));
    if (s.cublas != nullptr)
        CUADMM_CHECK_CUBLAS_NOTHROW(cublasDestroy(s.cublas));
    if (s.owns_stream && s.stream != nullptr)
        CUADMM_CHECK_CUDA_NOTHROW(cudaStreamDestroy(s.stream));
    s = Slot{};
}

StreamPool::StreamPool(const StreamPoolConfig &config) : config_(config)
{
    if (config_.medium_streams < 1)
        throw std::invalid_argument("StreamPool: medium_streams must be at least 1 (B17), got " + std::to_string(config_.medium_streams));
    main_ = make_slot(false); // the legacy default stream (Milestone 2)
    if (!config_.single_stream)
    {
        medium_.reserve(config_.medium_streams);
        for (int i = 0; i < config_.medium_streams; i++)
            medium_.push_back(make_slot(true));
    }
}

StreamPool::~StreamPool()
{
    for (Slot &s : medium_)
        destroy_slot(s);
    destroy_slot(main_);
}

const StreamPool::Slot &StreamPool::slot(StreamRole role, int index) const
{
    if (role == StreamRole::Medium && !medium_.empty())
    {
        if (index < 0)
            throw std::out_of_range("StreamPool: medium stream index " + std::to_string(index));
        return medium_[index % medium_.size()];
    }
    return main_; // main, small, large, copy: the legacy default stream until Phase 8; single_stream: everything
}

cudaStream_t StreamPool::stream(StreamRole role, int index) const { return slot(role, index).stream; }

ExecContext StreamPool::context(StreamRole role, int index) const
{
    const Slot &s = slot(role, index);
    ExecContext c;
    c.stream = s.stream;
    c.cublas = s.cublas;
    c.cusolver = s.cusolver;
    c.cusparse = nullptr;
    return c;
}

void StreamPool::wait(StreamRole to_role, int to_index, StreamRole from_role, int from_index)
{
    const Slot &from = slot(from_role, from_index);
    const Slot &to = slot(to_role, to_index);
    if (from.stream == to.stream)
        return; // the same queue: already ordered
    CUADMM_CHECK_CUDA(cudaEventRecord(from.event, from.stream));
    CUADMM_CHECK_CUDA(cudaStreamWaitEvent(to.stream, from.event, 0));
}

void StreamPool::synchronize(StreamRole role, int index) const
{
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(slot(role, index).stream));
}

void StreamPool::synchronize_all() const
{
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(main_.stream));
    for (const Slot &s : medium_)
        CUADMM_CHECK_CUDA(cudaStreamSynchronize(s.stream));
}

} // namespace core
} // namespace cuadmm
