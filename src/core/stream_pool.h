/*

    core/stream_pool.h

    StreamPool (plan Section 14, Phase 6b): the named streams of the solver (main, small, medium[N], large, copy)
    with a cuBLAS and a cuSOLVER handle bound to each, and events for cross-stream dependencies.

    Milestone 2 keeps the ported engine, whose kernels run on the legacy default stream and synchronize only with
    blocking streams: the main, small, large and copy roles are therefore the legacy default stream (nullptr)
    until Phase 8 moves the engine's work onto the pool, and only the medium streams are non-blocking streams of
    their own, as the engine's medium-tier streams were (DECISIONS 49). `single_stream` maps every role to the
    main stream (the mode that must reproduce the concurrent run bit for bit).

*/

#ifndef CUADMM_CORE_STREAM_POOL_H
#define CUADMM_CORE_STREAM_POOL_H

#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "core/gpu_context.h"

namespace cuadmm
{
namespace core
{

enum class StreamRole
{
    Main,
    Small,
    Medium,
    Large,
    Copy
};

struct StreamPoolConfig
{
    int medium_streams = 15;    // psd.medium_streams (>= 1, B17)
    bool single_stream = false; // every role on the main stream
};

class StreamPool
{
public:
    explicit StreamPool(const StreamPoolConfig &config);
    ~StreamPool();
    StreamPool(const StreamPool &) = delete;
    StreamPool &operator=(const StreamPool &) = delete;

    int medium_count() const { return static_cast<int>(medium_.size()); }
    bool single_stream() const { return config_.single_stream; }
    /// The stream of a role (index: the medium stream, taken modulo the count).
    cudaStream_t stream(StreamRole role, int index = 0) const;
    /// The stream of a role with its handles (cusparse: none; the engine's main handle serves the main stream).
    ExecContext context(StreamRole role, int index = 0) const;
    /// Makes the work queued on `to` wait for the work queued on `from` so far (an event on `from`).
    void wait(StreamRole to_role, int to_index, StreamRole from_role, int from_index = 0);
    void synchronize(StreamRole role, int index = 0) const;
    void synchronize_all() const;

private:
    struct Slot
    {
        cudaStream_t stream = nullptr; // nullptr: the legacy default stream
        cublasHandle_t cublas = nullptr;
        cusolverDnHandle_t cusolver = nullptr;
        cudaEvent_t event = nullptr;
        bool owns_stream = false;
    };
    const Slot &slot(StreamRole role, int index) const;
    static Slot make_slot(bool own_stream);
    static void destroy_slot(Slot &s);
    StreamPoolConfig config_;
    Slot main_;
    std::vector<Slot> medium_;
};

const char *stream_role_name(StreamRole role);

} // namespace core
} // namespace cuadmm

#endif // CUADMM_CORE_STREAM_POOL_H
