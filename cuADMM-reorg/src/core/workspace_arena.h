/*

    core/workspace_arena.h

    WorkspaceArena (plan Section 14, Phase 6b): components declare their workspace at setup, the arena lays the
    regions out with their alignment, allocates once (commit), and hands out spans. Private regions are laid out
    one after another in request order; the shared area (components that never run at the same time, such as
    the strategies of the large tier at different times, or the projections and the termination measures) is as
    large as the largest shared request and every shared handle starts at its beginning. No cudaMalloc or
    cudaFree happens inside the iteration loop: the arena is committed before the first iteration and released
    with its owner.

    The layout (planned_bytes, plan_text) is host logic: it is tested without a GPU; the allocation happens at
    commit().

*/

#ifndef CUADMM_CORE_WORKSPACE_ARENA_H
#define CUADMM_CORE_WORKSPACE_ARENA_H

#include <cstddef>
#include <string>
#include <vector>

#include "core/device_buffer.h"

namespace cuadmm
{
namespace core
{

class WorkspaceArena
{
public:
    struct Entry
    {
        std::string owner;
        std::size_t bytes = 0;
        std::size_t alignment = 256;
        bool shared = false;
        std::size_t offset = 0; // set by the layout
    };

    WorkspaceArena() = default;
    ~WorkspaceArena() = default;
    WorkspaceArena(const WorkspaceArena &) = delete;
    WorkspaceArena &operator=(const WorkspaceArena &) = delete;

    /// A private region of `bytes` bytes aligned to `alignment` (a power of two); returns its handle. Before commit().
    int request(const std::string &owner, std::size_t bytes, std::size_t alignment = 256);
    /// A region of the shared area; returns its handle. Before commit().
    int request_shared(const std::string &owner, std::size_t bytes, std::size_t alignment = 256);

    /// The total of the layout (private regions, then the shared area), 0 without requests.
    std::size_t planned_bytes() const;
    /// The size of the shared area (the largest shared request).
    std::size_t shared_bytes() const;
    /// One line per region ("owner: bytes at offset (shared)") and the total, for the setup print.
    std::string plan_text() const;
    const std::vector<Entry> &entries() const { return entries_; }

    /// The one allocation; nothing may be requested afterwards.
    void commit();
    bool committed() const { return committed_; }
    /// Frees the allocation; the handles stay valid for a new commit() after further requests.
    void release();

    std::size_t bytes(int handle) const;
    std::size_t offset(int handle) const;
    /// The device pointer of a region; throws before commit().
    void *data(int handle) const;
    template <typename T>
    DeviceSpan<T> span(int handle) const
    {
        return DeviceSpan<T>(static_cast<T *>(data(handle)), bytes(handle) / sizeof(T));
    }

private:
    void layout() const;
    const Entry &entry(int handle) const;
    std::vector<Entry> entries_;
    mutable std::size_t total_ = 0;
    mutable std::size_t shared_offset_ = 0;
    mutable bool laid_out_ = false;
    bool committed_ = false;
    DeviceBuffer<unsigned char> memory_;
};

} // namespace core
} // namespace cuadmm

#endif // CUADMM_CORE_WORKSPACE_ARENA_H
