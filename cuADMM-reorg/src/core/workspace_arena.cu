/*

    core/workspace_arena.cu

*/

#include "core/workspace_arena.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace cuadmm
{
namespace core
{

namespace
{
std::size_t align_up(std::size_t x, std::size_t a) { return (x + a - 1) / a * a; }

void check_alignment(std::size_t alignment)
{
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::invalid_argument("WorkspaceArena: the alignment must be a power of two (got " + std::to_string(alignment) + ")");
}
} // namespace

int WorkspaceArena::request(const std::string &owner, std::size_t bytes, std::size_t alignment)
{
    if (committed_)
        throw std::logic_error("WorkspaceArena: request after commit (" + owner + ")");
    check_alignment(alignment);
    entries_.push_back(Entry{owner, bytes, alignment, false, 0});
    laid_out_ = false;
    return static_cast<int>(entries_.size()) - 1;
}

int WorkspaceArena::request_shared(const std::string &owner, std::size_t bytes, std::size_t alignment)
{
    if (committed_)
        throw std::logic_error("WorkspaceArena: request after commit (" + owner + ")");
    check_alignment(alignment);
    entries_.push_back(Entry{owner, bytes, alignment, true, 0});
    laid_out_ = false;
    return static_cast<int>(entries_.size()) - 1;
}

void WorkspaceArena::layout() const
{
    if (laid_out_)
        return;
    std::size_t cursor = 0;
    std::size_t shared_size = 0, shared_align = 1;
    auto &entries = const_cast<std::vector<Entry> &>(entries_);
    for (Entry &e : entries)
    {
        if (e.shared)
        {
            shared_size = std::max(shared_size, e.bytes);
            shared_align = std::max(shared_align, e.alignment);
            continue;
        }
        cursor = align_up(cursor, e.alignment);
        e.offset = cursor;
        cursor += e.bytes;
    }
    shared_offset_ = align_up(cursor, shared_align);
    for (Entry &e : entries)
        if (e.shared)
            e.offset = shared_offset_;
    total_ = shared_size > 0 ? shared_offset_ + shared_size : cursor;
    laid_out_ = true;
}

std::size_t WorkspaceArena::planned_bytes() const
{
    layout();
    return total_;
}

std::size_t WorkspaceArena::shared_bytes() const
{
    std::size_t s = 0;
    for (const Entry &e : entries_)
        if (e.shared)
            s = std::max(s, e.bytes);
    return s;
}

std::string WorkspaceArena::plan_text() const
{
    layout();
    std::ostringstream os;
    for (std::size_t i = 0; i < entries_.size(); i++)
    {
        const Entry &e = entries_[i];
        os << "  " << e.owner << ": " << e.bytes << " bytes at " << e.offset << (e.shared ? " (shared area)" : "") << "\n";
    }
    os << "  total: " << total_ << " bytes (" << (double)total_ / (1024.0 * 1024.0) << " MiB)";
    return os.str();
}

void WorkspaceArena::commit()
{
    if (committed_)
        throw std::logic_error("WorkspaceArena: committed twice");
    layout();
    memory_.allocate(total_);
    committed_ = true;
}

void WorkspaceArena::release()
{
    memory_.release();
    committed_ = false;
}

const WorkspaceArena::Entry &WorkspaceArena::entry(int handle) const
{
    if (handle < 0 || handle >= static_cast<int>(entries_.size()))
        throw std::out_of_range("WorkspaceArena: handle " + std::to_string(handle));
    layout();
    return entries_[handle];
}

std::size_t WorkspaceArena::bytes(int handle) const { return entry(handle).bytes; }

std::size_t WorkspaceArena::offset(int handle) const { return entry(handle).offset; }

void *WorkspaceArena::data(int handle) const
{
    const Entry &e = entry(handle);
    if (!committed_)
        throw std::logic_error("WorkspaceArena: data() before commit (" + e.owner + ")");
    if (e.bytes == 0)
        return nullptr;
    return const_cast<unsigned char *>(memory_.data()) + e.offset;
}

} // namespace core
} // namespace cuadmm
