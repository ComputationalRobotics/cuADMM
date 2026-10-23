/*

    text_source.h

    Line reader over a plain or gzip-compressed text file (SDPA .dat-s and .dat-s.gz, plan Section 12), buffered,
    without loading the whole file. gzip support needs zlib (CUADMM_HAVE_ZLIB); otherwise a compressed file is
    rejected with a clear message.

*/

#ifndef CUADMM_IO_TEXT_SOURCE_H
#define CUADMM_IO_TEXT_SOURCE_H

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cuadmm
{
namespace io
{
namespace detail
{

/// @brief Whether the file starts with the gzip magic bytes.
bool is_gzip_file(const std::string &path);

class TextSource
{
  public:
    explicit TextSource(const std::string &path);
    ~TextSource();
    TextSource(const TextSource &) = delete;
    TextSource &operator=(const TextSource &) = delete;

    /// @brief The next line without its end-of-line characters; false at the end of the file.
    bool next_line(std::string_view &line);
    std::int64_t line_number() const { return line_number_; }
    bool compressed() const { return compressed_; }
    const std::string &path() const { return path_; }

  private:
    std::size_t fill(); // reads more bytes after end_; returns the number read
    std::string path_;
    bool compressed_ = false;
    std::FILE *file_ = nullptr;
    void *gz_ = nullptr; // gzFile
    std::vector<char> buf_;
    std::size_t begin_ = 0, end_ = 0;
    bool eof_ = false;
    std::int64_t line_number_ = 0;
};

} // namespace detail
} // namespace io
} // namespace cuadmm

#endif // CUADMM_IO_TEXT_SOURCE_H
