/*

    text_source.cc

*/

#include "io/text_source.h"

#include <cstring>
#include <stdexcept>

#ifdef CUADMM_HAVE_ZLIB
#include <zlib.h>
#endif

namespace cuadmm
{
namespace io
{
namespace detail
{

bool is_gzip_file(const std::string &path)
{
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f)
        return false;
    unsigned char magic[2] = {0, 0};
    const std::size_t n = std::fread(magic, 1, 2, f);
    std::fclose(f);
    return n == 2 && magic[0] == 0x1f && magic[1] == 0x8b;
}

TextSource::TextSource(const std::string &path) : path_(path), buf_(1 << 20)
{
    compressed_ = is_gzip_file(path);
    if (compressed_)
    {
#ifdef CUADMM_HAVE_ZLIB
        gz_ = gzopen(path.c_str(), "rb");
        if (!gz_)
            throw std::runtime_error("could not open the gzip file " + path);
        gzbuffer(static_cast<gzFile>(gz_), 1 << 20);
#else
        throw std::invalid_argument(path + " is gzip-compressed, and this build has no zlib (CUADMM_WITH_ZLIB): decompress the file first");
#endif
    }
    else
    {
        file_ = std::fopen(path.c_str(), "rb");
        if (!file_)
            throw std::runtime_error("could not open " + path);
    }
}

TextSource::~TextSource()
{
    if (file_)
        std::fclose(file_);
#ifdef CUADMM_HAVE_ZLIB
    if (gz_)
        gzclose(static_cast<gzFile>(gz_));
#endif
}

std::size_t TextSource::fill()
{
    if (eof_)
        return 0;
    // move the unread tail to the front; grow the buffer when a single line fills it
    if (begin_ > 0)
    {
        std::memmove(buf_.data(), buf_.data() + begin_, end_ - begin_);
        end_ -= begin_;
        begin_ = 0;
    }
    if (end_ == buf_.size())
        buf_.resize(buf_.size() * 2);
    std::size_t got = 0;
    if (file_)
        got = std::fread(buf_.data() + end_, 1, buf_.size() - end_, file_);
#ifdef CUADMM_HAVE_ZLIB
    else if (gz_)
    {
        const int r = gzread(static_cast<gzFile>(gz_), buf_.data() + end_, static_cast<unsigned>(buf_.size() - end_));
        if (r < 0)
        {
            int err = 0;
            const char *msg = gzerror(static_cast<gzFile>(gz_), &err);
            throw std::runtime_error("error while decompressing " + path_ + ": " + (msg ? msg : ""));
        }
        got = static_cast<std::size_t>(r);
    }
#endif
    if (got == 0)
        eof_ = true;
    end_ += got;
    return got;
}

bool TextSource::next_line(std::string_view &line)
{
    for (;;)
    {
        const char *start = buf_.data() + begin_;
        const char *nl = static_cast<const char *>(std::memchr(start, '\n', end_ - begin_));
        if (nl)
        {
            std::size_t len = static_cast<std::size_t>(nl - start);
            begin_ += len + 1;
            if (len > 0 && start[len - 1] == '\r')
                len--;
            line = std::string_view(start, len);
            line_number_++;
            return true;
        }
        if (fill() == 0)
        {
            // the last line without a newline
            if (end_ > begin_)
            {
                std::size_t len = end_ - begin_;
                const char *s = buf_.data() + begin_;
                begin_ = end_;
                if (len > 0 && s[len - 1] == '\r')
                    len--;
                line = std::string_view(s, len);
                line_number_++;
                return true;
            }
            return false;
        }
    }
}

} // namespace detail
} // namespace io
} // namespace cuadmm
