/*

    sha256.h

    SHA-256 of a file (the `source_sha256` attribute of the native HDF5 format, plan Section 12): a small
    self-contained implementation of FIPS 180-4, so that the writer needs no cryptography library.

*/

#ifndef CUADMM_IO_SHA256_H
#define CUADMM_IO_SHA256_H

#include <cstdint>
#include <string>

namespace cuadmm
{
namespace io
{
namespace detail
{

class Sha256
{
  public:
    Sha256();
    void update(const unsigned char *data, std::size_t len);
    /// @brief The digest as 64 lowercase hex characters; the object cannot be updated afterwards.
    std::string hex_digest();

  private:
    void transform(const unsigned char block[64]);
    std::uint32_t h_[8];
    unsigned char buffer_[64];
    std::size_t buffer_len_ = 0;
    std::uint64_t total_len_ = 0;
};

/// @brief Hex SHA-256 of a file; throws std::runtime_error when it cannot be read.
std::string sha256_of_file(const std::string &path);

} // namespace detail
} // namespace io
} // namespace cuadmm

#endif // CUADMM_IO_SHA256_H
