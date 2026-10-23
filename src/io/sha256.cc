/*

    sha256.cc  (FIPS 180-4)

*/

#include "io/sha256.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cuadmm
{
namespace io
{
namespace detail
{

namespace
{
constexpr std::uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
} // namespace

Sha256::Sha256()
{
    const std::uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h_, init, sizeof(h_));
}

void Sha256::transform(const unsigned char block[64])
{
    std::uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (std::uint32_t(block[4 * i]) << 24) | (std::uint32_t(block[4 * i + 1]) << 16) | (std::uint32_t(block[4 * i + 2]) << 8) |
               std::uint32_t(block[4 * i + 3]);
    for (int i = 16; i < 64; i++)
    {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; i++)
    {
        const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t t1 = h + S1 + ch + kK[i] + w[i];
        const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
}

void Sha256::update(const unsigned char *data, std::size_t len)
{
    total_len_ += len;
    while (len > 0)
    {
        const std::size_t take = std::min(len, static_cast<std::size_t>(64) - buffer_len_);
        std::memcpy(buffer_ + buffer_len_, data, take);
        buffer_len_ += take;
        data += take;
        len -= take;
        if (buffer_len_ == 64)
        {
            transform(buffer_);
            buffer_len_ = 0;
        }
    }
}

std::string Sha256::hex_digest()
{
    const std::uint64_t bits = total_len_ * 8;
    unsigned char pad = 0x80;
    update(&pad, 1);
    const unsigned char zero = 0;
    while (buffer_len_ != 56)
        update(&zero, 1);
    unsigned char len_bytes[8];
    for (int i = 0; i < 8; i++)
        len_bytes[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    total_len_ -= 9 + (64 - 56); // not meaningful after finalization
    update(len_bytes, 8);
    static const char *hex = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
        {
            const unsigned char byte = static_cast<unsigned char>(h_[i] >> (24 - 8 * j));
            out += hex[byte >> 4];
            out += hex[byte & 0xf];
        }
    return out;
}

std::string sha256_of_file(const std::string &path)
{
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("could not open " + path + " for hashing");
    Sha256 h;
    std::vector<unsigned char> buf(1 << 20);
    std::size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0)
        h.update(buf.data(), n);
    const bool error = std::ferror(f) != 0;
    std::fclose(f);
    if (error)
        throw std::runtime_error("error while reading " + path + " for hashing");
    return h.hex_digest();
}

} // namespace detail
} // namespace io
} // namespace cuadmm
