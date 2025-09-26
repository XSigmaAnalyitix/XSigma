#include "util/xtea.h"

#include <cstring>  // for size_t, memset

namespace xsigma::details
{
#define XTEA_DELTA 0x9E3779B9
#define BLOCK_SIZE 8  // XTEA uses 64-bit blocks, 64 bits is 8 bytes
#define XTEA_START 0xfade273d

inline void encipher(
    unsigned int num_rounds, std::array<unsigned int, 2>& v, const std::array<unsigned int, 4>& key)
{
    unsigned int i;
    unsigned int v0  = v[0];
    unsigned int v1  = v[1];
    unsigned int sum = 0;
    for (i = 0; i < num_rounds; i++)
    {
        v0 += (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + key[sum & 3]);
        sum += XTEA_DELTA;
        v1 += (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + key[(sum >> 11) & 3]);
    }
    v[0] = v0;
    v[1] = v1;
}

#ifdef DECIPHER

inline void decipher(
    unsigned int num_rounds, std::array<unsigned int, 2>& v, const std::array<unsigned int, 4>& key)
{
    unsigned int v0  = v[0];
    unsigned int v1  = v[1];
    unsigned int sum = XTEA_DELTA * num_rounds;

    for (unsigned int i = 0; i < num_rounds; i++)
    {
        v1 -= (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + key[(sum >> 11) & 3]);
        sum -= XTEA_DELTA;
        v0 -= (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + key[sum & 3]);
    }

    v[0] = v0;
    v[1] = v1;
}
#endif  // 0

unsigned int xtea_encipher_helper(
    const unsigned int                 start,
    const unsigned int*                src,
    size_t                             size,
    unsigned int*                      dst,
    const std::array<unsigned int, 4>& key)
{
    std::array<unsigned int, 2> v;
    v[1] = start;
    for (size_t i = 0; i < size; ++i)
    {
        v[0] = *src++;

        encipher(32, v, key);

        *dst++ = v[1];
        v[1]   = v[0];
    }
    return v[1];
}

void xtea_encipher(
    const unsigned int*                src,
    size_t                             src_size,
    unsigned int*                      dst,
    size_t                             dst_size,
    const std::array<unsigned int, 4>& key)
{
    // XSIGMA_CHECK_DEBUG(!(src == nullptr || dst == nullptr || key.empty() || dst_size < 1),
    // "ERROR");

    std::memset(dst, 0, dst_size * sizeof(unsigned int));
    unsigned int start = XTEA_START;
    while (src_size != 0U)
    {
        const size_t n = (src_size >= dst_size) ? dst_size - 1 : src_size;
        for (size_t i = 0; i < n; ++i)
        {
            dst[i] ^= *src++;
        }

        src_size -= n;

        start = xtea_encipher_helper(start, dst, dst_size - 1, dst, key);
    }

    dst[dst_size - 1] = xtea_encipher_helper(start, dst, dst_size - 1, dst, key);
}
}  // namespace xsigma::details
