#pragma once

#include <array>    // for array
#include <cstddef>  // for size_t

  #include "common/macros.h"

namespace xsigma
{
namespace details
{
constexpr std::array<unsigned int, 4> default_key = {
    0X9CF4F3C7,
    0XBF715880,
    0X8AED2A6A,
    0XB7E15162};  //{0xDC529BD3, 0xDEADBEEF, 0xC0FFEE33, 0x29211663};

XSIGMA_API void xtea_encipher(
    const unsigned int*                src,
    size_t                             src_size,
    unsigned int*                      dst,
    size_t                             dst_size,
    const std::array<unsigned int, 4>& key = default_key);
}  // namespace details
}  // namespace xsigma
