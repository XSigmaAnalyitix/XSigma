#pragma once

#ifndef __XSIGMA_WRAP__
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "common/macros.h"
#include "util/hash_util.h"

namespace xsigma
{
// Forward declaration - implementation is in hash_util.h
//static size_t hash_pair(size_t left, size_t right);

// Deprecated - use hash_pair from hash_util.h instead
[[deprecated("Use hash_pair from hash_util.h instead")]]
static size_t hash_pair_fnv(size_t left, size_t right)
{
    constexpr size_t FNV_PRIME  = 1099511628211ULL;
    constexpr size_t FNV_OFFSET = 14695981039346656037ULL;

    size_t hash = FNV_OFFSET;
    hash        = (hash ^ left) * FNV_PRIME;
    hash        = (hash ^ right) * FNV_PRIME;
    return hash;
}

template <size_t N>
class merkle_hash
{
    XSIGMA_DELETE_CLASS(merkle_hash);

public:
    template <typename... Ts>
    static size_t hash(const Ts&... data)
    {
        static_assert(sizeof...(Ts) == N, "Must provide N data items");

        std::array<size_t, N> hashes = {data->hash()...};
        return hash_array(hashes);
    }

    template <size_t Size>
    static size_t hash_array(const std::array<size_t, Size>& arr)
    {
        if constexpr (Size == 0)
            return 0;
        if constexpr (Size == 1)
            return arr[0];

        constexpr size_t            NewSize = (Size + 1) / 2;
        std::array<size_t, NewSize> next;

        for (size_t i = 0; i < Size - 1; i += 2)
        {
            next[i / 2] = hash_pair(arr[i], arr[i + 1]);
        }
        if constexpr (Size % 2 == 1)
        {
            next[NewSize - 1] = arr[Size - 1];
        }

        return hash_array(next);
    }
};
}  // namespace xsigma
#endif  // 0
