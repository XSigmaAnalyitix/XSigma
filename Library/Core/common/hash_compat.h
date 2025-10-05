/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _LIBCPP_NOESCAPE

/**
 * @file hash_compat.h
 * @brief Compatibility layer for hash functions across different libc++ versions
 *
 * This file provides implementations of hash-related functions that may not be
 * available or exported in all versions of libc++, particularly when using
 * Homebrew LLVM on macOS.
 *
 * The __hash_memory function is declared in libc++ headers but not always exported
 * from the library. This file provides a compatible implementation.
 */

// Only provide the implementation if we're using libc++ and the function is not available
#if defined(_LIBCPP_VERSION) && !defined(__GLIBCXX__)

_LIBCPP_BEGIN_NAMESPACE_STD

// Provide implementation of __hash_memory if it's not exported
// This is based on the MurmurHash2 algorithm, which is what libc++ uses internally
[[__gnu__::__pure__]] _LIBCPP_EXPORTED_FROM_ABI size_t __hash_memory(_LIBCPP_NOESCAPE const void* __key, size_t __len) _NOEXCEPT
{
    const size_t __m = 0xc6a4a7935bd1e995ULL;
    const int __r = 47;
    size_t __h = __len * __m;

    const unsigned char* __data = static_cast<const unsigned char*>(__key);
    const unsigned char* __end = __data + (__len & ~7ULL);

    while (__data != __end)
    {
        size_t __k;
        __builtin_memcpy(&__k, __data, sizeof(__k));

        __k *= __m;
        __k ^= __k >> __r;
        __k *= __m;

        __h ^= __k;
        __h *= __m;

        __data += 8;
    }

    switch (__len & 7)
    {
    case 7:
        __h ^= static_cast<size_t>(__data[6]) << 48;
        [[fallthrough]];
    case 6:
        __h ^= static_cast<size_t>(__data[5]) << 40;
        [[fallthrough]];
    case 5:
        __h ^= static_cast<size_t>(__data[4]) << 32;
        [[fallthrough]];
    case 4:
        __h ^= static_cast<size_t>(__data[3]) << 24;
        [[fallthrough]];
    case 3:
        __h ^= static_cast<size_t>(__data[2]) << 16;
        [[fallthrough]];
    case 2:
        __h ^= static_cast<size_t>(__data[1]) << 8;
        [[fallthrough]];
    case 1:
        __h ^= static_cast<size_t>(__data[0]);
        __h *= __m;
    }

    __h ^= __h >> __r;
    __h *= __m;
    __h ^= __h >> __r;

    return __h;
}

_LIBCPP_END_NAMESPACE_STD

#endif  // _LIBCPP_VERSION && !__GLIBCXX__
#endif
