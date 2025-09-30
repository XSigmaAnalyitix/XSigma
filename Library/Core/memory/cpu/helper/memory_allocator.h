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
#include <atomic>  // for allocation statistics

#if __cplusplus >= 202002L
#include <bit>  // for std::has_single_bit (C++20)
#endif

#include <cstdlib>
#include <cstring>  // for std::memset

#include "common/configure.h"
#include "common/macros.h"
#include "logging/logger.h"
#include "util/exception.h"

#if defined(XSIGMA_ENABLE_TBB)

#ifdef _MSC_VER
#pragma push_macro("__TBB_NO_IMPLICIT_LINKAGE")
#define __TBB_NO_IMPLICIT_LINKAGE 1
#endif

#include <tbb/scalable_allocator.h>

#ifdef _MSC_VER
#pragma pop_macro("__TBB_NO_IMPLICIT_LINKAGE")
#endif
#endif

#ifdef XSIGMA_NUMA_ENABLED
#include "memory/numa.h"
#endif  // XSIGMA_NUMA_ENABLED

namespace xsigma
{
namespace cpu
{
namespace memory_allocator
{
// Memory initialization options
enum class init_policy_enum : uint8_t
{
    UNINITIALIZED = 0,  // Don't initialize memory (fastest)
    ZERO          = 1,  // Zero-fill memory
    PATTERN       = 2   // Fill with debug pattern (debug builds only)
};

// Validate alignment is power of 2 and >= sizeof(void*)
XSIGMA_FORCE_INLINE bool is_valid_alignment(std::size_t alignment) noexcept
{
    return alignment >= sizeof(void*) &&
#if __cplusplus >= 202002L
           std::has_single_bit(alignment);  // C++20
#else
           (alignment & (alignment - 1)) == 0;  // Power of 2 check
#endif
}

// Get default alignment for the platform
XSIGMA_FORCE_INLINE XSIGMA_FUNCTION_CONSTEXPR std::size_t default_alignment() noexcept
{
    return XSIGMA_ALIGNMENT;
}

XSIGMA_FORCE_INLINE void* allocate(
    std::size_t      nbytes,
    std::size_t      alignment = default_alignment(),
    init_policy_enum init      = init_policy_enum::UNINITIALIZED) noexcept
{
    // Input validation
    if XSIGMA_UNLIKELY (nbytes == 0 || static_cast<std::ptrdiff_t>(nbytes) < 0)
    {
        XSIGMA_LOG_WARNING("cpu allocate() called with negative or zero size: {}", nbytes);
        return nullptr;
    }

#ifndef NDEBUG
    if XSIGMA_UNLIKELY (!is_valid_alignment(alignment))
    {
        XSIGMA_LOG_WARNING(
            "cpu allocate() called with invalid alignment: {} (must be power of 2 >= {})",
            alignment,
            sizeof(void*));
        return nullptr;
    }
#endif

    void* ptr = nullptr;

    // Platform-specific allocation
#ifdef XSIGMA_ENABLE_TBB
    ptr = scalable_aligned_malloc(nbytes, alignment);
#elif defined(__ANDROID__)
    ptr = memalign(alignment, nbytes);
#elif defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    ptr = _aligned_malloc(nbytes, alignment);  // Fixed syntax error
#else
    // POSIX systems
    if (alignment < sizeof(void*))
    {
        ptr = malloc(nbytes);
    }
    else
    {
        if XSIGMA_UNLIKELY (posix_memalign(&ptr, alignment, nbytes) != 0)
        {
            return nullptr;
        }
    }
#endif

    if XSIGMA_UNLIKELY (ptr == nullptr)
    {
        return nullptr;
    }

    // NUMA optimization
#ifdef XSIGMA_NUMA_ENABLED
    NUMAMove(ptr, nbytes, GetCurrentNUMANode());
#endif

    // Memory initialization
    switch (init)
    {
    case init_policy_enum::ZERO:
        std::memset(ptr, 0, nbytes);
        break;
    case init_policy_enum::PATTERN:
#ifndef NDEBUG
        std::memset(ptr, 0xCC, nbytes);
#endif
        break;
    case init_policy_enum::UNINITIALIZED:
    default:
        // Do nothing - fastest option
        break;
    }

    return ptr;
}

// Zero-initialized allocation
XSIGMA_FORCE_INLINE void* allocate_zero(
    std::size_t nbytes, std::size_t alignment = default_alignment()) noexcept
{
    return allocate(nbytes, alignment, init_policy_enum::ZERO);
}

XSIGMA_FORCE_INLINE void free(void* ptr, XSIGMA_UNUSED std::size_t nbytes = 0) noexcept
{
    if XSIGMA_LIKELY (ptr != nullptr)
    {
#ifdef XSIGMA_ENABLE_TBB
        scalable_aligned_free(ptr);
#elif defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        _aligned_free(ptr);  // Fixed syntax error
#else
        ::free(ptr);
#endif

#ifdef XSIGMA_ENABLE_ALLOCATION_STATS
        if (nbytes > 0)
        {
            update_free_stats(nbytes);
        }
#endif
    }
}
}  // namespace memory_allocator
}  // namespace cpu
}  // namespace xsigma
