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

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "common/configure.h"
#include "memory/cpu/helper/memory_allocator.h"
#include "util/exception.h"
#include "xsigmaTest.h"

// Platform-specific includes
#ifdef XSIGMA_ENABLE_TBB
#include <tbb/cache_aligned_allocator.h>
#include <tbb/scalable_allocator.h>
#endif

#ifdef XSIGMA_ENABLE_MIMALLOC
#include "mimalloc.h"
#endif

// Standard aligned allocation
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace xsigma
{
namespace testing
{

/**
 * @brief Standardized allocator interface for testing different memory allocators
 * 
 * This wrapper provides a unified interface for testing mimalloc, TBB allocators,
 * and standard aligned allocators with consistent behavior and error handling.
 */
class allocator_test_interface
{
public:
    virtual ~allocator_test_interface() = default;

    virtual void*       allocate(std::size_t size, std::size_t alignment = 64) noexcept = 0;
    virtual void        deallocate(void* ptr, std::size_t size = 0) noexcept            = 0;
    virtual const char* name() const noexcept                                           = 0;
    virtual bool        supports_alignment(std::size_t alignment) const noexcept        = 0;
    virtual bool        is_thread_safe() const noexcept                                 = 0;
};

/**
 * @brief mimalloc allocator wrapper
 */
class mimalloc_test_allocator : public allocator_test_interface
{
public:
    void* allocate(std::size_t size, std::size_t alignment = 64) noexcept override
    {
#ifdef XSIGMA_ENABLE_MIMALLOC
        return mi_malloc_aligned(size, alignment);
#else
        (void)size;
        (void)alignment;
        return nullptr;
#endif
    }

    void deallocate(void* ptr, XSIGMA_UNUSED std::size_t size = 0) noexcept override
    {
#ifdef XSIGMA_ENABLE_MIMALLOC
        mi_free(ptr);
#else
        (void)ptr;
#endif
    }

    const char* name() const noexcept override { return "mimalloc"; }
    bool        supports_alignment(std::size_t alignment) const noexcept override
    {
        return alignment > 0 && (alignment & (alignment - 1)) == 0;  // Power of 2
    }
    bool is_thread_safe() const noexcept override { return true; }
};

/**
 * @brief TBB scalable allocator wrapper
 */
class tbb_scalable_test_allocator : public allocator_test_interface
{
public:
    void* allocate(std::size_t size, std::size_t alignment = 64) noexcept override
    {
#ifdef XSIGMA_ENABLE_TBB
        return scalable_aligned_malloc(size, alignment);
#else
        (void)size;
        (void)alignment;
        return nullptr;
#endif
    }

    void deallocate(void* ptr, XSIGMA_UNUSED std::size_t size = 0) noexcept override
    {
#ifdef XSIGMA_ENABLE_TBB
        scalable_aligned_free(ptr);
#else
        (void)ptr;
#endif
    }

    const char* name() const noexcept override { return "tbb_scalable"; }
    bool        supports_alignment(std::size_t alignment) const noexcept override
    {
        return alignment > 0 && (alignment & (alignment - 1)) == 0;  // Power of 2
    }
    bool is_thread_safe() const noexcept override { return true; }
};

/**
 * @brief Standard aligned allocator wrapper
 */
class standard_aligned_test_allocator : public allocator_test_interface
{
public:
    void* allocate(std::size_t size, std::size_t alignment = 64) noexcept override
    {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        return _aligned_malloc(size, alignment);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0)
        {
            return nullptr;
        }
        return ptr;
#endif
    }

    void deallocate(void* ptr, XSIGMA_UNUSED std::size_t size = 0) noexcept override
    {
        if (ptr != nullptr)
        {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
            _aligned_free(ptr);
#else
            ::free(ptr);
#endif
        }
    }

    const char* name() const noexcept override { return "standard_aligned"; }
    bool        supports_alignment(std::size_t alignment) const noexcept override
    {
        return alignment >= sizeof(void*) &&
               (alignment & (alignment - 1)) == 0;  // Power of 2, >= pointer size
    }
    bool is_thread_safe() const noexcept override { return true; }
};

/**
 * @brief XSigma's default CPU allocator wrapper
 */
class xsigma_cpu_test_allocator : public allocator_test_interface
{
public:
    void* allocate(std::size_t size, std::size_t alignment = 64) noexcept override
    {
        return xsigma::cpu::memory_allocator::allocate(size, alignment);
    }

    void deallocate(void* ptr, std::size_t size = 0) noexcept override
    {
        xsigma::cpu::memory_allocator::free(ptr, size);
    }

    const char* name() const noexcept override { return "xsigma_cpu"; }
    bool        supports_alignment(std::size_t alignment) const noexcept override
    {
        return xsigma::cpu::memory_allocator::is_valid_alignment(alignment);
    }
    bool is_thread_safe() const noexcept override { return true; }
};

// Helper function to get test allocators
std::vector<std::unique_ptr<allocator_test_interface>> get_test_allocators()
{
    std::vector<std::unique_ptr<allocator_test_interface>> allocators;

    // Always add XSigma's default allocator
    allocators.emplace_back(std::make_unique<xsigma_cpu_test_allocator>());

    // Add standard aligned allocator
    allocators.emplace_back(std::make_unique<standard_aligned_test_allocator>());

#ifdef XSIGMA_ENABLE_MIMALLOC
    allocators.emplace_back(std::make_unique<mimalloc_test_allocator>());
#endif

#ifdef XSIGMA_ENABLE_TBB
    allocators.emplace_back(std::make_unique<tbb_scalable_test_allocator>());
#endif

    return allocators;
}
}  // namespace testing
}  // namespace xsigma
using namespace xsigma::testing;

XSIGMATEST(CPUMemoryAllocators, basic_allocation_deallocation)
{
    auto allocators = get_test_allocators();

    for (const auto& allocator : allocators)
    {
        // Test various sizes
        std::vector<std::size_t> sizes = {1, 8, 16, 64, 256, 1024, 4096, 65536};

        for (std::size_t size : sizes)
        {
            void* ptr = allocator->allocate(size);
            EXPECT_NE(nullptr, ptr);

            if (ptr != nullptr)
            {
                // Write to memory to ensure it's accessible
                std::memset(ptr, 0xAA, size);

                // Verify memory is writable and readable
                auto* byte_ptr = static_cast<unsigned char*>(ptr);
                for (std::size_t i = 0; i < size; ++i)
                {
                    EXPECT_EQ(0xAA, byte_ptr[i]);
                }

                allocator->deallocate(ptr, size);
            }
        }
    }

    END_TEST();
}

XSIGMATEST(CPUMemoryAllocators, zero_size_allocation)
{
    auto allocators = get_test_allocators();

    for (const auto& allocator : allocators)
    {
        void* ptr = allocator->allocate(0);
        // Zero-size allocation behavior is implementation-defined
        // Some allocators return nullptr, others return a valid pointer

        if (ptr != nullptr)
        {
            allocator->deallocate(ptr, 0);
        }

        // Should not crash
    }

    END_TEST();
}

XSIGMATEST(CPUMemoryAllocators, comprehensive_tests)
{
    // Test basic allocation and deallocation
    XSIGMATEST_CALL(CPUMemoryAllocators, basic_allocation_deallocation);

    // Test zero size allocation
    XSIGMATEST_CALL(CPUMemoryAllocators, zero_size_allocation);

    END_TEST();
}
