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

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "common/configure.h"
#include "common/macros.h"
#include "logging/logger.h"
#include "memory/cpu/helper/memory_allocator.h"
#include "util/exception.h"

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

/**
 * @brief Test fixture for CPU memory allocator testing
 */
class cpu_memory_allocator_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize all available allocators
        allocators_.clear();

        // Always add XSigma's default allocator
        allocators_.emplace_back(std::make_unique<xsigma_cpu_test_allocator>());

        // Add standard aligned allocator
        allocators_.emplace_back(std::make_unique<standard_aligned_test_allocator>());

#ifdef XSIGMA_ENABLE_MIMALLOC
        allocators_.emplace_back(std::make_unique<mimalloc_test_allocator>());
#endif

#ifdef XSIGMA_ENABLE_TBB
        allocators_.emplace_back(std::make_unique<tbb_scalable_test_allocator>());
#endif

        XSIGMA_LOG_INFO("Initialized {} allocators for testing", allocators_.size());
        for (const auto& alloc : allocators_)
        {
            XSIGMA_LOG_INFO("  - {}", alloc->name());
        }
    }

    void TearDown() override { allocators_.clear(); }

    std::vector<std::unique_ptr<allocator_test_interface>> allocators_;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(cpu_memory_allocator_test, basic_allocation_deallocation)
{
    XSIGMA_LOG_INFO("Testing basic allocation and deallocation...");

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        // Test various sizes
        std::vector<std::size_t> sizes = {1, 8, 16, 64, 256, 1024, 4096, 65536};

        for (std::size_t size : sizes)
        {
            void* ptr = allocator->allocate(size);
            EXPECT_NE(nullptr, ptr) << "Failed to allocate " << size << " bytes";

            if (ptr != nullptr)
            {
                // Write to memory to ensure it's accessible
                std::memset(ptr, 0xAA, size);

                // Verify memory is writable and readable
                auto* byte_ptr = static_cast<unsigned char*>(ptr);
                for (std::size_t i = 0; i < size; ++i)
                {
                    EXPECT_EQ(0xAA, byte_ptr[i]) << "Memory corruption at offset " << i;
                }

                allocator->deallocate(ptr, size);
            }
        }
    }
}

TEST_F(cpu_memory_allocator_test, zero_size_allocation)
{
    XSIGMA_LOG_INFO("Testing zero-size allocation behavior...");

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        void* ptr = allocator->allocate(0);
        // Zero-size allocation behavior is implementation-defined
        // Some allocators return nullptr, others return a valid pointer

        if (ptr != nullptr)
        {
            allocator->deallocate(ptr, 0);
        }

        // Should not crash
        SUCCEED();
    }
}

TEST_F(cpu_memory_allocator_test, null_pointer_deallocation)
{
    XSIGMA_LOG_INFO("Testing null pointer deallocation safety...");

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        // Should not crash when deallocating nullptr
        EXPECT_NO_THROW(allocator->deallocate(nullptr, 0));
        EXPECT_NO_THROW(allocator->deallocate(nullptr, 1024));
    }
}

// =============================================================================
// Alignment Tests
// =============================================================================

TEST_F(cpu_memory_allocator_test, memory_alignment_verification)
{
    XSIGMA_LOG_INFO("Testing memory alignment requirements...");

    std::vector<std::size_t> alignments = {16, 32, 64, 128, 256, 512, 1024};
    std::vector<std::size_t> sizes      = {64, 256, 1024, 4096};

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        for (std::size_t alignment : alignments)
        {
            if (!allocator->supports_alignment(alignment))
            {
                continue;  // Skip unsupported alignments
            }

            for (std::size_t size : sizes)
            {
                void* ptr = allocator->allocate(size, alignment);
                EXPECT_NE(nullptr, ptr) << "Failed to allocate " << size << " bytes with "
                                        << alignment << "-byte alignment";

                if (ptr != nullptr)
                {
                    // Verify alignment
                    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
                    EXPECT_EQ(0u, addr % alignment)
                        << "Memory not aligned to " << alignment << " bytes. Address: 0x"
                        << std::hex << addr << std::dec;

                    // Test memory accessibility
                    std::memset(ptr, 0x55, size);

                    allocator->deallocate(ptr, size);
                }
            }
        }
    }
}

TEST_F(cpu_memory_allocator_test, cache_line_alignment)
{
    XSIGMA_LOG_INFO("Testing cache line alignment (64-byte)...");

    constexpr std::size_t    cache_line_size = 64;
    std::vector<std::size_t> sizes           = {1, 32, 64, 128, 256, 1024, 4096};

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        if (!allocator->supports_alignment(cache_line_size))
        {
            continue;
        }

        for (std::size_t size : sizes)
        {
            void* ptr = allocator->allocate(size, cache_line_size);
            EXPECT_NE(nullptr, ptr);

            if (ptr != nullptr)
            {
                auto addr = reinterpret_cast<std::uintptr_t>(ptr);
                EXPECT_EQ(0u, addr % cache_line_size)
                    << "Memory not cache-line aligned. Address: 0x" << std::hex << addr << std::dec;

                allocator->deallocate(ptr, size);
            }
        }
    }
}

// =============================================================================
// Large Allocation Tests
// =============================================================================

TEST_F(cpu_memory_allocator_test, large_allocations)
{
    XSIGMA_LOG_INFO("Testing large memory allocations...");

    std::vector<std::size_t> large_sizes = {
        1024 * 1024,       // 1 MB
        16 * 1024 * 1024,  // 16 MB
        64 * 1024 * 1024   // 64 MB
    };

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        for (std::size_t size : large_sizes)
        {
            void* ptr = allocator->allocate(size);

            if (ptr != nullptr)
            {
                // Test first and last pages
                auto* byte_ptr = static_cast<unsigned char*>(ptr);

                // Write to first page
                std::memset(byte_ptr, 0xCC, 4096);

                // Write to last page
                std::memset(byte_ptr + size - 4096, 0xDD, 4096);

                // Verify first page
                for (std::size_t i = 0; i < 4096; ++i)
                {
                    EXPECT_EQ(0xCC, byte_ptr[i]) << "First page corruption at offset " << i;
                }

                // Verify last page
                for (std::size_t i = size - 4096; i < size; ++i)
                {
                    EXPECT_EQ(0xDD, byte_ptr[i]) << "Last page corruption at offset " << i;
                }

                allocator->deallocate(ptr, size);
            }
            else
            {
                XSIGMA_LOG_WARNING(
                    "Failed to allocate {} MB with {}", size / (1024 * 1024), allocator->name());
            }
        }
    }
}

// =============================================================================
// Memory Pattern Tests
// =============================================================================

TEST_F(cpu_memory_allocator_test, memory_pattern_integrity)
{
    XSIGMA_LOG_INFO("Testing memory pattern integrity...");

    constexpr std::size_t test_size    = 8192;
    constexpr std::size_t pattern_size = 256;

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        void* ptr = allocator->allocate(test_size);
        EXPECT_NE(nullptr, ptr);

        if (ptr != nullptr)
        {
            auto* byte_ptr = static_cast<unsigned char*>(ptr);

            // Fill with repeating pattern
            for (std::size_t i = 0; i < test_size; ++i)
            {
                byte_ptr[i] = static_cast<unsigned char>(i % pattern_size);
            }

            // Verify pattern
            for (std::size_t i = 0; i < test_size; ++i)
            {
                unsigned char expected = static_cast<unsigned char>(i % pattern_size);
                EXPECT_EQ(expected, byte_ptr[i])
                    << "Pattern mismatch at offset " << i << " (expected "
                    << static_cast<int>(expected) << ", got " << static_cast<int>(byte_ptr[i])
                    << ")";
            }

            allocator->deallocate(ptr, test_size);
        }
    }
}

// =============================================================================
// Multi-threading Tests
// =============================================================================

TEST_F(cpu_memory_allocator_test, concurrent_allocation_deallocation)
{
    XSIGMA_LOG_INFO("Testing concurrent allocation and deallocation...");

    constexpr std::size_t num_threads            = 4;
    constexpr std::size_t allocations_per_thread = 1000;
    constexpr std::size_t allocation_size        = 1024;

    for (const auto& allocator : allocators_)
    {
        if (!allocator->is_thread_safe())
        {
            continue;  // Skip non-thread-safe allocators
        }

        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        std::atomic<std::size_t> successful_allocations{0};
        std::atomic<std::size_t> successful_deallocations{0};
        std::vector<std::thread> threads;

        auto worker = [&]()
        {
            std::vector<void*> ptrs;
            ptrs.reserve(allocations_per_thread);

            // Allocation phase
            for (std::size_t i = 0; i < allocations_per_thread; ++i)
            {
                void* ptr = allocator->allocate(allocation_size);
                if (ptr != nullptr)
                {
                    ptrs.push_back(ptr);
                    successful_allocations.fetch_add(1, std::memory_order_relaxed);

                    // Write pattern to verify memory integrity
                    std::memset(ptr, static_cast<int>(i & 0xFF), allocation_size);
                }
            }

            // Deallocation phase
            for (void* ptr : ptrs)
            {
                allocator->deallocate(ptr, allocation_size);
                successful_deallocations.fetch_add(1, std::memory_order_relaxed);
            }
        };

        // Launch threads
        for (std::size_t i = 0; i < num_threads; ++i)
        {
            threads.emplace_back(worker);
        }

        // Wait for completion
        for (auto& thread : threads)
        {
            thread.join();
        }

        // Verify results
        std::size_t expected_total = num_threads * allocations_per_thread;
        EXPECT_GE(successful_allocations.load(), expected_total * 0.95)
            << "Too many allocation failures";
        EXPECT_EQ(successful_allocations.load(), successful_deallocations.load())
            << "Allocation/deallocation count mismatch";

        XSIGMA_LOG_INFO(
            "Allocator {}: {}/{} successful allocations",
            allocator->name(),
            successful_allocations.load(),
            expected_total);
    }
}

TEST_F(cpu_memory_allocator_test, thread_local_allocation_patterns)
{
    XSIGMA_LOG_INFO("Testing thread-local allocation patterns...");

    constexpr std::size_t num_threads            = 8;
    constexpr std::size_t allocations_per_thread = 500;

    for (const auto& allocator : allocators_)
    {
        if (!allocator->is_thread_safe())
        {
            continue;
        }

        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        std::atomic<bool>        test_failed{false};
        std::vector<std::thread> threads;

        auto worker = [&](std::size_t thread_id)
        {
            std::mt19937                               rng(static_cast<unsigned int>(thread_id));
            std::uniform_int_distribution<std::size_t> size_dist(64, 4096);

            std::vector<std::pair<void*, std::size_t>> allocations;
            allocations.reserve(allocations_per_thread);

            // Mixed allocation/deallocation pattern
            for (std::size_t i = 0; i < allocations_per_thread; ++i)
            {
                std::size_t size = size_dist(rng);
                void*       ptr  = allocator->allocate(size);

                if (ptr != nullptr)
                {
                    // Fill with thread-specific pattern
                    std::memset(ptr, static_cast<int>(thread_id), size);
                    allocations.emplace_back(ptr, size);

                    // Randomly deallocate some allocations
                    if (allocations.size() > 10 && (rng() % 4) == 0)
                    {
                        std::size_t idx          = rng() % allocations.size();
                        auto [old_ptr, old_size] = allocations[idx];

                        // Verify pattern before deallocation
                        auto* byte_ptr = static_cast<unsigned char*>(old_ptr);
                        for (std::size_t j = 0; j < old_size; ++j)
                        {
                            if (byte_ptr[j] != static_cast<unsigned char>(thread_id))
                            {
                                test_failed.store(true, std::memory_order_relaxed);
                                return;
                            }
                        }

                        allocator->deallocate(old_ptr, old_size);
                        allocations.erase(allocations.begin() + idx);
                    }
                }
            }

            // Clean up remaining allocations
            for (auto [ptr, size] : allocations)
            {
                allocator->deallocate(ptr, size);
            }
        };

        // Launch threads
        for (std::size_t i = 0; i < num_threads; ++i)
        {
            threads.emplace_back(worker, i);
        }

        // Wait for completion
        for (auto& thread : threads)
        {
            thread.join();
        }

        EXPECT_FALSE(test_failed.load()) << "Memory corruption detected in thread-local test";
    }
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================
TEST_F(cpu_memory_allocator_test, stress_test_mixed_sizes)
{
    XSIGMA_LOG_INFO("Running stress test with mixed allocation sizes...");

    constexpr std::size_t num_iterations = 10000;

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        std::mt19937                               rng(42);  // Fixed seed for reproducibility
        std::uniform_int_distribution<std::size_t> size_dist(1, 65536);
        std::vector<std::pair<void*, std::size_t>> active_allocations;

        std::size_t total_allocated  = 0;
        std::size_t peak_allocations = 0;

        for (std::size_t i = 0; i < num_iterations; ++i)
        {
            if (active_allocations.empty() || (rng() % 3) != 0)
            {
                // Allocate
                std::size_t size = size_dist(rng);
                void*       ptr  = allocator->allocate(size);

                if (ptr != nullptr)
                {
                    // Fill with pattern
                    std::memset(ptr, static_cast<int>(i & 0xFF), size);
                    active_allocations.emplace_back(ptr, size);
                    total_allocated += size;
                    peak_allocations = std::max(peak_allocations, active_allocations.size());
                }
            }
            else
            {
                // Deallocate random allocation
                std::size_t idx  = rng() % active_allocations.size();
                auto [ptr, size] = active_allocations[idx];

                allocator->deallocate(ptr, size);
                active_allocations.erase(active_allocations.begin() + idx);
                total_allocated -= size;
            }
        }

        // Clean up remaining allocations
        for (auto [ptr, size] : active_allocations)
        {
            allocator->deallocate(ptr, size);
        }

        XSIGMA_LOG_INFO(
            "Allocator {}: Peak {} allocations, {} total bytes allocated",
            allocator->name(),
            peak_allocations,
            total_allocated);
    }
}

// =============================================================================
// Performance Comparison Tests
// =============================================================================

TEST_F(cpu_memory_allocator_test, allocation_speed_comparison)
{
    XSIGMA_LOG_INFO("Comparing allocation speeds across allocators...");

    constexpr std::size_t num_allocations = 10000;
    constexpr std::size_t allocation_size = 1024;

    for (const auto& allocator : allocators_)
    {
        SCOPED_TRACE("Allocator: " + std::string(allocator->name()));

        std::vector<void*> ptrs;
        ptrs.reserve(num_allocations);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Allocation phase
        for (std::size_t i = 0; i < num_allocations; ++i)
        {
            void* ptr = allocator->allocate(allocation_size);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        auto mid_time = std::chrono::high_resolution_clock::now();

        // Deallocation phase
        for (void* ptr : ptrs)
        {
            allocator->deallocate(ptr, allocation_size);
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        auto alloc_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time);
        auto dealloc_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - mid_time);

        XSIGMA_LOG_INFO(
            "Allocator {}: {} successful allocations, alloc: {}μs, dealloc: {}μs",
            allocator->name(),
            ptrs.size(),
            alloc_duration.count(),
            dealloc_duration.count());

        // Verify we got most allocations
        EXPECT_GE(ptrs.size(), num_allocations * 0.95) << "Too many allocation failures";
    }
}

}  // namespace testing
}  // namespace xsigma
