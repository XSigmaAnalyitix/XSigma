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

#include <cstdint>      // for uint8_t, uint64_t, uintptr_t, SIZE_MAX, UINT64_MAX, int64_t
#include <cstdlib>      // for size_t, free, malloc
#include <cstring>      // for memset
#include <functional>   // for function, _Func_class
#include <limits>       // for numeric_limits
#include <memory>       // for make_unique, unique_ptr, _Simple_types, allocator
#include <optional>     // for optional
#include <string>       // for char_traits, operator<<, string, operator+, to_string, basic...
#include <type_traits>  // for is_trivial
#include <utility>      // for move, min, max, max_element, min_element
#include <vector>       // for vector, _Vector_iterator, _Vector_const_iterator

#include "common/pointer.h"        // for make_ptr_unique_mutable
#include "memory/cpu/allocator.h"  // for sub_allocator, allocation_attributes, Allocator, allocator_m...
#include "memory/cpu/allocator_bfc.h"     // for allocator_bfc
#include "memory/cpu/allocator_device.h"  // for allocator_device
#include "memory/cpu/allocator_pool.h"  // for basic_cpu_allocator, allocator_pool, NoopRounder, round_up_i...
#include "memory/cpu/allocator_tracking.h"  // for allocator_tracking, enhanced_alloc_record, tracking_log_level
#include "memory/cpu/helper/memory_allocator.h"  // for free, allocate
#include "memory/cpu/helper/process_state.h"     // for process_state
#include "memory/unified_memory_stats.h"  // for atomic_timing_stats, unified_resource_stats, memory_fragment...
#include "xsigmaTest.h"  // for XSIGMATEST_CALL, XSIGMATEST_VOID, END_TEST, XSIGMATEST

using namespace xsigma;
namespace xsigma
{

// Helper class for testing allocators (trivial type for allocator_typed)
struct TestStruct
{
    int    x;
    double y;
    char   data[64];
};

// Helper function to check memory alignment
bool IsAligned(void* ptr, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

// Helper function to validate memory content
void FillMemory(void* ptr, size_t size, uint8_t pattern)
{
    memset(ptr, pattern, size);
}

bool ValidateMemory(void* ptr, size_t size, uint8_t pattern)
{
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i)
    {
        if (bytes[i] != pattern)
        {
            return false;
        }
    }
    return true;
}
}  // namespace xsigma

// ============================================================================
// ALLOCATOR_DEVICE TESTS
// ============================================================================

XSIGMATEST(CPUMemory, allocator_device_basic_allocation)
{
    auto allocator = std::make_unique<allocator_device>();

    // Test zero allocation
    void* ptr_zero = allocator->allocate_raw(64, 0);
    EXPECT_EQ(nullptr, ptr_zero);

    // Test small allocation
    void* ptr_small = allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr_small);

    // Verify memory is writable
    if (ptr_small)
    {
        std::memset(ptr_small, 0xAA, 1024);
        EXPECT_EQ(static_cast<unsigned char*>(ptr_small)[0], 0xAA);
        EXPECT_EQ(static_cast<unsigned char*>(ptr_small)[1023], 0xAA);
        allocator->deallocate_raw(ptr_small);
    }

    // Test large allocation
    void* ptr_large = allocator->allocate_raw(64, 1024 * 1024);  // 1MB
    EXPECT_NE(nullptr, ptr_large);

    if (ptr_large)
    {
        // Verify memory is writable
        std::memset(ptr_large, 0x55, 1024 * 1024);
        EXPECT_EQ(static_cast<unsigned char*>(ptr_large)[0], 0x55);
        EXPECT_EQ(static_cast<unsigned char*>(ptr_large)[1024 * 1024 - 1], 0x55);
        allocator->deallocate_raw(ptr_large);
    }

    END_TEST();
}

XSIGMATEST(CPUMemory, allocator_device_interface)
{
    auto allocator = std::make_unique<allocator_device>();

    // Test Name() method
    std::string name = allocator->Name();
    EXPECT_EQ("allocator_device", name);

    // Test GetMemoryType() method
    allocator_memory_enum memory_type = allocator->GetMemoryType();
    EXPECT_EQ(allocator_memory_enum::HOST_PINNED, memory_type);

    // Test TracksAllocationSizes() - should return false by default
    EXPECT_FALSE(allocator->TracksAllocationSizes());

    // Test AllocatesOpaqueHandle() - should return false
    EXPECT_FALSE(allocator->AllocatesOpaqueHandle());

    // Test GetStats() - should return nullopt by default
    auto stats = allocator->GetStats();
    EXPECT_FALSE(stats.has_value());

    // Test ClearStats() - should return false by default
    EXPECT_FALSE(allocator->ClearStats());

    END_TEST();
}

XSIGMATEST(CPUMemory, allocator_device_memory_type)
{
    auto allocator = std::make_unique<allocator_device>();

    // Verify memory type is HOST_PINNED
    EXPECT_EQ(allocator_memory_enum::HOST_PINNED, allocator->GetMemoryType());

    END_TEST();
}

XSIGMATEST(CPUMemory, allocator_device_error_handling)
{
    auto allocator = std::make_unique<allocator_device>();

    // Test deallocate with nullptr (should not crash)
    allocator->deallocate_raw(nullptr);

    // Test zero allocation
    void* ptr_zero = allocator->allocate_raw(64, 0);
    EXPECT_EQ(nullptr, ptr_zero);

    // Test very large allocation (may fail gracefully)
    void* ptr_huge = allocator->allocate_raw(64, SIZE_MAX);
    if (ptr_huge)
    {
        allocator->deallocate_raw(ptr_huge);
    }
    // If it doesn't throw, that's also acceptable

    END_TEST();
}

XSIGMATEST(CPUMemory, allocator_bfc_basic_allocation)
{
    // Create a BFC allocator with 1MB memory limit
    const size_t memory_limit  = 1024 * 1024;  // 1MB
    auto         sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    allocator_bfc allocator(std::move(sub_allocator), memory_limit, "test_bfc", opts);

    // Test basic allocation
    void* ptr1 = allocator.allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(IsAligned(ptr1, 64));

    // Test memory content
    FillMemory(ptr1, 1024, 0xAA);
    EXPECT_TRUE(ValidateMemory(ptr1, 1024, 0xAA));

    // Test deallocation
    allocator.deallocate_raw(ptr1);

    // Test multiple allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator.allocate_raw(32, 512);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(IsAligned(ptr, 32));
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs)
    {
        allocator.deallocate_raw(ptr);
    }

    END_TEST();
}
// Test allocator_bfc edge cases
XSIGMATEST_VOID(BFCAllocatorTest, EdgeCases)
{
    const size_t memory_limit  = 1024 * 1024;  // 1MB
    auto         sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    allocator_bfc allocator(std::move(sub_allocator), memory_limit, "test_bfc_edge", opts);

    // Test zero-size allocation
    void* ptr_zero = allocator.allocate_raw(64, 0);
    EXPECT_EQ(nullptr, ptr_zero);  // Should return nullptr for zero size

    // Test large alignment
    void* ptr_aligned = allocator.allocate_raw(4096, 1024);
    EXPECT_NE(nullptr, ptr_aligned);
    EXPECT_TRUE(IsAligned(ptr_aligned, 4096));
    allocator.deallocate_raw(ptr_aligned);

    // Test very small allocation
    void* ptr_small = allocator.allocate_raw(1, 1);
    EXPECT_NE(nullptr, ptr_small);
    allocator.deallocate_raw(ptr_small);

    // Test allocation statistics if available
    auto stats = allocator.GetStats();
    if (stats.has_value())
    {
        EXPECT_GE(stats->num_allocs, 0);
        EXPECT_GE(stats->bytes_in_use, 0);
        EXPECT_GE(stats->peak_bytes_in_use, 0);
    }
}

// Test allocator_bfc memory tracking
XSIGMATEST_VOID(BFCAllocatorTest, MemoryTracking)
{
    const size_t memory_limit  = 1024 * 1024;  // 1MB
    auto         sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    allocator_bfc allocator(std::move(sub_allocator), memory_limit, "test_bfc_tracking", opts);

    // Test allocation size tracking
    if (allocator.TracksAllocationSizes())
    {
        void* ptr = allocator.allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);

        size_t requested_size = allocator.RequestedSize(ptr);
        size_t allocated_size = allocator.AllocatedSize(ptr);

        EXPECT_EQ(requested_size, 1024);
        EXPECT_GE(allocated_size, 1024);

        allocator.deallocate_raw(ptr);
    }
}
// Test pool basic functionality
XSIGMATEST_VOID(PoolAllocatorTest, BasicFunctionality)
{
    // Create a pool allocator with size limit of 5
    auto sub_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto size_rounder = util::make_ptr_unique_mutable<NoopRounder>();

    allocator_pool pool(5, false, std::move(sub_allocator), std::move(size_rounder), "test_pool");

    // Test basic allocation
    void* ptr1 = pool.allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(IsAligned(ptr1, 64));

    // Test memory content
    FillMemory(ptr1, 1024, 0xBB);
    EXPECT_TRUE(ValidateMemory(ptr1, 1024, 0xBB));

    // Test deallocation (should go to pool)
    pool.deallocate_raw(ptr1);

    // Test reallocation (should come from pool)
    void* ptr2 = pool.allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr2);

    // Clean up
    pool.deallocate_raw(ptr2);
    pool.Clear();
}

// Test pool zero-size handling
XSIGMATEST_VOID(PoolAllocatorTest, ZeroSizeHandling)
{
    auto sub_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto size_rounder = util::make_ptr_unique_mutable<NoopRounder>();

    allocator_pool pool(
        2, false, std::move(sub_allocator), std::move(size_rounder), "test_pool_zero");

    // Test zero-size allocation
    void* ptr_zero = pool.allocate_raw(4, 0);
    EXPECT_EQ(nullptr, ptr_zero);  // Should return nullptr for zero size

    // Should not crash on nullptr deallocation
    pool.deallocate_raw(nullptr);

    // Test normal allocations still work
    void* ptr1 = pool.allocate_raw(4, 64);
    void* ptr2 = pool.allocate_raw(4, 128);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_NE(nullptr, ptr2);

    pool.deallocate_raw(ptr1);
    pool.deallocate_raw(ptr2);
    pool.Clear();
}

// Test pool alignment requirements
XSIGMATEST_VOID(PoolAllocatorTest, AlignmentRequirements)
{
    auto sub_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto size_rounder = util::make_ptr_unique_mutable<NoopRounder>();

    allocator_pool pool(
        0, false, std::move(sub_allocator), std::move(size_rounder), "test_pool_alignment");

    // Test various alignment requirements
    for (int i = 0; i < 12; ++i)  // Test up to 4KB alignment
    {
        size_t alignment = size_t{1} << i;
        void*  ptr       = pool.allocate_raw(alignment, 256);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(IsAligned(ptr, alignment));

        // Test memory access
        FillMemory(ptr, 256, static_cast<uint8_t>(i));
        EXPECT_TRUE(ValidateMemory(ptr, 256, static_cast<uint8_t>(i)));

        pool.deallocate_raw(ptr);
    }

    pool.Clear();
}

// Test allocator_tracking functionality
XSIGMATEST_VOID(TrackingAllocatorTest, BasicTracking)
{
    // Create a base allocator
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        10,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "base_pool");

    // Create tracking allocator (use pointer due to protected destructor)
    auto tracker = new xsigma::allocator_tracking(pool.get(), true);

    // Test basic allocation tracking
    void* ptr1 = tracker->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(IsAligned(ptr1, 64));

    // Test memory content
    FillMemory(ptr1, 1024, 0xCC);
    EXPECT_TRUE(ValidateMemory(ptr1, 1024, 0xCC));

    // Test statistics
    auto stats = tracker->GetStats();
    if (stats.has_value())
    {
        EXPECT_GT(stats->num_allocs, 0);
        EXPECT_GT(stats->bytes_in_use, 0);
    }

    // Test deallocation
    tracker->deallocate_raw(ptr1);

    // Test multiple allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = tracker->allocate_raw(32, 512);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();
}

// Test allocator stress scenarios
XSIGMATEST_VOID(AllocatorTest, StressTest)
{
#if 0
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        50,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "stress_pool");

    const int    num_threads            = 4;
    const int    allocations_per_thread = 100;
    const size_t max_alloc_size         = 4096;

    std::vector<std::thread>        threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch multiple threads doing allocations
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                thread_ptrs[t].reserve(allocations_per_thread);

                for (int i = 0; i < allocations_per_thread; ++i)
                {
                    size_t size      = (i % max_alloc_size) + 1;
                    size_t alignment = size_t{1} << (i % 8);  // 1, 2, 4, 8, 16, 32, 64, 128

                    void* ptr = pool->allocate_raw(alignment, size);
                    if (ptr != nullptr)
                    {
                        // Test memory access
                        FillMemory(ptr, size, static_cast<uint8_t>(t + i));
                        thread_ptrs[t].push_back(ptr);
                    }
                }
            });
    }

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    // Verify memory content and deallocate
    for (int t = 0; t < num_threads; ++t)
    {
        for (size_t i = 0; i < thread_ptrs[t].size(); ++i)
        {
            void*  ptr  = thread_ptrs[t][i];
            size_t size = (i % max_alloc_size) + 1;

            // Verify memory content
            EXPECT_TRUE(ValidateMemory(ptr, size, static_cast<uint8_t>(t + i)));

            pool->deallocate_raw(ptr);
        }
    }
#endif
}

// Test error handling and edge cases
XSIGMATEST_VOID(AllocatorTest, ErrorHandling)
{
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        10,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "error_pool");

    // Test null pointer deallocation (should not crash)
    pool->deallocate_raw(nullptr);

    // Test invalid alignment (should handle gracefully)
    void* ptr_invalid_align = pool->allocate_raw(3, 1024);  // 3 is not power of 2
    // Implementation may round up alignment or handle differently
    if (ptr_invalid_align != nullptr)
    {
        pool->deallocate_raw(ptr_invalid_align);
    }

    // Test very large allocation (may fail, but should not crash)
    {
        void* ptr_large = pool->allocate_raw(64, SIZE_MAX / 2);
        if (ptr_large != nullptr)
        {
            pool->deallocate_raw(ptr_large);
        }
    };

    // Test double deallocation protection (if implemented)
    void* ptr_normal = pool->allocate_raw(64, 1024);
    if (ptr_normal != nullptr)
    {
        pool->deallocate_raw(ptr_normal);
        // Second deallocation should be handled gracefully
        pool->deallocate_raw(ptr_normal);
    }
}

// Test memory leak detection
XSIGMATEST_VOID(AllocatorTest, MemoryLeakDetection)
{
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        10,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "leak_pool");
    auto allocator_tracking = new xsigma::allocator_tracking(pool.get(), true);

    // Get initial statistics
    auto initial_stats = allocator_tracking->GetStats();

    // Allocate some memory
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator_tracking->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Check that allocations are tracked
    auto mid_stats = allocator_tracking->GetStats();
    if (initial_stats.has_value() && mid_stats.has_value())
    {
        EXPECT_GT(mid_stats->num_allocs, initial_stats->num_allocs);
        EXPECT_GT(mid_stats->bytes_in_use, initial_stats->bytes_in_use);
    }

    // Deallocate all memory
    for (void* ptr : ptrs)
    {
        allocator_tracking->deallocate_raw(ptr);
    }

    // Check that memory is properly freed
    auto final_stats = allocator_tracking->GetStats();
    if (initial_stats.has_value() && final_stats.has_value())
    {
        EXPECT_EQ(final_stats->bytes_in_use, initial_stats->bytes_in_use);
    }

    // Properly cleanup tracking allocator by releasing reference
    allocator_tracking->GetRecordsAndUnRef();
}

// Test allocator statistics and monitoring
XSIGMATEST_VOID(AllocatorTest, StatisticsAndMonitoring)
{
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        10,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "stats_pool");
    auto allocator_tracking = new xsigma::allocator_tracking(pool.get(), true);

    // Test statistics collection
    auto stats = allocator_tracking->GetStats();
    if (stats.has_value())
    {
        EXPECT_GE(stats->num_allocs, 0);
        EXPECT_GE(stats->bytes_in_use, 0);
        EXPECT_GE(stats->peak_bytes_in_use, 0);
        EXPECT_GE(stats->largest_alloc_size, 0);
    }

    // Test allocation size tracking
    if (allocator_tracking->TracksAllocationSizes())
    {
        void* ptr = allocator_tracking->allocate_raw(64, 2048);
        EXPECT_NE(nullptr, ptr);

        size_t requested_size = allocator_tracking->RequestedSize(ptr);
        size_t allocated_size = allocator_tracking->AllocatedSize(ptr);

        EXPECT_EQ(requested_size, 2048);
        EXPECT_GE(allocated_size, 2048);

        allocator_tracking->deallocate_raw(ptr);
    }

    // Test allocator name (allocator_tracking returns the underlying allocator's name)
    EXPECT_EQ(std::string(allocator_tracking->Name()), std::string("stats_pool"));

    // Properly cleanup tracking allocator by releasing reference
    allocator_tracking->GetRecordsAndUnRef();
}

// ============================================================================
// MEMORY PORT ABSTRACTION LAYER TESTS (mem.h)
// ============================================================================

// Test memory port basic functionality
XSIGMATEST_VOID(MemoryPortTest, BasicMemoryOperations)
{
    XSIGMA_LOG_INFO("Testing memory port basic operations...");

    // Test aligned malloc/free (these are available)
    void* ptr2 = cpu::memory_allocator::allocate(2048, 64);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_TRUE(IsAligned(ptr2, 64));
    FillMemory(ptr2, 2048, 0xBB);
    EXPECT_TRUE(ValidateMemory(ptr2, 2048, 0xBB));
    cpu::memory_allocator::free(ptr2);

    XSIGMA_LOG_INFO("Memory port basic operations tests completed successfully");
}

// Test memory port alignment requirements
XSIGMATEST_VOID(MemoryPortTest, AlignmentRequirements)
{
    XSIGMA_LOG_INFO("Testing memory port alignment requirements...");

    // Test various alignment values
    std::vector<int> alignments = {8, 16, 32, 64, 128, 256, 512, 1024};

    for (int alignment : alignments)
    {
        void* ptr = cpu::memory_allocator::allocate(1024, alignment);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(IsAligned(ptr, alignment));

        // Test memory access
        FillMemory(ptr, 1024, static_cast<uint8_t>(alignment & 0xFF));
        EXPECT_TRUE(ValidateMemory(ptr, 1024, static_cast<uint8_t>(alignment & 0xFF)));

        cpu::memory_allocator::free(ptr);
    }

    XSIGMA_LOG_INFO("Memory port alignment requirements tests completed successfully");
}

// Test memory port edge cases
XSIGMATEST_VOID(MemoryPortTest, EdgeCases)
{
    XSIGMA_LOG_INFO("Testing memory port edge cases...");

    // Test null pointer free (should not crash)
    cpu::memory_allocator::free(nullptr);

    // Test zero-size allocation
    void* ptr_zero = cpu::memory_allocator::allocate(0, 64);
    // Implementation may return nullptr or valid pointer for zero size
    if (ptr_zero != nullptr)
    {
        cpu::memory_allocator::free(ptr_zero);
    }

    XSIGMA_LOG_INFO("Memory port edge cases tests completed successfully");
}

// Test memory port system information
XSIGMATEST_VOID(MemoryPortTest, SystemInformation)
{
    XSIGMA_LOG_INFO("Testing memory port system information...");

    // Test NUMA affinity constant
    EXPECT_EQ(xsigma::NUMANOAFFINITY, -1);

    XSIGMA_LOG_INFO("Memory port system information tests completed successfully");
}

// ============================================================================
// ALLOCATION ATTRIBUTES TESTS (allocator.h)
// ============================================================================

// Test allocation attributes construction and behavior
XSIGMATEST_VOID(AllocationAttributesTest, ConstructionAndBehavior)
{
    XSIGMA_LOG_INFO("Testing allocation attributes construction and behavior...");

    // Test default construction
    xsigma::allocation_attributes default_attrs;
    EXPECT_TRUE(default_attrs.retry_on_failure);
    EXPECT_FALSE(default_attrs.allocation_will_be_logged);
    EXPECT_EQ(nullptr, default_attrs.freed_by_func);

    // Test parameterized construction
    std::function<uint64_t()>     timing_func = []() { return 12345; };
    xsigma::allocation_attributes custom_attrs(true, true, &timing_func);
    EXPECT_TRUE(custom_attrs.retry_on_failure);
    EXPECT_TRUE(custom_attrs.allocation_will_be_logged);
    EXPECT_EQ(&timing_func, custom_attrs.freed_by_func);

    // Test move construction
    xsigma::allocation_attributes moved_attrs = std::move(custom_attrs);
    EXPECT_TRUE(moved_attrs.retry_on_failure);
    EXPECT_TRUE(moved_attrs.allocation_will_be_logged);
    EXPECT_EQ(&timing_func, moved_attrs.freed_by_func);

    // Test move assignment
    xsigma::allocation_attributes assigned_attrs;
    assigned_attrs = std::move(moved_attrs);
    EXPECT_TRUE(assigned_attrs.retry_on_failure);
    EXPECT_TRUE(assigned_attrs.allocation_will_be_logged);
    EXPECT_EQ(&timing_func, assigned_attrs.freed_by_func);

    XSIGMA_LOG_INFO("Allocation attributes construction and behavior tests completed successfully");
}

// Test allocation attributes with timing constraints
XSIGMATEST_VOID(AllocationAttributesTest, TimingConstraints)
{
    XSIGMA_LOG_INFO("Testing allocation attributes timing constraints...");

    uint64_t                  counter     = 0;
    std::function<uint64_t()> timing_func = [&counter]() { return ++counter; };

    xsigma::allocation_attributes attrs(true, true, &timing_func);

    // Test timing function calls
    EXPECT_EQ(&timing_func, attrs.freed_by_func);

    if (attrs.freed_by_func)
    {
        uint64_t time1 = (*attrs.freed_by_func)();
        uint64_t time2 = (*attrs.freed_by_func)();

        EXPECT_EQ(time1, 1);
        EXPECT_EQ(time2, 2);
        EXPECT_GT(time2, time1);
    }

    XSIGMA_LOG_INFO("Allocation attributes timing constraints tests completed successfully");
}

// ============================================================================
// PROCESS STATE TESTS (process_state.h) - COMMENTED OUT DUE TO LINKING ISSUES
// ============================================================================
// Note: These tests are commented out due to linking issues with FLAGS_brain_gpu_record_mem_types
// and other dependencies that are not available in the current build configuration

// Test process state singleton functionality and thread safety
XSIGMATEST_VOID(ProcessStateTest, SingletonFunctionality)
{
    XSIGMA_LOG_INFO("Testing process state singleton functionality...");

    // Test singleton access
    xsigma::process_state* state1 = xsigma::process_state::singleton();
    xsigma::process_state* state2 = xsigma::process_state::singleton();

    EXPECT_NE(nullptr, state1);
    EXPECT_EQ(state1, state2);  // Should be the same instance

    state1->TestOnlyReset();

    XSIGMA_LOG_INFO("Process state singleton functionality tests completed successfully");
}

// Test CPU allocator retrieval and management
XSIGMATEST_VOID(ProcessStateTest, CPUAllocatorManagement)
{
    XSIGMA_LOG_INFO("Testing CPU allocator retrieval and management...");

    xsigma::process_state* state = xsigma::process_state::singleton();

    // Test CPU allocator access for different NUMA nodes
    xsigma::Allocator* cpu_alloc1 = state->GetCPUAllocator(0);
    xsigma::Allocator* cpu_alloc2 = state->GetCPUAllocator(xsigma::NUMANOAFFINITY);

    auto ptr = cpu_alloc1->allocate_raw(64, 1000);
    cpu_alloc1->deallocate_raw(ptr);

    EXPECT_NE(nullptr, cpu_alloc1);
    EXPECT_NE(nullptr, cpu_alloc2);

    // Test allocator consistency
    xsigma::Allocator* cpu_alloc3 = state->GetCPUAllocator(0);
    EXPECT_EQ(cpu_alloc1, cpu_alloc3);  // Should return same allocator for same node

    XSIGMA_LOG_INFO("CPU allocator retrieval and management tests completed successfully");
}
// Test NUMA enablement and affinity handling
XSIGMATEST_VOID(ProcessStateTest, NUMAHandling)
{
    XSIGMA_LOG_INFO("Testing NUMA enablement and affinity handling...");

    xsigma::process_state* state = xsigma::process_state::singleton();

    // Test NUMA enablement (should not crash)
    state->EnableNUMA();

    // Test allocator access after NUMA enablement
    xsigma::Allocator* numa_alloc = state->GetCPUAllocator(0);
    EXPECT_NE(nullptr, numa_alloc);

    // Test NUMANOAFFINITY handling
    xsigma::Allocator* no_affinity_alloc = state->GetCPUAllocator(xsigma::NUMANOAFFINITY);
    EXPECT_NE(nullptr, no_affinity_alloc);

    XSIGMA_LOG_INFO("NUMA enablement and affinity handling tests completed successfully");
}

// Test memory description and pointer type detection
XSIGMATEST_VOID(ProcessStateTest, MemoryDescription)
{
    XSIGMA_LOG_INFO("Testing memory description and pointer type detection...");

    xsigma::process_state* state = xsigma::process_state::singleton();

    // Test memory description for CPU pointer
    void* cpu_ptr = std::malloc(1024);
    EXPECT_NE(nullptr, cpu_ptr);

    auto mem_desc = state->PtrType(cpu_ptr);
    EXPECT_EQ(mem_desc.loc, xsigma::process_state::MemDesc::CPU);
    EXPECT_EQ(mem_desc.dev_index, 0);
    EXPECT_FALSE(mem_desc.gpu_registered);
    EXPECT_FALSE(mem_desc.nic_registered);

    // Test debug string generation
    std::string debug_str = mem_desc.debug_string();
    EXPECT_FALSE(debug_str.empty());

    // Test default MemDesc construction
    xsigma::process_state::MemDesc default_desc;
    EXPECT_EQ(default_desc.loc, xsigma::process_state::MemDesc::CPU);
    EXPECT_EQ(default_desc.dev_index, 0);
    EXPECT_FALSE(default_desc.gpu_registered);
    EXPECT_FALSE(default_desc.nic_registered);

    std::free(cpu_ptr);

    state->TestOnlyReset();
    XSIGMA_LOG_INFO("Memory description and pointer type detection tests completed successfully");
}

// Test visitor registration for allocation/deallocation callbacks
XSIGMATEST_VOID(ProcessStateTest, VisitorRegistration)
{
    XSIGMA_LOG_INFO("Testing visitor registration for allocation/deallocation callbacks...");

    xsigma::process_state* state = xsigma::process_state::singleton();

    // Test CPU allocator visitor registration
    bool alloc_visitor_called = false;
    bool free_visitor_called  = false;

    auto alloc_visitor = [&alloc_visitor_called](void* ptr, int index, size_t num_bytes)
    { alloc_visitor_called = true; };

    auto free_visitor = [&free_visitor_called](void* ptr, int index, size_t num_bytes)
    { free_visitor_called = true; };

    // Note: These must be called before GetCPUAllocator according to the API
    // In practice, we test that the calls don't crash
    state->AddCPUAllocVisitor(alloc_visitor);
    state->AddCPUFreeVisitor(free_visitor);

    xsigma::Allocator* cpu_alloc = state->GetCPUAllocator(0);

    auto ptr = cpu_alloc->allocate_raw(64, 1000);
    cpu_alloc->deallocate_raw(ptr);

    XSIGMA_LOG_INFO("Visitor registration tests completed successfully");
}

// ============================================================================
// TRACKING ALLOCATOR TESTS (allocator_tracking.h)
// ============================================================================

// Test memory allocation tracking and leak detection
XSIGMATEST_VOID(TrackingAllocatorTest, AllocationTracking)
{
    XSIGMA_LOG_INFO("Testing memory allocation tracking and leak detection...");

    // Create underlying BFC allocator for testing
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    xsigma::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 1024 * 1024, "test_tracking_bfc", opts);

    // Create tracking allocator with size tracking enabled (use pointer due to protected destructor)
    auto tracker = new xsigma::allocator_tracking(&underlying_alloc, true);

    // Test basic allocation tracking
    void* ptr1 = tracker->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);

    // Verify tracking capabilities
    EXPECT_TRUE(tracker->TracksAllocationSizes());
    EXPECT_EQ(tracker->RequestedSize(ptr1), 1024);
    EXPECT_GE(tracker->AllocatedSize(ptr1), 1024);

    // Test allocation ID generation
    int64_t alloc_id = tracker->AllocationId(ptr1);
    EXPECT_GT(alloc_id, 0);

    // Test multiple allocations
    void* ptr2 = tracker->allocate_raw(32, 512);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_NE(ptr1, ptr2);

    // Verify different allocation IDs
    int64_t alloc_id2 = tracker->AllocationId(ptr2);
    EXPECT_GT(alloc_id2, 0);
    EXPECT_NE(alloc_id, alloc_id2);

    // Test deallocation
    tracker->deallocate_raw(ptr1);
    tracker->deallocate_raw(ptr2);

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    XSIGMA_LOG_INFO("Memory allocation tracking and leak detection tests completed successfully");
}

// Test allocation statistics collection and reporting
XSIGMATEST_VOID(TrackingAllocatorTest, StatisticsCollection)
{
    XSIGMA_LOG_INFO("Testing allocation statistics collection and reporting...");

    // Create underlying allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    xsigma::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 1024 * 1024, "test_stats_bfc", opts);

    // Create tracking allocator (use pointer due to protected destructor)
    auto tracker = new xsigma::allocator_tracking(&underlying_alloc, true);

    // Get initial statistics
    auto initial_stats = tracker->GetStats();
    EXPECT_TRUE(initial_stats.has_value());

    // Get initial sizes
    auto [total_initial, high_initial, current_initial] = tracker->GetSizes();
    EXPECT_EQ(total_initial, 0);
    EXPECT_EQ(high_initial, 0);
    EXPECT_EQ(current_initial, 0);

    // Perform allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = tracker->allocate_raw(64, 1024 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Check updated statistics
    auto [total_after, high_after, current_after] = tracker->GetSizes();
    EXPECT_GT(total_after, 0);
    EXPECT_GT(high_after, 0);
    EXPECT_GT(current_after, 0);

    // Test statistics reset
    bool reset_success = tracker->ClearStats();
    EXPECT_TRUE(reset_success);

    // Clean up allocations
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    XSIGMA_LOG_INFO("Allocation statistics collection and reporting tests completed successfully");
}

// Test memory usage monitoring and bounds checking
XSIGMATEST_VOID(TrackingAllocatorTest, MemoryUsageMonitoring)
{
    XSIGMA_LOG_INFO("Testing memory usage monitoring and bounds checking...");

    // Create underlying allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    xsigma::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 1024 * 1024, "test_monitoring_bfc", opts);

    // Create tracking allocator (use pointer due to protected destructor)
    auto tracker = new xsigma::allocator_tracking(&underlying_alloc, true);

    // Test allocation records collection
    std::vector<void*> ptrs;
    for (int i = 0; i < 3; ++i)
    {
        void* ptr = tracker->allocate_raw(64, 512 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Get current allocation records
    auto current_records = tracker->GetCurrentRecords();
    EXPECT_EQ(current_records.size(), 3);

    // Verify record contents
    for (const auto& record : current_records)
    {
        EXPECT_GT(record.alloc_bytes, 0);
        EXPECT_GT(record.alloc_micros, 0);
    }

    // Test high watermark tracking
    auto [total_bytes, high_watermark, current_bytes] = tracker->GetSizes();
    EXPECT_GT(high_watermark, 0);
    EXPECT_EQ(current_bytes, high_watermark);  // Should be at peak

    // Deallocate one allocation
    tracker->deallocate_raw(ptrs[0]);
    ptrs.erase(ptrs.begin());

    // Check that current usage decreased but high watermark remained
    auto [total_after, high_after, current_after] = tracker->GetSizes();
    EXPECT_EQ(high_after, high_watermark);    // High watermark should not decrease
    EXPECT_LT(current_after, current_bytes);  // Current usage should decrease

    // Clean up remaining allocations
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    XSIGMA_LOG_INFO("Memory usage monitoring and bounds checking tests completed successfully");
}

// Test integration with underlying allocator implementations
XSIGMATEST_VOID(TrackingAllocatorTest, UnderlyingAllocatorIntegration)
{
    XSIGMA_LOG_INFO("Testing integration with underlying allocator implementations...");

    // Test with BFC allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    xsigma::allocator_bfc bfc_alloc(
        std::move(sub_allocator), 1024 * 1024, "test_integration_bfc", opts);
    auto bfc_tracker = new xsigma::allocator_tracking(&bfc_alloc, true);

    // Test name delegation
    std::string tracker_name = bfc_tracker->Name();
    EXPECT_FALSE(tracker_name.empty());

    // Test memory type delegation
    auto memory_type = bfc_tracker->GetMemoryType();
    EXPECT_EQ(memory_type, bfc_alloc.GetMemoryType());

    // Test allocation through tracking wrapper
    void* ptr = bfc_tracker->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);

    // Verify size tracking works
    EXPECT_TRUE(bfc_tracker->TracksAllocationSizes());
    EXPECT_EQ(bfc_tracker->RequestedSize(ptr), 1024);
    EXPECT_GE(bfc_tracker->AllocatedSize(ptr), 1024);

    bfc_tracker->deallocate_raw(ptr);

    // Test with pool allocator
    auto pool_sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});
    auto size_rounder = util::make_ptr_unique_mutable<NoopRounder>();

    xsigma::allocator_pool pool_alloc(
        10, false, std::move(pool_sub_allocator), std::move(size_rounder), "test_integration_pool");
    auto pool_tracker = new xsigma::allocator_tracking(&pool_alloc, false);  // No local tracking

    void* pool_ptr = pool_tracker->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, pool_ptr);

    pool_tracker->deallocate_raw(pool_ptr);

    // Properly cleanup tracking allocator by releasing reference
    bfc_tracker->GetRecordsAndUnRef();
    pool_tracker->GetRecordsAndUnRef();

    XSIGMA_LOG_INFO(
        "Integration with underlying allocator implementations tests completed successfully");
}

// Test enhanced tracking functionality with comprehensive analytics
XSIGMATEST_VOID(TrackingAllocatorTest, EnhancedTrackingAnalytics)
{
    XSIGMA_LOG_INFO("Testing enhanced tracking analytics and performance profiling...");

    // Create underlying allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    xsigma::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 2 * 1024 * 1024, "test_enhanced_bfc", opts);

    // Create tracking allocator with enhanced tracking enabled
    auto tracker =
        new xsigma::allocator_tracking(&underlying_alloc, true, true);  // Enhanced tracking enabled

    // Test enhanced timing statistics
    auto initial_timing = tracker->GetTimingStats();
    EXPECT_EQ(initial_timing.total_allocations, 0);
    EXPECT_EQ(initial_timing.total_deallocations, 0);

    // Perform multiple allocations to generate timing data
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = tracker->allocate_raw(64, 1024 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Test timing statistics after allocations
    auto post_alloc_timing = tracker->GetTimingStats();
    EXPECT_EQ(post_alloc_timing.total_allocations, 10);
    EXPECT_GT(post_alloc_timing.total_alloc_time_us, 0);
    EXPECT_GT(post_alloc_timing.max_alloc_time_us, 0);
    EXPECT_LT(post_alloc_timing.min_alloc_time_us, UINT64_MAX);

    // Test average allocation time calculation
    double avg_alloc_time = post_alloc_timing.average_alloc_time_us();
    EXPECT_GT(avg_alloc_time, 0.0);

    // Test enhanced allocation records
    auto enhanced_records = tracker->GetEnhancedRecords();
    EXPECT_EQ(enhanced_records.size(), 10);

    // Verify enhanced record data
    for (size_t i = 0; i < enhanced_records.size(); ++i)
    {
        const auto& record = enhanced_records[i];
        EXPECT_GT(record.requested_bytes, 0);
        EXPECT_GE(record.alloc_bytes, record.requested_bytes);
        EXPECT_GT(record.allocation_id, 0);
        EXPECT_GE(record.alloc_duration_us, 0);
        EXPECT_GT(record.alloc_micros, 0);
        EXPECT_EQ(record.alignment, 64);
    }

    // Test fragmentation metrics
    auto frag_metrics = tracker->GetFragmentationMetrics();
    EXPECT_GE(frag_metrics.fragmentation_ratio, 0.0);
    EXPECT_LE(frag_metrics.fragmentation_ratio, 1.0);
    EXPECT_GE(frag_metrics.wasted_bytes, 0);
    EXPECT_GE(frag_metrics.largest_free_block, 0);

    // Test efficiency metrics
    auto [memory_efficiency, allocation_efficiency, overall_efficiency] =
        tracker->GetEfficiencyMetrics();
    EXPECT_GE(memory_efficiency, 0.0);
    EXPECT_LE(memory_efficiency, 1.0);
    EXPECT_GE(allocation_efficiency, 0.0);
    EXPECT_LE(allocation_efficiency, 1.0);
    EXPECT_GE(overall_efficiency, 0.0);
    EXPECT_LE(overall_efficiency, 1.0);

    // Deallocate memory and test deallocation timing
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Test timing statistics after deallocations
    auto post_dealloc_timing = tracker->GetTimingStats();
    EXPECT_EQ(post_dealloc_timing.total_deallocations, 10);
    EXPECT_GE(post_dealloc_timing.total_dealloc_time_us, 0);
    EXPECT_GE(post_dealloc_timing.max_dealloc_time_us, 0);
    EXPECT_LT(post_dealloc_timing.min_dealloc_time_us, UINT64_MAX);

    // Test average deallocation time calculation
    double avg_dealloc_time = post_dealloc_timing.average_dealloc_time_us();
    EXPECT_GE(avg_dealloc_time, 0.0);

    // Verify enhanced records have deallocation timing
    auto final_records = tracker->GetEnhancedRecords();
    for (const auto& record : final_records)
    {
        EXPECT_GE(record.dealloc_duration_us, 0);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    XSIGMA_LOG_INFO(
        "Enhanced tracking analytics and performance profiling tests completed successfully");
}

// Test logging levels and comprehensive reporting
XSIGMATEST_VOID(TrackingAllocatorTest, LoggingAndReporting)
{
    XSIGMA_LOG_INFO("Testing logging levels and comprehensive reporting...");

    // Create underlying allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    xsigma::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 1024 * 1024, "test_logging_bfc", opts);

    // Create tracking allocator with enhanced tracking
    auto tracker = new xsigma::allocator_tracking(&underlying_alloc, true, true);

    // Test logging level configuration
    tracker->SetLoggingLevel(tracking_log_level::DEBUG);
    EXPECT_EQ(tracker->GetLoggingLevel(), tracking_log_level::DEBUG);

    tracker->SetLoggingLevel(tracking_log_level::INFO);
    EXPECT_EQ(tracker->GetLoggingLevel(), tracking_log_level::INFO);

    // Perform some allocations for report generation
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = tracker->allocate_raw(512 * (i + 1), 32);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Test comprehensive report generation
    std::string basic_report = tracker->GenerateReport(false);
    EXPECT_FALSE(basic_report.empty());
    EXPECT_NE(basic_report.find("Memory Allocation Tracking Report"), std::string::npos);
    EXPECT_NE(basic_report.find("Performance Statistics"), std::string::npos);
    EXPECT_NE(basic_report.find("Efficiency Metrics"), std::string::npos);

    // Test detailed report with allocations
    std::string detailed_report = tracker->GenerateReport(true);
    EXPECT_FALSE(detailed_report.empty());
    EXPECT_GT(detailed_report.length(), basic_report.length());
    EXPECT_NE(detailed_report.find("Recent Allocations"), std::string::npos);

    // Test timing statistics reset
    tracker->ResetTimingStats();
    auto reset_timing = tracker->GetTimingStats();
    EXPECT_EQ(reset_timing.total_allocations, 0);
    EXPECT_EQ(reset_timing.total_deallocations, 0);
    EXPECT_EQ(reset_timing.total_alloc_time_us, 0);
    EXPECT_EQ(reset_timing.total_dealloc_time_us, 0);

    // Clean up
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    XSIGMA_LOG_INFO("Logging levels and comprehensive reporting tests completed successfully");
}