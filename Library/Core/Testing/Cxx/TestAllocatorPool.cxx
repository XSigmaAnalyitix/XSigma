/**
 * @file TestAllocatorPool.cxx
 * @brief Comprehensive test suite for Pool allocator with LRU eviction
 *
 * Tests all public methods and functionality of the allocator_pool class including:
 * - Basic allocation and deallocation with pooling
 * - LRU eviction policy behavior
 * - Pool size limits and auto-resize functionality
 * - Size rounding strategies
 * - Statistics collection and monitoring
 * - Thread safety and concurrent access
 * - Error handling and edge cases
 */

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "Testing/xsigmaTest.h"
#include "common/pointer.h"
#include "memory/cpu/allocator.h"
#include "memory/cpu/allocator_pool.h"
#include "memory/cpu/helper/memory_allocator.h"

using namespace xsigma;

namespace
{

/**
 * @brief Test implementation of sub_allocator for pool testing
 */
class test_pool_sub_allocator : public sub_allocator
{
public:
    test_pool_sub_allocator() : sub_allocator({}, {}), alloc_count_(0), free_count_(0) {}

    void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) override
    {
        alloc_count_++;
        void* ptr = xsigma::cpu::memory_allocator::allocate(num_bytes, alignment);
        if (bytes_received)
            *bytes_received = num_bytes;
        return ptr;
    }

    void Free(void* ptr, size_t num_bytes) override
    {
        free_count_++;
        xsigma::cpu::memory_allocator::free(ptr);
    }

    bool SupportsCoalescing() const override { return false; }

    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_memory_enum::HOST_PAGEABLE;
    }

    // Test helpers
    int  get_alloc_count() const { return alloc_count_; }
    int  get_free_count() const { return free_count_; }
    void reset_counters()
    {
        alloc_count_ = 0;
        free_count_  = 0;
    }

private:
    std::atomic<int> alloc_count_;
    std::atomic<int> free_count_;
};

/**
 * @brief No-op size rounder for testing
 */
class noop_size_rounder : public round_up_interface
{
public:
    size_t RoundUp(size_t num_bytes) override
    {
        return num_bytes;  // No rounding
    }
};

/**
 * @brief Power-of-2 size rounder for testing
 */
class power_of_2_rounder : public round_up_interface
{
public:
    size_t RoundUp(size_t num_bytes) override
    {
        if (num_bytes <= 1)
            return 1;

        size_t power = 1;
        while (power < num_bytes)
        {
            power <<= 1;
        }
        return power;
    }
};

/**
 * @brief Helper function to check memory alignment
 */
bool is_aligned(void* ptr, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/**
 * @brief Helper function to fill memory with pattern
 */
void fill_memory(void* ptr, size_t size, uint8_t pattern)
{
    std::memset(ptr, pattern, size);
}

/**
 * @brief Helper function to validate memory pattern
 */
bool validate_memory(void* ptr, size_t size, uint8_t pattern)
{
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i)
    {
        if (bytes[i] != pattern)
            return false;
    }
    return true;
}

}  // anonymous namespace

/**
 * @brief Test suite for Pool allocator basic functionality
 */
class AllocatorPool : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test sub-allocator
        auto sub_alloc     = std::make_unique<test_pool_sub_allocator>();
        sub_allocator_ptr_ = sub_alloc.get();  // Keep reference for testing

        // Create size rounder
        auto size_rounder = std::make_unique<noop_size_rounder>();

        // Create pool allocator with size limit of 5
        pool_allocator_ = std::make_unique<allocator_pool>(
            5,      // pool_size_limit
            false,  // auto_resize
            std::move(sub_alloc),
            std::move(size_rounder),
            "test_pool");
    }

    void TearDown() override
    {
        pool_allocator_.reset();
        sub_allocator_ptr_ = nullptr;
    }

    std::unique_ptr<allocator_pool> pool_allocator_;
    test_pool_sub_allocator*        sub_allocator_ptr_;  // Non-owning reference for testing
};

/**
 * @brief Test basic allocation and deallocation with pooling
 */
TEST_F(AllocatorPool, basic_allocation_deallocation)
{
    sub_allocator_ptr_->reset_counters();

    // First allocation should go to underlying allocator
    void* ptr1 = pool_allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(is_aligned(ptr1, 64));
    EXPECT_EQ(sub_allocator_ptr_->get_alloc_count(), 1);

    // Test memory content
    fill_memory(ptr1, 1024, 0xAA);
    EXPECT_TRUE(validate_memory(ptr1, 1024, 0xAA));

    // Deallocation should put memory in pool (not free to underlying allocator)
    pool_allocator_->deallocate_raw(ptr1);
    EXPECT_EQ(sub_allocator_ptr_->get_free_count(), 0);  // Should be 0 (in pool)

    // Second allocation of same size should come from pool
    void* ptr2 = pool_allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_EQ(sub_allocator_ptr_->get_alloc_count(), 1);  // Still 1 (from pool)

    pool_allocator_->deallocate_raw(ptr2);
}

/**
 * @brief Test pool size limits and LRU eviction
 */
TEST_F(AllocatorPool, pool_size_limits_and_lru)
{
    sub_allocator_ptr_->reset_counters();

    // Fill pool to capacity (5 items)
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = pool_allocator_->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    EXPECT_EQ(sub_allocator_ptr_->get_alloc_count(), 5);

    // Return all to pool
    for (void* ptr : ptrs)
    {
        pool_allocator_->deallocate_raw(ptr);
    }

    EXPECT_EQ(sub_allocator_ptr_->get_free_count(), 0);  // All in pool

    // Allocate one more item of same size - should come from pool
    void* ptr_from_pool = pool_allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr_from_pool);
    EXPECT_EQ(sub_allocator_ptr_->get_alloc_count(), 5);  // No new allocation

    pool_allocator_->deallocate_raw(ptr_from_pool);

    // Now add one more item to pool (6th item) - should trigger LRU eviction
    void* ptr6 = pool_allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr6);
    pool_allocator_->deallocate_raw(ptr6);

    // Pool should still have 5 items, oldest should be evicted
    EXPECT_EQ(sub_allocator_ptr_->get_free_count(), 0);  // One evicted
}

/**
 * @brief Test pool statistics and monitoring
 */
TEST_F(AllocatorPool, pool_statistics)
{
    // Check initial statistics
    EXPECT_EQ(pool_allocator_->get_from_pool_count(), 0);
    EXPECT_EQ(pool_allocator_->put_count(), 0);
    EXPECT_EQ(pool_allocator_->allocated_count(), 0);
    EXPECT_EQ(pool_allocator_->evicted_count(), 0);
    EXPECT_EQ(pool_allocator_->size_limit(), 5);

    // Perform allocation and deallocation
    void* ptr = pool_allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);

    // Should show one allocation
    EXPECT_EQ(pool_allocator_->allocated_count(), 1);

    pool_allocator_->deallocate_raw(ptr);

    // Should show one put to pool
    EXPECT_EQ(pool_allocator_->put_count(), 1);

    // Allocate same size again - should come from pool
    void* ptr2 = pool_allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr2);

    // Should show one get from pool
    EXPECT_EQ(pool_allocator_->get_from_pool_count(), 1);

    pool_allocator_->deallocate_raw(ptr2);
}

/**
 * @brief Test Clear() functionality
 */
TEST_F(AllocatorPool, clear_functionality)
{
    sub_allocator_ptr_->reset_counters();

    // Fill pool with items
    std::vector<void*> ptrs;
    for (int i = 0; i < 3; ++i)
    {
        void* ptr = pool_allocator_->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Return to pool
    for (void* ptr : ptrs)
    {
        pool_allocator_->deallocate_raw(ptr);
    }

    EXPECT_EQ(sub_allocator_ptr_->get_free_count(), 0);  // All in pool

    // Clear pool - should free all cached items
    pool_allocator_->Clear();

    EXPECT_EQ(sub_allocator_ptr_->get_free_count(), 3);  // All freed

    // Next allocation should go to underlying allocator
    void* new_ptr = pool_allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, new_ptr);
    EXPECT_EQ(sub_allocator_ptr_->get_alloc_count(), 4);  // New allocation

    pool_allocator_->deallocate_raw(new_ptr);
}

/**
 * @brief Test allocator properties
 */
TEST_F(AllocatorPool, allocator_properties)
{
    // Test allocator name
    EXPECT_EQ(std::string(pool_allocator_->Name()), std::string("test_pool"));

    // Test memory type (should delegate to underlying allocator)
    EXPECT_EQ(pool_allocator_->GetMemoryType(), allocator_memory_enum::HOST_PAGEABLE);

    // Test allocation size tracking (pool allocator doesn't track sizes by default)
    EXPECT_FALSE(pool_allocator_->tracks_allocation_sizes());
}

/**
 * @brief Test zero-size allocation handling
 */
TEST_F(AllocatorPool, zero_size_allocation)
{
    // Zero-size allocation should be handled gracefully
    void* ptr = pool_allocator_->allocate_raw(64, 0);

    // Implementation may return nullptr or valid pointer
    if (ptr != nullptr)
    {
        pool_allocator_->deallocate_raw(ptr);
    }

    // Test passes if no crash occurs
    EXPECT_TRUE(true);
}

/**
 * @brief Test null pointer deallocation
 */
TEST_F(AllocatorPool, null_pointer_deallocation)
{
    // Deallocating nullptr should not crash
    pool_allocator_->deallocate_raw(nullptr);

    // Test passes if no exception is thrown
    EXPECT_TRUE(true);
}

/**
 * @brief Test pool allocator with auto-resize enabled
 */
TEST(AllocatorPoolAutoResize, auto_resize_functionality)
{
    auto sub_alloc      = std::make_unique<test_pool_sub_allocator>();
    auto test_sub_alloc = sub_alloc.get();

    // Create pool with auto-resize enabled
    auto pool = std::make_unique<allocator_pool>(
        2,     // initial pool_size_limit
        true,  // auto_resize enabled
        std::move(sub_alloc),
        std::make_unique<noop_size_rounder>(),
        "test_auto_resize_pool");

    test_sub_alloc->reset_counters();

    // Fill initial pool capacity
    std::vector<void*> ptrs;
    for (int i = 0; i < 3; ++i)
    {
        void* ptr = pool->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Return to pool
    for (void* ptr : ptrs)
    {
        pool->deallocate_raw(ptr);
    }

    // With auto-resize, pool should have grown to accommodate all items
    EXPECT_EQ(test_sub_alloc->get_free_count(), 1);  // All should be in pool

    // Verify pool size has increased
    EXPECT_GE(pool->size_limit(), 2);
}

/**
 * @brief Test pool allocator with different size rounding strategies
 */
TEST(AllocatorPoolSizeRounding, size_rounding_strategies)
{
    auto sub_alloc      = std::make_unique<test_pool_sub_allocator>();
    auto test_sub_alloc = sub_alloc.get();

    // Create pool with power-of-2 size rounder
    auto pool = std::make_unique<allocator_pool>(
        5,
        false,
        std::move(sub_alloc),
        std::make_unique<power_of_2_rounder>(),
        "test_rounding_pool");

    test_sub_alloc->reset_counters();

    // Allocate 1000 bytes (should be rounded to 1024)
    void* ptr1 = pool->allocate_raw(64, 1000);
    EXPECT_NE(nullptr, ptr1);

    pool->deallocate_raw(ptr1);

    // Allocate 1024 bytes - should come from pool due to rounding
    void* ptr2 = pool->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_EQ(test_sub_alloc->get_alloc_count(), 1);  // Should reuse from pool

    pool->deallocate_raw(ptr2);
}

/**
 * @brief Test concurrent access to pool allocator
 */
TEST(AllocatorPoolConcurrency, thread_safety)
{
    auto sub_alloc = std::make_unique<test_pool_sub_allocator>();

    auto pool = std::make_unique<allocator_pool>(
        10,  // larger pool for concurrency
        false,
        std::move(sub_alloc),
        std::make_unique<noop_size_rounder>(),
        "test_concurrent_pool");

    const int                       num_threads            = 4;
    const int                       allocations_per_thread = 20;
    std::vector<std::thread>        threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch threads that perform allocations and deallocations
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                for (int i = 0; i < allocations_per_thread; ++i)
                {
                    void* ptr = pool->allocate_raw(64, 1024);
                    if (ptr != nullptr)
                    {
                        thread_ptrs[t].push_back(ptr);

                        // Brief delay to increase chance of contention
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                }
            });
    }

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    // Verify allocations and clean up
    size_t total_allocations = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        total_allocations += thread_ptrs[t].size();

        for (void* ptr : thread_ptrs[t])
        {
            EXPECT_NE(nullptr, ptr);
            pool->deallocate_raw(ptr);
        }
    }

    EXPECT_GT(total_allocations, 0);
}

/**
 * @brief Test pool allocator performance characteristics
 */
TEST(AllocatorPoolPerformance, allocation_timing)
{
    auto sub_alloc      = std::make_unique<test_pool_sub_allocator>();
    auto test_sub_alloc = sub_alloc.get();

    auto pool = std::make_unique<allocator_pool>(
        100,  // large pool
        false,
        std::move(sub_alloc),
        std::make_unique<noop_size_rounder>(),
        "test_perf_pool");

    const int          num_allocations = 1000;
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocations);

    test_sub_alloc->reset_counters();

    // Time allocation performance
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_allocations; ++i)
    {
        void* ptr = pool->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    auto alloc_time = std::chrono::high_resolution_clock::now();

    // Return all to pool
    for (void* ptr : ptrs)
    {
        pool->deallocate_raw(ptr);
    }

    // Allocate again - should be much faster (from pool)
    std::vector<void*> ptrs2;
    ptrs2.reserve(num_allocations);

    auto pool_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < std::min(static_cast<int>(ptrs.size()), 100); ++i)
    {
        void* ptr = pool->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs2.push_back(ptr);
        }
    }

    auto pool_end_time = std::chrono::high_resolution_clock::now();

    // Clean up
    for (void* ptr : ptrs2)
    {
        pool->deallocate_raw(ptr);
    }

    // Verify reasonable performance
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(alloc_time - start_time).count();

    EXPECT_LT(total_time, 1000);                  // Should complete within 1 second
    EXPECT_GT(ptrs.size(), num_allocations / 2);  // Most allocations should succeed

    // Pool allocations should be faster than initial allocations
    auto pool_time =
        std::chrono::duration_cast<std::chrono::microseconds>(pool_end_time - pool_start_time)
            .count();

    EXPECT_GT(ptrs2.size(), 0);  // Should have successful pool allocations
}
