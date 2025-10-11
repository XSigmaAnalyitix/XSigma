/**
 * @file TestAllocatorBfc.cxx
 * @brief Comprehensive test suite for BFC (Best-Fit with Coalescing) allocator
 *
 * Tests all public methods and functionality of the allocator_bfc class including:
 * - Basic allocation and deallocation
 * - Alignment requirements
 * - Memory tracking capabilities
 * - Configuration options
 * - Statistics collection
 * - Error handling and edge cases
 */

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "Testing/xsigmaTest.h"
#include "common/pointer.h"
#include "memory/cpu/allocator_bfc.h"
#include "memory/cpu/allocator_pool.h"
#include "memory/cpu/helper/memory_allocator.h"

using namespace xsigma;

namespace
{

/**
 * @brief Helper class for testing - provides basic CPU sub-allocator
 */
class test_cpu_sub_allocator : public sub_allocator
{
public:
    test_cpu_sub_allocator() : sub_allocator({}, {}) {}

    void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) override
    {
        void* ptr = xsigma::cpu::memory_allocator::allocate(num_bytes, alignment);
        if (bytes_received)
            *bytes_received = num_bytes;
        return ptr;
    }

    void Free(void* ptr, size_t num_bytes) override { xsigma::cpu::memory_allocator::free(ptr); }

    bool SupportsCoalescing() const override { return true; }

    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_memory_enum::HOST_PAGEABLE;
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
 * @brief Test suite for BFC allocator basic functionality
 */
class AllocatorBfc : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create sub-allocator for testing
        auto sub_alloc = std::make_unique<test_cpu_sub_allocator>();

        // Create BFC allocator with default options
        allocator_bfc::Options opts;
        opts.allow_growth           = true;
        opts.garbage_collection     = false;
        opts.fragmentation_fraction = 0.0;

        allocator_ = std::make_unique<allocator_bfc>(
            std::move(sub_alloc),
            1024 * 1024,  // 1MB initial size
            "test_bfc",
            opts);
    }

    void TearDown() override { allocator_.reset(); }

    std::unique_ptr<allocator_bfc> allocator_;
};

/**
 * @brief Test basic allocation and deallocation functionality
 */
TEST_F(AllocatorBfc, basic_allocation_deallocation)
{
    // Test basic allocation
    void* ptr1 = allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(is_aligned(ptr1, 64));

    // Test memory content
    fill_memory(ptr1, 1024, 0xAA);
    EXPECT_TRUE(validate_memory(ptr1, 1024, 0xAA));

    // Test deallocation
    allocator_->deallocate_raw(ptr1);

    // Test multiple allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator_->allocate_raw(32, 512);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(is_aligned(ptr, 32));
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs)
    {
        allocator_->deallocate_raw(ptr);
    }
}

/**
 * @brief Test allocation tracking capabilities
 */
TEST_F(AllocatorBfc, allocation_tracking)
{
    // Verify tracking is enabled
    EXPECT_TRUE(allocator_->tracks_allocation_sizes());

    // Test allocation with size tracking
    void* ptr = allocator_->allocate_raw(128, 2048);
    EXPECT_NE(nullptr, ptr);

    // Test size queries
    EXPECT_EQ(allocator_->RequestedSize(ptr), 2048);
    EXPECT_GE(allocator_->AllocatedSize(ptr), 2048);

    // Test allocation ID
    int64_t alloc_id = allocator_->AllocationId(ptr);
    EXPECT_GT(alloc_id, 0);

    allocator_->deallocate_raw(ptr);
}

/**
 * @brief Test different alignment requirements
 */
TEST_F(AllocatorBfc, alignment_requirements)
{
    // Test various alignment values
    std::vector<size_t> alignments = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    for (size_t alignment : alignments)
    {
        void* ptr = allocator_->allocate_raw(alignment, 1024);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(is_aligned(ptr, std::max(alignment, static_cast<size_t>(64))));
        allocator_->deallocate_raw(ptr);
    }
}

/**
 * @brief Test allocation with attributes
 */
TEST_F(AllocatorBfc, allocation_with_attributes)
{
    allocation_attributes attrs;
    attrs.retry_on_failure = true;

    void* ptr = allocator_->allocate_raw(64, 1024, attrs);
    EXPECT_NE(nullptr, ptr);
    EXPECT_TRUE(is_aligned(ptr, 64));

    allocator_->deallocate_raw(ptr);
}

/**
 * @brief Test statistics collection
 */
TEST_F(AllocatorBfc, statistics_collection)
{
    // Get initial statistics
    auto stats_opt = allocator_->GetStats();
    EXPECT_TRUE(stats_opt.has_value());

    auto initial_stats = stats_opt.value();

    // Perform some allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = allocator_->allocate_raw(64, 1024 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Check updated statistics
    auto updated_stats_opt = allocator_->GetStats();
    EXPECT_TRUE(updated_stats_opt.has_value());

    auto updated_stats = updated_stats_opt.value();
    EXPECT_GT(updated_stats.num_allocs, initial_stats.num_allocs);
    EXPECT_GT(updated_stats.bytes_in_use, initial_stats.bytes_in_use);

    // Clean up
    for (void* ptr : ptrs)
    {
        allocator_->deallocate_raw(ptr);
    }
}

/**
 * @brief Test allocator name and memory type
 */
TEST_F(AllocatorBfc, allocator_properties)
{
    // Test allocator name
    EXPECT_EQ(std::string(allocator_->Name()), std::string("test_bfc"));

    // Test memory type
    EXPECT_EQ(allocator_->GetMemoryType(), allocator_memory_enum::HOST_PAGEABLE);
}

/**
 * @brief Test zero-size allocation handling
 */
TEST_F(AllocatorBfc, zero_size_allocation)
{
    // Zero-size allocation should return nullptr or handle gracefully
    void* ptr = allocator_->allocate_raw(64, 0);

    // Implementation may return nullptr or a valid pointer for zero-size
    if (ptr != nullptr)
    {
        allocator_->deallocate_raw(ptr);
    }
}

/**
 * @brief Test null pointer deallocation
 */
TEST_F(AllocatorBfc, null_pointer_deallocation)
{
    // Deallocating nullptr should not crash
    allocator_->deallocate_raw(nullptr);

    // This test passes if no exception is thrown
    EXPECT_TRUE(true);
}

/**
 * @brief Test large allocation handling
 */
TEST_F(AllocatorBfc, large_allocations)
{
    // Test allocation larger than initial pool size
    void* large_ptr = allocator_->allocate_raw(64, 2 * 1024 * 1024);  // 2MB

    if (large_ptr != nullptr)
    {
        EXPECT_TRUE(is_aligned(large_ptr, 64));

        // Test memory access
        fill_memory(large_ptr, 1024, 0xBB);
        EXPECT_TRUE(validate_memory(large_ptr, 1024, 0xBB));

        allocator_->deallocate_raw(large_ptr);
    }
}

/**
 * @brief Test fragmentation and coalescing behavior
 */
TEST_F(AllocatorBfc, fragmentation_and_coalescing)
{
    // Allocate several blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator_->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Deallocate every other block to create fragmentation
    for (size_t i = 1; i < ptrs.size(); i += 2)
    {
        allocator_->deallocate_raw(ptrs[i]);
        ptrs[i] = nullptr;
    }

    // Try to allocate a larger block (should trigger coalescing)
    void* large_ptr = allocator_->allocate_raw(64, 2048);
    if (large_ptr != nullptr)
    {
        allocator_->deallocate_raw(large_ptr);
    }

    // Clean up remaining blocks
    for (void* ptr : ptrs)
    {
        if (ptr != nullptr)
        {
            allocator_->deallocate_raw(ptr);
        }
    }
}

/**
 * @brief Test BFC allocator with different configuration options
 */
TEST(AllocatorBfcConfig, configuration_options)
{
    auto sub_alloc = std::make_unique<test_cpu_sub_allocator>();

    // Test with garbage collection enabled
    allocator_bfc::Options gc_opts;
    gc_opts.allow_growth           = true;
    gc_opts.garbage_collection     = true;
    gc_opts.fragmentation_fraction = 0.1;

    auto gc_allocator = std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        512 * 1024,  // 512KB
        "test_bfc_gc",
        gc_opts);

    // Test basic functionality with GC enabled
    void* ptr = gc_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);
    EXPECT_TRUE(is_aligned(ptr, 64));

    gc_allocator->deallocate_raw(ptr);
}

/**
 * @brief Test BFC allocator with growth disabled
 */
TEST(AllocatorBfcConfig, no_growth_configuration)
{
    auto sub_alloc = std::make_unique<test_cpu_sub_allocator>();

    allocator_bfc::Options no_growth_opts;
    no_growth_opts.allow_growth       = false;
    no_growth_opts.garbage_collection = false;

    auto fixed_allocator = std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        64 * 1024,  // 64KB fixed size
        "test_bfc_fixed",
        no_growth_opts);

    // Test allocation within fixed size
    void* ptr = fixed_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);

    fixed_allocator->deallocate_raw(ptr);
}

/**
 * @brief Test concurrent access to BFC allocator
 */
TEST(AllocatorBfcConcurrency, thread_safety)
{
    auto sub_alloc = std::make_unique<test_cpu_sub_allocator>();

    allocator_bfc::Options opts;
    opts.allow_growth = true;

    auto allocator = std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        1024 * 1024,  // 1MB
        "test_bfc_concurrent",
        opts);

    const int                       num_threads            = 4;
    const int                       allocations_per_thread = 10;
    std::vector<std::thread>        threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch threads that perform allocations
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                for (int i = 0; i < allocations_per_thread; ++i)
                {
                    void* ptr = allocator->allocate_raw(64, 1024 + i * 64);
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
            allocator->deallocate_raw(ptr);
        }
    }

    EXPECT_GT(total_allocations, 0);
}

/**
 * @brief Test BFC allocator performance characteristics
 */
TEST(AllocatorBfcPerformance, allocation_timing)
{
    auto sub_alloc = std::make_unique<test_cpu_sub_allocator>();

    allocator_bfc::Options opts;
    opts.allow_growth = true;

    auto allocator = std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        2 * 1024 * 1024,  // 2MB
        "test_bfc_perf",
        opts);

    const int          num_allocations = 1000;
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocations);

    // Time allocation performance
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_allocations; ++i)
    {
        void* ptr = allocator->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    auto alloc_time = std::chrono::high_resolution_clock::now();

    // Time deallocation performance
    for (void* ptr : ptrs)
    {
        allocator->deallocate_raw(ptr);
    }

    auto dealloc_time = std::chrono::high_resolution_clock::now();

    // Verify reasonable performance (should complete in reasonable time)
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(dealloc_time - start_time).count();

    EXPECT_LT(total_time, 1000);                  // Should complete within 1 second
    EXPECT_GT(ptrs.size(), num_allocations / 2);  // Most allocations should succeed
}
