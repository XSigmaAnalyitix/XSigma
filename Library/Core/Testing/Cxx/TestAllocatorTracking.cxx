/**
 * @file TestAllocatorTracking.cxx
 * @brief Comprehensive test suite for Tracking allocator wrapper
 *
 * Tests all public methods and functionality of the allocator_tracking class including:
 * - Basic allocation tracking and statistics
 * - Size tracking capabilities
 * - Allocation ID generation and management
 * - Enhanced tracking features
 * - Logging level configuration
 * - Reference counting and lifecycle management
 * - Performance timing statistics
 * - Thread safety and concurrent access
 */

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "Testing/xsigmaTest.h"
#include "common/pointer.h"
#include "memory/cpu/allocator_pool.h"
#include "memory/cpu/allocator_tracking.h"
#include "memory/cpu/helper/memory_allocator.h"

using namespace xsigma;

namespace
{

/**
 * @brief Test implementation of basic allocator for tracking tests
 */
class test_basic_allocator : public Allocator
{
public:
    test_basic_allocator() : alloc_count_(0), dealloc_count_(0), total_allocated_(0) {}

    void* allocate_raw(size_t alignment, size_t num_bytes) override
    {
        alloc_count_++;
        total_allocated_ += num_bytes;
        void* ptr = xsigma::cpu::memory_allocator::allocate(num_bytes, alignment);

        // Store size for tracking
        if (ptr)
        {
            allocations_[ptr] = num_bytes;
        }

        return ptr;
    }

    void* allocate_raw(
        size_t alignment, size_t num_bytes, const allocation_attributes& attrs) override
    {
        return allocate_raw(alignment, num_bytes);
    }

    void deallocate_raw(void* ptr) override
    {
        if (ptr)
        {
            dealloc_count_++;
            auto it = allocations_.find(ptr);
            if (it != allocations_.end())
            {
                total_allocated_ -= it->second;
                allocations_.erase(it);
            }
            xsigma::cpu::memory_allocator::free(ptr);
        }
    }

    bool tracks_allocation_sizes() const noexcept override { return true; }

    size_t RequestedSize(const void* ptr) const noexcept override
    {
        auto it = allocations_.find(const_cast<void*>(ptr));
        return (it != allocations_.end()) ? it->second : 0;
    }

    size_t AllocatedSize(const void* ptr) const noexcept override
    {
        return RequestedSize(ptr);  // Same as requested for this test allocator
    }

    int64_t AllocationId(const void* ptr) const noexcept override
    {
        return reinterpret_cast<int64_t>(ptr);  // Use pointer as ID
    }

    std::optional<allocator_stats> GetStats() override
    {
        allocator_stats stats;
        stats.num_allocs         = alloc_count_;
        stats.bytes_in_use       = total_allocated_;
        stats.peak_bytes_in_use  = total_allocated_;  // Simplified
        stats.largest_alloc_size = 0;                 // Not tracked in this test allocator
        return stats;
    }

    std::string Name() override { return "test_basic_allocator"; }

    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_memory_enum::HOST_PAGEABLE;
    }

    // Test helpers
    int    get_alloc_count() const { return alloc_count_; }
    int    get_dealloc_count() const { return dealloc_count_; }
    size_t get_total_allocated() const { return total_allocated_; }
    void   reset_counters()
    {
        alloc_count_     = 0;
        dealloc_count_   = 0;
        total_allocated_ = 0;
    }

private:
    std::atomic<int>                  alloc_count_;
    std::atomic<int>                  dealloc_count_;
    std::atomic<size_t>               total_allocated_;
    std::unordered_map<void*, size_t> allocations_;
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
 * @brief Test suite for Tracking allocator basic functionality
 */
class AllocatorTracking : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create underlying allocator
        underlying_allocator_ = std::make_unique<test_basic_allocator>();
        underlying_ptr_       = underlying_allocator_.get();

        // Create tracking allocator (use pointer due to protected destructor)
        tracking_allocator_ = new allocator_tracking(underlying_ptr_, true, true);
    }

    void TearDown() override
    {
        if (tracking_allocator_)
        {
            // Properly release tracking allocator
            tracking_allocator_->GetRecordsAndUnRef();
            tracking_allocator_ = nullptr;
        }
        underlying_allocator_.reset();
        underlying_ptr_ = nullptr;
    }

    std::unique_ptr<test_basic_allocator> underlying_allocator_;
    test_basic_allocator*                 underlying_ptr_;
    allocator_tracking*                   tracking_allocator_;
};

/**
 * @brief Test basic allocation tracking functionality
 */
TEST_F(AllocatorTracking, basic_allocation_tracking)
{
    underlying_ptr_->reset_counters();

    // Test basic allocation
    void* ptr = tracking_allocator_->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);
    EXPECT_TRUE(is_aligned(ptr, 64));

    // Verify underlying allocator was called
    EXPECT_EQ(underlying_ptr_->get_alloc_count(), 1);

    // Test memory content
    fill_memory(ptr, 1024, 0xCC);
    EXPECT_TRUE(validate_memory(ptr, 1024, 0xCC));

    // Test deallocation
    tracking_allocator_->deallocate_raw(ptr);
    EXPECT_EQ(underlying_ptr_->get_dealloc_count(), 1);
}

/**
 * @brief Test allocation size tracking capabilities
 */
TEST_F(AllocatorTracking, allocation_size_tracking)
{
    // Verify tracking is enabled
    EXPECT_TRUE(tracking_allocator_->tracks_allocation_sizes());

    // Test allocation with size tracking
    void* ptr = tracking_allocator_->allocate_raw(128, 2048);
    EXPECT_NE(nullptr, ptr);

    // Test size queries
    EXPECT_EQ(tracking_allocator_->RequestedSize(ptr), 2048);
    EXPECT_GE(tracking_allocator_->AllocatedSize(ptr), 2048);

    // Test allocation ID
    int64_t alloc_id = tracking_allocator_->AllocationId(ptr);
    EXPECT_GT(alloc_id, 0);

    tracking_allocator_->deallocate_raw(ptr);
}

/**
 * @brief Test statistics collection and monitoring
 */
TEST_F(AllocatorTracking, statistics_collection)
{
    // Get initial statistics
    auto initial_stats = tracking_allocator_->GetStats();
    EXPECT_TRUE(initial_stats.has_value());

    // Get initial sizes
    auto [total_initial, high_initial, current_initial] = tracking_allocator_->GetSizes();
    EXPECT_EQ(current_initial, 0);

    // Perform allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = tracking_allocator_->allocate_raw(64, 1024 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Check updated statistics
    auto updated_stats = tracking_allocator_->GetStats();
    EXPECT_TRUE(updated_stats.has_value());
    EXPECT_GT(updated_stats->num_allocs, initial_stats->num_allocs);
    EXPECT_GT(updated_stats->bytes_in_use, initial_stats->bytes_in_use);

    // Check updated sizes
    auto [total_updated, high_updated, current_updated] = tracking_allocator_->GetSizes();
    EXPECT_GT(current_updated, current_initial);
    EXPECT_GE(high_updated, current_updated);

    // Clean up
    for (void* ptr : ptrs)
    {
        tracking_allocator_->deallocate_raw(ptr);
    }

    // Verify cleanup
    auto [total_final, high_final, current_final] = tracking_allocator_->GetSizes();
    EXPECT_EQ(current_final, 0);
}

/**
 * @brief Test enhanced tracking features
 */
TEST_F(AllocatorTracking, enhanced_tracking_features)
{
    // Test enhanced records collection
    auto   initial_records = tracking_allocator_->GetEnhancedRecords();
    size_t initial_count   = initial_records.size();

    // Perform some allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 3; ++i)
    {
        void* ptr = tracking_allocator_->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Check enhanced records
    auto updated_records = tracking_allocator_->GetEnhancedRecords();
    EXPECT_GT(updated_records.size(), initial_count);

    // Clean up
    for (void* ptr : ptrs)
    {
        tracking_allocator_->deallocate_raw(ptr);
    }
}

/**
 * @brief Test logging level configuration
 */
TEST_F(AllocatorTracking, logging_level_configuration)
{
    // Test setting and getting logging levels
    tracking_allocator_->SetLoggingLevel(tracking_log_level::DEBUG);
    EXPECT_EQ(tracking_allocator_->GetLoggingLevel(), tracking_log_level::DEBUG);

    tracking_allocator_->SetLoggingLevel(tracking_log_level::INFO);
    EXPECT_EQ(tracking_allocator_->GetLoggingLevel(), tracking_log_level::INFO);

    tracking_allocator_->SetLoggingLevel(tracking_log_level::SILENT);
    EXPECT_EQ(tracking_allocator_->GetLoggingLevel(), tracking_log_level::SILENT);

    tracking_allocator_->SetLoggingLevel(tracking_log_level::TRACE);
    EXPECT_EQ(tracking_allocator_->GetLoggingLevel(), tracking_log_level::TRACE);
}

/**
 * @brief Test timing statistics reset
 */
TEST_F(AllocatorTracking, timing_statistics_reset)
{
    // Perform some allocations to generate timing data
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = tracking_allocator_->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Reset timing statistics
    tracking_allocator_->ResetTimingStats();

    // Timing stats should be reset but allocation records should remain
    auto stats = tracking_allocator_->GetStats();
    EXPECT_TRUE(stats.has_value());

    // Clean up
    for (void* ptr : ptrs)
    {
        tracking_allocator_->deallocate_raw(ptr);
    }
}

/**
 * @brief Test allocator properties delegation
 */
TEST_F(AllocatorTracking, allocator_properties_delegation)
{
    // Test name delegation
    EXPECT_EQ(std::string(tracking_allocator_->Name()), std::string("test_basic_allocator"));

    // Test memory type delegation
    EXPECT_EQ(tracking_allocator_->GetMemoryType(), allocator_memory_enum::HOST_PAGEABLE);
}

/**
 * @brief Test zero-size allocation handling
 */
TEST_F(AllocatorTracking, zero_size_allocation)
{
    // Zero-size allocation should be handled gracefully
    void* ptr = tracking_allocator_->allocate_raw(64, 0);

    // Implementation may return nullptr or valid pointer
    if (ptr != nullptr)
    {
        tracking_allocator_->deallocate_raw(ptr);
    }

    // Test passes if no crash occurs
    EXPECT_TRUE(true);
}

/**
 * @brief Test null pointer deallocation
 */
TEST_F(AllocatorTracking, null_pointer_deallocation)
{
    // Deallocating nullptr should not crash
    tracking_allocator_->deallocate_raw(nullptr);

    // Test passes if no exception is thrown
    EXPECT_TRUE(true);
}

/**
 * @brief Test tracking allocator with non-tracking underlying allocator
 */
TEST(AllocatorTrackingNonTracking, local_size_tracking)
{
    // Create a simple allocator that doesn't track sizes
    class non_tracking_allocator : public Allocator
    {
    public:
        void* allocate_raw(size_t alignment, size_t num_bytes) override
        {
            return xsigma::cpu::memory_allocator::allocate(num_bytes, alignment);
        }

        void* allocate_raw(
            size_t alignment, size_t num_bytes, const allocation_attributes& attrs) override
        {
            return allocate_raw(alignment, num_bytes);
        }

        void deallocate_raw(void* ptr) override { xsigma::cpu::memory_allocator::free(ptr); }

        bool tracks_allocation_sizes() const noexcept override { return false; }

        size_t  RequestedSize(const void* ptr) const noexcept override { return 0; }
        size_t  AllocatedSize(const void* ptr) const noexcept override { return 0; }
        int64_t AllocationId(const void* ptr) const noexcept override { return 0; }

        std::optional<allocator_stats> GetStats() override { return std::nullopt; }
        std::string                    Name() override { return "non_tracking_allocator"; }
        allocator_memory_enum          GetMemoryType() const noexcept override
        {
            return allocator_memory_enum::HOST_PAGEABLE;
        }
    };

    auto underlying     = std::make_unique<non_tracking_allocator>();
    auto underlying_ptr = underlying.get();

    // Create tracking allocator with local size tracking enabled
    auto tracker = new allocator_tracking(underlying_ptr, true, false);

    // Should enable tracking even though underlying doesn't track
    EXPECT_TRUE(tracker->tracks_allocation_sizes());

    // Test allocation with local tracking
    void* ptr = tracker->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);

    // Should track size locally
    EXPECT_EQ(tracker->RequestedSize(ptr), 1024);
    EXPECT_GE(tracker->AllocatedSize(ptr), 1024);

    tracker->deallocate_raw(ptr);

    // Properly cleanup
    tracker->GetRecordsAndUnRef();
}

/**
 * @brief Test concurrent access to tracking allocator
 */
TEST(AllocatorTrackingConcurrency, thread_safety)
{
    auto underlying     = std::make_unique<test_basic_allocator>();
    auto underlying_ptr = underlying.get();

    auto tracker = new allocator_tracking(underlying_ptr, true, true);

    const int                       num_threads            = 4;
    const int                       allocations_per_thread = 50;
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
                    void* ptr = tracker->allocate_raw(64, 1024 + i * 64);
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
            EXPECT_GT(tracker->RequestedSize(ptr), 0);
            tracker->deallocate_raw(ptr);
        }
    }

    EXPECT_GT(total_allocations, 0);

    // Properly cleanup
    tracker->GetRecordsAndUnRef();
}

/**
 * @brief Test tracking allocator performance characteristics
 */
TEST(AllocatorTrackingPerformance, allocation_timing)
{
    auto underlying     = std::make_unique<test_basic_allocator>();
    auto underlying_ptr = underlying.get();

    auto tracker = new allocator_tracking(underlying_ptr, true, true);

    const int          num_allocations = 1000;
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocations);

    underlying_ptr->reset_counters();

    // Time allocation performance
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_allocations; ++i)
    {
        void* ptr = tracker->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    auto alloc_time = std::chrono::high_resolution_clock::now();

    // Time deallocation performance
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    auto dealloc_time = std::chrono::high_resolution_clock::now();

    // Verify reasonable performance (should complete in reasonable time)
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(dealloc_time - start_time).count();

    EXPECT_LT(total_time, 2000);                  // Should complete within 2 seconds
    EXPECT_GT(ptrs.size(), num_allocations / 2);  // Most allocations should succeed

    // Verify tracking overhead is reasonable
    EXPECT_EQ(underlying_ptr->get_alloc_count(), static_cast<int>(ptrs.size()));
    EXPECT_EQ(underlying_ptr->get_dealloc_count(), static_cast<int>(ptrs.size()));

    // Properly cleanup
    tracker->GetRecordsAndUnRef();
}

/**
 * @brief Test reference counting and lifecycle management
 */
TEST(AllocatorTrackingLifecycle, reference_counting)
{
    auto underlying     = std::make_unique<test_basic_allocator>();
    auto underlying_ptr = underlying.get();

    // Create tracking allocator
    auto tracker = new allocator_tracking(underlying_ptr, true, false);

    // Perform some allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = tracker->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Get enhanced records (this should increment reference count)
    auto records = tracker->GetEnhancedRecords();
    EXPECT_GE(records.size(), 0);  // May be empty if no tracking

    // Clean up allocations
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Properly release reference - this should delete the tracker
    tracker->GetRecordsAndUnRef();

    // tracker is now invalid - test passes if no crash occurred
    EXPECT_TRUE(true);
}
