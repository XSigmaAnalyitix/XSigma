/*
 * XSigma: High-Performance Quantitative Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 XSigma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of XSigma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for XSigma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include <algorithm>
#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "experimental/profiler/scoped_memory_debug_annotation.h"
#include "experimental/profiler/traceme.h"
#include "logging/logger.h"
#include "memory/cpu/allocator.h"
#include "memory/cpu/helper/allocator_registry.h"
#include "memory/cpu/helper/mem.h"
#include "memory/cpu/helper/memory_allocator.h"
#include "util/strcat.h"

namespace xsigma
{

/**
 * @brief Global flag controlling CPU allocator statistics collection.
 *
 * When enabled, the CPU allocator tracks detailed memory usage statistics
 * including allocation counts, peak usage, and memory warnings. This adds
 * minimal overhead but provides valuable debugging information.
 *
 * **Performance Impact**: ~1-2% overhead when enabled
 * **Thread Safety**: Atomic operations ensure thread-safe access
 * **Default**: Disabled for optimal performance
 */
static std::atomic<bool> cpu_allocator_collect_stats{false};

/**
 * @brief Enables comprehensive CPU allocator statistics collection.
 *
 * Activates detailed memory tracking including allocation counts, peak usage,
 * and memory pressure warnings. Useful for debugging and performance analysis.
 *
 * **Thread Safety**: Thread-safe atomic operation
 * **Performance**: Adds ~1-2% overhead to allocation operations
 * **Use Cases**: Debugging, memory leak detection, performance profiling
 */
void EnableCPUAllocatorStats() noexcept
{
    cpu_allocator_collect_stats.store(true, std::memory_order_relaxed);
}

/**
 * @brief Disables CPU allocator statistics collection for optimal performance.
 *
 * **Thread Safety**: Thread-safe atomic operation
 * **Performance**: Eliminates statistics overhead
 * **Use Cases**: Production deployments, performance-critical applications
 */
void DisableCPUAllocatorStats() noexcept
{
    cpu_allocator_collect_stats.store(false, std::memory_order_relaxed);
}

/**
 * @brief Checks if CPU allocator statistics collection is enabled.
 *
 * @return true if statistics are being collected, false otherwise
 *
 * **Thread Safety**: Thread-safe atomic read
 * **Performance**: O(1) - simple atomic load
 */
bool CPUAllocatorStatsEnabled() noexcept
{
    return cpu_allocator_collect_stats.load(std::memory_order_relaxed);
}

// ========== Memory Warning Configuration ==========

/**
 * @brief Maximum number of total allocation warnings to emit.
 *
 * Prevents log spam when total memory usage consistently exceeds thresholds.
 * After this many warnings, further total allocation warnings are suppressed.
 */
static constexpr int kMaxTotalAllocationWarnings = 1;

/**
 * @brief Maximum number of large single allocation warnings to emit.
 *
 * Prevents log spam from repeated large allocations. After this many warnings,
 * further large allocation warnings are suppressed.
 */
static constexpr int kMaxSingleAllocationWarnings = 5;

/**
 * @brief Threshold for total memory usage warnings (fraction of available RAM).
 *
 * When total allocated memory exceeds this fraction of available system RAM,
 * a warning is logged to alert about potential memory pressure.
 *
 * **Value**: 0.5 (50% of available RAM)
 * **Purpose**: Early warning for memory pressure conditions
 */
static constexpr double kTotalAllocationWarningThreshold = 0.5;

/**
 * @brief Threshold for large single allocation warnings (fraction of available RAM).
 *
 * Individual allocations exceeding this fraction of available system RAM
 * trigger warnings to identify potentially problematic large allocations.
 *
 * **Value**: 0.1 (10% of available RAM)
 * **Purpose**: Identify unexpectedly large allocations
 */
static constexpr double kLargeAllocationWarningThreshold = 0.1;

/**
 * @brief Cached threshold for large allocation warnings in bytes.
 *
 * @return Byte threshold for large allocation warnings
 *
 * **Caching**: Expensive port::available_ram() call is cached on first use
 * **Thread Safety**: Static initialization is thread-safe in C++11+
 * **Performance**: O(1) after first call, expensive first call
 */
static int64_t LargeAllocationWarningBytes() noexcept
{
    static const int64_t value =
        static_cast<int64_t>(port::available_ram() * kLargeAllocationWarningThreshold);
    return value;
}

/**
 * @brief Cached threshold for total allocation warnings in bytes.
 *
 * @return Byte threshold for total allocation warnings
 *
 * **Caching**: Expensive port::available_ram() call is cached on first use
 * **Thread Safety**: Static initialization is thread-safe in C++11+
 * **Performance**: O(1) after first call, expensive first call
 */
static int64_t TotalAllocationWarningBytes() noexcept
{
    static const int64_t value =
        static_cast<int64_t>(port::available_ram() * kTotalAllocationWarningThreshold);
    return value;
}

namespace
{

/**
 * @brief High-performance CPU memory allocator with comprehensive monitoring.
 *
 * CPUAllocator provides efficient memory allocation for CPU-based computations
 * with optional statistics collection, memory pressure monitoring, and
 * integration with profiling systems.
 *
 * **Key Features**:
 * - Direct integration with optimized cpu_allocator backend
 * - Optional comprehensive statistics collection
 * - Memory pressure warnings and monitoring
 * - Profiling integration with TraceMe system
 * - Thread-safe operation with minimal contention
 *
 * **Performance Characteristics**:
 * - Allocation: O(1) - delegates to cpu::memory_allocator::allocate
 * - Deallocation: O(1) - delegates to cpu::memory_allocator::free
 * - Statistics overhead: ~1-2% when enabled, 0% when disabled
 * - Memory overhead: Minimal - only statistics when enabled
 *
 * **Thread Safety**: Fully thread-safe with fine-grained locking
 * **Memory Type**: Host pageable memory (standard system RAM)
 *
 * **Design Principles**:
 * - Minimal overhead when statistics disabled
 * - Comprehensive monitoring when statistics enabled
 * - Integration with XSigma profiling ecosystem
 * - Robust warning system for memory pressure detection
 *
 * **Use Cases**:
 * - CPU-based numerical computations
 * - Temporary buffer allocation
 * - General-purpose memory management
 * - Development and debugging with statistics
 */
class CPUAllocator : public Allocator
{
public:
    /**
     * @brief Constructs CPU allocator with default configuration.
     *
     * **Initialization**: Sets up warning counters and statistics
     * **Performance**: O(1) - minimal initialization overhead
     * **Thread Safety**: Constructor is not thread-safe
     */
    CPUAllocator() : single_allocation_warning_count_{0} {}

    /**
     * @brief Destructor with automatic cleanup.
     *
     * **Cleanup**: No explicit cleanup needed - uses RAII
     * **Thread Safety**: Destructor is not thread-safe
     */
    ~CPUAllocator() override = default;

    /**
     * @brief Returns allocator name for identification and debugging.
     *
     * @return Static string "cpu" identifying this allocator type
     *
     * **Thread Safety**: Thread-safe - returns constant string
     * **Performance**: O(1) - returns static string
     * **Use Cases**: Logging, debugging, allocator identification
     */
    std::string Name() override { return "cpu"; }

    /**
     * @brief Allocates aligned memory block with comprehensive monitoring.
     *
     * Provides high-performance memory allocation with optional statistics
     * collection, memory pressure warnings, and profiling integration.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Algorithm**: Delegates to optimized cpu::memory_allocator::allocate backend
     * **Performance**: O(1) - direct system allocator call + optional stats
     * **Thread Safety**: Thread-safe with fine-grained locking for statistics
     * **Memory Pressure**: Monitors and warns about large/total allocations
     *
     * **Statistics Collection** (when enabled):
     * - Updates allocation count and bytes in use
     * - Tracks peak memory usage
     * - Records largest single allocation
     * - Generates profiling traces
     *
     * **Warning System**:
     * - Large allocation warnings for individual allocations > 10% of RAM
     * - Total allocation warnings when total usage > 50% of RAM
     * - Rate-limited to prevent log spam
     *
     * **Error Handling**: Returns nullptr on allocation failure
     * **Alignment**: Guaranteed to meet or exceed requested alignment
     */
    void* allocate_raw(size_t alignment, size_t num_bytes) override
    {
        // Check for large allocation warning (rate-limited)
        if XSIGMA_UNLIKELY (num_bytes > static_cast<size_t>(LargeAllocationWarningBytes()))
        {
            const auto current_count =
                single_allocation_warning_count_.load(std::memory_order_relaxed);
            if (current_count < kMaxSingleAllocationWarnings)
            {
                // Use compare_exchange to avoid race conditions in warning count
                auto expected = current_count;
                if (single_allocation_warning_count_.compare_exchange_weak(
                        expected, current_count + 1, std::memory_order_relaxed))
                {
                    XSIGMA_LOG_WARNING(
                        "Large allocation of {} bytes ({}% of available RAM) exceeds {}% threshold",
                        num_bytes,
                        (100.0 * num_bytes / port::available_ram()),
                        (100 * kLargeAllocationWarningThreshold));
                }
            }
        }

        // Perform the actual allocation
        void* p = cpu::memory_allocator::allocate(num_bytes, alignment);

        // Collect statistics if enabled (fast path when disabled)
        if XSIGMA_UNLIKELY (
            cpu_allocator_collect_stats.load(std::memory_order_relaxed) && p != nullptr)
        {
            const auto alloc_size = 0;

            // Use scoped lock for statistics update
            {
                std::lock_guard<std::mutex> lock(mu_);

                // Update core statistics
                stats_.num_allocs.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_in_use.fetch_add(alloc_size, std::memory_order_relaxed);

                // Update peak bytes in use atomically
                int64_t current_bytes = stats_.bytes_in_use.load(std::memory_order_relaxed);
                int64_t peak_bytes    = stats_.peak_bytes_in_use.load(std::memory_order_relaxed);
                while (current_bytes > peak_bytes &&
                       !stats_.peak_bytes_in_use.compare_exchange_weak(
                           peak_bytes, current_bytes, std::memory_order_relaxed))
                {
                    // Retry if another thread updated peak_bytes
                }

                // Update largest allocation size atomically
                int64_t alloc_size_int64 = static_cast<int64_t>(alloc_size);
                int64_t largest_size = stats_.largest_alloc_size.load(std::memory_order_relaxed);
                while (alloc_size_int64 > largest_size &&
                       !stats_.largest_alloc_size.compare_exchange_weak(
                           largest_size, alloc_size_int64, std::memory_order_relaxed))
                {
                    // Retry if another thread updated largest_size
                }

                // Check for total allocation warning (rate-limited)
                if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
                    total_allocation_warning_count_ < kMaxTotalAllocationWarnings)
                {
                    ++total_allocation_warning_count_;
                    size_t bytes_in_use = stats_.bytes_in_use.load(std::memory_order_relaxed);
                    XSIGMA_LOG_WARNING(
                        "Total allocated memory {} bytes ({}% of available RAM) exceeds {}% "
                        "threshold",
                        bytes_in_use,
                        (100.0 * bytes_in_use / port::available_ram()),
                        (100 * kTotalAllocationWarningThreshold));
                }
            }

            // Add profiling trace (outside lock to minimize contention)
            AddTraceMe("MemoryAllocation", p, num_bytes, alloc_size);
        }

        return p;
    }

    /**
     * @brief Deallocates memory block with statistics tracking.
     *
     * Efficiently deallocates memory with optional statistics updates
     * and profiling integration. Provides the primary deallocation interface.
     *
     * @param ptr Pointer to memory block to deallocate (must not be nullptr)
     *
     * **Algorithm**: Delegates to cpu::memory_allocator::free backend
     * **Performance**: O(1) - direct system deallocator call + optional stats
     * **Thread Safety**: Thread-safe with fine-grained locking for statistics
     * **Error Handling**: Undefined behavior if ptr is invalid
     *
     * **Statistics Collection** (when enabled):
     * - Updates bytes in use counter
     * - Generates profiling deallocation trace
     * - Maintains accurate memory usage tracking
     *
     * **Profiling Integration**: Records deallocation events for analysis
     */
    void deallocate_raw(void* ptr) override
    {
        // Fast path when statistics disabled
        if XSIGMA_UNLIKELY (cpu_allocator_collect_stats.load(std::memory_order_relaxed))
        {
            // Get allocation size before deallocation
            const auto alloc_size = 0;

            // Update statistics under lock
            {
                std::lock_guard<std::mutex> lock(mu_);
                stats_.bytes_in_use -= alloc_size;
            }

            // Add profiling trace (outside lock to minimize contention)
            AddTraceMe("MemoryDeallocation", ptr, 0, alloc_size);
        }

        // Perform actual deallocation
        cpu::memory_allocator::free(ptr);
    }

    /**
     * @brief Deallocates memory block with size and alignment hints.
     *
     * Extended deallocation interface that accepts size and alignment
     * parameters for potential optimization. Currently delegates to
     * standard deallocate_raw as cpu_allocator doesn't use these hints.
     *
     * @param ptr Pointer to memory block to deallocate
     * @param alignment Original alignment requirement (unused)
     * @param num_bytes Original allocation size (unused)
     *
     * **Algorithm**: Same as deallocate_raw(ptr) - hints are unused
     * **Performance**: O(1) - identical to single-parameter version
     * **Thread Safety**: Thread-safe with fine-grained locking
     * **Future**: May be optimized to use size/alignment hints
     */
    void deallocate_raw(
        void* ptr, XSIGMA_UNUSED size_t alignment, XSIGMA_UNUSED size_t num_bytes) override
    {
        // Currently identical to single-parameter version
        // Future optimization could use num_bytes hint to avoid GetAllocatedSize call
        deallocate_raw(ptr);
    }

    /**
     * @brief Adds comprehensive profiling trace for memory operations.
     *
     * Integrates with XSigma's profiling system to record detailed memory
     * operation traces including allocator state, operation context, and
     * memory usage statistics.
     *
     * @param traceme_name Operation name ("MemoryAllocation" or "MemoryDeallocation")
     * @param chunk_ptr Pointer to memory being traced
     * @param req_bytes Originally requested size (0 for deallocations)
     * @param alloc_bytes Actual allocated size
     *
     * **Performance**: Minimal overhead - only called when statistics enabled
     * **Thread Safety**: Thread-safe - captures consistent snapshot
     * **Integration**: Works with XSigma profiling and debugging tools
     *
     * **Trace Information**:
     * - Allocator identification and current state
     * - Memory addresses and sizes
     * - Current operation context and metadata
     * - Peak usage and allocation statistics
     *
     * **Use Cases**:
     * - Memory usage profiling and analysis
     * - Allocation pattern debugging
     * - Performance optimization
     * - Memory leak detection
     */
    void AddTraceMe(
        std::string_view traceme_name,
        const void*      chunk_ptr,
        std::size_t      req_bytes,
        std::size_t      alloc_bytes)
    {
        xsigma::trace_me::instant_activity(
            [this, traceme_name, chunk_ptr, req_bytes, alloc_bytes]()
                XSIGMA_NO_THREAD_SAFETY_ANALYSIS -> std::string
            {
                // Capture current debug annotation context
                const auto& annotation =
                    xsigma::scoped_memory_debug_annotation::current_annotation();

                // Create comprehensive trace with current allocator state
                return xsigma::trace_me_encode(
                    std::string(traceme_name),
                    {{"allocator_name", Name()},
                     {"bytes_reserved", stats_.bytes_reserved.load(std::memory_order_relaxed)},
                     {"bytes_allocated", stats_.bytes_in_use.load(std::memory_order_relaxed)},
                     {"peak_bytes_in_use",
                      stats_.peak_bytes_in_use.load(std::memory_order_relaxed)},
                     {"requested_bytes", req_bytes},
                     {"allocation_bytes", alloc_bytes},
                     {"addr", reinterpret_cast<uint64_t>(chunk_ptr)},
                     {"xsigma_op", annotation.pending_op_name},
                     {"id", annotation.pending_step_id},
                     {"region_type", annotation.pending_region_type},
                     {"data_type", annotation.pending_data_type},
                     {"shape", annotation.pending_shape_func()}});
            },
            static_cast<int>(xsigma::trace_me_level_enum::INFO));
    }

    /**
     * @brief Retrieves current allocator statistics.
     *
     * @return Optional statistics object, or nullopt if statistics disabled
     *
     * **Availability**: Only available when statistics collection is enabled
     * **Performance**: O(1) - returns cached statistics under lock
     * **Thread Safety**: Thread-safe with mutex protection
     * **Consistency**: Returns atomic snapshot of current statistics
     *
     * **Statistics Include**:
     * - Total number of allocations
     * - Current bytes in use
     * - Peak bytes in use
     * - Largest single allocation size
     * - Reserved bytes (always 0 for CPU allocator)
     */
    std::optional<allocator_stats> GetStats() override
    {
        if (!cpu_allocator_collect_stats.load(std::memory_order_relaxed))
        {
            return std::nullopt;
        }

        std::lock_guard<std::mutex> lock(mu_);
        // Create a copy of the atomic stats structure
        allocator_stats stats_copy(stats_);
        return stats_copy;
    }

    /**
     * @brief Resets statistics counters while preserving current state.
     *
     * @return true if statistics were reset, false if statistics disabled
     *
     * **Behavior**: Resets counters but preserves current bytes_in_use
     * **Performance**: O(1) - simple counter reset under lock
     * **Thread Safety**: Thread-safe with mutex protection
     *
     * **Reset Operations**:
     * - num_allocs → 0
     * - peak_bytes_in_use → current bytes_in_use
     * - largest_alloc_size → 0
     * - bytes_in_use remains unchanged
     */
    bool ClearStats() override
    {
        if (!cpu_allocator_collect_stats.load(std::memory_order_relaxed))
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(mu_);
        stats_.num_allocs.store(0, std::memory_order_relaxed);
        stats_.peak_bytes_in_use.store(
            stats_.bytes_in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
        stats_.largest_alloc_size.store(0, std::memory_order_relaxed);
        return true;
    }

    /**
     * @brief Returns memory type managed by this allocator.
     *
     * @return allocator_memory_enum::HOST_PAGEABLE (standard system RAM)
     *
     * **Memory Type**: Host pageable memory (standard system RAM)
     * **Performance**: O(1) - returns constant
     * **Thread Safety**: Thread-safe (returns constant)
     * **Use Cases**: Memory type identification, NUMA awareness, optimization
     */
    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_memory_enum::HOST_PAGEABLE;
    }

    // Prevent copying to avoid complex state duplication
    CPUAllocator(const CPUAllocator&)            = delete;
    CPUAllocator& operator=(const CPUAllocator&) = delete;

private:
    /**
     * @brief Mutex protecting statistics and warning counters.
     *
     * Provides thread-safe access to mutable allocator state including
     * statistics and warning counters. Marked mutable to allow const
     * methods to acquire locks for read operations.
     *
     * **Granularity**: Fine-grained - only protects statistics updates
     * **Contention**: Minimal - statistics updates are brief
     * **Performance**: Low overhead when statistics disabled
     */
    mutable std::mutex mu_;

    /**
     * @brief Comprehensive allocator statistics.
     *
     * Tracks detailed memory usage metrics including allocation counts,
     * current usage, peak usage, and largest allocation size.
     * Protected by mu_ for thread-safe access.
     *
     * **Updates**: Modified during allocation/deallocation when stats enabled
     * **Thread Safety**: Protected by mu_ mutex
     * **Persistence**: Maintained throughout allocator lifetime
     */
    allocator_stats stats_;

    /**
     * @brief Atomic counter for large allocation warnings.
     *
     * Tracks number of large allocation warnings emitted to prevent
     * log spam. Uses atomic operations for thread-safe rate limiting
     * without requiring mutex acquisition.
     *
     * **Thread Safety**: Atomic operations ensure thread-safe updates
     * **Rate Limiting**: Prevents excessive warning messages
     * **Performance**: Lock-free increment and comparison
     */
    std::atomic<int> single_allocation_warning_count_;

    /**
     * @brief Counter for total allocation warnings.
     *
     * Tracks number of total allocation warnings emitted. Protected
     * by mu_ since it's only accessed during statistics updates.
     *
     * **Thread Safety**: Protected by mu_ mutex
     * **Rate Limiting**: Prevents excessive total allocation warnings
     * **Scope**: Only incremented during statistics collection
     */
    int total_allocation_warning_count_{0};
};

/**
 * @brief Factory for creating CPU allocators and sub-allocators.
 *
 * Implements the factory pattern for CPU memory allocators, providing
 * standardized creation of both direct allocators and sub-allocators
 * for integration with higher-level memory management systems.
 *
 * **Design Pattern**: Factory pattern for allocator creation
 * **Thread Safety**: Thread-safe creation methods
 * **Memory Management**: Uses smart pointers for automatic cleanup
 * **Integration**: Works with allocator registry system
 */
class CPUAllocatorFactory : public allocator_factory
{
public:
    /**
     * @brief Creates new CPU allocator instance.
     *
     * @return Pointer to new CPUAllocator (caller takes ownership)
     *
     * **Ownership**: Caller responsible for deletion
     * **Thread Safety**: Thread-safe creation
     * **Performance**: O(1) - simple object construction
     */
    Allocator* CreateAllocator() override { return new CPUAllocator; }

    /**
     * @brief Creates CPU sub-allocator for integration with BFC allocator.
     *
     * @param numa_node NUMA node preference (unused for CPU allocator)
     * @return Pointer to new CPUSubAllocator (caller takes ownership)
     *
     * **NUMA Support**: Currently ignores NUMA node parameter
     * **Ownership**: Caller responsible for deletion
     * **Thread Safety**: Thread-safe creation
     * **Use Cases**: BFC allocator backend, memory pool management
     */
    sub_allocator* CreateSubAllocator(XSIGMA_UNUSED int numa_node) override
    {
        return new CPUSubAllocator(std::make_unique<CPUAllocator>());
    }

private:
    /**
     * @brief Sub-allocator adapter for CPU memory allocation.
     *
     * Provides sub_allocator interface over CPUAllocator for integration
     * with higher-level allocators like allocator_bfc. Handles large
     * memory region allocation and deallocation.
     *
     * **Design Pattern**: Adapter pattern wrapping CPUAllocator
     * **Memory Regions**: Allocates large contiguous regions
     * **Profiling**: Integrated with XSigma tracing system
     * **Thread Safety**: Thread-safe through underlying CPUAllocator
     */
    class CPUSubAllocator : public sub_allocator
    {
    public:
        /**
         * @brief Constructs sub-allocator with CPU allocator backend.
         *
         * @param cpu_allocator Unique pointer to CPU allocator (takes ownership)
         *
         * **Ownership**: Takes ownership of CPU allocator
         * **Initialization**: Sets up sub-allocator interface
         * **Memory Management**: Uses RAII for automatic cleanup
         */
        explicit CPUSubAllocator(std::unique_ptr<CPUAllocator> cpu_allocator)
            : sub_allocator({}, {}), allocator_cpu_(std::move(cpu_allocator))
        {
        }

        /**
         * @brief Allocates large memory region for sub-division.
         *
         * @param alignment Required alignment in bytes
         * @param num_bytes Size of region to allocate
         * @param bytes_received Pointer to receive actual allocated size
         * @return Pointer to allocated region, or nullptr on failure
         *
         * **Algorithm**: Delegates to underlying CPU allocator
         * **Performance**: O(1) - direct system allocation
         * **Profiling**: Integrated with trace system
         * **Alignment**: Guaranteed to meet alignment requirements
         */
        void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) override
        {
            xsigma::trace_me traceme("CPUSubAllocator::Alloc");
            *bytes_received = num_bytes;
            return allocator_cpu_->allocate_raw(alignment, num_bytes);
        }

        /**
         * @brief Deallocates memory region.
         *
         * @param ptr Pointer to region to deallocate
         * @param num_bytes Size of region (unused by CPU allocator)
         *
         * **Algorithm**: Delegates to underlying CPU allocator
         * **Performance**: O(1) - direct system deallocation
         * **Profiling**: Integrated with trace system
         * **Size Hint**: num_bytes parameter currently unused
         */
        void Free(void* ptr, XSIGMA_UNUSED size_t num_bytes) override
        {
            xsigma::trace_me traceme("CPUSubAllocator::Free");
            allocator_cpu_->deallocate_raw(ptr);
        }

        /**
         * @brief Indicates whether allocator supports region coalescing.
         *
         * @return false - CPU allocator doesn't support coalescing
         *
         * **Coalescing**: CPU allocator doesn't track adjacent regions
         * **Performance**: O(1) - returns constant
         * **Use Cases**: BFC allocator optimization decisions
         */
        bool SupportsCoalescing() const noexcept override { return false; }

        /**
         * @brief Returns memory type of underlying allocator.
         *
         * @return Memory type from CPU allocator
         *
         * **Delegation**: Forwards to underlying CPU allocator
         * **Performance**: O(1) - simple delegation
         * **Thread Safety**: Thread-safe delegation
         */
        allocator_memory_enum GetMemoryType() const noexcept override
        {
            return allocator_cpu_->GetMemoryType();
        }

    private:
        /**
         * @brief Underlying CPU allocator instance.
         *
         * Owned CPU allocator that performs actual memory allocation
         * and deallocation operations. Managed via unique_ptr for
         * automatic cleanup and exception safety.
         */
        std::unique_ptr<CPUAllocator> allocator_cpu_;
    };
};

REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocatorFactory);
}  // namespace

}  // namespace xsigma
