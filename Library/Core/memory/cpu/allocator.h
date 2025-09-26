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

#pragma once

#include <stdlib.h>

#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

//#include "memory/cpu/mem.h"
#include "memory/numa.h"
#include "memory/unified_memory_stats.h"
#include "util/exception.h"
#include "util/logger.h"

namespace xsigma
{
constexpr int NUMANOAFFINITY = -1;
/**
 * @brief Attributes for a single allocation call specifying allocation behavior and constraints.
 *
 * Different calls to the same allocator can have different allocation attributes to control
 * retry behavior, logging, and memory lifecycle management. This structure provides fine-grained
 * control over allocation policies.
 *
 * @note This class is move-only to prevent accidental copying of function pointers.
 *
 * **Thread Safety**: Individual instances are not thread-safe, but multiple instances
 * can be used concurrently across different threads.
 *
 * **Performance**: O(1) construction and access to all members.
 */
struct allocation_attributes
{
    /**
     * @brief Default constructor with standard allocation behavior.
     *
     * Creates attributes with retry enabled, logging disabled, and no timing constraints.
     */
    allocation_attributes() = default;

    /**
     * @brief Constructs allocation attributes with specific behavior settings.
     *
     * @param retry_on_failure Whether to retry allocation on initial failure
     * @param allocation_will_be_logged Whether this allocation should be logged for tracking
     * @param freed_by_func Optional function providing timing constraints for memory reuse
     *
     * **Example Usage**:
     * ```cpp
     * // Critical allocation that should retry on failure
     * allocation_attributes critical_attrs(true, true, nullptr);
     *
     * // Optional scratch space that shouldn't retry
     * allocation_attributes scratch_attrs(false, false, nullptr);
     * ```
     */
    allocation_attributes(
        bool                       retry_on_failure,
        bool                       allocation_will_be_logged,
        std::function<uint64_t()>* freed_by_func) noexcept
        : retry_on_failure(retry_on_failure),
          allocation_will_be_logged(allocation_will_be_logged),
          freed_by_func(freed_by_func)
    {
    }

    /**
     * @brief Move constructor for efficient transfer of attributes.
     */
    allocation_attributes(allocation_attributes&&) noexcept = default;

    /**
     * @brief Move assignment operator for efficient transfer of attributes.
     */
    allocation_attributes& operator=(allocation_attributes&&) noexcept = default;

    /**
     * @brief Controls retry behavior on allocation failure.
     *
     * If true, the allocator will wait and retry allocation with exponential backoff
     * when the initial attempt fails. Set to false for optional allocations where
     * failure has only performance impact (e.g., scratch space allocation).
     *
     * **Default**: true (retry enabled)
     * **Performance Impact**: Retries can add significant latency but improve success rate
     */
    bool retry_on_failure = true;

    /**
     * @brief Controls whether allocation is tracked in execution logs.
     *
     * When true, this allocation will be properly attributed to the executing operation
     * for debugging and profiling purposes. Should be set to true for all tensor
     * allocations during normal execution.
     *
     * **Default**: false (logging disabled)
     * **Performance Impact**: Minimal overhead for tracking metadata
     */
    bool allocation_will_be_logged = false;

    /**
     * @brief Optional timing constraint function for memory reuse policies.
     *
     * **EXPERIMENTAL FEATURE**: When provided, this function returns a timing count
     * that constrains which freed memory chunks can be reused. Only chunks with
     * freed_at_count <= returned value are eligible for reuse.
     *
     * **Ownership**: Pointer is not owned by this structure - caller must ensure lifetime
     * **Thread Safety**: Function must be thread-safe if used across multiple threads
     * **Performance**: Called during allocation - should be fast (O(1) preferred)
     *
     * @warning This is an experimental feature and may change in future versions
     */
    std::function<uint64_t()>* freed_by_func = nullptr;  // Not owned.

    // Prevent accidental copying which could lead to dangling function pointers
    allocation_attributes(const allocation_attributes&)            = delete;
    allocation_attributes& operator=(const allocation_attributes&) = delete;
};

/**
 * @brief Enumeration of memory types supported by allocators.
 *
 * Specifies the physical location and characteristics of allocated memory.
 * This information is crucial for performance optimization, data transfer
 * planning, and ensuring memory access patterns are appropriate for the
 * target hardware.
 *
 * **Performance Implications**:
 * - HOST_PAGEABLE: Standard system memory, may be swapped to disk
 * - HOST_PINNED: Locked in physical RAM, faster for device transfers
 * - DEVICE: On-device memory (GPU/accelerator), highest bandwidth for compute
 *
 * **Thread Safety**: Enum values are immutable and thread-safe
 */
enum class allocator_memory_enum : uint8_t
{
    /**
     * @brief Memory type is unknown or not specified.
     *
     * Default value when memory type cannot be determined or is not relevant.
     * Should be avoided in performance-critical code paths.
     */
    UNKNOWN = 0,

    /**
     * @brief Memory located on a compute device (GPU, TPU, etc.).
     *
     * Provides highest bandwidth for device computations but may have
     * limited capacity and require special access patterns.
     *
     * **Characteristics**:
     * - High bandwidth for device operations
     * - Limited capacity compared to host memory
     * - May require asynchronous transfers to/from host
     */
    DEVICE = 1,

    /**
     * @brief Pageable host memory that can be swapped to disk.
     *
     * Standard system memory allocated through malloc/new. May be paged
     * out to disk by the operating system under memory pressure.
     *
     * **Characteristics**:
     * - Large capacity (limited by virtual memory)
     * - May be swapped to disk, causing access latency
     * - Standard CPU cache hierarchy applies
     */
    HOST_PAGEABLE = 2,

    /**
     * @brief Pinned (page-locked) host memory that cannot be swapped.
     *
     * Memory locked in physical RAM, providing faster and more predictable
     * access times. Often used for DMA transfers to/from devices.
     *
     * **Characteristics**:
     * - Guaranteed to remain in physical RAM
     * - Faster device transfer rates
     * - Limited by physical RAM capacity
     * - Higher system resource usage
     */
    HOST_PINNED = 3,
};

/**
 * @brief Abstract interface for high-performance memory allocation and deallocation.
 *
 * The Allocator class provides a unified interface for memory management across different
 * memory types (host, device, pinned) and allocation strategies (pooled, tracking, etc.).
 * All allocators guarantee proper alignment and provide optional statistics collection.
 *
 * **Design Principles**:
 * - Zero-overhead abstraction when statistics are disabled
 * - Consistent alignment guarantees across all implementations
 * - Support for both simple and attribute-based allocation
 * - Thread-safe implementations (implementation-dependent)
 *
 * **Performance Characteristics**:
 * - allocate_raw(): O(1) to O(log n) depending on implementation
 * - deallocate_raw(): O(1) to O(log n) depending on implementation
 * - Statistics collection adds minimal overhead when enabled
 *
 * **Memory Alignment**: All allocations are aligned to kAllocatorAlignment (64 bytes)
 * for optimal cache performance and SIMD instruction compatibility.
 *
 * @note Implementations must be thread-safe unless explicitly documented otherwise.
 */
class XSIGMA_API Allocator
{
public:
    /**
     * @brief Default memory alignment boundary for all allocations.
     *
     * 64-byte alignment ensures optimal performance for:
     * - CPU cache line alignment (typically 64 bytes)
     * - SIMD instructions (AVX-512 requires 64-byte alignment)
     * - GPU memory coalescing requirements
     *
     * **Compile-time constant**: Can be used in constexpr contexts
     */
    static constexpr size_t kAllocatorAlignment = 64;

    /**
     * @brief Virtual destructor ensuring proper cleanup of derived classes.
     *
     * **Thread Safety**: Destructor calls must be externally synchronized
     * **Exception Safety**: noexcept - implementations should not throw
     */
    virtual ~Allocator() = default;

    /**
     * @brief Returns a human-readable identifier for this allocator instance.
     *
     * Used for debugging, logging, and performance profiling. Should be
     * descriptive enough to distinguish between different allocator types
     * and configurations.
     *
     * @return String identifier (e.g., "allocator_bfc", "cpu_allocator", "gpu_allocator")
     *
     * **Performance**: O(1) - should return a cached string
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw
     *
     * **Example**:
     * ```cpp
     * auto allocator = std::make_unique<allocator_bfc>(...);
     * std::cout << "Using: " << allocator->Name() << std::endl;
     * // Output: "Using: allocator_bfc"
     * ```
     */
    virtual std::string Name() = 0;

    /**
     * @brief Allocates an uninitialized memory block with specified alignment.
     *
     * Returns a pointer to a memory block of at least num_bytes size, aligned
     * to the specified boundary. The memory content is uninitialized.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Requirements**:
     * - alignment must be a power of 2
     * - alignment must be >= sizeof(void*)
     * - num_bytes can be 0 (implementation-defined behavior)
     *
     * **Performance**: Implementation-dependent, typically O(1) to O(log n)
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw, return nullptr on failure
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(32, 1024);  // 1KB aligned to 32 bytes
     * if (ptr) {
     *     // Use memory...
     *     allocator->deallocate_raw(ptr);
     * }
     * ```
     */
    virtual void* allocate_raw(size_t alignment, size_t num_bytes) = 0;

    /**
     * @brief Allocates memory with specific allocation attributes and policies.
     *
     * Extended allocation interface supporting retry policies, logging,
     * and timing constraints. Provides fine-grained control over allocation
     * behavior for performance-critical applications.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @param allocation_attr Attributes controlling allocation behavior
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Default Implementation**: Delegates to simple allocate_raw() method
     * **Override Recommendation**: Implement for allocators supporting advanced features
     *
     * **Performance**: May be slower than simple allocate_raw() due to attribute processing
     * **Thread Safety**: Must be thread-safe
     *
     * **Example**:
     * ```cpp
     * allocation_attributes attrs(true, true, nullptr);  // retry=true, log=true
     * void* ptr = allocator->allocate_raw(64, 2048, attrs);
     * ```
     */
    virtual void* allocate_raw(
        size_t                                     alignment,
        size_t                                     num_bytes,
        XSIGMA_UNUSED const allocation_attributes& allocation_attr)
    {
        // Default implementation ignores attributes and delegates to simple version
        return allocate_raw(alignment, num_bytes);
    }

    /**
     * @brief Deallocates a previously allocated memory block.
     *
     * Releases memory pointed to by ptr back to the allocator. The pointer
     * must have been returned by a previous call to allocate_raw() on the
     * same allocator instance.
     *
     * @param ptr Pointer to memory block to deallocate (must not be nullptr)
     *
     * **Requirements**:
     * - ptr must have been returned by allocate_raw() on this allocator
     * - ptr must not be nullptr (undefined behavior)
     * - ptr must not be used after this call (undefined behavior)
     *
     * **Performance**: Implementation-dependent, typically O(1) to O(log n)
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(64, 1024);
     * // ... use memory ...
     * allocator->deallocate_raw(ptr);  // ptr is now invalid
     * ```
     */
    virtual void deallocate_raw(void* ptr) = 0;

    /**
     * @brief Extended deallocation with size and alignment hints.
     *
     * Some allocators can optimize deallocation when provided with the original
     * allocation size and alignment. This information can enable faster
     * deallocation and better memory pool management.
     *
     * @param ptr Pointer to memory block to deallocate
     * @param alignment Original alignment used for allocation
     * @param num_bytes Original size requested for allocation
     *
     * **Default Implementation**: Ignores hints and delegates to simple deallocate_raw()
     * **Override Recommendation**: Implement for allocators that can benefit from size hints
     *
     * **Performance**: May be faster than simple deallocate_raw() for some allocators
     * **Thread Safety**: Must be thread-safe
     */
    virtual void deallocate_raw(void* ptr, size_t alignment, size_t num_bytes)
    {
        // Default implementation ignores size hints
        static_cast<void>(alignment);
        static_cast<void>(num_bytes);
        deallocate_raw(ptr);
    }

    /**
     * @brief Indicates whether this allocator tracks allocation sizes and metadata.
     *
     * When true, the allocator maintains detailed information about each allocation
     * including requested size, actual allocated size, and unique identifiers.
     * This enables advanced debugging and profiling capabilities.
     *
     * @return true if size tracking is enabled, false otherwise
     *
     * **Implementation Note**: If overridden to return true, must also override:
     * - RequestedSize()
     * - AllocatedSize()
     * - AllocationId()
     *
     * **Performance Impact**: Size tracking adds memory overhead and slight CPU cost
     * **Thread Safety**: Must be thread-safe and return consistent values
     * **Default**: false (no tracking for optimal performance)
     *
     * **Example**:
     * ```cpp
     * if (allocator->TracksAllocationSizes()) {
     *     size_t actual_size = allocator->AllocatedSize(ptr);
     *     size_t requested_size = allocator->RequestedSize(ptr);
     *     std::cout << "Overhead: " << (actual_size - requested_size) << " bytes\n";
     * }
     * ```
     */
    virtual bool TracksAllocationSizes() const noexcept { return false; }

    /**
     * @brief Indicates whether allocator returns opaque handles instead of memory pointers.
     *
     * Special-purpose allocators may return opaque handles for tracking tensor usage
     * rather than actual memory pointers. This is used by advanced memory management
     * systems that need to intercept all memory access.
     *
     * @return true if allocator returns opaque handles, false for normal memory pointers
     *
     * **Important**: When true:
     * - allocate_raw() should be called even for num_bytes=0
     * - Returned pointers are handles, not memory addresses
     * - No constructors/destructors should be run on "allocated" memory
     * - Caller must track handle vs. pointer semantics
     *
     * **Performance**: Handle-based allocators may have different performance characteristics
     * **Thread Safety**: Must be thread-safe and return consistent values
     * **Default**: false (returns actual memory pointers)
     *
     * **Use Cases**:
     * - Memory usage tracking and profiling
     * - Lazy allocation strategies
     * - Memory access pattern analysis
     */
    virtual bool AllocatesOpaqueHandle() const noexcept { return false; }

    /**
     * @brief Returns the user-requested size for a previously allocated pointer.
     *
     * Retrieves the original size requested by the user when the memory was allocated.
     * The actual allocated size may be larger due to alignment or allocator overhead.
     *
     * @param ptr Pointer to previously allocated memory (must not be nullptr)
     * @return Original requested size in bytes
     *
     * **Requirements**:
     * - TracksAllocationSizes() must return true
     * - ptr must not be nullptr
     * - ptr must have been allocated by this allocator instance
     *
     * **Performance**: O(1) to O(log n) depending on tracking implementation
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw, logs error and returns 0 on failure
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(64, 1000);
     * if (allocator->TracksAllocationSizes()) {
     *     size_t requested = allocator->RequestedSize(ptr);  // Returns 1000
     *     size_t actual = allocator->AllocatedSize(ptr);     // May return 1024
     * }
     * ```
     */
    virtual size_t RequestedSize(XSIGMA_UNUSED const void* ptr) const
    {
        //XSIGMA_LOG_ERROR("Allocator '" << Name() << "' doesn't track allocation sizes");
        return size_t{0};
    }

    /**
     * @brief Returns the actual allocated size for a previously allocated pointer.
     *
     * Retrieves the actual size of the memory block allocated, which may be larger
     * than the requested size due to alignment requirements or allocator overhead.
     * Always >= RequestedSize(ptr).
     *
     * @param ptr Pointer to previously allocated memory (must not be nullptr)
     * @return Actual allocated size in bytes
     *
     * **Requirements**:
     * - TracksAllocationSizes() must return true
     * - ptr must not be nullptr
     * - ptr must have been allocated by this allocator instance
     *
     * **Guarantee**: AllocatedSize(ptr) >= RequestedSize(ptr)
     * **Performance**: O(1) to O(log n) depending on tracking implementation
     * **Thread Safety**: Must be thread-safe
     * **Default Implementation**: Returns RequestedSize(ptr)
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(32, 100);  // Request 100 bytes, 32-byte aligned
     * size_t actual = allocator->AllocatedSize(ptr);  // May return 128 due to alignment
     * ```
     */
    virtual size_t AllocatedSize(const void* ptr) const { return RequestedSize(ptr); }

    /**
     * @brief Returns a unique identifier for the allocation, if available.
     *
     * Provides a unique ID assigned to each allocation for tracking and debugging
     * purposes. IDs are unique within this allocator instance and non-zero for
     * valid allocations.
     *
     * @param ptr Pointer to previously allocated memory (must not be nullptr)
     * @return Unique allocation ID (>0), or 0 if not available
     *
     * **Requirements**:
     * - TracksAllocationSizes() must return true
     * - ptr must not be nullptr
     * - ptr must have been allocated by this allocator instance
     *
     * **Uniqueness**: Each allocation gets a different non-zero ID
     * **Performance**: O(1) to O(log n) depending on tracking implementation
     * **Thread Safety**: Must be thread-safe
     * **Default**: Returns 0 (no ID tracking)
     *
     * **Use Cases**:
     * - Memory leak detection
     * - Allocation lifetime tracking
     * - Performance profiling correlation
     *
     * **Example**:
     * ```cpp
     * void* ptr1 = allocator->allocate_raw(64, 1024);
     * void* ptr2 = allocator->allocate_raw(64, 1024);
     * int64_t id1 = allocator->AllocationId(ptr1);  // e.g., returns 1
     * int64_t id2 = allocator->AllocationId(ptr2);  // e.g., returns 2
     * assert(id1 != id2);  // IDs are unique
     * ```
     */
    virtual int64_t AllocationId(XSIGMA_UNUSED const void* ptr) const noexcept { return 0; }

    /**
     * @brief Returns allocated size with potentially slow computation.
     *
     * Attempts to determine the allocated size even when TracksAllocationSizes()
     * returns false. May use expensive system calls or memory introspection.
     * Should only be used for debugging or when performance is not critical.
     *
     * @param ptr Pointer to previously allocated memory (must not be nullptr)
     * @return Allocated size in bytes, or 0 if cannot be determined
     *
     * **Requirements**:
     * - ptr must not be nullptr
     * - ptr must have been allocated by this allocator instance
     *
     * **Performance**: Can be extremely slow (O(n) or system call overhead)
     * **Thread Safety**: Must be thread-safe
     * **Use Cases**: Debugging, memory analysis tools, diagnostics
     *
     * **Implementation Strategy**:
     * - If TracksAllocationSizes() is true: delegate to fast AllocatedSize()
     * - Otherwise: attempt system-specific size queries (malloc_usable_size, etc.)
     * - Return 0 if size cannot be determined
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(64, 1024);
     * // This may be slow but works even without size tracking
     * size_t size = allocator->AllocatedSizeSlow(ptr);
     * ```
     */
    virtual size_t AllocatedSizeSlow(const void* ptr) const
    {
        if constexpr (true)
        {  // Use C++17 if constexpr for potential optimization
            if (TracksAllocationSizes())
            {
                return AllocatedSize(ptr);
            }
        }
        return 0;
    }

    /**
     * @brief Retrieves comprehensive allocator statistics.
     *
     * Returns detailed runtime statistics about allocator performance and memory usage.
     * Statistics may include allocation counts, memory usage, fragmentation metrics,
     * and performance counters.
     *
     * @return Optional statistics structure, std::nullopt if not supported
     *
     * **Performance**: O(1) for cached stats, O(n) if computed on-demand
     * **Thread Safety**: Must be thread-safe
     * **Consistency**: Statistics should represent a consistent snapshot
     *
     * **Implementation Notes**:
     * - Return std::nullopt for allocators without statistics support
     * - Consider caching expensive computations
     * - Ensure thread-safe access to internal counters
     *
     * **Example**:
     * ```cpp
     * if (auto stats = allocator->GetStats()) {
     *     std::cout << "Memory in use: " << stats->bytes_in_use << " bytes\n";
     *     std::cout << "Peak usage: " << stats->peak_bytes_in_use << " bytes\n";
     *     std::cout << "Allocations: " << stats->num_allocs << "\n";
     * }
     * ```
     */
    virtual std::optional<allocator_stats> GetStats() { return std::nullopt; }

    /**
     * @brief Resets allocator statistics to initial state.
     *
     * Clears accumulated statistics while preserving current usage information.
     * Peak values are reset to current values, counters are reset to zero.
     *
     * @return true if statistics were cleared, false if not supported
     *
     * **Requirements**: GetStats() must be overridden to return valid statistics
     * **Behavior**:
     * - Preserves: bytes_in_use, allocated memory state
     * - Resets: peak_bytes_in_use = bytes_in_use
     * - Clears: num_allocs, allocation counters
     *
     * **Performance**: O(1) - simple counter reset
     * **Thread Safety**: Must be thread-safe
     * **Use Cases**: Periodic monitoring, benchmark resets, memory profiling
     *
     * **Example**:
     * ```cpp
     * // Reset stats before benchmark
     * if (allocator->ClearStats()) {
     *     run_benchmark();
     *     auto stats = allocator->GetStats();
     *     // stats now reflect only benchmark allocations
     * }
     * ```
     */
    virtual bool ClearStats() { return false; }

    /**
     * @brief Sets the safe frontier for timestamped memory management.
     *
     * **EXPERIMENTAL FEATURE**: Establishes a timestamp boundary for safe memory reuse.
     * Memory freed before this timestamp can be safely reused without synchronization
     * concerns. Used by advanced allocators with temporal memory safety guarantees.
     *
     * @param count Timestamp representing the safe frontier
     *
     * **Thread Safety**: Must be thread-safe
     * **Performance**: O(1) - simple atomic update
     * **Default**: No-op for allocators without timestamp support
     *
     * **Use Cases**:
     * - GPU memory management with stream synchronization
     * - Asynchronous computation memory safety
     * - Temporal memory reuse optimization
     *
     * @warning This is an experimental feature subject to change
     */
    virtual void SetSafeFrontier(XSIGMA_UNUSED uint64_t count) noexcept {}

    /**
     * @brief Returns the type of memory managed by this allocator.
     *
     * Indicates the physical location and characteristics of memory returned
     * by this allocator. Critical for performance optimization and ensuring
     * appropriate usage patterns.
     *
     * @return Memory type enumeration value
     *
     * **Performance**: O(1) - should return cached value
     * **Thread Safety**: Must be thread-safe and return consistent values
     * **Default**: UNKNOWN (should be overridden by implementations)
     *
     * **Usage Guidelines**:
     * - HOST_PAGEABLE: Standard CPU memory, may be swapped
     * - HOST_PINNED: Locked CPU memory, optimal for device transfers
     * - DEVICE: GPU/accelerator memory, highest compute bandwidth
     * - UNKNOWN: Avoid in performance-critical code
     *
     * **Example**:
     * ```cpp
     * switch (allocator->GetMemoryType()) {
     *     case allocator_memory_enum::DEVICE:
     *         // Optimize for device computation
     *         break;
     *     case allocator_memory_enum::HOST_PINNED:
     *         // Optimize for host-device transfers
     *         break;
     *     default:
     *         // Generic handling
     *         break;
     * }
     * ```
     */
    virtual allocator_memory_enum GetMemoryType() const noexcept
    {
        return allocator_memory_enum::UNKNOWN;
    }
};

/**
 * @brief Decorator pattern implementation for extending allocator functionality.
 *
 * allocator_wrapper provides a base class for implementing the decorator pattern
 * with allocators. It delegates all calls to a wrapped allocator while allowing
 * derived classes to override specific methods to add functionality like logging,
 * statistics collection, or access control.
 *
 * **Design Pattern**: Decorator - adds behavior without altering the interface
 * **Performance**: Zero overhead delegation - all calls are simple forwards
 * **Thread Safety**: Inherits thread safety from wrapped allocator
 *
 * **Common Use Cases**:
 * - Adding logging to existing allocators
 * - Implementing access control or quotas
 * - Collecting additional statistics
 * - Adding debugging instrumentation
 *
 * **Example Usage**:
 * ```cpp
 * class LoggingAllocator : public allocator_wrapper {
 * public:
 *     LoggingAllocator(Allocator* wrapped) : allocator_wrapper(wrapped) {}
 *
 *     void* allocate_raw(size_t alignment, size_t num_bytes) override {
 *         std::cout << "Allocating " << num_bytes << " bytes\n";
 *         return allocator_wrapper::allocate_raw(alignment, num_bytes);
 *     }
 * };
 * ```
 */
class allocator_wrapper : public Allocator
{
public:
    /**
     * @brief Constructs wrapper around an existing allocator.
     *
     * @param wrapped Pointer to allocator to wrap (must not be nullptr)
     *
     * **Ownership**: Does not take ownership of wrapped allocator
     * **Lifetime**: Wrapped allocator must outlive this wrapper
     * **Thread Safety**: Safe if wrapped allocator is thread-safe
     */
    explicit allocator_wrapper(Allocator* wrapped) noexcept : wrapped_(wrapped)
    {
        assert(wrapped != nullptr && "Wrapped allocator cannot be nullptr");
    }

    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     *
     * **Note**: Does not delete the wrapped allocator - caller retains ownership
     */
    ~allocator_wrapper() override = default;

    /**
     * @brief Returns the wrapped allocator instance.
     *
     * Provides access to the underlying allocator for direct manipulation
     * or unwrapping in decorator chains.
     *
     * @return Pointer to wrapped allocator (never nullptr)
     *
     * **Thread Safety**: Safe to call concurrently
     * **Performance**: O(1) - simple pointer access
     */
    Allocator* wrapped() const noexcept { return wrapped_; }

    // Delegated interface methods - all forward to wrapped allocator
    std::string Name() override { return wrapped_->Name(); }

    void* allocate_raw(size_t alignment, size_t num_bytes) override
    {
        return wrapped_->allocate_raw(alignment, num_bytes);
    }

    void* allocate_raw(
        size_t alignment, size_t num_bytes, const allocation_attributes& allocation_attr) override
    {
        return wrapped_->allocate_raw(alignment, num_bytes, allocation_attr);
    }

    void deallocate_raw(void* ptr) override { wrapped_->deallocate_raw(ptr); }

    void deallocate_raw(void* ptr, size_t alignment, size_t num_bytes) override
    {
        wrapped_->deallocate_raw(ptr, alignment, num_bytes);
    }

    bool TracksAllocationSizes() const noexcept override
    {
        return wrapped_->TracksAllocationSizes();
    }

    bool AllocatesOpaqueHandle() const noexcept override
    {
        return wrapped_->AllocatesOpaqueHandle();
    }

    size_t RequestedSize(const void* ptr) const override { return wrapped_->RequestedSize(ptr); }

    size_t AllocatedSize(const void* ptr) const override { return wrapped_->AllocatedSize(ptr); }

    int64_t AllocationId(const void* ptr) const noexcept override
    {
        return wrapped_->AllocationId(ptr);
    }

    size_t AllocatedSizeSlow(const void* ptr) const override
    {
        return wrapped_->AllocatedSizeSlow(ptr);
    }

    std::optional<allocator_stats> GetStats() override { return wrapped_->GetStats(); }

    bool ClearStats() override { return wrapped_->ClearStats(); }

    void SetSafeFrontier(uint64_t count) noexcept override { wrapped_->SetSafeFrontier(count); }

    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return wrapped_->GetMemoryType();
    }

private:
    Allocator* const wrapped_;  ///< Wrapped allocator instance (not owned)
};

/**
 * @brief Specifies memory allocation requirements and device compatibility constraints.
 *
 * allocator_attributes provides a flexible way to specify memory allocation requirements
 * when operations need access to different types of memory (host, device, pinned, etc.).
 * This enables fine-grained control over memory placement for optimal performance.
 *
 * **Design**: Bit-packed structure for efficient storage and comparison
 * **Performance**: O(1) operations for all attribute queries and modifications
 * **Thread Safety**: Individual instances are not thread-safe, but immutable after construction
 *
 * **Use Cases**:
 * - Cross-device memory allocation (GPU ops needing CPU memory)
 * - DMA-optimized memory regions
 * - Network-compatible memory for distributed computing
 * - Specialized allocator selection
 *
 * **Example Usage**:
 * ```cpp
 * // Request CPU memory for a GPU operation
 * allocator_attributes cpu_attrs;
 * cpu_attrs.set_on_host(true);
 * Allocator* cpu_alloc = device->GetAllocator(cpu_attrs);
 *
 * // Request GPU-compatible memory for transfers
 * allocator_attributes gpu_attrs;
 * gpu_attrs.set_gpu_compatible(true);
 * Allocator* gpu_alloc = device->GetAllocator(gpu_attrs);
 * ```
 */
struct XSIGMA_API allocator_attributes
{
    /**
     * @brief Sets whether memory should be allocated on host (CPU).
     *
     * @param v true to request host memory, false for device memory
     *
     * **Performance Impact**: Host memory typically has larger capacity but lower bandwidth
     * **Use Cases**: Large buffers, intermediate results, CPU-only operations
     */
    void set_on_host(bool v) noexcept { value = (value & ~0x1u) | static_cast<uint32_t>(v); }

    /**
     * @brief Checks if host memory allocation is requested.
     * @return true if host memory is requested
     */
    bool on_host() const noexcept { return (value & 0x1u) != 0; }

    /**
     * @brief Sets whether memory should be compatible with network operations.
     *
     * @param v true to request network-compatible memory
     *
     * **Performance Impact**: May require special alignment or memory regions
     * **Use Cases**: Distributed computing, RDMA operations, network transfers
     */
    void set_nic_compatible(bool v) noexcept
    {
        value = (value & ~0x2u) | (static_cast<uint32_t>(v) << 1);
    }

    /**
     * @brief Checks if network-compatible memory is requested.
     * @return true if network-compatible memory is requested
     */
    bool nic_compatible() const noexcept { return (value & 0x2u) != 0; }

    /**
     * @brief Sets whether memory should be compatible with GPU operations.
     *
     * @param v true to request GPU-compatible memory
     *
     * **Performance Impact**: Optimized for GPU access patterns and transfers
     * **Use Cases**: GPU computations, device-to-device transfers, unified memory
     */
    void set_gpu_compatible(bool v) noexcept
    {
        value = (value & ~0x4u) | (static_cast<uint32_t>(v) << 2);
    }

    /**
     * @brief Checks if GPU-compatible memory is requested.
     * @return true if GPU-compatible memory is requested
     */
    bool gpu_compatible() const noexcept { return (value & 0x4u) != 0; }

    /**
     * @brief Merges attributes from another instance using bitwise OR.
     *
     * Combines attribute flags and handles scope_id conflicts. If both instances
     * have non-zero scope_ids, they must match or one must be zero.
     *
     * @param other Attributes to merge with this instance
     *
     * **Exception Safety**: May throw if scope_ids conflict
     * **Performance**: O(1) bitwise operations
     *
     * **Example**:
     * ```cpp
     * allocator_attributes base;
     * base.set_on_host(true);
     *
     * allocator_attributes extra;
     * extra.set_gpu_compatible(true);
     *
     * base.Merge(extra);  // Now has both on_host and gpu_compatible
     * ```
     */
    void Merge(const allocator_attributes& other)
    {
        value |= other.value;
        if (scope_id != other.scope_id)
        {
            XSIGMA_CHECK(
                scope_id == 0 || other.scope_id == 0,
                "Cannot merge allocator_attributes with conflicting scope_ids: ",
                scope_id,
                " vs ",
                other.scope_id,
                ". At least one must be zero.");

            scope_id = (scope_id == 0) ? other.scope_id : scope_id;
        }
    }

    /**
     * @brief Checks if this instance's requirements are subset of another's.
     *
     * Returns true if all attributes set in this instance are also set in other.
     * Useful for determining allocator compatibility and requirement satisfaction.
     *
     * @param other Attributes to compare against
     * @return true if this is subset of or equal to other
     *
     * **Performance**: O(1) bitwise comparison
     * **Use Cases**: Allocator selection, compatibility checking
     *
     * **Example**:
     * ```cpp
     * allocator_attributes required;
     * required.set_on_host(true);
     *
     * allocator_attributes available;
     * available.set_on_host(true);
     * available.set_gpu_compatible(true);
     *
     * assert(required.IsEqualOrLessRestrictiveThan(available));  // true
     * ```
     */
    bool IsEqualOrLessRestrictiveThan(const allocator_attributes& other) const noexcept
    {
        return (value | other.value) == other.value;
    }

    /**
     * @brief Bit-packed attribute flags.
     *
     * **Layout**:
     * - Bits 0-3: Standard attributes (host, nic, gpu, pjrt)
     * - Bits 4-23: Reserved for future use
     * - Bits 24-31: Device-specific attributes
     *
     * **Device-Specific Usage**: Upper 8 bits (24-31) are reserved for
     * device-specific interpretations. Device implementations can use these
     * bits for custom allocation policies.
     */
    uint32_t value{0};

    /**
     * @brief Scope identifier for specialized allocators.
     *
     * **EXPERIMENTAL**: When non-zero, delegates allocation to a named
     * special-purpose allocator on the same device. Enables fine-grained
     * control over memory allocation policies.
     *
     * **Values**:
     * - 0: Use default allocator selection
     * - >0: Use specialized allocator with this ID
     *
     * **Thread Safety**: Should be set once during initialization
     * **Use Cases**: Memory pools, specialized allocation strategies
     */
    int32_t scope_id{0};

    /**
     * @brief Generates human-readable string representation of attributes.
     *
     * @return Formatted string showing all set attributes
     *
     * **Performance**: O(1) - simple string formatting
     * **Thread Safety**: Safe to call concurrently on different instances
     *
     * **Example Output**:
     * ```
     * "allocator_attributes: on_host=true, gpu_compatible=true, scope_id=0"
     * ```
     */
    std::string debug_string() const;
};

/**
 * @brief Returns the base CPU allocator singleton instance.
 *
 * Provides access to a simple, process-wide CPU allocator implementation.
 * This is a fallback allocator intended for infrastructure use only.
 *
 * @return Pointer to singleton CPU allocator (never nullptr)
 *
 * **Thread Safety**: Thread-safe singleton initialization
 * **Performance**: O(1) after initialization
 * **Lifetime**: Lives for entire process duration
 *
 * @warning Intended for restricted infrastructure use only.
 *          Prefer process_state::GetCPUAllocator() when available.
 */
Allocator* allocator_cpu_base();

/**
 * @brief Returns a NUMA-aware CPU allocator for the specified node.
 *
 * Attempts to use process_state::GetCPUAllocator() for NUMA-optimized allocation.
 * Falls back to allocator_cpu_base() if process_state is not available.
 *
 * @param numa_node NUMA node ID, or NUMANOAFFINITY for no preference
 * @return Pointer to CPU allocator optimized for the specified NUMA node
 *
 * **NUMA Optimization**: When available, allocates memory local to specified node
 * **Fallback**: Uses base CPU allocator if NUMA support unavailable
 * **Thread Safety**: Thread-safe
 * **Performance**: O(1) - cached allocator instances
 *
 * **Example**:
 * ```cpp
 * // Get allocator for current NUMA node
 * Allocator* local_alloc = cpu_allocator();
 *
 * // Get allocator for specific NUMA node
 * Allocator* node_alloc = cpu_allocator(1);
 * ```
 */
XSIGMA_API Allocator* cpu_allocator(int numa_node = NUMANOAFFINITY);

/**
 * @brief Enables statistics collection in the default CPU allocator.
 *
 * Activates comprehensive statistics tracking including allocation counts,
 * memory usage, and performance metrics. Statistics are disabled by default
 * for optimal performance.
 *
 * **Performance Impact**: Minimal overhead for counter updates
 * **Thread Safety**: Safe to call from any thread
 * **Global Effect**: Affects all CPU allocator instances
 *
 * **Example**:
 * ```cpp
 * EnableCPUAllocatorStats();
 * // Now CPU allocators will collect statistics
 * auto stats = cpu_allocator()->GetStats();
 * ```
 */
XSIGMA_API void EnableCPUAllocatorStats() noexcept;

/**
 * @brief Disables statistics collection in the default CPU allocator.
 *
 * Deactivates statistics tracking to minimize performance overhead.
 * This is the default state for optimal performance.
 *
 * **Performance Benefit**: Eliminates statistics collection overhead
 * **Thread Safety**: Safe to call from any thread
 * **Global Effect**: Affects all CPU allocator instances
 */
XSIGMA_API void DisableCPUAllocatorStats() noexcept;

/**
 * @brief Checks if CPU allocator statistics collection is enabled.
 *
 * @return true if statistics are being collected, false otherwise
 *
 * **Thread Safety**: Safe to call from any thread
 * **Performance**: O(1) - simple flag check
 */
bool CPUAllocatorStatsEnabled() noexcept;

/**
 * @brief Enables comprehensive statistics collection in CPU allocators.
 *
 * Activates detailed statistics including fragmentation metrics, allocation
 * patterns, and performance counters. More comprehensive than basic stats
 * but with higher overhead.
 *
 * **Performance Impact**: Higher overhead than basic statistics
 * **Thread Safety**: Safe to call from any thread
 * **Use Cases**: Detailed performance analysis, memory debugging
 */
void EnableCPUAllocatorFullStats();

/**
 * @brief Checks if full CPU allocator statistics collection is enabled.
 *
 * @return true if full statistics are being collected, false otherwise
 *
 * **Thread Safety**: Safe to call from any thread
 * **Performance**: O(1) - simple flag check
 */
bool CPUAllocatorFullStatsEnabled();

/**
 * @brief Low-level memory allocator interface for high-level allocator backends.
 *
 * sub_allocator provides the underlying memory allocation/deallocation services
 * for higher-level allocators like allocator_bfc. It's designed for infrequent
 * large allocations that are then managed by sophisticated pooling and caching
 * strategies in the higher-level allocator.
 *
 * **Design Principle**: Coarse-grained allocation with visitor pattern for monitoring
 * **Performance**: Optimized for large, infrequent allocations (not per-object allocation)
 * **Thread Safety**: Implementation-dependent, but typically thread-safe
 *
 * **Typical Usage Pattern**:
 * 1. Higher-level allocator requests large memory regions from sub_allocator
 * 2. Higher-level allocator subdivides regions for individual allocations
 * 3. sub_allocator is called infrequently compared to user allocation requests
 *
 * **Example**:
 * ```cpp
 * // allocator_bfc might request 64MB from sub_allocator
 * size_t received;
 * void* region = sub_alloc->Alloc(64, 64*1024*1024, &received);
 * // Then serve thousands of small allocations from this region
 * ```
 */
class sub_allocator
{
public:
    /**
     * @brief Visitor function type for allocation/deallocation monitoring.
     *
     * Called whenever memory is allocated or freed to enable tracking,
     * profiling, and debugging. The index parameter identifies the memory
     * domain (NUMA node for CPU, device ID for GPU).
     *
     * @param ptr Pointer to allocated/freed memory
     * @param index Memory domain identifier (NUMA node, GPU ID, etc.)
     * @param size Size of memory region in bytes
     *
     * **Performance**: Should be fast (O(1)) to avoid allocation overhead
     * **Thread Safety**: Must be thread-safe if used with concurrent allocators
     */
    using Visitor = std::function<void(void*, int index, size_t)>;

    /**
     * @brief Constructs sub_allocator with visitor callbacks for monitoring.
     *
     * @param alloc_visitors Functions called on each allocation
     * @param free_visitors Functions called on each deallocation
     *
     * **Visitor Execution**: Visitors are called in registration order
     * **Exception Safety**: Visitor exceptions may propagate
     * **Performance**: Visitor overhead is per large allocation, not per user allocation
     */
    XSIGMA_API sub_allocator(
        const std::vector<Visitor>& alloc_visitors, const std::vector<Visitor>& free_visitors);

    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~sub_allocator() = default;

    /**
     * @brief Allocates a large memory region with specified alignment.
     *
     * Allocates at least num_bytes of memory aligned to the specified boundary.
     * Returns the actual allocated size which may be larger than requested.
     * The entire allocated region is safe to use.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Minimum size to allocate
     * @param bytes_received Output parameter for actual allocated size
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Requirements**:
     * - alignment must be power of 2
     * - bytes_received must not be nullptr
     * - *bytes_received >= num_bytes on success
     *
     * **Performance**: Typically O(1) but may involve system calls
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw, return nullptr on failure
     *
     * **Example**:
     * ```cpp
     * size_t received;
     * void* ptr = sub_alloc->Alloc(4096, 1024*1024, &received);
     * if (ptr) {
     *     // Can safely use 'received' bytes starting at 'ptr'
     *     assert(received >= 1024*1024);
     * }
     * ```
     */
    virtual void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) = 0;

    /**
     * @brief Frees a previously allocated memory region.
     *
     * Releases memory back to the system. The pointer and size must exactly
     * match a previous successful Alloc() call.
     *
     * @param ptr Pointer returned by previous Alloc() call
     * @param num_bytes Size returned in bytes_received from Alloc()
     *
     * **Requirements**:
     * - ptr must have been returned by Alloc() on this instance
     * - num_bytes must match bytes_received from the Alloc() call
     * - ptr must not be used after this call
     *
     * **Performance**: Typically O(1) but may involve system calls
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw
     */
    virtual void Free(void* ptr, size_t num_bytes) = 0;

    /**
     * @brief Indicates whether adjacent allocations can be safely coalesced.
     *
     * Returns true if the BFC allocator can treat adjacent memory regions
     * from this sub_allocator as contiguous for coalescing purposes. This
     * enables more efficient memory management and reduced fragmentation.
     *
     * @return true if coalescing is safe, false otherwise
     *
     * **Safety Consideration**: Only return true if adjacent allocations
     * are guaranteed to be physically contiguous in the address space.
     *
     * **Performance Impact**: Coalescing enables better memory utilization
     * **Thread Safety**: Must return consistent values across threads
     *
     * **Example Use Cases**:
     * - System malloc: typically true (virtual memory is contiguous)
     * - GPU memory: may be false if allocations use different memory pools
     * - NUMA memory: may be false across NUMA boundaries
     */
    virtual bool SupportsCoalescing() const = 0;

    /**
     * @brief Returns the type of memory managed by this sub_allocator.
     *
     * @return Memory type enumeration indicating physical memory characteristics
     *
     * **Thread Safety**: Must be thread-safe and return consistent values
     * **Performance**: O(1) - should return cached value
     * **Default**: UNKNOWN (should be overridden by implementations)
     */
    virtual allocator_memory_enum GetMemoryType() const noexcept
    {
        return allocator_memory_enum::UNKNOWN;
    }

protected:
    /**
     * @brief Notifies allocation visitors of new memory allocation.
     *
     * Must be called by Alloc() implementations immediately after successful
     * allocation. Executes all registered allocation visitors in order.
     *
     * @param ptr Pointer to newly allocated memory
     * @param index Memory domain identifier (NUMA node, device ID, etc.)
     * @param num_bytes Size of allocated region
     *
     * **Call Requirement**: Must be called by every Alloc() implementation
     * **Exception Handling**: Visitor exceptions may propagate
     * **Performance**: O(n) where n is number of visitors
     */
    void VisitAlloc(void* ptr, int index, size_t num_bytes);

    /**
     * @brief Notifies deallocation visitors of memory release.
     *
     * Must be called by Free() implementations immediately before actual
     * deallocation. Executes all registered deallocation visitors in order.
     *
     * @param ptr Pointer to memory being freed
     * @param index Memory domain identifier (NUMA node, device ID, etc.)
     * @param num_bytes Size of region being freed
     *
     * **Call Requirement**: Must be called by every Free() implementation
     * **Exception Handling**: Visitor exceptions may propagate
     * **Performance**: O(n) where n is number of visitors
     */
    void VisitFree(void* ptr, int index, size_t num_bytes);

    const std::vector<Visitor> alloc_visitors_;  ///< Allocation monitoring callbacks
    const std::vector<Visitor> free_visitors_;   ///< Deallocation monitoring callbacks
};

}  // namespace xsigma
