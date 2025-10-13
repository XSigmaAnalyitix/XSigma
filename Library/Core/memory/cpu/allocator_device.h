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

#include <cstddef>  // for size_t
#include <string>   // for string

#include "common/export.h"         // for XSIGMA_API
#include "memory/cpu/allocator.h"  // for Allocator, allocator_memory_enum

namespace xsigma
{
/**
 * @brief High-performance device-aware memory allocator for pinned CPU memory.
 *
 * The allocator_device class provides pinned (page-locked) CPU memory that can be
 * transferred to/from GPU memory more efficiently than regular CPU memory. It inherits
 * from the Allocator base class to provide a unified interface while specializing
 * in device-optimized memory allocation.
 *
 * **Key Features**:
 * - Pinned CPU memory allocation for optimal GPU transfer performance
 * - CUDA-aware allocation when CUDA support is enabled
 * - Fallback to standard aligned allocation when CUDA is not available
 * - Thread-safe allocation and deallocation
 * - Comprehensive error handling and logging
 * - Memory copy operations optimized for device transfers
 *
 * **Performance Characteristics**:
 * - allocate_raw(): O(1) - direct system call to CUDA or standard allocator
 * - deallocate_raw(): O(1) - direct system call to CUDA or standard allocator
 * - Memory transfers: Optimized for high-bandwidth device communication
 *
 * **Thread Safety**: All operations are thread-safe and can be called concurrently
 * from multiple threads without external synchronization.
 *
 * **Memory Type**: Returns HOST_PINNED memory type for optimal device transfer performance.
 *
 * **Example Usage**:
 * ```cpp
 * auto allocator = std::make_unique<allocator_device>();
 * 
 * // Allocate 1MB of pinned memory
 * void* ptr = allocator->allocate_raw(64, 1024ULL);
 * if (ptr) {
 *     // Use memory for GPU transfers...
 *     allocator->deallocate_raw(ptr);
 * }
 * ```
 */
class XSIGMA_API allocator_device : public Allocator
{
public:
    /**
     * @brief Constructs a new device allocator instance.
     *
     * Initializes the allocator for pinned CPU memory allocation. No additional
     * configuration is required as the allocator automatically detects CUDA
     * availability and configures itself accordingly.
     *
     * **Exception Safety**: noexcept - constructor cannot fail
     * **Thread Safety**: Safe to construct from multiple threads
     */
    allocator_device() noexcept = default;

    /**
     * @brief Virtual destructor ensuring proper cleanup.
     *
     * **Exception Safety**: noexcept - destructor cannot throw
     * **Thread Safety**: Destructor calls must be externally synchronized
     */
    ~allocator_device() override = default;

    /**
     * @brief Returns the allocator name for debugging and profiling.
     *
     * @return String identifier "allocator_device"
     *
     * **Performance**: O(1) - returns cached string
     * **Thread Safety**: Thread-safe
     * **Exception Safety**: noexcept
     */
    std::string Name() const override;

    /**
     * @brief Allocates pinned CPU memory with specified alignment.
     *
     * Allocates pinned (page-locked) CPU memory that can be efficiently transferred
     * to/from GPU devices. Uses CUDA pinned memory allocation when available,
     * falls back to standard aligned allocation otherwise.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to allocated pinned memory, or nullptr on failure
     *
     * **Requirements**:
     * - alignment must be a power of 2
     * - alignment must be >= sizeof(void*)
     * - num_bytes can be 0 (returns nullptr)
     *
     * **Performance**: O(1) - direct system call
     * **Thread Safety**: Thread-safe
     * **Exception Safety**: May throw std::bad_alloc on allocation failure
     *
     * **CUDA Path**: Uses cudaMallocHost() for optimal GPU transfer performance
     * **Fallback Path**: Uses standard aligned allocation with 64-byte alignment
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(64, 4096);  // 4KB pinned memory
     * ```
     */
    void* allocate_raw(size_t alignment, size_t num_bytes) override;

    /**
     * @brief Deallocates previously allocated pinned memory.
     *
     * Releases pinned memory back to the system. Uses CUDA deallocation when
     * the memory was allocated via CUDA, standard deallocation otherwise.
     *
     * @param ptr Pointer to memory block to deallocate (must not be nullptr)
     *
     * **Requirements**:
     * - ptr must have been returned by allocate_raw() on this allocator
     * - ptr must not be nullptr (undefined behavior)
     * - ptr must not be used after this call (undefined behavior)
     *
     * **Performance**: O(1) - direct system call
     * **Thread Safety**: Thread-safe
     * **Exception Safety**: noexcept - logs errors but does not throw
     *
     * **Error Handling**: Logs errors but continues execution to maintain
     * exception safety in destructors and cleanup code.
     */
    void deallocate_raw(void* ptr) override;

    /**
     * @brief Returns the memory type managed by this allocator.
     *
     * @return allocator_memory_enum::HOST_PINNED
     *
     * **Performance**: O(1) - returns constant value
     * **Thread Safety**: Thread-safe
     * **Exception Safety**: noexcept
     */
    allocator_memory_enum GetMemoryType() const noexcept override;
};

}  // namespace xsigma
