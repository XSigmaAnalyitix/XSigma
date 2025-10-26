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

#include "memory_allocator.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>  // for std::memset

#include "common/macros.h"
#include "util/exception.h"

#ifdef XSIGMA_ENABLE_TBB

#ifdef _MSC_VER
#pragma push_macro("__TBB_NO_IMPLICIT_LINKAGE")
#define __TBB_NO_IMPLICIT_LINKAGE 1  //NOLINT
#endif

#include <tbb/scalable_allocator.h>

#ifdef _MSC_VER
#pragma pop_macro("__TBB_NO_IMPLICIT_LINKAGE")
#endif
#endif

#ifdef XSIGMA_NUMA_ENABLED
#include "memory/numa.h"
#endif  // XSIGMA_NUMA_ENABLED

#ifdef XSIGMA_ENABLE_MIMALLOC
#include <mimalloc.h>
#endif

#ifdef XSIGMA_ENABLE_CUDA
#include <cuda.h>  // For CUDA Driver API
#include <cuda_runtime.h>
#endif

#ifdef XSIGMA_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "logging/logger.h"

namespace xsigma::cpu::memory_allocator
{

void* allocate(std::size_t nbytes, std::size_t alignment, init_policy_enum init)
{
    XSIGMA_CHECK(
        static_cast<std::ptrdiff_t>(nbytes) > 0,
        "cpu allocate() called with negative or zero size: {}",
        nbytes);

    XSIGMA_CHECK_DEBUG(
        is_valid_alignment(alignment),
        "cpu allocate() called with invalid alignment: {} (must be power of 2 >= {})",
        alignment,
        sizeof(void*));

    void* ptr = nullptr;

    // Platform-specific allocation
#ifdef XSIGMA_ENABLE_MIMALLOC
    ptr = mi_aligned_alloc(alignment, nbytes);
#elif defined(XSIGMA_ENABLE_TBB)
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
        // cppcheck-suppress syntaxError
        if XSIGMA_UNLIKELY (posix_memalign(&ptr, alignment, nbytes) != 0)
        {
            return nullptr;
        }
    }
#endif
    // cppcheck-suppress syntaxError
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
#ifndef NDEBUG
    case init_policy_enum::PATTERN:
        std::memset(ptr, 0xCC, nbytes);
        break;
#endif
    case init_policy_enum::UNINITIALIZED:
    default:
        // Do nothing - fastest option
        break;
    }

    return ptr;
}

void free(void* ptr, XSIGMA_UNUSED std::size_t nbytes) noexcept
{
    if XSIGMA_LIKELY (ptr != nullptr)
    {
#ifdef XSIGMA_ENABLE_MIMALLOC
        mi_free(ptr);
#elif defined(XSIGMA_ENABLE_TBB)
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

void* allocate_tbb(XSIGMA_UNUSED std::size_t nbytes, XSIGMA_UNUSED std::size_t alignment)
{
#ifdef XSIGMA_ENABLE_TBB
    return scalable_aligned_malloc(nbytes, alignment);
#else
    return nullptr;
#endif
}

void free_tbb(XSIGMA_UNUSED void* ptr, XSIGMA_UNUSED std::size_t nbytes) noexcept
{
#ifdef XSIGMA_ENABLE_TBB
    scalable_aligned_free(ptr);
#endif
}

void* allocate_mi(std::size_t nbytes, std::size_t alignment)
{
#ifdef XSIGMA_ENABLE_MIMALLOC
    return mi_malloc_aligned(nbytes, alignment);
#else
    return nullptr;
#endif
}

void free_mi(void* ptr, XSIGMA_UNUSED std::size_t nbytes) noexcept
{
#ifdef XSIGMA_ENABLE_MIMALLOC
    mi_free(ptr);
#endif
}
}  // namespace xsigma::cpu::memory_allocator

namespace xsigma::gpu::memory_allocator
{

// CUDA error checking helper macros
#ifdef XSIGMA_ENABLE_CUDA
#define CUDA_CHECK_RETURN_NULL(call)                                                    \
    do                                                                                  \
    {                                                                                   \
        cudaError_t error = call;                                                       \
        if (error != cudaSuccess)                                                       \
        {                                                                               \
            XSIGMA_LOG_ERROR("CUDA error in {}: {}", #call, cudaGetErrorString(error)); \
            return nullptr;                                                             \
        }                                                                               \
    } while (0)

#define CUDA_CHECK_RETURN_FALSE(call)                                                   \
    do                                                                                  \
    {                                                                                   \
        cudaError_t error = call;                                                       \
        if (error != cudaSuccess)                                                       \
        {                                                                               \
            XSIGMA_LOG_ERROR("CUDA error in {}: {}", #call, cudaGetErrorString(error)); \
            return false;                                                               \
        }                                                                               \
    } while (0)
#else
#define CUDA_CHECK_RETURN_NULL(call) return nullptr
#define CUDA_CHECK_RETURN_FALSE(call) return false
#endif

// HIP error checking helper macros
#ifdef XSIGMA_ENABLE_HIP
#define HIP_CHECK_RETURN_NULL(call)                                                   \
    do                                                                                \
    {                                                                                 \
        hipError_t error = call;                                                      \
        if (error != hipSuccess)                                                      \
        {                                                                             \
            XSIGMA_LOG_ERROR("HIP error in {}: {}", #call, hipGetErrorString(error)); \
            return nullptr;                                                           \
        }                                                                             \
    } while (0)

#define HIP_CHECK_RETURN_FALSE(call)                                                  \
    do                                                                                \
    {                                                                                 \
        hipError_t error = call;                                                      \
        if (error != hipSuccess)                                                      \
        {                                                                             \
            XSIGMA_LOG_ERROR("HIP error in {}: {}", #call, hipGetErrorString(error)); \
            return false;                                                             \
        }                                                                             \
    } while (0)
#else
#define HIP_CHECK_RETURN_NULL(call) return nullptr
#define HIP_CHECK_RETURN_FALSE(call) return false
#endif

void* allocate(
    std::size_t nbytes, int device_id, XSIGMA_UNUSED void* stream, XSIGMA_UNUSED void* memory_pool)
{
    XSIGMA_CHECK(
        static_cast<std::ptrdiff_t>(nbytes) > 0,
        "gpu allocate() called with negative or zero size: {}",
        nbytes);

    if (nbytes == 0)
    {
        return nullptr;
    }

    // Set device context
    if (!set_device(device_id))
    {
        return nullptr;
    }

    void* ptr = nullptr;  //NOLINT

#ifdef XSIGMA_ENABLE_CUDA

#ifdef XSIGMA_CUDA_ALLOC_SYNC
    // Use synchronous CUDA Runtime API allocation (more commonly available)
    cudaError_t result = cudaMalloc(&ptr, nbytes);
    if (result != cudaSuccess)
    {
        XSIGMA_LOG_WARNING(
            "GPU synchronous allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            cudaGetErrorString(result));
        return nullptr;
    }

#elif defined(XSIGMA_CUDA_ALLOC_ASYNC)
    // Use asynchronous CUDA Runtime API allocation
    cudaStream_t gpu_stream_cuda = static_cast<cudaStream_t>(stream);
    cudaError_t  result          = cudaMallocAsync(&ptr, nbytes, gpu_stream_cuda);
    if (result != cudaSuccess)
    {
        XSIGMA_LOG_WARNING(
            "GPU asynchronous allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            cudaGetErrorString(result));
        return nullptr;
    }

#elif defined(XSIGMA_CUDA_ALLOC_POOL_ASYNC)
    // Use pool-based asynchronous CUDA Runtime API allocation
    cudaStream_t  gpu_stream_cuda = static_cast<cudaStream_t>(stream);
    cudaMemPool_t cuda_pool       = static_cast<cudaMemPool_t>(memory_pool);

    cudaError_t result;
    if (cuda_pool != nullptr)
    {
        result = cudaMallocFromPoolAsync(&ptr, nbytes, cuda_pool, gpu_stream_cuda);
    }
    else
    {
        // Fall back to regular async allocation if no pool is specified
        result = cudaMallocAsync(&ptr, nbytes, gpu_stream_cuda);
    }

    if (result != cudaSuccess)
    {
        XSIGMA_LOG_WARNING(
            "GPU pool-based allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            cudaGetErrorString(result));
        return nullptr;
    }

#else
    // Default to synchronous allocation using CUDA Runtime API
    cudaError_t const result = cudaMalloc(&ptr, nbytes);
    if (result != cudaSuccess)
    {
        XSIGMA_LOG_WARNING(
            "GPU default allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            cudaGetErrorString(result));
        return nullptr;
    }
#endif

#elif defined(XSIGMA_ENABLE_HIP)

#if defined(XSIGMA_HIP_ALLOC_SYNC)
    // Use synchronous HIP allocation
    hipError_t result = hipMalloc(&ptr, nbytes);
    if (result != hipSuccess)
    {
        XSIGMA_LOG_WARNING(
            "HIP synchronous allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            hipGetErrorString(result));
        return nullptr;
    }

#elif defined(XSIGMA_HIP_ALLOC_ASYNC)
    // Use asynchronous HIP allocation
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t  result     = hipMallocAsync(&ptr, nbytes, hip_stream);
    if (result != hipSuccess)
    {
        XSIGMA_LOG_WARNING(
            "HIP asynchronous allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            hipGetErrorString(result));
        return nullptr;
    }

#elif defined(XSIGMA_HIP_ALLOC_POOL_ASYNC)
    // Use pool-based asynchronous HIP allocation
    hipStream_t  hip_stream = static_cast<hipStream_t>(stream);
    hipMemPool_t hip_pool   = static_cast<hipMemPool_t>(memory_pool);

    hipError_t result;
    if (hip_pool != nullptr)
    {
        result = hipMallocFromPoolAsync(&ptr, nbytes, hip_pool, hip_stream);
    }
    else
    {
        // Fallback to regular async allocation if no pool is specified
        result = hipMallocAsync(&ptr, nbytes, hip_stream);
    }

    if (result != hipSuccess)
    {
        XSIGMA_LOG_WARNING(
            "HIP pool allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            hipGetErrorString(result));
        return nullptr;
    }

#else
    // Default to synchronous HIP allocation
    hipError_t result = hipMalloc(&ptr, nbytes);
    if (result != hipSuccess)
    {
        XSIGMA_LOG_WARNING(
            "HIP default allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            hipGetErrorString(result));
        return nullptr;
    }
#endif

#else
    (void)stream;
    (void)memory_pool;
    XSIGMA_LOG_ERROR("GPU support not enabled in this build");
#endif

    return ptr;
}

void free(
    void* ptr, XSIGMA_UNUSED std::size_t nbytes, int device_id, XSIGMA_UNUSED void* stream) noexcept
{
    if (ptr == nullptr)
    {
        return;
    }

    // Set device context (ignore errors in free)
    set_device(device_id);

#ifdef XSIGMA_ENABLE_CUDA

#ifdef XSIGMA_CUDA_ALLOC_SYNC
    // Use synchronous CUDA Runtime API deallocation
    cudaError_t result = cudaFree(ptr);
    if (result != cudaSuccess)
    {
        XSIGMA_LOG_ERROR(
            "GPU synchronous deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            cudaGetErrorString(result));
    }

#elif defined(XSIGMA_CUDA_ALLOC_ASYNC)
    // Use asynchronous CUDA Runtime API deallocation
    cudaStream_t gpu_stream_cuda = static_cast<cudaStream_t>(stream);
    cudaError_t  result          = cudaFreeAsync(ptr, gpu_stream_cuda);
    if (result != cudaSuccess)
    {
        XSIGMA_LOG_ERROR(
            "GPU asynchronous deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            cudaGetErrorString(result));
    }

#elif defined(XSIGMA_CUDA_ALLOC_POOL_ASYNC)
    // Use asynchronous CUDA Runtime API deallocation (same for pool and non-pool)
    cudaStream_t gpu_stream_cuda = static_cast<cudaStream_t>(stream);
    cudaError_t  result          = cudaFreeAsync(ptr, gpu_stream_cuda);
    if (result != cudaSuccess)
    {
        XSIGMA_LOG_ERROR(
            "GPU pool-based deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            cudaGetErrorString(result));
    }

#else
    // Default to synchronous deallocation using CUDA Runtime API
    cudaError_t const result = cudaFree(ptr);
    if (result != cudaSuccess)
    {
        XSIGMA_LOG_ERROR(
            "GPU default deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            cudaGetErrorString(result));
    }
#endif

#elif defined(XSIGMA_ENABLE_HIP)

#if defined(XSIGMA_HIP_ALLOC_SYNC)
    // Use synchronous HIP deallocation
    hipError_t result = hipFree(ptr);
    if (result != hipSuccess)
    {
        XSIGMA_LOG_ERROR(
            "HIP synchronous deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            hipGetErrorString(result));
    }

#elif defined(XSIGMA_HIP_ALLOC_ASYNC)
    // Use asynchronous HIP deallocation
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t  result     = hipFreeAsync(ptr, hip_stream);
    if (result != hipSuccess)
    {
        XSIGMA_LOG_ERROR(
            "HIP asynchronous deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            hipGetErrorString(result));
    }

#elif defined(XSIGMA_HIP_ALLOC_POOL_ASYNC)
    // Use asynchronous HIP deallocation (same for pool and non-pool)
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t  result     = hipFreeAsync(ptr, hip_stream);
    if (result != hipSuccess)
    {
        XSIGMA_LOG_ERROR(
            "HIP pool deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            hipGetErrorString(result));
    }

#else
    // Default to synchronous HIP deallocation
    hipError_t result = hipFree(ptr);
    if (result != hipSuccess)
    {
        XSIGMA_LOG_ERROR(
            "HIP default deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            hipGetErrorString(result));
    }
#endif

#else
    (void)nbytes;
    (void)device_id;
    (void)stream;
#endif
}

bool set_device(int device_id) noexcept
{
#ifdef XSIGMA_ENABLE_CUDA
    CUDA_CHECK_RETURN_FALSE(cudaSetDevice(device_id));
    return true;
#elif defined(XSIGMA_ENABLE_HIP)
    HIP_CHECK_RETURN_FALSE(hipSetDevice(device_id));
    return true;
#else
    (void)device_id;
    return false;
#endif
}

int get_current_device() noexcept
{
#ifdef XSIGMA_ENABLE_CUDA
    int               device_id = -1;
    cudaError_t const error     = cudaGetDevice(&device_id);
    if (error != cudaSuccess)
    {
        XSIGMA_LOG_ERROR("Failed to get current CUDA device: {}", cudaGetErrorString(error));
        return -1;
    }
    return device_id;
#elif defined(XSIGMA_ENABLE_HIP)
    int        device_id = -1;
    hipError_t error     = hipGetDevice(&device_id);
    if (error != hipSuccess)
    {
        XSIGMA_LOG_ERROR("Failed to get current HIP device: {}", hipGetErrorString(error));
        return -1;
    }
    return device_id;
#else
    return -1;
#endif
}

}  // namespace xsigma::gpu::memory_allocator