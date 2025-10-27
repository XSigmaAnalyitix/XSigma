# XSigma Memory Allocator - Technical Implementation Guide

**Date:** October 5, 2025
**Audience:** Developers, System Architects
**Purpose:** Technical reference for memory allocator selection and configuration

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [CPU Allocator Implementation](#cpu-allocator-implementation)
3. [GPU Allocator Implementation](#gpu-allocator-implementation)
4. [Configuration Guidelines](#configuration-guidelines)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Memory Management Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Monte Carlo, PDE Solvers, Risk Calculations)              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              XSigma Memory Management API                    │
│  • allocator_typed<T>  • allocation_attributes              │
│  • Allocator interface • allocator_registry                 │
└────────────┬───────────────────────────┬────────────────────┘
             │                           │
    ┌────────▼────────┐         ┌───────▼────────┐
    │  CPU Allocators │         │ GPU Allocators  │
    └────────┬────────┘         └───────┬─────────┘
             │                           │
    ┌────────▼────────────────┐ ┌───────▼──────────────────┐
    │ • XSigma CPU Allocator  │ │ • CUDA Caching Allocator │
    │ • mimalloc              │ │ • GPU Memory Pool        │
    │ • TBB Scalable          │ │ • Direct CUDA            │
    │ • Standard Aligned      │ │ • Unified Memory         │
    └────────┬────────────────┘ └───────┬──────────────────┘
             │                           │
    ┌────────▼────────────────┐ ┌───────▼──────────────────┐
    │   System Allocators     │ │   CUDA Runtime           │
    │ • malloc/free           │ │ • cudaMalloc/cudaFree    │
    │ • _aligned_malloc       │ │ • cudaMallocHost         │
    │ • posix_memalign        │ │ • cudaMallocManaged      │
    └─────────────────────────┘ └──────────────────────────┘
```

### Design Principles

1. **Unified Interface**: Single API for CPU and GPU memory
2. **Zero-Cost Abstraction**: Template-based design with no runtime overhead
3. **Type Safety**: Compile-time type checking with allocator_typed<T>
4. **Configurability**: Runtime selection of allocation strategies
5. **Observability**: Built-in tracking and profiling support

---

## CPU Allocator Implementation

### 1. XSigma CPU Allocator

**File:** `Library/Core/memory/cpu/allocator_cpu_impl.cxx`

#### Key Features

```cpp
class CPUAllocator : public Allocator {
public:
    void* allocate_raw(size_t alignment, size_t num_bytes) override {
        // Statistics collection (optional)
        if (cpu_allocator_collect_stats.load(std::memory_order_relaxed)) {
            UpdateStats(num_bytes);
        }

        // Delegate to backend
        void* p = cpu::memory_allocator::allocate(num_bytes, alignment);

        // Memory pressure warnings
        CheckMemoryPressure(num_bytes);

        return p;
    }

    void deallocate_raw(void* ptr) override {
        if (ptr) {
            cpu::memory_allocator::free(ptr);
        }
    }
};
```

#### Configuration

**Enable Statistics:**
```cpp
// In your application initialization
xsigma::cpu_allocator_collect_stats.store(true);
```

**Backend Selection:**
```cpp
// Compile-time selection via CMake
// -DXSIGMA_ENABLE_MIMALLOC=ON  -> Use mimalloc
// -DXSIGMA_ENABLE_TBB=ON       -> Use TBB scalable_malloc
// Default: Standard aligned allocator
```

#### Performance Characteristics

- **Allocation**: O(1) - delegates to backend
- **Deallocation**: O(1) - delegates to backend
- **Statistics Overhead**: ~1-2% when enabled, 0% when disabled
- **Thread Safety**: Fully thread-safe

### 2. mimalloc Integration

**File:** `Library/Core/memory/helper/memory_allocator.h`

#### Implementation

```cpp
#ifdef XSIGMA_ENABLE_MIMALLOC
inline void* allocate(size_t nbytes, int alignment = 64) {
    return mi_malloc_aligned(nbytes, alignment);
}

inline void free(void* ptr) {
    mi_free(ptr);
}
#endif
```

#### Configuration Options

```cpp
// Environment variables for mimalloc tuning
// MI_SHOW_STATS=1          - Show statistics on exit
// MI_VERBOSE=1             - Verbose output
// MI_EAGER_COMMIT=1        - Eager memory commit
// MI_LARGE_OS_PAGES=1      - Use large OS pages (2MB)
// MI_RESERVE_HUGE_OS_PAGES=N - Reserve N huge pages
```

#### Best Practices

1. **Use for high-frequency allocations** (<64KB)
2. **Enable large pages** for performance-critical applications
3. **Monitor statistics** during development
4. **Disable statistics** in production for maximum performance

### 3. TBB Scalable Allocator

**File:** `Library/Core/memory/helper/memory_allocator.h`

#### Implementation

```cpp
#ifdef XSIGMA_ENABLE_TBB
inline void* allocate(size_t nbytes, int alignment = 64) {
    return scalable_aligned_malloc(nbytes, alignment);
}

inline void free(void* ptr) {
    scalable_aligned_free(ptr);
}
#endif
```

#### Configuration

```cpp
// TBB memory pool configuration
#include <tbb/scalable_allocator.h>

// Set memory pool parameters
scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, 1);
```

#### Use Cases

- **Multi-threaded applications** with high contention
- **NUMA-aware** allocation patterns
- **Cache-aligned** data structures

### 4. BFC Allocator (Best-Fit with Coalescing)

**File:** `Library/Core/memory/cpu/allocator_bfc.cxx`

#### Architecture

```cpp
class allocator_bfc : public Allocator {
private:
    struct Chunk {
        size_t size;
        void* ptr;
        bool in_use;
        Chunk* prev;
        Chunk* next;
    };

    std::vector<Chunk*> bins_;  // Size-based bins
    std::mutex mutex_;

public:
    void* allocate_raw(size_t alignment, size_t num_bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Find best-fit chunk
        Chunk* chunk = FindBestFit(num_bytes);

        if (!chunk) {
            // Allocate new chunk from sub-allocator
            chunk = AllocateNewChunk(num_bytes);
        }

        return chunk->ptr;
    }
};
```

#### Configuration

```cpp
// Create BFC allocator with custom configuration
allocator_bfc::Options opts;
opts.allow_growth = true;
opts.garbage_collection = true;
opts.coalesce_on_free = true;

auto sub_alloc = std::make_unique<CPUSubAllocator>();
auto allocator = std::make_unique<allocator_bfc>(
    std::move(sub_alloc),
    1024*1024*1024,  // 1GB initial size
    "MainAllocator",
    opts
);
```

#### Performance Tuning

- **Initial Size**: Set to expected peak memory usage
- **Growth Policy**: Enable for dynamic workloads
- **Garbage Collection**: Enable for long-running applications
- **Coalescing**: Enable to reduce fragmentation

---

## GPU Allocator Implementation

### 1. CUDA Caching Allocator

**File:** `Library/Core/memory/gpu/cuda_caching_allocator.cxx`

#### Architecture

```cpp
class cuda_caching_allocator {
private:
    struct Block {
        void* ptr;
        size_t size;
        cudaStream_t stream;
        cudaEvent_t event;
        bool in_use;
    };

    std::map<size_t, std::vector<Block*>> free_blocks_;
    std::map<void*, Block*> allocated_blocks_;
    size_t cached_bytes_;
    size_t max_cached_bytes_;

public:
    void* allocate(size_t size, cudaStream_t stream = 0) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Try to find cached block
        Block* block = find_suitable_block(size);

        if (block) {
            // Cache hit - reuse block
            stats_.cache_hits++;
            return block->ptr;
        }

        // Cache miss - allocate new block
        void* ptr;
        cudaMalloc(&ptr, size);

        block = new Block{ptr, size, stream, nullptr, true};
        allocated_blocks_[ptr] = block;
        stats_.cache_misses++;

        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) return;

        Block* block = it->second;

        // Add to cache if under limit
        if (cached_bytes_ + size <= max_cached_bytes_) {
            block->in_use = false;
            free_blocks_[size].push_back(block);
            cached_bytes_ += size;
        } else {
            // Cache full - free immediately
            cudaFree(ptr);
            delete block;
        }

        allocated_blocks_.erase(it);
    }
};
```

#### Configuration

```cpp
// Create caching allocator
cuda_caching_allocator allocator(
    0,                    // Device ID
    256 * 1024 * 1024     // 256MB cache size
);

// For Monte Carlo simulations
auto config = gpu_allocator_config::create_monte_carlo_optimized(0);
// config.cache_max_bytes = 1GB
// config.alignment = 512
```

#### Performance Tuning

**Cache Size Selection:**
```cpp
// Small workloads: 64-128 MB
cuda_caching_allocator allocator(0, 64 * 1024 * 1024);

// Medium workloads: 256-512 MB
cuda_caching_allocator allocator(0, 256 * 1024 * 1024);

// Large workloads: 1-2 GB
cuda_caching_allocator allocator(0, 1024 * 1024 * 1024);
```

**Stream-Aware Caching:**
```cpp
// Allocate with stream association
void* ptr = allocator.allocate(size, my_stream);

// Deallocation respects stream ordering
allocator.deallocate(ptr, size);
```

### 2. GPU Memory Pool

**File:** `Library/Core/memory/gpu/gpu_memory_pool.cxx`

#### Configuration

```cpp
gpu_memory_pool_config config;
config.min_block_size = 1024;              // 1KB
config.max_block_size = 64 * 1024 * 1024;  // 64MB
config.block_growth_factor = 2.0;
config.max_pool_size = 256 * 1024 * 1024;  // 256MB
config.max_cached_blocks = 16;
config.enable_alignment = true;
config.alignment_boundary = 256;
config.enable_tracking = true;

auto pool = gpu_memory_pool::create(config);
```

#### Use Cases

- **Regular allocation patterns** (predictable sizes)
- **Medium-sized allocations** (1KB - 1MB)
- **Moderate frequency** (100-1000 ops/sec)

### 3. Direct CUDA Allocation

**File:** `Library/Core/memory/cpu/allocator_device.cxx`

#### Implementation

```cpp
void* allocator_device::allocate_raw(size_t alignment, size_t num_bytes) {
    void* ptr = nullptr;

#ifdef XSIGMA_ENABLE_CUDA
    cudaError_t result = cudaMallocHost(&ptr, num_bytes);
    if (result != cudaSuccess) {
        XSIGMA_LOG_WARNING("CUDA allocation failed: {}", result);
        return nullptr;
    }
#else
    ptr = cpu::memory_allocator::allocate(num_bytes, alignment);
#endif

    return ptr;
}
```

#### Use Cases

- **Large allocations** (>1MB)
- **Infrequent allocations** (<100 ops/sec)
- **Pinned host memory** for transfers

---

## Configuration Guidelines

### Application Profile: Monte Carlo Simulation

**Characteristics:**
- High-frequency small allocations
- GPU-intensive computation
- Parallel path generation

**Recommended Configuration:**

```cpp
// CPU side
#define XSIGMA_ENABLE_MIMALLOC ON

// GPU side
auto gpu_config = gpu_allocator_config::create_monte_carlo_optimized(0);
cuda_caching_allocator gpu_alloc(0, gpu_config.cache_max_bytes);

// Expected performance
// - CPU: 10M+ allocations/sec
// - GPU: 99% cache hit rate after warmup
```

### Application Profile: PDE Solver

**Characteristics:**
- Large matrix allocations
- Moderate frequency
- Memory-intensive

**Recommended Configuration:**

```cpp
// CPU side
#define XSIGMA_ENABLE_TBB ON  // For NUMA awareness

// GPU side
gpu_memory_pool_config pool_config;
pool_config.max_block_size = 128 * 1024 * 1024;  // 128MB
pool_config.max_pool_size = 2 * 1024 * 1024 * 1024;  // 2GB

auto pool = gpu_memory_pool::create(pool_config);
```

### Application Profile: Risk Calculation

**Characteristics:**
- Mixed allocation sizes
- Batch processing
- CPU and GPU hybrid

**Recommended Configuration:**

```cpp
// CPU side
#define XSIGMA_ENABLE_MIMALLOC ON

// GPU side - hybrid approach
cuda_caching_allocator small_alloc(0, 256 * 1024 * 1024);  // For <64KB
// Use direct CUDA for large allocations

// Allocation strategy
if (size < 64 * 1024) {
    ptr = small_alloc.allocate(size);
} else {
    cudaMalloc(&ptr, size);
}
```

---

## Performance Tuning

### CPU Allocator Tuning

#### 1. Enable Large Pages (Windows)

```cpp
// Requires administrator privileges
// Enable in mimalloc
_putenv("MI_LARGE_OS_PAGES=1");
_putenv("MI_RESERVE_HUGE_OS_PAGES=4");  // Reserve 4 huge pages
```

#### 2. NUMA-Aware Allocation (Linux)

```cpp
#ifdef XSIGMA_NUMA_ENABLED
// Bind thread to NUMA node
numa_run_on_node(node_id);

// Allocate on specific node
void* ptr = numa_alloc_onnode(size, node_id);
#endif
```

#### 3. Disable Statistics in Production

```cpp
// Development
xsigma::cpu_allocator_collect_stats.store(true);

// Production
xsigma::cpu_allocator_collect_stats.store(false);  // ~2% speedup
```

### GPU Allocator Tuning

#### 1. Optimize Cache Size

```cpp
// Profile your application
cuda_caching_allocator allocator(0, initial_cache_size);

// Monitor cache statistics
auto stats = allocator.get_statistics();
double hit_rate = static_cast<double>(stats.cache_hits) /
                  (stats.cache_hits + stats.cache_misses);

// Adjust cache size
if (hit_rate < 0.90) {
    // Increase cache size
    allocator.set_max_cached_bytes(larger_cache_size);
}
```

#### 2. Stream-Aware Allocation

```cpp
// Create streams for parallel work
cudaStream_t streams[4];
for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&streams[i]);
}

// Allocate with stream association
for (int i = 0; i < 4; i++) {
    void* ptr = allocator.allocate(size, streams[i]);
    // Launch kernel on stream[i]
    my_kernel<<<blocks, threads, 0, streams[i]>>>(ptr);
}
```

#### 3. Prefetch for Unified Memory

```cpp
#ifdef XSIGMA_ENABLE_CUDA
// Allocate unified memory
void* ptr;
cudaMallocManaged(&ptr, size);

// Prefetch to GPU
cudaMemPrefetchAsync(ptr, size, device_id, stream);

// Use in kernel
my_kernel<<<blocks, threads, 0, stream>>>(ptr);

// Prefetch back to CPU
cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream);
#endif
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory on GPU

**Symptoms:**
```
CUDA error: out of memory (error code 2)
```

**Solutions:**

1. **Reduce cache size:**
```cpp
// Reduce from 1GB to 256MB
cuda_caching_allocator allocator(0, 256 * 1024 * 1024);
```

2. **Clear cache periodically:**
```cpp
allocator.clear_cache();
```

3. **Use memory pool with limits:**
```cpp
pool_config.max_pool_size = 512 * 1024 * 1024;  // 512MB limit
```

#### Issue 2: Slow CPU Allocations

**Symptoms:**
- Allocation time > 1μs
- High contention in multi-threaded code

**Solutions:**

1. **Switch to mimalloc:**
```cmake
set(XSIGMA_ENABLE_MIMALLOC ON)
```

2. **Use thread-local allocators:**
```cpp
thread_local std::unique_ptr<Allocator> local_allocator;
```

3. **Batch allocations:**
```cpp
// Instead of many small allocations
std::vector<void*> ptrs;
ptrs.reserve(1000);
for (int i = 0; i < 1000; i++) {
    ptrs.push_back(allocator->allocate_raw(64, size));
}
```

#### Issue 3: Memory Fragmentation

**Symptoms:**
- Allocation failures despite available memory
- Increasing memory usage over time

**Solutions:**

1. **Use BFC allocator:**
```cpp
allocator_bfc::Options opts;
opts.garbage_collection = true;
opts.coalesce_on_free = true;
```

2. **Periodic defragmentation:**
```cpp
// For GPU memory pool
pool->defragment();
```

3. **Use fixed-size allocations:**
```cpp
// Allocate in power-of-2 sizes
size_t rounded_size = next_power_of_2(requested_size);
```

---

## Best Practices Summary

### CPU Allocations

✅ **DO:**
- Use mimalloc for high-frequency allocations
- Enable statistics during development
- Disable statistics in production
- Use alignment for cache-sensitive data
- Profile allocation patterns

❌ **DON'T:**
- Mix allocators for the same memory
- Allocate in hot loops without caching
- Ignore alignment requirements
- Use global locks for thread-local data

### GPU Allocations

✅ **DO:**
- Use CUDA caching for small frequent allocations
- Use memory pool for medium regular allocations
- Use direct CUDA for large infrequent allocations
- Monitor cache hit rates
- Prefetch unified memory

❌ **DON'T:**
- Allocate on GPU in tight loops
- Ignore stream associations
- Exceed device memory limits
- Mix allocation strategies without profiling

---

## Conclusion

This technical guide provides the foundation for optimal memory allocator selection and configuration in XSigma applications. Always profile your specific workload and adjust configurations accordingly.

**For Support:**
- GitHub Issues: https://github.com/xsigma/xsigma
- Documentation: https://docs.xsigma.co.uk
- Email: support@xsigma.co.uk
