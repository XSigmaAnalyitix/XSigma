# XSigma Memory Allocator Selection Strategy

## Executive Summary

This document provides a comprehensive strategy for selecting the optimal memory allocator in the XSigma framework based on workload characteristics, performance requirements, and memory usage patterns. The strategy is designed to maximize performance while minimizing memory fragmentation and overhead.

## Table of Contents

1. [Allocator Overview](#allocator-overview)
2. [Selection Criteria](#selection-criteria)
3. [Decision Matrix](#decision-matrix)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Performance Characteristics](#performance-characteristics)
6. [Recommendations by Use Case](#recommendations-by-use-case)

---

## Allocator Overview

### Available Allocators

XSigma provides four primary CPU memory allocators, each optimized for different scenarios:

#### 1. **allocator_cpu** (Basic CPU Allocator)
- **Type**: Direct system allocator wrapper
- **Backend**: mimalloc, TBB scalable_allocator, or system malloc
- **Best For**: General-purpose allocations, unpredictable patterns
- **Overhead**: Minimal (~1-2% with stats enabled)
- **Thread Safety**: Fully thread-safe

#### 2. **allocator_bfc** (Best-Fit with Coalescing)
- **Type**: Sophisticated memory pool with coalescing
- **Backend**: Custom implementation with sub-allocator
- **Best For**: Large, long-lived allocations with predictable patterns
- **Overhead**: Moderate (~5-10% for management)
- **Thread Safety**: Fully thread-safe with fine-grained locking

#### 3. **allocator_pool** (LRU Pool Allocator)
- **Type**: Fixed-size pool with LRU eviction
- **Backend**: Configurable sub-allocator
- **Best For**: Repeated allocations of similar sizes
- **Overhead**: Low (~2-5% for pool management)
- **Thread Safety**: Fully thread-safe

#### 4. **allocator_tracking** (Debugging Wrapper)
- **Type**: Wrapper providing comprehensive tracking
- **Backend**: Any underlying allocator
- **Best For**: Development, debugging, profiling
- **Overhead**: Moderate (~5-15% depending on tracking level)
- **Thread Safety**: Fully thread-safe

---

## Selection Criteria

### Primary Factors

#### 1. **Allocation Size Pattern**

| Size Range | Recommended Allocator | Rationale |
|------------|----------------------|-----------|
| < 64 bytes | `allocator_cpu` | System allocators optimized for small objects |
| 64 B - 4 KB | `allocator_pool` | Excellent cache reuse for common sizes |
| 4 KB - 1 MB | `allocator_bfc` | Efficient coalescing reduces fragmentation |
| > 1 MB | `allocator_cpu` or `allocator_bfc` | Direct allocation or managed pool |

#### 2. **Allocation Frequency**

| Frequency | Recommended Allocator | Rationale |
|-----------|----------------------|-----------|
| Very High (>10K/sec) | `allocator_pool` | Minimal overhead, excellent cache reuse |
| High (1K-10K/sec) | `allocator_cpu` or `allocator_pool` | Balance between speed and flexibility |
| Moderate (100-1K/sec) | `allocator_bfc` | Coalescing benefits outweigh overhead |
| Low (<100/sec) | Any allocator | Performance difference negligible |

#### 3. **Allocation Lifetime**

| Lifetime | Recommended Allocator | Rationale |
|----------|----------------------|-----------|
| Very Short (<1ms) | `allocator_pool` | Pool reuse eliminates deallocation cost |
| Short (1ms-1s) | `allocator_cpu` | Fast allocation/deallocation |
| Medium (1s-1min) | `allocator_bfc` | Efficient memory management |
| Long (>1min) | `allocator_bfc` or `allocator_cpu` | Fragmentation management important |

#### 4. **Memory Pressure**

| Pressure Level | Recommended Allocator | Configuration |
|----------------|----------------------|---------------|
| Low (<50% RAM) | Any allocator | Default settings |
| Medium (50-75% RAM) | `allocator_bfc` | Enable garbage collection |
| High (75-90% RAM) | `allocator_bfc` | Aggressive GC, limited growth |
| Critical (>90% RAM) | `allocator_cpu` | Minimal overhead, direct allocation |

#### 5. **Thread Contention**

| Contention Level | Recommended Allocator | Rationale |
|------------------|----------------------|-----------|
| Single-threaded | Any allocator | No contention concerns |
| Low (2-4 threads) | `allocator_cpu` or `allocator_pool` | Good scalability |
| Medium (4-16 threads) | `allocator_cpu` | Excellent multi-threaded performance |
| High (>16 threads) | `allocator_cpu` with TBB | Designed for high concurrency |

---

## Decision Matrix

### Quick Selection Guide

```
START
  |
  ├─ Development/Debugging? ──YES──> allocator_tracking
  |                                   (wrap any allocator)
  NO
  |
  ├─ Allocation size predictable? ──YES──> Repeated similar sizes? ──YES──> allocator_pool
  |                                  |                                       (pool_size_limit: 20-100)
  |                                  NO
  |                                  |
  |                                  └──> Large allocations (>4KB)? ──YES──> allocator_bfc
  |                                                                           (allow_growth: true)
  NO
  |
  ├─ High frequency (>10K/sec)? ──YES──> allocator_pool
  |                                       (large pool_size_limit)
  NO
  |
  ├─ Memory constrained? ──YES──> allocator_bfc
  |                                (garbage_collection: true)
  NO
  |
  └──> allocator_cpu (default choice)
       (enable stats for monitoring)
```

---

## Implementation Guidelines

### Configuration Examples

#### Example 1: High-Frequency Small Allocations (Trading Engine)

```cpp
// Scenario: 50K allocations/sec, sizes 128-512 bytes, short-lived
auto sub_alloc = std::make_unique<basic_cpu_allocator>(0, {}, {});
auto size_rounder = std::make_unique<power_of_2_rounder>();

auto allocator = std::make_unique<allocator_pool>(
    100,    // Large pool for high reuse
    true,   // Auto-resize enabled
    std::move(sub_alloc),
    std::move(size_rounder),
    "trading_pool");
```

#### Example 2: Large Matrix Operations (PDE Solver)

```cpp
// Scenario: Large matrices (1MB-100MB), long-lived, predictable
auto sub_alloc = std::make_unique<basic_cpu_allocator>(0, {}, {});

allocator_bfc::Options opts;
opts.allow_growth = true;
opts.garbage_collection = true;
opts.allow_retry_on_failure = true;

auto allocator = std::make_unique<allocator_bfc>(
    std::move(sub_alloc),
    2ULL * 1024ULL * 1024ULL * 1024ULL,  // 2GB pool
    "matrix_allocator",
    opts);
```

#### Example 3: General-Purpose with Monitoring

```cpp
// Scenario: Mixed workload, need statistics
EnableCPUAllocatorStats();
Allocator* allocator = cpu_allocator(0);

// Wrap with tracking for detailed analysis
auto tracking = new allocator_tracking(allocator, true, true);
tracking->SetLoggingLevel(tracking_log_level::INFO);
```

#### Example 4: Memory-Constrained Environment

```cpp
// Scenario: Limited RAM, need aggressive management
auto sub_alloc = std::make_unique<basic_cpu_allocator>(0, {}, {});

allocator_bfc::Options opts;
opts.allow_growth = false;  // Fixed pool
opts.garbage_collection = true;
opts.allow_retry_on_failure = true;
opts.fragmentation_fraction = 0.1;  // Aggressive defragmentation

auto allocator = std::make_unique<allocator_bfc>(
    std::move(sub_alloc),
    512ULL * 1024ULL * 1024ULL,  // 512MB fixed
    "constrained_allocator",
    opts);
```

---

## Performance Characteristics

### Benchmark Results Summary

Based on comprehensive benchmarking (see `BenchmarkCPUMemoryAllocators.cxx`):

| Allocator | Small (<64B) | Medium (64B-4KB) | Large (>4KB) | Thread Scaling |
|-----------|--------------|------------------|--------------|----------------|
| allocator_cpu | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| allocator_bfc | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★★☆ |
| allocator_pool | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| allocator_tracking | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ |

**Legend**: ★★★★★ Excellent | ★★★★☆ Good | ★★★☆☆ Fair | ★★☆☆☆ Poor

### Memory Overhead

| Allocator | Per-Allocation Overhead | Pool Overhead | Total Overhead |
|-----------|------------------------|---------------|----------------|
| allocator_cpu | 0-16 bytes | None | Minimal |
| allocator_bfc | 32-64 bytes | 1-5% of pool | Moderate |
| allocator_pool | 24-48 bytes | 2-10% of pool | Low-Moderate |
| allocator_tracking | +16-32 bytes | +tracking data | High (debug only) |

---

## Recommendations by Use Case

### 1. **Real-Time Trading Systems**
- **Primary**: `allocator_pool` (pool_size_limit: 100-500)
- **Rationale**: Predictable latency, minimal jitter
- **Configuration**: Large pool, power-of-2 rounding, no auto-resize

### 2. **Monte Carlo Simulations**
- **Primary**: `allocator_bfc` (2-8GB pool)
- **Rationale**: Large temporary allocations, good coalescing
- **Configuration**: Allow growth, enable GC, moderate fragmentation tolerance

### 3. **PDE/FEM Solvers**
- **Primary**: `allocator_bfc` (4-16GB pool)
- **Rationale**: Very large matrices, long-lived allocations
- **Configuration**: Large pool, allow growth, aggressive coalescing

### 4. **Data Analytics/Processing**
- **Primary**: `allocator_cpu` (with TBB if available)
- **Rationale**: Unpredictable patterns, high parallelism
- **Configuration**: Enable statistics, NUMA-aware if multi-socket

### 5. **Machine Learning Training**
- **Primary**: `allocator_bfc` (GPU-compatible sub-allocator)
- **Rationale**: Large tensors, predictable lifecycle
- **Configuration**: Very large pool (16-64GB), allow growth

### 6. **Development/Testing**
- **Primary**: `allocator_tracking` wrapping appropriate allocator
- **Rationale**: Comprehensive debugging and profiling
- **Configuration**: Enhanced tracking enabled, INFO logging level

---

## Advanced Topics

### Dynamic Allocator Switching

For applications with varying workload phases, consider implementing dynamic allocator selection:

```cpp
class adaptive_allocator_manager {
public:
    Allocator* get_allocator(allocation_context ctx) {
        if (ctx.size < 1024 && ctx.frequency > 10000) {
            return pool_allocator_;
        } else if (ctx.size > 1024 * 1024) {
            return bfc_allocator_;
        }
        return cpu_allocator_;
    }

private:
    Allocator* pool_allocator_;
    Allocator* bfc_allocator_;
    Allocator* cpu_allocator_;
};
```

### NUMA Considerations

For multi-socket systems, use NUMA-aware allocation:

```cpp
// Allocate on specific NUMA node
Allocator* numa_alloc = cpu_allocator(numa_node_id);

// Or use NUMANOAFFINITY for automatic placement
Allocator* auto_alloc = cpu_allocator(NUMANOAFFINITY);
```

### Monitoring and Adaptation

Implement runtime monitoring to adapt allocator selection:

```cpp
void monitor_and_adapt() {
    auto stats = allocator->GetStats();
    if (stats.has_value()) {
        double efficiency = stats->memory_efficiency();
        if (efficiency < 0.7) {
            // Consider switching allocator or adjusting configuration
            XSIGMA_LOG_WARNING("Low memory efficiency: {:.2f}%", efficiency * 100.0);
        }
    }
}
```

---

## Conclusion

Selecting the right allocator is crucial for optimal performance in the XSigma framework. Use this guide as a starting point, but always profile your specific workload to validate the choice. The provided benchmarks and tests can help you make data-driven decisions.

For questions or contributions to this strategy, please contact the XSigma development team.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Authors**: XSigma Development Team
