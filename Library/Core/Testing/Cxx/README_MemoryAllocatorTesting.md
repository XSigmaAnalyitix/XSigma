# CPU Memory Allocator Testing and Benchmarking

This document describes the comprehensive testing and benchmarking suite for CPU memory allocators in the XSigma project.

## Overview

The XSigma project includes extensive testing and performance benchmarking for multiple CPU memory allocators:

1. **mimalloc** - Microsoft's high-performance memory allocator
2. **TBB allocators** - Intel TBB's scalable_malloc and cache_aligned_allocator
3. **Standard aligned allocators** - POSIX aligned_alloc, Windows _aligned_malloc
4. **XSigma CPU allocator** - XSigma's default CPU memory allocation wrapper

## Test Files

### TestCPUMemoryAllocators.cxx
Comprehensive unit tests covering:
- Basic allocation/deallocation functionality
- Memory alignment verification (16, 32, 64, 128, 256, 512, 1024 bytes)
- Large allocation handling (1MB, 16MB, 64MB)
- Memory pattern integrity testing
- Multi-threaded allocation scenarios
- Edge cases and error handling
- Stress testing with mixed allocation sizes

### BenchmarkCPUMemoryAllocators.cxx
Performance benchmarks measuring:
- Simple allocation/deallocation speed
- Batch allocation performance
- Mixed-size allocation patterns
- Memory access patterns and cache performance
- Alignment-specific performance
- Memory fragmentation resistance

## Test Methodology

### Unit Testing Approach

#### 1. **Standardized Interface**
All allocators are wrapped in a common `allocator_test_interface` providing:
```cpp
virtual void* allocate(std::size_t size, std::size_t alignment = 64) noexcept = 0;
virtual void deallocate(void* ptr, std::size_t size = 0) noexcept = 0;
virtual const char* name() const noexcept = 0;
virtual bool supports_alignment(std::size_t alignment) const noexcept = 0;
virtual bool is_thread_safe() const noexcept = 0;
```

#### 2. **Comprehensive Test Coverage**
- **Basic Functionality**: Allocation sizes from 1 byte to 64KB
- **Alignment Testing**: Power-of-2 alignments from 16 to 1024 bytes
- **Large Allocations**: Up to 64MB allocations with boundary testing
- **Memory Integrity**: Pattern-based corruption detection
- **Multi-threading**: Concurrent allocation with 4-8 threads
- **Edge Cases**: Zero-size allocations, null pointer handling
- **Stress Testing**: 10,000 mixed-size allocations with random patterns

#### 3. **Thread Safety Verification**
Multi-threaded tests with:
- Concurrent allocation/deallocation across multiple threads
- Thread-local allocation patterns with cross-thread verification
- Memory pattern integrity checks to detect race conditions

### Benchmarking Methodology

#### 1. **Performance Metrics**
- **Allocation Speed**: Operations per second
- **Memory Throughput**: Bytes processed per second
- **Latency**: Microsecond-level timing precision
- **Scalability**: Performance across different allocation sizes

#### 2. **Benchmark Categories**

##### Simple Allocation Benchmarks
- Single allocation/deallocation cycles
- Size range: 64 bytes to 64KB
- Measures raw allocation speed

##### Batch Allocation Benchmarks
- Multiple allocations followed by batch deallocation
- Tests: 100/1000 allocations of 1KB/4KB blocks
- Measures allocation efficiency and memory management overhead

##### Mixed Size Benchmarks
- Random allocation sizes (64-4096 bytes)
- Tests realistic workload patterns
- Measures fragmentation resistance

##### Memory Access Pattern Benchmarks
- Sequential read/write after allocation
- Tests cache performance and memory locality
- Size range: 1KB to 1MB

##### Alignment-Specific Benchmarks
- Fixed 1KB allocations with varying alignment (16-512 bytes)
- Measures alignment overhead

##### Fragmentation Resistance Benchmarks
- Allocate many small blocks, free every other block
- Attempt large allocations in fragmented space
- Tests memory management efficiency

## Allocator Comparison

### Expected Performance Characteristics

#### mimalloc
- **Strengths**: Excellent general-purpose performance, low fragmentation
- **Use Cases**: High-frequency allocations, multi-threaded applications
- **Alignment**: Efficient for all power-of-2 alignments

#### TBB Scalable Allocator
- **Strengths**: Superior multi-threaded scalability, cache-aware allocation
- **Use Cases**: Parallel algorithms, NUMA-aware applications
- **Alignment**: Optimized for cache-line alignment (64 bytes)

#### Standard Aligned Allocators
- **Strengths**: Predictable behavior, system integration
- **Use Cases**: System compatibility, simple alignment requirements
- **Alignment**: Platform-dependent efficiency

#### XSigma CPU Allocator
- **Strengths**: Integrated with XSigma ecosystem, configurable backends
- **Use Cases**: XSigma applications, unified memory management
- **Alignment**: Depends on configured backend (TBB/standard)

## Running the Tests

### Prerequisites
```bash
# Enable required allocators in CMake
cmake -DXSIGMA_ENABLE_MIMALLOC=ON \
      -DXSIGMA_ENABLE_TBB=ON \
      -DXSIGMA_ENABLE_BENCHMARK=ON \
      -S . -B build
```

### Unit Tests
```bash
# Run all memory allocator tests
cd build
ctest -R CoreCxxTests -V

# Run specific test patterns
./Library/Core/Testing/Cxx/CoreCxxTests --gtest_filter="*cpu_memory_allocator_test*"
```

### Benchmarks
```bash
# Run all memory allocator benchmarks
./Library/Core/Testing/Cxx/CoreCxxBenchmark --benchmark_filter="BM_.*Allocation"

# Run specific allocator benchmarks
./Library/Core/Testing/Cxx/CoreCxxBenchmark --benchmark_filter="BM_Mimalloc_.*"

# Generate detailed reports
./Library/Core/Testing/Cxx/CoreCxxBenchmark \
    --benchmark_format=json \
    --benchmark_out=memory_allocator_results.json
```

## Interpreting Results

### Unit Test Results
- **PASS**: All allocators should pass basic functionality tests
- **Performance Warnings**: Logged for allocation failures or slow operations
- **Thread Safety**: Multi-threaded tests verify concurrent access safety

### Benchmark Results
Results are reported in microseconds with the following metrics:
- **Time**: Average time per operation
- **Bytes/sec**: Memory throughput
- **Items/sec**: Operations per second

#### Example Output
```
BM_Mimalloc_SimpleAllocation/1024        125 ns       125 ns   5623415   7.62GB/s
BM_TBBScalable_SimpleAllocation/1024      98 ns        98 ns    7142857   9.73GB/s
BM_StandardAligned_SimpleAllocation/1024  156 ns      156 ns   4487179   6.11GB/s
```

### Performance Analysis Guidelines

#### Allocation Speed
- **Excellent**: < 100ns per allocation
- **Good**: 100-200ns per allocation  
- **Acceptable**: 200-500ns per allocation
- **Poor**: > 500ns per allocation

#### Memory Throughput
- **High-performance allocators**: > 5GB/s
- **Standard allocators**: 1-5GB/s
- **System allocators**: < 1GB/s

#### Multi-threaded Scalability
- **Linear scaling**: Performance increases proportionally with threads
- **Contention**: Performance plateaus or decreases with more threads
- **Lock-free**: Minimal performance degradation under contention

## Allocator Selection Guidelines

### High-Frequency Small Allocations
**Recommended**: mimalloc or TBB scalable_malloc
- Optimized for frequent allocation/deallocation cycles
- Low per-allocation overhead
- Excellent fragmentation resistance

### Multi-threaded Applications
**Recommended**: TBB scalable_malloc
- Superior thread scalability
- NUMA-aware allocation strategies
- Cache-aligned allocation support

### Cache-Sensitive Applications
**Recommended**: TBB cache_aligned_allocator or mimalloc with 64-byte alignment
- Ensures cache-line alignment
- Reduces false sharing
- Improves memory access patterns

### Large Block Allocations
**Recommended**: Standard aligned allocators or mimalloc
- Efficient for infrequent large allocations
- Lower memory overhead for large blocks
- Direct system memory mapping

### XSigma Integration
**Recommended**: XSigma CPU allocator with appropriate backend
- Seamless integration with XSigma memory management
- Configurable backend selection
- Unified error handling and logging

## Continuous Integration

The memory allocator tests are integrated into the XSigma CI pipeline:
- **Unit tests** run on every commit across all platforms
- **Benchmarks** run on performance-critical changes
- **Regression detection** alerts on significant performance changes
- **Coverage reporting** ensures comprehensive test coverage

## Future Enhancements

Planned improvements to the testing suite:
1. **NUMA-aware testing** for multi-socket systems
2. **Memory pressure testing** under low-memory conditions
3. **Long-running stability tests** for memory leak detection
4. **Cross-platform performance comparison** charts
5. **Automated performance regression detection**
