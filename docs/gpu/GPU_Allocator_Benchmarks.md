# XSigma GPU Memory Allocator Benchmarks

## Overview

This document describes the comprehensive benchmark suite for XSigma's GPU memory allocators. The benchmarks compare different allocation strategies and provide performance analysis for various workload patterns.

## Benchmark Architecture

### Allocators Tested

1. **allocator_gpu (Direct)** - Direct GPU API wrapper with compile-time strategy selection
   - SYNC: `cuMemAlloc`/`cuMemFree` (CUDA) or `hipMalloc`/`hipFree` (HIP)
   - ASYNC: `cuMemAllocAsync`/`cuMemFreeAsync` (CUDA) or `hipMallocAsync`/`hipFreeAsync` (HIP)
   - POOL_ASYNC: `cuMemAllocFromPoolAsync` (CUDA) or `hipMallocFromPoolAsync` (HIP)

2. **cuda_caching_allocator** - Intelligent caching allocator with stream awareness
   - Multiple cache sizes: 64MB, 256MB, 512MB
   - Stream-aware memory management
   - CUDA event synchronization

3. **Direct CUDA Allocator** - Baseline `cudaMalloc`/`cudaFree` for comparison

### Benchmark Scenarios

#### 1. Small Allocations Test
- **Size Range**: 16 bytes to 1MB
- **Pattern**: High frequency, mixed allocation/deallocation
- **Purpose**: Test overhead and cache efficiency

#### 2. Large Allocations Test  
- **Size Range**: 1MB to 16MB
- **Pattern**: Lower frequency, larger blocks
- **Purpose**: Test memory management for large data structures

#### 3. Multi-threaded Test
- **Threads**: 4 concurrent threads
- **Pattern**: Mixed allocation patterns with contention
- **Purpose**: Test thread safety and scalability

#### 4. High Frequency Test
- **Pattern**: Rapid allocation/deallocation cycles
- **Size Range**: 256 bytes to 256KB
- **Purpose**: Stress test allocator performance

### Performance Metrics

- **Allocation Time**: Average time per allocation (nanoseconds)
- **Deallocation Time**: Average time per deallocation (nanoseconds)
- **Throughput**: Memory bandwidth (MB/second)
- **Total Time**: Overall benchmark duration
- **Allocation Failures**: Number of failed allocations
- **Peak Memory Usage**: Maximum memory consumption
- **Size Class Performance**: Performance breakdown by allocation size

## Running Benchmarks

### Quick Benchmark (Current Build)

```bash
# Run benchmarks with current build configuration
python scripts/run_gpu_benchmarks.py .
```

This runs benchmarks on the currently built allocation method and generates:
- Console output with performance metrics
- `gpu_allocator_benchmark_report.txt` - Detailed results
- Performance comparison tables

### Comprehensive Benchmark (All Methods)

```bash
# Test all allocation methods (requires rebuilding)
python scripts/benchmark_gpu_allocators.py .
```

This script:
1. Builds XSigma with each allocation method (SYNC, ASYNC, POOL_ASYNC)
2. Runs comprehensive benchmarks for each method
3. Generates comparative analysis report
4. Provides recommendations based on results

### Manual Method Testing

To test specific allocation methods manually:

```bash
cd Scripts

# Test SYNC method
python setup.py config.ninja.clang.python.build.test -DXSIGMA_CUDA_ALLOC=SYNC
python setup.py ninja.clang.python.build.test

# Test ASYNC method  
python setup.py config.ninja.clang.python.build.test -DXSIGMA_CUDA_ALLOC=ASYNC
python setup.py ninja.clang.python.build.test

# Test POOL_ASYNC method
python setup.py config.ninja.clang.python.build.test -DXSIGMA_CUDA_ALLOC=POOL_ASYNC
python setup.py ninja.clang.python.build.test
```

Then run the test executable:
```bash
./build_ninja_python/xsigmaTest --gtest_filter=GpuAllocatorBenchmark.*
```

## Benchmark Implementation

### Core Components

#### 1. GPU Timer (`gpu_timer`)
- Uses CUDA events for precise GPU timing
- Fallback to CPU timing when CUDA unavailable
- Nanosecond precision for allocation measurements

#### 2. Statistics Collector (`gpu_benchmark_stats`)
- Thread-safe atomic counters
- Size class performance tracking
- Comprehensive metric calculation

#### 3. Allocation Pattern Generator
- Realistic workload simulation
- Configurable allocation/deallocation ratios
- Size class distribution matching real applications

#### 4. Allocator Wrappers
- Unified interface for different allocators
- Consistent measurement methodology
- Proper resource cleanup

### Test Configuration

```cpp
struct gpu_benchmark_config {
    size_t num_threads = 1;           // Concurrency level
    size_t num_iterations = 1000;     // Operations per thread
    size_t min_alloc_size = 16;       // Minimum allocation size
    size_t max_alloc_size = 16*1024*1024; // Maximum allocation size
    double allocation_ratio = 0.7;    // Allocation vs deallocation ratio
    std::string test_name;            // Test identifier
    std::vector<size_t> size_classes; // Predefined size classes
};
```

## Interpreting Results

### Performance Indicators

#### Good Performance
- **Allocation Time**: < 1000 ns for small allocations
- **Cache Hit Rate**: > 90% for caching allocators
- **Throughput**: > 1000 MB/s for large allocations
- **Zero Failures**: No allocation failures under normal load

#### Performance Issues
- **High Allocation Time**: > 10000 ns indicates driver overhead
- **Low Cache Hit Rate**: < 70% suggests poor caching strategy
- **High Failure Rate**: > 0% indicates memory pressure
- **Poor Scalability**: Performance degrades with thread count

### Allocation Method Comparison

#### SYNC Method
- **Best For**: Simple, single-threaded applications
- **Characteristics**: Predictable performance, lower complexity
- **Trade-offs**: Higher latency, no stream parallelism

#### ASYNC Method  
- **Best For**: Stream-based parallel workloads
- **Characteristics**: Lower latency, stream parallelism
- **Trade-offs**: Requires stream management, complexity

#### POOL_ASYNC Method
- **Best For**: High-frequency allocation patterns
- **Characteristics**: Lowest latency, memory pool efficiency
- **Trade-offs**: Higher memory usage, pool management overhead

### Caching Allocator Analysis

#### Cache Size Impact
- **64MB Cache**: Good for small applications, lower memory overhead
- **256MB Cache**: Balanced performance and memory usage
- **512MB+ Cache**: Best performance for large applications

#### Cache Efficiency Metrics
- **Hit Rate**: Percentage of allocations served from cache
- **Driver Call Reduction**: Factor of reduction in GPU driver calls
- **Memory Overhead**: Additional memory used for caching

## Optimization Recommendations

### Based on Workload

#### Monte Carlo Simulations
- **Recommended**: CUDA caching allocator with 256MB+ cache
- **Alternative**: POOL_ASYNC method for consistent allocation sizes
- **Avoid**: SYNC method for high-frequency patterns

#### PDE Solvers
- **Recommended**: ASYNC method with stream management
- **Alternative**: Direct allocator for large, infrequent allocations
- **Consider**: Memory pools for fixed-size grids

#### Machine Learning Workloads
- **Recommended**: POOL_ASYNC method with appropriate pool sizes
- **Alternative**: Caching allocator with large cache
- **Monitor**: Memory fragmentation and peak usage

### Performance Tuning

1. **Profile First**: Run benchmarks to establish baseline
2. **Match Workload**: Choose allocator based on allocation patterns
3. **Monitor Metrics**: Track cache hit rates and failure rates
4. **Adjust Parameters**: Tune cache sizes and pool configurations
5. **Validate**: Re-run benchmarks after changes

## Troubleshooting

### Common Issues

#### Build Failures
- Ensure CUDA/HIP development tools are installed
- Check CMake configuration flags
- Verify compiler compatibility

#### Benchmark Failures
- Check GPU memory availability
- Ensure CUDA runtime is properly installed
- Verify device permissions and access

#### Performance Issues
- Monitor GPU memory usage during benchmarks
- Check for memory leaks or fragmentation
- Verify optimal GPU device selection

### Debug Information

Enable detailed logging:
```bash
export XSIGMA_LOG_LEVEL=DEBUG
./xsigmaTest --gtest_filter=GpuAllocatorBenchmark.*
```

Check GPU status:
```bash
nvidia-smi  # For NVIDIA GPUs
rocm-smi    # For AMD GPUs
```

## Future Enhancements

### Planned Improvements

1. **HIP Benchmark Support**: Extend benchmarks to AMD GPUs
2. **Memory Pattern Analysis**: Advanced allocation pattern detection
3. **Automated Tuning**: Self-optimizing allocator parameters
4. **Cross-Platform Testing**: Windows, Linux, macOS validation
5. **Integration Testing**: Real-world application benchmarks

### Contributing

To add new benchmark scenarios:

1. Extend `gpu_benchmark_config` with new parameters
2. Implement test logic in benchmark worker threads
3. Add result analysis in report generation
4. Update documentation with new metrics

The benchmark suite is designed to be extensible and maintainable, providing comprehensive performance analysis for XSigma's GPU memory management system.
