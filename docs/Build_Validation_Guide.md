# XSigma GPU Allocator Build Validation Guide

## Overview

This guide provides comprehensive instructions for building and validating the refactored XSigma GPU memory allocator system. The new architecture supports multiple GPU allocation strategies with cross-platform compatibility.

## Prerequisites

### Required Dependencies

#### Windows
- **Visual Studio 2019/2022** or **Clang 12+**
- **CUDA Toolkit 11.0+** (for CUDA support)
- **ROCm 5.0+** (for HIP/AMD GPU support)
- **CMake 3.18+**
- **Python 3.7+**

#### Linux
- **GCC 9+** or **Clang 12+**
- **CUDA Toolkit 11.0+** (for CUDA support)
- **ROCm 5.0+** (for HIP/AMD GPU support)
- **CMake 3.18+**
- **Python 3.7+**

#### macOS
- **Xcode 12+** or **Clang 12+**
- **CMake 3.18+**
- **Python 3.7+**
- **Note**: CUDA support limited on newer macOS versions

## Build Configuration

### GPU Allocation Strategy Selection

The build system supports three GPU allocation strategies via the `XSIGMA_CUDA_ALLOC` CMake flag:

```bash
# Synchronous allocation (default)
-DXSIGMA_CUDA_ALLOC=SYNC

# Asynchronous allocation
-DXSIGMA_CUDA_ALLOC=ASYNC

# Pool-based asynchronous allocation
-DXSIGMA_CUDA_ALLOC=POOL_ASYNC
```

### GPU Platform Selection

```bash
# CUDA support (NVIDIA GPUs)
-DXSIGMA_ENABLE_CUDA=ON

# HIP support (AMD GPUs)
-DXSIGMA_USE_HIP=ON

# CPU-only build
-DXSIGMA_ENABLE_CUDA=OFF -DXSIGMA_USE_HIP=OFF
```

## Build Process

### Standard Build (CUDA with SYNC allocation)

```bash
cd Scripts
python setup.py config.ninja.clang.python.build.test
python setup.py ninja.clang.python.build.test
```

### Custom Configuration Build

```bash
cd Scripts

# Configure with specific allocation method
python setup.py config.ninja.clang.python.build.test \
    -DXSIGMA_CUDA_ALLOC=ASYNC \
    -DXSIGMA_ENABLE_CUDA=ON

# Build
python setup.py ninja.clang.python.build.test
```

### HIP Build (AMD GPUs)

```bash
cd Scripts

# Configure for HIP
python setup.py config.ninja.clang.python.build.test \
    -DXSIGMA_USE_HIP=ON \
    -DXSIGMA_CUDA_ALLOC=SYNC

# Build
python setup.py ninja.clang.python.build.test
```

## Validation Process

### 1. Build Validation

#### Check Build Success
```bash
# Verify build completed without errors
echo $?  # Should return 0 on Unix systems
```

#### Verify Executables
```bash
# Check test executable exists
ls -la build_ninja_python/xsigmaTest
ls -la build_ninja_python/Library/Core/Testing/Cxx/xsigmaTest
```

### 2. Test Validation

#### Run All GPU Tests
```bash
# Run all GPU-related tests
./build_ninja_python/xsigmaTest --gtest_filter=*Cuda*
./build_ninja_python/xsigmaTest --gtest_filter=*Gpu*
./build_ninja_python/xsigmaTest --gtest_filter=*CudaCaching*
```

#### Run Specific Test Suites
```bash
# Core allocator tests
./build_ninja_python/xsigmaTest --gtest_filter=AllocatorCuda.*

# Benchmark tests
./build_ninja_python/xsigmaTest --gtest_filter=GpuAllocatorBenchmark.*

# Factory tests
./build_ninja_python/xsigmaTest --gtest_filter=GpuAllocatorFactory.*
```

#### Expected Test Results
- **All tests should PASS**
- **No memory leaks** reported
- **No segmentation faults** or crashes
- **Performance benchmarks** complete successfully

### 3. Performance Validation

#### Quick Performance Test
```bash
# Run performance benchmarks
python scripts/run_gpu_benchmarks.py .
```

#### Comprehensive Performance Analysis
```bash
# Test all allocation methods
python scripts/benchmark_gpu_allocators.py .
```

#### Expected Performance Metrics
- **Allocation Time**: < 1000 ns for small allocations
- **Throughput**: > 1000 MB/s for large allocations
- **Cache Hit Rate**: > 90% for caching allocator
- **Zero Allocation Failures** under normal conditions

### 4. Cross-Platform Validation

#### Test Different Allocation Methods
```bash
# Test SYNC method
cd Scripts
python setup.py config.ninja.clang.python.build.test -DXSIGMA_CUDA_ALLOC=SYNC
python setup.py ninja.clang.python.build.test
./build_ninja_python/xsigmaTest --gtest_filter=GpuAllocatorBenchmark.AllocationMethodComparison

# Test ASYNC method
python setup.py config.ninja.clang.python.build.test -DXSIGMA_CUDA_ALLOC=ASYNC
python setup.py ninja.clang.python.build.test
./build_ninja_python/xsigmaTest --gtest_filter=GpuAllocatorBenchmark.AllocationMethodComparison

# Test POOL_ASYNC method
python setup.py config.ninja.clang.python.build.test -DXSIGMA_CUDA_ALLOC=POOL_ASYNC
python setup.py ninja.clang.python.build.test
./build_ninja_python/xsigmaTest --gtest_filter=GpuAllocatorBenchmark.AllocationMethodComparison
```

## Troubleshooting

### Common Build Issues

#### 1. Missing CUDA Toolkit
```
Error: CUDA not found
Solution: Install CUDA Toolkit 11.0+ and ensure nvcc is in PATH
```

#### 2. Missing HIP/ROCm
```
Error: HIP not found
Solution: Install ROCm 5.0+ and ensure HIP is properly configured
```

#### 3. Compiler Issues
```
Error: C++ compiler not found
Solution: Install Visual Studio 2019+, GCC 9+, or Clang 12+
```

#### 4. CMake Configuration Errors
```
Error: CMake configuration failed
Solution: Ensure CMake 3.18+ is installed and all dependencies are available
```

### Common Test Issues

#### 1. No CUDA Devices Available
```
Test Output: "No CUDA devices available"
Solution: Ensure CUDA-capable GPU is installed and drivers are up to date
```

#### 2. Memory Allocation Failures
```
Test Output: Allocation failures > 0
Solution: Check GPU memory availability and close other GPU applications
```

#### 3. Performance Issues
```
Test Output: Low throughput or high latency
Solution: Check GPU utilization and thermal throttling
```

## Validation Checklist

### Build Validation ✅
- [ ] Build completes without errors
- [ ] All source files compile successfully
- [ ] Test executable is generated
- [ ] No linker errors or warnings

### Test Validation ✅
- [ ] All GPU allocator tests pass
- [ ] Benchmark tests complete successfully
- [ ] No memory leaks detected
- [ ] No crashes or segmentation faults

### Performance Validation ✅
- [ ] Allocation latency < 1000 ns
- [ ] Throughput > 1000 MB/s for large allocations
- [ ] Cache hit rate > 90% for caching allocator
- [ ] Zero allocation failures under normal load

### Cross-Platform Validation ✅
- [ ] SYNC allocation method works
- [ ] ASYNC allocation method works
- [ ] POOL_ASYNC allocation method works
- [ ] CUDA support functional (if available)
- [ ] HIP support functional (if available)

### Code Quality Validation ✅
- [ ] No compilation warnings
- [ ] Follows XSigma coding standards
- [ ] Proper error handling (no exceptions)
- [ ] Memory alignment requirements met

## Success Criteria

### Build Success Indicators
1. **Zero compilation errors**
2. **Zero linker errors**
3. **Test executable generated**
4. **All dependencies resolved**

### Test Success Indicators
1. **All tests pass** (100% pass rate)
2. **Test coverage ≥ 98%**
3. **No memory leaks**
4. **Performance benchmarks complete**

### Performance Success Indicators
1. **Sub-microsecond allocation latency**
2. **High throughput (>1000 MB/s)**
3. **Efficient cache utilization**
4. **Scalable multi-threaded performance**

## Final Validation Report

After completing all validation steps, generate a final report:

```bash
# Generate comprehensive test coverage report
python scripts/analyze_gpu_test_coverage.py .

# Generate performance benchmark report
python scripts/benchmark_gpu_allocators.py .

# Check for any remaining issues
./build_ninja_python/xsigmaTest --gtest_output=xml:test_results.xml
```

## Continuous Integration

### Automated Build Pipeline
1. **Configure** with all allocation methods
2. **Build** on multiple platforms
3. **Test** all functionality
4. **Benchmark** performance
5. **Report** results

### Quality Gates
- **Build Success**: 100% success rate
- **Test Coverage**: ≥ 98%
- **Performance**: Within acceptable thresholds
- **Memory Safety**: Zero leaks or errors

The validation process ensures the refactored GPU allocator system meets all quality, performance, and reliability requirements for production use.
