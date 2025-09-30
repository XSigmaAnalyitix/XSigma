# XSigma - High-performance C++ library with modern CMake build system

XSigma is a high-performance C++ library with a comprehensive CMake build system that provides cross-platform compatibility, advanced optimization, and flexible third-party dependency management.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Build Configuration](#build-configuration)
- [Third-Party Dependencies](#third-party-dependencies)
- [Vectorization Support](#vectorization-support)
- [Cross-Platform Building](#cross-platform-building)
- [Advanced Configuration](#advanced-configuration)
- [Sanitizers](#sanitizers)
- [Code Coverage](#code-coverage)
- [Include-What-You-Use (IWYU)](#include-what-you-use-iwyu)
- [Cppcheck Static Analysis](#cppcheck-static-analysis)
- [Usage Examples](#usage-examples)
- [Migration Guide](#migration-guide)
- [Troubleshooting](#troubleshooting)

## Overview

The XSigma project features a modern CMake build system (3.16+ required) with:

- **Full cross-platform compatibility** (Windows, Linux, macOS)
- **Advanced build optimization** with compiler-specific flags and LTO
- **Conditional compilation** for third-party dependencies
- **Enhanced vectorization support** including CUDA
- **Modern CMake best practices** with target-based configuration

### Key Features

- **Platform Detection**: Automatic Windows, Linux, and macOS support
- **Compiler Support**: MSVC, GCC, and Clang with optimized flags
- **Vectorization**: SSE, AVX, AVX2, AVX512, and CUDA acceleration
- **Third-Party Integration**: Conditional compilation with Git submodules
- **Testing Framework**: Integrated Google Test and benchmarking support

## Quick Start

### Basic Build

```bash
# Clone and build with default settings
git clone <repository-url>
cd XSigma

# Configure and build
cmake -B build -S .
cmake --build build
```

### Optimized Release Build

```bash
# High-performance build with AVX2 and LTO
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DXSIGMA_ENABLE_LTO=ON \
    -DXSIGMA_VECTORIZATION_TYPE=avx2

cmake --build build --config Release
```

## Build Configuration

### Build Types

| Build Type | Description | Optimization |
|------------|-------------|--------------|
| `Debug` | Development build | No optimization, debug info |
| `Release` | Production build | Full optimization |
| `RelWithDebInfo` | Release with debug info | Optimized + debug symbols |
| `MinSizeRel` | Size-optimized build | Optimize for size |

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
```

### C++ Standard Selection

```bash
# C++17 (default)
cmake -B build -S . -DXSIGMA_CXX_STANDARD=17

# C++20
cmake -B build -S . -DXSIGMA_CXX_STANDARD=20

# C++23
cmake -B build -S . -DXSIGMA_CXX_STANDARD=23
```

### Optimization Options

```bash
# Enable Link-Time Optimization
cmake -B build -S . -DXSIGMA_ENABLE_LTO=ON

# Enable sanitizers for debugging
cmake -B build -S . -DXSIGMA_ENABLE_SANITIZER=ON -DXSIGMA_SANITIZER_TYPE=address
```

## Third-Party Dependencies

XSigma uses a **conditional compilation pattern** where each library is controlled by `XSIGMA_ENABLE_XXX` options.

### Mandatory Core Libraries (Always Included)

| Library | Description | Target Alias |
|---------|-------------|--------------|
| fmt | Modern C++ formatting | `XSigma::fmt` |
| cpuinfo | CPU feature detection | `XSigma::cpuinfo` |

### Optional Libraries (Enabled by Default)

| Library | Option | Description | Target Alias |
|---------|--------|-------------|--------------|
| magic_enum | `XSIGMA_ENABLE_MAGICENUM=ON` | Enum reflection | `XSigma::magic_enum` |
| loguru | `XSIGMA_ENABLE_LOGURU=ON` | Lightweight logging | `XSigma::loguru` |

### Optional Libraries (Disabled by Default)

| Library | Option | Description | Target Alias |
|---------|--------|-------------|--------------|
| mimalloc | `XSIGMA_ENABLE_MIMALLOC=OFF` | High-performance allocator | `XSigma::mimalloc` |
| Google Test | `XSIGMA_ENABLE_GTEST=OFF` | Testing framework | `XSigma::gtest` |
| Benchmark | `XSIGMA_ENABLE_BENCHMARK=OFF` | Microbenchmarking | `XSigma::benchmark` |

### Dependency Management Pattern

**Mandatory Libraries (fmt, cpuinfo):**
- Always included in the build
- Always create `XSigma::xxx` target aliases
- Always add `XSIGMA_HAS_XXX` compile definitions
- Always linked to Core target

**Optional Libraries (controlled by `XSIGMA_ENABLE_XXX`):**

**When `XSIGMA_ENABLE_XXX=ON`:**
1. Include the library in the build
2. Create `XSigma::xxx` target alias
3. Add `XSIGMA_HAS_XXX` compile definition
4. Link to Core target

**When `XSIGMA_ENABLE_XXX=OFF`:**
1. Skip library completely
2. No target aliases created
3. No compile definitions added
4. No linking attempted

### Setting Up Dependencies

#### Option 1: Git Submodules (Recommended)

```bash
# Core libraries
git submodule add https://github.com/fmtlib/fmt.git ThirdParty/fmt
git submodule add https://github.com/pytorch/cpuinfo.git ThirdParty/cpuinfo
git submodule add https://github.com/Neargye/magic_enum.git ThirdParty/magic_enum
git submodule add https://github.com/emilk/loguru.git ThirdParty/loguru

# Optional libraries
git submodule add https://github.com/microsoft/mimalloc.git ThirdParty/mimalloc
git submodule add https://github.com/google/benchmark.git ThirdParty/benchmark
git submodule add https://github.com/google/googletest.git ThirdParty/googletest

# Initialize
git submodule update --init --recursive
```

#### Option 2: External Libraries

```bash
# Use system-installed libraries
cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON
```

### Build Configuration for Third-Party Libraries

Third-party targets are configured to:
- Suppress warnings (using -w or /w)
- Use the same C++ standard as the main project
- Avoid altering the main project's compiler/linker settings
- Provide consistent target aliases with the XSigma:: prefix

### Notes
- When XSIGMA_USE_EXTERNAL=ON, system-installed libraries are preferred over bundled submodules.
- Some libraries may require additional system packages; consult their upstream documentation if find_package() fails.


## Vectorization Support

XSigma supports multiple vectorization backends including CPU SIMD instructions and CUDA GPU acceleration.

### Vectorization Options

| Type | Description | GCC/Clang Flags | MSVC Flags |
|------|-------------|-----------------|------------|
| `no` | No vectorization | None | None |
| `sse` | SSE/SSE2 instructions | `-msse -msse2` | `/arch:SSE2` |
| `avx` | AVX instructions | `-mavx` | `/arch:AVX` |
| `avx2` | AVX2 instructions (default) | `-mavx -mavx2` | `/arch:AVX2` |
| `avx512` | AVX-512 instructions | `-mavx -mavx2 -mavx512f` | `/arch:AVX512` |

### CPU Vectorization Examples

```bash
# AVX2 vectorization (default)
cmake -B build -S . -DXSIGMA_VECTORIZATION_TYPE=avx2

# SSE for older CPUs
cmake -B build -S . -DXSIGMA_VECTORIZATION_TYPE=sse

# Disable vectorization
cmake -B build -S . -DXSIGMA_VECTORIZATION_TYPE=no
```



## Cross-Platform Building

### Windows (Visual Studio)

```bash
# Release build with optimizations
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DXSIGMA_ENABLE_LTO=ON \
    -DXSIGMA_VECTORIZATION_TYPE=avx2

cmake --build build --config Release
```

### Linux (GCC/Clang)

```bash
# Release build with parallel compilation
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DXSIGMA_ENABLE_LTO=ON \
    -DXSIGMA_VECTORIZATION_TYPE=avx2

cmake --build build -j$(nproc)
```

### macOS (Clang)

```bash
# Optimized for Apple Silicon
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DXSIGMA_ENABLE_LTO=ON

cmake --build build -j$(sysctl -n hw.ncpu)
```

## Advanced Configuration

### Custom Compiler Flags

```bash
# Override optimization flags
cmake -B build -S . -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -DNDEBUG"
```


### CMake Optimization Modules Overview

XSigma ships optimized CMake modules that improve configuration speed and runtime performance. These are included automatically by the top-level CMakeLists.txt and require no manual setup:

- build_type.cmake: Optimized build-type flags (Release, Debug, RelWithDebInfo, MinSizeRel), LTO handling, MSVC runtime selection
- checks.cmake: Fast, cached platform/compiler validation and C++17 feature checks
- platform.cmake: Platform-specific optimizations (MSVC/GCC/Clang), vectorization flags, parallel builds, safe flag application

Paths: Cmake/flags/build_type.cmake, Cmake/flags/checks.cmake, Cmake/flags/platform.cmake

## Sanitizers

XSigma provides comprehensive sanitizer support for memory debugging and analysis with all modern sanitizers across multiple compilers.

### Supported Sanitizers

| Sanitizer | Purpose | GCC | Clang | Apple Clang | MSVC |
|-----------|---------|-----|-------|-------------|------|
| **AddressSanitizer** | Memory errors, buffer overflows | ✅ | ✅ | ✅ | ✅ |
| **UndefinedBehaviorSanitizer** | Undefined behavior detection | ✅ | ✅ | ✅ | ❌ |
| **ThreadSanitizer** | Data race detection | ✅ | ✅ | ✅ | ❌ |
| **MemorySanitizer** | Uninitialized memory reads | ❌ | ✅ | ✅ | ❌ |
| **LeakSanitizer** | Memory leak detection | ✅ | ✅ | ✅ | ❌ |

### Command-Line Usage

**Python Setup Script (Recommended):**
```bash
# AddressSanitizer - detects memory errors and buffer overflows
python setup.py ninja.clang.test.config --sanitizer.address

# UndefinedBehaviorSanitizer - detects undefined behavior
python setup.py ninja.clang.test.config --sanitizer.undefined

# ThreadSanitizer - detects data races and threading issues
python setup.py ninja.clang.test.config --sanitizer.thread

# MemorySanitizer - detects uninitialized memory reads (Clang only)
python setup.py ninja.clang.test.config --sanitizer.memory

# LeakSanitizer - detects memory leaks
python setup.py ninja.clang.test.config --sanitizer.leak

# Alternative syntax
python setup.py vs22.test.build --sanitizer-type=address
```

**Direct CMake Configuration:**
```bash
# AddressSanitizer
cmake -B build -S . \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=address

# UndefinedBehaviorSanitizer
cmake -B build -S . \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=undefined

# ThreadSanitizer
cmake -B build -S . \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=thread
```

### Sanitizer Descriptions and Use Cases

**AddressSanitizer (ASan)**
- **Purpose**: Detects buffer overflows, use-after-free, double-free, memory leaks
- **Performance**: ~2x slowdown, ~2-3x memory usage
- **Best for**: General memory debugging, CI/CD pipelines
- **Platforms**: All supported (Windows, Linux, macOS)

**UndefinedBehaviorSanitizer (UBSan)**
- **Purpose**: Detects undefined behavior like integer overflow, null pointer dereference
- **Performance**: ~20% slowdown, minimal memory overhead
- **Best for**: Code quality assurance, detecting subtle bugs
- **Platforms**: GCC, Clang (not MSVC)

**ThreadSanitizer (TSan)**
- **Purpose**: Detects data races, deadlocks, thread safety issues
- **Performance**: ~5-15x slowdown, ~5-10x memory usage
- **Best for**: Multithreaded code debugging
- **Platforms**: GCC, Clang (not MSVC)
- **Note**: Cannot be used with AddressSanitizer simultaneously

**MemorySanitizer (MSan)**
- **Purpose**: Detects reads of uninitialized memory
- **Performance**: ~3x slowdown, ~3x memory usage
- **Best for**: Finding uninitialized variable bugs
- **Platforms**: Clang only
- **Note**: Requires rebuilding all dependencies with MSan

**LeakSanitizer (LSan)**
- **Purpose**: Detects memory leaks
- **Performance**: Minimal runtime overhead
- **Best for**: Memory leak detection in long-running applications
- **Platforms**: GCC, Clang (not MSVC)
- **Note**: Can be used standalone or with AddressSanitizer

### Customizing Sanitizer Behavior

**Sanitizer Ignore File**

XSigma uses `Scripts/sanitizer_ignore.txt` to exclude files, functions, or types from sanitizer checks:

```bash
# Edit the ignore file to customize sanitizer behavior
vim Scripts/sanitizer_ignore.txt
```

**Common ignore patterns:**
```
# Ignore third-party libraries
src:*/ThirdParty/*
src:*/external/*

# Ignore specific functions
fun:*test*
fun:benchmark_*

# Ignore specific types
type:std::*
type:boost::*

# Sanitizer-specific exclusions
# AddressSanitizer only
src:*/performance_critical.cpp

# ThreadSanitizer only
fun:*lockfree*
```

**Environment Variables**

Control sanitizer behavior at runtime:

```bash
# AddressSanitizer options
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:check_initialization_order=1"

# UndefinedBehaviorSanitizer options
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"

# ThreadSanitizer options
export TSAN_OPTIONS="detect_deadlocks=1:second_deadlock_stack=1"

# MemorySanitizer options
export MSAN_OPTIONS="print_stats=1:halt_on_error=1"

# LeakSanitizer options
export LSAN_OPTIONS="suppressions=leak_suppressions.txt"
```

### Platform-Specific Considerations

**Windows (MSVC)**
- Only AddressSanitizer is supported
- Requires Visual Studio 2019 16.9+ or Visual Studio 2022
- May require `/MD` runtime library flag
- Debug information (`/Zi`) recommended for better stack traces

**Linux (GCC/Clang)**
- All sanitizers supported (MSan only with Clang)
- Runtime libraries automatically detected and preloaded
- Best performance with debug symbols (`-g`)
- Consider using `gold` linker for faster linking

**macOS (Apple Clang)**
- All sanitizers supported except MemorySanitizer
- System Integrity Protection (SIP) may interfere with some sanitizers
- Use `DYLD_INSERT_LIBRARIES` for runtime library preloading
- Xcode integration available through build schemes

### Best Practices

**Development Workflow**
1. **Start with AddressSanitizer** - catches most common memory errors
2. **Add UndefinedBehaviorSanitizer** - minimal overhead, catches subtle bugs
3. **Use ThreadSanitizer** for multithreaded code - run separately from ASan
4. **Apply MemorySanitizer** for critical code paths - requires clean build
5. **Enable LeakSanitizer** for long-running applications

**CI/CD Integration**
```bash
# Separate CI jobs for different sanitizers
- name: "AddressSanitizer Tests"
  run: python setup.py ninja.clang.test.build --sanitizer.address

- name: "UndefinedBehavior Tests"
  run: python setup.py ninja.clang.test.build --sanitizer.undefined

- name: "Thread Safety Tests"
  run: python setup.py ninja.clang.test.build --sanitizer.thread
```

**Performance Testing**
- Use sanitizers in debug/testing builds only
- Disable sanitizers for performance benchmarks
- Consider sanitizer overhead when setting test timeouts
- Use sanitizer-specific optimization flags when needed

### Troubleshooting Sanitizer Issues

**Common Problems and Solutions**

1. **"Sanitizer runtime library not found"**
   ```bash
   # Install sanitizer runtime libraries
   # Ubuntu/Debian
   sudo apt-get install libc6-dbg gcc-multilib

   # CentOS/RHEL
   sudo yum install glibc-debuginfo

   # macOS
   xcode-select --install
   ```

2. **"Cannot combine AddressSanitizer with ThreadSanitizer"**
   - These sanitizers are mutually exclusive
   - Run separate builds for each sanitizer
   - Use different build directories

3. **"MemorySanitizer requires rebuilding dependencies"**
   ```bash
   # Build all dependencies with MemorySanitizer
   export CC=clang
   export CXX=clang++
   export CFLAGS="-fsanitize=memory -fsanitize-memory-track-origins=2"
   export CXXFLAGS="-fsanitize=memory -fsanitize-memory-track-origins=2"
   ```

4. **"Sanitizer reports false positives"**
   - Add patterns to `Scripts/sanitizer_ignore.txt`
   - Use sanitizer-specific suppressions
   - Check for third-party library issues

5. **"Slow build times with sanitizers"**
   ```bash
   # Use faster linker
   export LDFLAGS="-fuse-ld=gold"  # Linux
   export LDFLAGS="-fuse-ld=lld"   # Clang

   # Reduce optimization level
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
   ```

6. **"Out of memory during sanitizer build"**
   - Increase system swap space
   - Use distributed compilation
   - Build with fewer parallel jobs: `cmake --build build -j2`

**Debugging Sanitizer Output**

```bash
# Get detailed stack traces

```

## Code Coverage

Generate code coverage reports to measure test effectiveness and identify untested code paths. XSigma uses LLVM coverage tools (llvm-profdata and llvm-cov) for source-based coverage analysis.

### Automatic Coverage Analysis (setup.py integration)

Coverage analysis is integrated into the Python build helper in `Scripts/setup.py`. Running `python setup.py ninja.clang.config.build.test.coverage` (from the `Scripts` directory) will automatically analyze coverage after data collection—no need to add an extra `.analyze` step. If you want to re-analyze existing coverage data without rebuilding, use the standalone `python setup.py analyze` command. For more details, see the documentation in the `Scripts/` folder (e.g., `Scripts/README_COVERAGE.md`).

### Quick Start

```bash
cd Scripts
python setup.py ninja.clang.config.tbb.build.coverage
```

The HTML coverage report will be generated at `build_ninja_tbb_coverage/coverage_report/html/index.html`

### What Gets Measured

Coverage analysis tracks:
- **Line coverage**: Which lines of code were executed during tests
- **Function coverage**: Which functions were called
- **Region coverage**: Which code branches were taken

Third-party libraries and test files are automatically excluded from coverage reports.

### Manual Coverage Commands

After building with coverage enabled, you can generate reports manually:

```bash
# Navigate to build directory
cd build_ninja_tbb_coverage

# Merge raw coverage data
cmake --build . --target coverage-merge

# Generate text report
cmake --build . --target coverage-report

# Generate HTML report
cmake --build . --target coverage-html
```

### Coverage Requirements

- **Compiler**: Clang with LLVM tools (llvm-profdata, llvm-cov)
- **Build type**: Coverage builds use Debug configuration with instrumentation flags
- **Tests**: Google Test framework must be enabled

For detailed coverage configuration and advanced usage, see `Cmake/tools/COVERAGE_USAGE.md`

## Include-What-You-Use (IWYU)

IWYU helps reduce unnecessary includes and enforces clean header dependencies across the codebase.

- CMake option: `XSIGMA_ENABLE_IWYU` (default: OFF)
- Applies to: XSigma targets only (ThirdParty targets are skipped)
- Logs: `build/iwyu.log` with per-file analysis also recorded under `build/iwyu_logs/`
- Mapping file (optional): `Scripts/iwyu_exclusion.imp` (used if present)

### Install IWYU
- Ubuntu/Debian: `sudo apt-get install iwyu`
- Fedora/CentOS/RHEL: `sudo dnf install iwyu`
- macOS (Homebrew): `brew install include-what-you-use`
- Windows: Download from https://include-what-you-use.org/ or build from source

### Enable and Run
```bash
# Configure with IWYU enabled
cmake -B build -S . -DXSIGMA_ENABLE_IWYU=ON

# Build (IWYU runs during compilation and writes logs)
cmake --build build -j

# Inspect the log for include suggestions
less build/iwyu.log
```

Notes:
- IWYU is crash-resistant and uses conservative flags configured in `Cmake/tools/iwyu.cmake`.
- If IWYU is not found and the option is ON, configuration fails with a helpful install hint.

## Cppcheck Static Analysis

Cppcheck provides static analysis for C/C++ code quality, style, performance, and portability.

- CMake option: `XSIGMA_ENABLE_CPPCHECK` (default: OFF)
- Optional: `XSIGMA_ENABLE_AUTOFIX` (WARNING: enables `--fix`, modifies source files!)
- Suppressions file (optional): `Scripts/cppcheck_suppressions.txt`
- Output log: `${CMAKE_BINARY_DIR}/cppcheckoutput.log`
- Third-party code is skipped automatically

### Install Cppcheck
- Ubuntu/Debian: `sudo apt-get install cppcheck`
- Fedora/CentOS/RHEL: `sudo dnf install cppcheck`
- macOS (Homebrew): `brew install cppcheck`
- Windows: `choco install cppcheck` or `winget install cppcheck`

### Enable and Run
```bash
# Configure with Cppcheck enabled
cmake -B build -S . -DXSIGMA_ENABLE_CPPCHECK=ON

# Optionally enable automatic fixes (use with caution)
cmake -B build -S . -DXSIGMA_ENABLE_CPPCHECK=ON -DXSIGMA_ENABLE_AUTOFIX=ON

# Build (cppcheck runs as part of compilation and writes the log file)
cmake --build build -j

# Review analysis results
less ${CMAKE_BINARY_DIR}/cppcheckoutput.log
```

Tips:
- Customize suppressions in `Scripts/cppcheck_suppressions.txt` to silence known safe patterns.

- The analysis is configured in `Cmake/tools/cppcheck.cmake` with platform-appropriate options.

```bash

export ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer

export MSAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer

# Save sanitizer output to file
export ASAN_OPTIONS="log_path=./asan_log"
export TSAN_OPTIONS="log_path=./tsan_log"

# Enable additional debugging
export ASAN_OPTIONS="verbosity=1:debug=1"
```

### Testing Configuration

```bash
# Enable testing with Google Test
cmake -B build -S . \
    -DXSIGMA_BUILD_TESTING=ON \
    -DXSIGMA_GOOGLE_TEST=ON

# Run tests
cmake --build build
ctest --test-dir build
```
## Usage Examples

### Enhanced Profiler (Experimental)

An advanced profiler for the XSigma Core module providing nanosecond-precision timing, memory tracking, hierarchical scopes, thread safety, and multiple output formats (console/JSON/CSV/XML).

- Location: Library/Core/experimental/profiler
- Quick start: see Library/Core/Testing/Cxx/TestEnhancedProfiler.cxx for end-to-end examples
- Macros: XSIGMA_PROFILE_SCOPE("name"), XSIGMA_PROFILE_FUNCTION()


### Example 1: Minimal Build (Fastest)

```bash
# Disable all optional libraries for fastest build
cmake -B build_minimal -S . \
    -DXSIGMA_ENABLE_MAGIC_ENUM=OFF \
    -DXSIGMA_ENABLE_LOGURU=OFF \
    -DXSIGMA_VECTORIZATION_TYPE=no
```

**Result:** Fastest build time, smallest binary, core functionality only.

### Example 2: High-Performance Build

```bash
# Enable performance libraries with optimizations
cmake -B build_performance -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DXSIGMA_ENABLE_LTO=ON \
    -DXSIGMA_ENABLE_TBB=ON \
    -DXSIGMA_ENABLE_MIMALLOC=ON \
    -DXSIGMA_VECTORIZATION_TYPE=avx2
```

**Result:** Maximum runtime performance with parallel processing and optimized memory allocation.

### Example 3: Development Build

```bash
# Enable testing and debugging tools
cmake -B build_dev -S . \
    -DCMAKE_BUILD_TYPE=Debug \
    -DXSIGMA_BUILD_TESTING=ON \
    -DXSIGMA_GOOGLE_TEST=ON \
    -DXSIGMA_ENABLE_BENCHMARK=ON \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=address
```

**Result:** Full development environment with testing, benchmarking, and debugging capabilities.

### Example 4: External Libraries Build

```bash
# Use system-installed libraries (faster build)
cmake -B build_external -S . \
    -DXSIGMA_USE_EXTERNAL=ON \
    -DXSIGMA_ENABLE_TBB=ON \
    -DXSIGMA_GOOGLE_TEST=ON
```

**Result:** Faster build using pre-installed system libraries.

## Code Integration

### Using Third-Party Libraries in C++

```cpp
#include "xsigma_features.h"

void example_function() {
    #ifdef XSIGMA_HAS_FMT
        // Use fmt library for formatting
        fmt::print("Hello, {}!\n", "World");
    #else
        // Fallback to standard library
        std::cout << "Hello, World!" << std::endl;
    #endif

    #ifdef XSIGMA_HAS_TBB
        // Use TBB for parallel algorithms
        tbb::parallel_for(/*...*/);
    #else
        // Use standard threading
        std::thread t(/*...*/);
    #endif

    #ifdef XSIGMA_HAS_MIMALLOC
        // mimalloc is available as drop-in replacement
        // No code changes needed - just link with XSigma::mimalloc
    #endif
}
```

### CMake Target Usage

```cmake
# Your custom target
add_executable(my_app main.cpp)

# Link with XSigma Core (always available)
target_link_libraries(my_app PRIVATE XSigma::Core)

# Conditionally link with third-party libraries
if(TARGET XSigma::fmt)
    target_link_libraries(my_app PRIVATE XSigma::fmt)
endif()

if(TARGET XSigma::benchmark)
    target_link_libraries(my_app PRIVATE XSigma::benchmark)
endif()
```

## Build Performance Comparison

| Configuration | Build Time | Binary Size | Features |
|---------------|------------|-------------|----------|
| Minimal | Fastest | Smallest | Core only |
| Core (Default) | Fast | Small | Essential libraries |
| Performance | Medium | Medium | High-performance libraries |
| Development | Slow | Large | All development tools |
| External | Fast | Small | Uses system libraries |

## Migration Guide

### From Previous Build System

The new CMake system maintains **complete backward compatibility**:

- ✅ Same CMake options and values
- ✅ Same function names and signatures
- ✅ Same compiler flags and definitions
- ✅ Same target configuration behavior

### Vectorization Configuration

The vectorization system supports CPU SIMD instruction sets:

**CPU Vectorization:**
- SSE, AVX, AVX2, AVX512 instruction sets
- Cross-platform compiler support (MSVC, GCC, Clang)
- Automatic detection and configuration

**Configuration:**
- Set via `XSIGMA_VECTORIZATION_TYPE` option
- Options: `no`, `sse`, `avx`, `avx2`, `avx512`

**No migration required** - all existing configurations continue to work.

## Troubleshooting

### Missing Third-Party Libraries

If you see warnings about missing third-party libraries:

1. **Add Git submodules** (recommended):
   ```bash
   git submodule add https://github.com/fmtlib/fmt.git ThirdParty/fmt
   git submodule update --init --recursive
   ```

2. **Use external libraries**:
   ```bash
   cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON
   ```

3. **Disable unused optional features**:
   ```bash
   cmake -B build -S . -DXSIGMA_ENABLE_MAGIC_ENUM=OFF
   ```

### Vectorization Issues

If vectorization fails to compile:

1. **Check compiler support** - the system auto-detects capabilities
2. **Use lower vectorization**:
   ```bash
   cmake -B build -S . -DXSIGMA_VECTORIZATION_TYPE=avx  # instead of avx2
   ```
3. **Disable vectorization**:
   ```bash
   cmake -B build -S . -DXSIGMA_VECTORIZATION_TYPE=no
   ```



### Link-Time Optimization (LTO) Issues

If LTO fails:

1. **Disable LTO**:
   ```bash
   cmake -B build -S . -DXSIGMA_ENABLE_LTO=OFF
   ```

2. **Check compiler version** - ensure recent GCC/Clang/MSVC
3. **Increase available memory** - LTO can be memory-intensive

### Build Performance Issues

For slow builds:

1. **Use external libraries**:
   ```bash
   cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON
   ```

2. **Disable unused features**:
   ```bash
   cmake -B build -S . -DXSIGMA_ENABLE_BENCHMARK=OFF -DXSIGMA_GOOGLE_TEST=OFF
   ```

3. **Use parallel compilation**:
   ```bash
   cmake --build build -j$(nproc)  # Linux/macOS
   cmake --build build -j%NUMBER_OF_PROCESSORS%  # Windows
   ```

## Best Practices

### Development Workflow

1. **Development builds**:
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DXSIGMA_GOOGLE_TEST=ON
   ```

2. **Production builds**:
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DXSIGMA_ENABLE_LTO=ON
   ```

3. **CI/CD builds**:
   ```bash
   cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON -DXSIGMA_BUILD_TESTING=ON
   ```

### Performance Optimization

1. **Enable high-performance libraries**:
   ```bash
   cmake -B build -S . -DXSIGMA_ENABLE_TBB=ON -DXSIGMA_ENABLE_MIMALLOC=ON
   ```

2. **Use appropriate vectorization**:
   - Modern CPUs: `avx2` or `avx512`
   - Older CPUs: `sse` or `avx`
   - Disable for compatibility: `no`

3. **Enable optimizations**:
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DXSIGMA_ENABLE_LTO=ON
   ```

### Resource-Constrained Environments

1. **Minimal builds**:
   ```bash
   cmake -B build -S . -DXSIGMA_ENABLE_MAGIC_ENUM=OFF -DXSIGMA_ENABLE_LOGURU=OFF
   ```

2. **Disable vectorization**:
   ```bash
   cmake -B build -S . -DXSIGMA_VECTORIZATION_TYPE=no
   ```

3. **Use external libraries**:
   ```bash
   cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON
   ```



## Contributing

When contributing to XSigma:

1. **Follow CMake best practices** - use modern target-based configuration
2. **Maintain backward compatibility** - existing configurations should continue working
3. **Test cross-platform** - verify Windows, Linux, and macOS compatibility
4. **Document changes** - update this README for new features
5. **Use conditional compilation** - follow the `XSIGMA_ENABLE_XXX` pattern

## License

See [LICENSE](LICENSE) file for details.

---

**XSigma** - High-performance C++ library with modern CMake build system
