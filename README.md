# XSigma - Modern CMake Build System

XSigma is a high-performance C++ library with a comprehensive CMake build system that provides cross-platform compatibility, advanced optimization, and flexible third-party dependency management.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Build Configuration](#build-configuration)
- [Third-Party Dependencies](#third-party-dependencies)
- [Vectorization Support](#vectorization-support)
- [Cross-Platform Building](#cross-platform-building)
- [Advanced Configuration](#advanced-configuration)
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
| TBB | `XSIGMA_ENABLE_TBB=OFF` | Threading Building Blocks | `XSigma::tbb` |
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
git submodule add https://github.com/oneapi-src/oneTBB.git ThirdParty/tbb
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

### Sanitizers

```bash
# Address sanitizer
cmake -B build -S . \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=address

# Thread sanitizer
cmake -B build -S . \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=thread
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

## Usage Examples

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

## Project Structure

```
XSigma/
├── CMakeLists.txt              # Main CMake configuration
├── README.md                   # This file
├── LICENSE                     # Project license
├── Library/                    # Core library source code
│   ├── CMakeLists.txt         # Library configuration
│   └── Core/                  # Core module
│       ├── CMakeLists.txt     # Core module configuration
│       └── Testing/           # Core tests
├── ThirdParty/                # Third-party dependencies
│   ├── CMakeLists.txt         # Dependency configuration
│   ├── README.md              # Dependency setup guide
│   └── [submodules]           # Git submodules for dependencies
├── Cmake/                     # CMake utilities and tools
│   └── tools/                 # CMake helper modules
│       ├── xsigmaUtils.cmake  # Core utilities + vectorization
│       ├── cuda.cmake         # CUDA configuration
│       ├── xsigmaTestUtils.cmake # Testing utilities
│       └── [other tools]      # Additional CMake modules
└── Data/                      # Project data files
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
```
