# XSigma

A high-performance C++ library with a modern CMake build system providing cross-platform compatibility, advanced optimization, and flexible dependency management.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Features](#features)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

XSigma is a production-ready C++ library featuring:

- **Cross-platform compatibility** - Windows, Linux, and macOS support
- **Modern CMake build system** - CMake 3.16+ with best practices
- **Advanced optimization** - LTO, vectorization, and compiler-specific flags
- **Flexible logging** - Three backend options (LOGURU, GLOG, NATIVE)
- **Comprehensive testing** - Sanitizers, coverage analysis, and static analysis tools

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

### Using setup.py (Recommended)

```bash
cd Scripts
python setup.py ninja.clang.python.build.test
```

## Features

### Build Configuration

Configure your build with multiple options including build types (Debug, Release, RelWithDebInfo, MinSizeRel), C++ standard selection (C++17/20/23), and optimization settings.

**Key capabilities:**
- Multiple build types for different use cases
- Link-Time Optimization (LTO) for maximum performance
- Custom compiler flags and optimization levels

📖 **[Read more: Build Configuration Guide](docs/build-configuration.md)**

---

### Logging System

Flexible logging with three mutually-exclusive backends that can be selected at compile time, each offering different trade-offs between features, performance, and dependencies.

**Available backends:**
- **LOGURU** (default) - Full-featured logging with scopes and callbacks
- **GLOG** - Google's production-grade logging library
- **NATIVE** - Minimal native implementation with no dependencies

📖 **[Read more: Logging System Guide](docs/logging-system.md)**

---

### Third-Party Dependencies

Conditional compilation pattern where each library is controlled by `XSIGMA_ENABLE_XXX` options, allowing you to customize your build by including only the dependencies you need.

**Key features:**
- Git submodules or system-installed libraries
- Mandatory core libraries (fmt, cpuinfo)
- Optional libraries (magic_enum, loguru, mimalloc, Google Test, Benchmark)

📖 **[Read more: Third-Party Dependencies Guide](docs/third-party-dependencies.md)**

---

### Vectorization Support

CPU SIMD (Single Instruction, Multiple Data) instruction sets for high-performance computing, allowing the CPU to process multiple data elements simultaneously.

**Supported instruction sets:**
- SSE/SSE2 - Legacy compatibility
- AVX/AVX2 - Modern CPUs (default: AVX2)
- AVX-512 - Latest high-end systems

📖 **[Read more: Vectorization Guide](docs/vectorization.md)**

---

### Sanitizers

Comprehensive sanitizer support for memory debugging and analysis with all modern sanitizers across multiple compilers.

**Available sanitizers:**
- **AddressSanitizer** - Memory errors, buffer overflows
- **UndefinedBehaviorSanitizer** - Undefined behavior detection
- **ThreadSanitizer** - Data race detection
- **MemorySanitizer** - Uninitialized memory reads
- **LeakSanitizer** - Memory leak detection

📖 **[Read more: Sanitizers Guide](docs/sanitizers.md)**

---

### Code Coverage

Generate code coverage reports to measure test effectiveness and identify untested code paths using LLVM coverage tools (llvm-profdata and llvm-cov).

**Key features:**
- Line, function, and region coverage tracking
- HTML, text, and JSON report formats
- Automatic exclusion of third-party code
- Integrated with setup.py for easy use

📖 **[Read more: Code Coverage Guide](docs/code-coverage.md)**

---

### Static Analysis Tools

Integrated static analysis tools to improve code quality: Include-What-You-Use (IWYU) for header dependency management and Cppcheck for comprehensive static code analysis.

**Available tools:**
- **IWYU** - Reduces unnecessary includes and enforces clean header dependencies
- **Cppcheck** - Detects errors, undefined behavior, style issues, and performance problems

📖 **[Read more: Static Analysis Guide](docs/static-analysis.md)**

---

### Cross-Platform Building

Full cross-platform compatibility across Windows, Linux, and macOS with platform-specific optimizations and build instructions.

**Supported platforms:**
- Windows (MSVC 2019+, Clang)
- Linux (GCC 9+, Clang 10+)
- macOS (Apple Clang, including Apple Silicon)

📖 **[Read more: Cross-Platform Building Guide](docs/cross-platform-building.md)**

---

## Documentation

### Core Documentation

- **[Build Configuration](docs/build-configuration.md)** - Build types, C++ standards, optimization options
- **[Logging System](docs/logging-system.md)** - Flexible logging with three backend options
- **[Third-Party Dependencies](docs/third-party-dependencies.md)** - Dependency management and integration
- **[Vectorization](docs/vectorization.md)** - CPU SIMD optimization (SSE, AVX, AVX2, AVX-512)
- **[Sanitizers](docs/sanitizers.md)** - Memory debugging and analysis tools
- **[Code Coverage](docs/code-coverage.md)** - Coverage analysis and reporting
- **[Static Analysis](docs/static-analysis.md)** - IWYU and Cppcheck tools
- **[Cross-Platform Building](docs/cross-platform-building.md)** - Platform-specific build instructions
- **[Usage Examples](docs/usage-examples.md)** - Practical build configuration examples

### Additional Documentation

- **[CI/CD Pipeline](docs/CI_CD_PIPELINE.md)** - Continuous integration setup
- **[CI Quick Start](docs/CI_QUICK_START.md)** - Getting started with CI
- **[Valgrind Setup](docs/VALGRIND_SETUP.md)** - Memory debugging with Valgrind

### Legacy Documentation

The following documentation files are referenced in the codebase but may be located elsewhere:
- **LOGGING_BACKEND_USAGE_GUIDE.md** - Detailed logging usage guide
- **LOGGING_BACKEND_IMPLEMENTATION.md** - Logging implementation details
- **LOGGING_BACKEND_TEST_RESULTS.md** - Logging test results

## Getting Started

### Prerequisites

- CMake 3.16 or later
- C++17 compatible compiler:
  - Windows: MSVC 2019+ or Clang
  - Linux: GCC 9+ or Clang 10+
  - macOS: Apple Clang or Clang

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd XSigma

# Initialize submodules (if using Git submodules)
git submodule update --init --recursive

# Configure and build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Running Tests

```bash
# Enable testing during configuration
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Debug \
    -DXSIGMA_BUILD_TESTING=ON \
    -DXSIGMA_GOOGLE_TEST=ON

# Build and run tests
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Troubleshooting

### Common Issues

**Missing Third-Party Libraries**

If you see warnings about missing third-party libraries:

1. Add Git submodules:
   ```bash
   git submodule add https://github.com/fmtlib/fmt.git ThirdParty/fmt
   git submodule update --init --recursive
   ```

2. Use external libraries:
   ```bash
   cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON
   ```

3. Disable unused features:
   ```bash
   cmake -B build -S . -DXSIGMA_ENABLE_MAGIC_ENUM=OFF
   ```

**Vectorization Issues**

If vectorization fails to compile:

1. Check compiler support
2. Use lower vectorization: `-DXSIGMA_VECTORIZATION_TYPE=avx`
3. Disable vectorization: `-DXSIGMA_VECTORIZATION_TYPE=no`

**LTO Issues**

If Link-Time Optimization fails:

1. Disable LTO: `-DXSIGMA_ENABLE_LTO=OFF`
2. Check compiler version
3. Increase available memory

**Build Performance**

For faster builds:

1. Use external libraries: `-DXSIGMA_USE_EXTERNAL=ON`
2. Disable unused features
3. Use parallel compilation: `cmake --build build -j`

For more detailed troubleshooting, see the specific feature documentation.

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

## Best Practices

### Development Workflow

**For daily development:**
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DXSIGMA_BUILD_TESTING=ON
```

**For production releases:**
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DXSIGMA_ENABLE_LTO=ON
```

**For CI/CD pipelines:**
```bash
cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON -DXSIGMA_BUILD_TESTING=ON
```

### Performance Optimization

1. **Enable high-performance libraries**: TBB, mimalloc
2. **Use appropriate vectorization**: AVX2 for modern CPUs
3. **Enable LTO**: For maximum optimization in release builds

### Resource-Constrained Environments

1. **Use minimal builds**: Disable optional features
2. **Disable vectorization**: For maximum compatibility
3. **Use external libraries**: Reduce build time and repository size




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
