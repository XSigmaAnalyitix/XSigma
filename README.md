# XSigma

[![CI Status](https://github.com/XSigmaAnalyitix/XSigma/actions/workflows/ci.yml/badge.svg)](https://github.com/XSigmaAnalyitix/XSigma/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/XSigmaAnalyitix/XSigma/branch/main/graph/badge.svg)](https://codecov.io/gh/XSigmaAnalyitix/XSigma)
[![License: GPL-3.0 or Commercial](https://img.shields.io/badge/License-GPL--3.0%20or%20Commercial-blue.svg)](LICENSE)

## Project Introduction

**XSigma** is a modern, high-performance quantitative analysis library designed for both CPU and GPU computing. Built with a production-ready C++ foundation and a modern CMake build system, XSigma provides cross-platform compatibility, advanced optimization capabilities, and flexible dependency management for demanding computational workloads.

### Key Features

- **High-Performance Computing** - Optimized for CPU and GPU acceleration
- **Cross-Platform Compatibility** - Windows, Linux, and macOS support
- **Modern CMake Build System** - CMake 3.16+ with best practices and flexible configuration
- **Advanced Optimization** - Link-Time Optimization (LTO), vectorization (SSE/AVX/AVX2/AVX-512), and compiler-specific flags
- **Flexible Logging System** - Three backend options (LOGURU, GLOG, NATIVE) with configurable levels
- **Comprehensive Testing & Analysis** - Sanitizers, code coverage analysis, static analysis tools, and memory profiling
- **Production-Ready** - Thoroughly tested across multiple platforms and compilers

## Table of Contents

- [Project Introduction](#project-introduction)
- [Quick Start](#quick-start)
- [Build Optimizations](#build-optimizations)
- [Analysis Tools](#analysis-tools)
- [Third-Party Dependencies](#third-party-dependencies)
- [High-Performance Computing](#high-performance-computing)
- [Features](#features)
- [Documentation](#documentation)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

### Prerequisites

XSigma requires the following minimum versions:

- **CMake** 3.16 or later
- **C++17** compatible compiler:
  - **Windows**: MSVC 2019+ or Clang 10+
  - **Linux**: GCC 9+ or Clang 10+
  - **macOS**: Apple Clang 12+ or Clang 10+
- **Python** 3.9+ (for build scripts and testing)
- **Git** (for cloning and submodule management)

### Prerequisites: Install Build Script Dependencies

Before using the setup.py script, install the required Python dependencies:

```bash
# Clone the repository
git clone https://github.com/XSigmaAnalyitix/XSigma.git
cd XSigma

# Install Python dependencies
pip install -r requirements.txt

# Initialize Git submodules (for third-party dependencies)
git submodule update --init --recursive
```

**Note**: The setup.py script requires `colorama` for colored terminal output. Install it before running any setup.py commands.

### Understanding setup.py Flags

The `setup.py` script uses a standardized flag ordering convention to organize build configuration:

- **config** - Runs the CMake configuration phase to generate build files
- **build** - Compiles the source code and builds the project
- **test** - Runs the test suite after building

### Platform-Specific Build Instructions

#### Unix/Linux (GCC and Clang)

**Using setup.py (Recommended):**
```bash
cd Scripts

# Debug build with Clang
python setup.py config.build.ninja.clang.debug

# Release build with optimizations (LTO enabled by default)
python setup.py config.build.ninja.clang.release.avx2

# With testing enabled
python setup.py config.build.test.ninja.clang.debug
```

**Using raw CMake:**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ ..
cmake --build . --parallel $(nproc)
```

#### macOS (Clang)

**Using setup.py (Recommended):**
```bash
cd Scripts

# Debug build with Apple Clang
python setup.py config.build.ninja.clang.debug

# Release build with optimizations (LTO enabled by default)
python setup.py config.build.ninja.clang.release
```

**Using raw CMake:**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ ..
cmake --build . --parallel $(sysctl -n hw.ncpu)
```

#### Windows (MSVC and Clang)

**Using setup.py (Recommended):**
```bash
cd Scripts

# Debug build with MSVC
python setup.py config.build.vs22.debug

# Release build with Clang (LTO enabled by default)
python setup.py config.build.ninja.clang.release

# With testing enabled
python setup.py config.build.test.vs22.debug
```

**Using raw CMake:**
```bash
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --parallel %NUMBER_OF_PROCESSORS%
```

### Setup Script Documentation

For detailed information about the `setup.py` script, available CMake flags, and advanced configuration options, see:

📖 **[Setup Guide](Docs/readme/setup.md)** - Comprehensive guide to setup.py and all XSIGMA CMake flags

## Build Optimizations
### Cache Build Systems

XSigma supports multiple compiler caching systems to dramatically speed up incremental builds. Each caching solution is optimized for specific use cases:

#### BuildCache (Windows)

**BuildCache** is the recommended caching solution for Windows builds, providing incremental build caching with minimal configuration.

```bash
cd Scripts

# Use BuildCache for Windows
python setup.py config.build.ninja.clang.buildcache
```

**Use Case**: Windows developers working on local machines who want fast incremental builds with persistent caching.

#### ccache (CI/CD Pipelines - Linux/macOS)

**ccache** is a distributed compiler cache optimized for CI/CD pipelines on Linux and macOS, providing excellent performance for continuous integration environments.

```bash
cd Scripts

# Use ccache for faster incremental builds (Linux/macOS CI)
python setup.py config.build.ninja.clang.ccache
```

**Use Case**: CI/CD pipelines on Linux and macOS where multiple builds run on different machines and benefit from shared caching.

#### sccache (Alternative CI/CD Caching)

**sccache** is an alternative distributed cache with cloud storage support, providing flexibility for complex CI/CD environments with remote caching capabilities.

```bash
cd Scripts

# Use sccache for distributed caching with cloud storage
python setup.py config.build.ninja.clang.sccache
```

**Use Case**: CI/CD pipelines requiring cloud-based caching or distributed build environments across multiple machines.

#### Disabling Caching

```bash
cd Scripts

# Disable all caching (default)
python setup.py config.build.ninja.clang.none
```

📖 **[Read more: Compiler Caching Guide](Docs/readme/cache.md)** - Complete caching configuration and performance tuning

### Link-Time Optimization (LTO)

Link-Time Optimization is **enabled by default** in release builds. Use the `lto` flag to toggle it.

```bash
cd Scripts

# Release build with LTO enabled (default - no flag needed)
python setup.py config.build.ninja.clang.release

# Disable LTO if needed (add 'lto' flag to toggle OFF)
python setup.py config.build.ninja.clang.release.lto  # (LTO disabled - enabled by default in release builds)
```

**LTO Behavior**:
- **Default**: LTO is ON for release builds
- **Toggle**: Adding `lto` flag toggles it OFF (since it defaults to ON)
- **Benefits**: 10-30% performance improvement in release builds
- **Trade-off**: Increases build time (use for final releases)

### Linker Optimizations

The build system automatically detects and uses the fastest available linker for your platform and compiler combination, providing significant performance improvements during the linking phase.

#### Supported Linkers

| Platform | Compiler | Preferred Linker | Fallback |
|----------|----------|------------------|----------|
| Linux | Clang | mold | lld |
| Linux | GCC | mold | gold |
| macOS | Clang/Apple Clang | lld | system linker |
| Windows | Clang | lld-link | default |
| Windows | MSVC | default | - |

#### Installation

**Linux:**
```bash
# Install faster linkers (mold, lld)
sudo apt-get install lld mold
```

**macOS:**
```bash
# Install lld (optional)
brew install llvm
```

**Windows:**
Linker optimization is handled automatically by the build system. No additional installation required.

## Analysis Tools

### Runtime Analysis

XSigma includes comprehensive runtime analysis tools for detecting memory errors, data races, and undefined behavior:

#### Sanitizers

Enable memory debugging and analysis with modern sanitizers:

```bash
cd Scripts

# Address Sanitizer (memory errors, buffer overflows)
python setup.py config.build.test.ninja.clang.debug --sanitizer.address

# Thread Sanitizer (data race detection)
python setup.py config.build.test.ninja.clang.debug --sanitizer.thread

# Undefined Behavior Sanitizer
python setup.py config.build.test.ninja.clang.debug --sanitizer.undefined

# Memory Sanitizer (uninitialized memory reads)
python setup.py config.build.test.ninja.clang.debug --sanitizer.memory

# Leak Sanitizer (memory leaks)
python setup.py config.build.test.ninja.clang.debug --sanitizer.leak
```

📖 **[Sanitizer Setup Guide](Docs/readme/sanitizer.md)** - Complete sanitizer configuration and usage

#### Valgrind Memory Profiling

For detailed memory leak detection and profiling:

```bash
cd Scripts

# Enable Valgrind support
python setup.py config.build.test.ninja.clang.debug.valgrind
```

📖 **[Valgrind Setup Guide](Docs/readme/valgrind.md)** - Complete Valgrind configuration and usage

### Static Analysis

#### Clang-Tidy

Perform static code analysis and automatic fixes:

```bash
cd Scripts

# Run clang-tidy checks
python setup.py config.build.ninja.clang.clangtidy

# Run with automatic fixes
python setup.py config.build.ninja.clang.clangtidy.fix
```

#### Cppcheck

Comprehensive static analysis for additional code quality checks:

```bash
cd Scripts

# Run cppcheck analysis
python setup.py config.build.ninja.clang.cppcheck
```

#### Include-What-You-Use (IWYU)

Optimize header dependencies and reduce unnecessary includes:

```bash
cd Scripts

# Run IWYU analysis
python setup.py config.build.ninja.clang.iwyu
```

📖 **[Static Analysis Guide](Docs/readme/static-analysis.md)** - Complete configuration and usage

### Linting System

XSigma uses a comprehensive linting framework to maintain code quality and consistency across the codebase.

#### Running Linters

```bash
cd Tools/linter

# Run all linters
python -m lintrunner

# Run specific linter
python -m lintrunner --only=clangtidy

# Run with automatic fixes
python -m lintrunner --fix
```

#### Linter Configuration

The linting system is configured through:
- **`.lintrunner.toml`** - Main linter configuration file
- **`Tools/linter/config/xsigma_linter_config.yaml`** - XSigma-specific paths and settings
- **`Tools/linter/adapters/`** - Individual linter adapter implementations

📖 **[Linter Documentation](Docs/readme/linter.md)** - Complete linter guide with configuration details and adapter documentation

### Code Coverage

Generate comprehensive code coverage reports to measure test effectiveness and identify untested code paths. XSigma supports multiple compilers (Clang, GCC, MSVC) with unified reporting.

#### Quick Start

**Generate coverage with Clang:**
```bash
cd Scripts
python setup.py config.build.test.ninja.clang.debug.coverage
# View HTML report: ../build_ninja_coverage/coverage_report/html/index.html
```

**Generate coverage with MSVC:**
```bash
cd Scripts
python setup.py config.build.test.vs22.debug.coverage
# View HTML report: ../build_vs22_coverage/coverage_report/html/index.html
```

**Generate coverage with GCC:**
```bash
cd Scripts
python setup.py config.build.test.ninja.gcc.debug.coverage
# View HTML report: ../build_ninja_coverage/coverage_report/html/index.html
```

#### Coverage Tools

- **Clang/GCC**: Uses gcov/lcov for coverage data collection and reporting
- **MSVC**: Uses OpenCppCoverage for native Windows coverage analysis
- **Unified Runner**: `Tools/coverage/run_coverage.py` provides consistent interface across all compilers

#### Report Formats

- **HTML** - Interactive visual reports with source code highlighting
- **JSON** - Machine-readable format for CI/CD integration
- **LCOV** - Standard coverage format compatible with Codecov

📖 **[Code Coverage Guide](Docs/readme/code-coverage.md)** - Comprehensive multi-compiler coverage documentation

## Third-Party Dependencies

XSigma uses carefully selected third-party libraries to provide robust functionality while maintaining minimal external dependencies. All dependencies are managed through Git submodules or system package managers.

### Core Dependencies (Always Included)

| Library | Purpose | Version |
|---------|---------|---------|
| **fmt** | String formatting and output | Latest |
| **cpuinfo** | CPU feature detection | Latest |

### Optional Dependencies

| Library | Purpose | CMake Flag | Default |
|---------|---------|-----------|---------|
| **Google Test** | Unit testing framework | `XSIGMA_ENABLE_GTEST` | ON |
| **Benchmark** | Performance benchmarking | `XSIGMA_ENABLE_BENCHMARK` | OFF |
| **Loguru** | Advanced logging backend | `XSIGMA_ENABLE_LOGURU` | ON |
| **magic_enum** | Enum reflection utilities | `XSIGMA_ENABLE_MAGICENUM` | ON |
| **mimalloc** | High-performance memory allocator | `XSIGMA_ENABLE_MIMALLOC` | ON |
| **TBB** | Intel Threading Building Blocks | `XSIGMA_ENABLE_TBB` | OFF |
| **MKL** | Intel Math Kernel Library | `XSIGMA_ENABLE_MKL` | OFF |
| **CUDA** | GPU acceleration | `XSIGMA_ENABLE_CUDA` | OFF |

### Enabling/Disabling Dependencies

```bash
cd Scripts

# Enable CUDA support
python setup.py config.build.ninja.clang.cuda

# Enable TBB for parallel computing
python setup.py config.build.ninja.clang.tbb

# Use external system libraries instead of submodules
python setup.py config.build.ninja.clang.external

# Disable optional features for minimal builds
python setup.py config.build.ninja.clang.test.magic_enum
```

### Logging System

XSigma provides a flexible logging system with multiple backend options to suit different use cases and performance requirements.

#### Logging Backends

- **LOGURU** (default) - Full-featured logging with scopes, callbacks, and advanced formatting
- **GLOG** - Google's production-grade logging library with minimal overhead
- **NATIVE** - Minimal native implementation with zero external dependencies

#### Configuring Logging

```bash
cd Scripts

# Use GLOG backend
python setup.py config.build.test.ninja.clang --logging.GLOG

# Use native logging (no dependencies)
python setup.py config.build.test.ninja.clang --logging.NATIVE
```

📖 **[Logging System Guide](Docs/readme/logging.md)** - Complete logging documentation with configuration examples

📖 **[Third-Party Dependencies Guide](Docs/readme/third-party-dependencies.md)** - Dependency management and integration

## High-Performance Computing

XSigma provides comprehensive support for high-performance computing through CPU vectorization (SIMD), GPU acceleration (CUDA/HIP), and multithreading(TBB/native).

### Quick Start

```bash
cd Scripts

# CPU vectorization (AVX2 - recommended)
python setup.py config.build.ninja.clang.release.avx2

# GPU acceleration (CUDA)
python setup.py config.build.ninja.clang.release.cuda

# Multithreading (TBB)
python setup.py config.build.ninja.clang.release.tbb

# Combined: CUDA + AVX2 + LTO (LTO enabled by default)
python setup.py config.build.ninja.clang.release.cuda.avx2
```

### Features

- **Vectorization (SIMD)**: SSE, AVX, AVX2, AVX-512 instruction sets (2-8x speedup)
- **GPU Acceleration**: NVIDIA CUDA and AMD HIP support (10-100x speedup)
- **Multithreading**: Intel TBB and native C++17/20 threading
- **Flexible Combinations**: Mix and match features for optimal performance

### Supported Platforms

- **CPU Vectorization**: All platforms (Windows, Linux, macOS)
- **CUDA**: Linux, Windows, macOS (with NVIDIA GPU)
- **HIP**: Linux (primary), Windows (experimental)
- **Multithreading**: All platforms

📖 **[Read more: High-Performance Computing Guide](Docs/readme/high-performance-computing.md)** - Comprehensive guide to SIMD, GPU acceleration, and multithreading

---

## Features

### Build Configuration

Configure your build with multiple options including build types (Debug, Release, RelWithDebInfo, MinSizeRel), C++ standard selection (C++17/20/23), and optimization settings.

**Key capabilities:**
- Multiple build types for different use cases
- Link-Time Optimization (LTO) for maximum performance
- Custom compiler flags and optimization levels

📖 **[Read more: Build Configuration Guide](Docs/readme/build/build-configuration.md)**

---

### Cross-Platform Building

Full cross-platform compatibility across Windows, Linux, and macOS with platform-specific optimizations and build instructions.

**Supported platforms:**
- Windows (MSVC 2019+, Clang)
- Linux (GCC 9+, Clang 10+)
- macOS (Apple Clang, including Apple Silicon)

📖 **[Read more: Cross-Platform Building Guide](Docs/readme/cross-platform-building.md)**

---

## Documentation

### Core Documentation

- **[Setup Guide](Docs/readme/setup.md)** - Detailed setup.py script and CMake flags reference
- **[Build Configuration](Docs/readme/build/build-configuration.md)** - Build types, C++ standards, optimization options
- **[High-Performance Computing](Docs/readme/high-performance-computing.md)** - SIMD, GPU acceleration, and multithreading
- **[Logging System](Docs/readme/logging.md)** - Flexible logging with three backend options
- **[Third-Party Dependencies](Docs/readme/third-party-dependencies.md)** - Dependency management and integration
- **[Vectorization](Docs/readme/vectorization.md)** - CPU SIMD optimization (SSE, AVX, AVX2, AVX-512)
- **[Sanitizers](Docs/readme/sanitizer.md)** - Memory debugging and analysis tools
- **[Code Coverage](Docs/readme/code-coverage.md)** - Multi-compiler coverage analysis and reporting
- **[Static Analysis](Docs/readme/static-analysis.md)** - IWYU and Cppcheck tools
- **[Compiler Caching](Docs/readme/cache.md)** - Compiler cache types, installation, and configuration
- **[Cross-Platform Building](Docs/readme/cross-platform-building.md)** - Platform-specific build instructions
- **[Linting System](Docs/readme/linter.md)** - Comprehensive linter documentation and configuration
- **[Usage Examples](Docs/readme/usage-examples.md)** - Practical build configuration examples

### Additional Documentation

<!-- [CI/CD Pipeline](Docs/ci/CI_CD_PIPELINE.md) - Continuous integration setup (File not found: Docs/ci/ directory does not exist) -->
<!-- [CI Quick Start](Docs/ci/CI_QUICK_START.md) - Getting started with CI (File not found: Docs/ci/ directory does not exist) -->
- **[Valgrind Setup](Docs/readme/valgrind.md)** - Memory debugging with Valgrind

## Running Tests

```bash
# Enable testing during configuration
cd Scripts
python setup.py config.build.test.ninja.clang.debug.gtest
```

## Troubleshooting

### Common Issues

**Missing Third-Party Libraries**

If you see warnings about missing third-party libraries:

1. **Initialize existing submodules** (recommended):
   ```bash
   # Sync and initialize all submodules from the repository
   git submodule sync --recursive
   git submodule update --init --recursive
   ```

   **Note**: Do NOT use `git submodule add` unless you're adding a new dependency. The repository already has submodules configured; this command just initializes them locally.

2. Use external system libraries:
   ```bash
   cd Scripts
   python setup.py config.build.ninja.clang.external
   ```

3. Disable unused features:
   ```bash
   cd Scripts
   python setup.py config.build.ninja.clang.magic_enum
   ```

**Vectorization Issues**

If vectorization fails to compile:

1. Check compiler support
2. Use lower vectorization: add `avx` flag to setup.py command
3. Disable vectorization: omit vectorization flags from setup.py command

**LTO Issues**

If Link-Time Optimization fails:

1. **Disable LTO**: Add `lto` flag to setup.py command (toggles default ON to OFF)
   ```bash
   cd Scripts
   python setup.py config.build.ninja.clang.release.lto  # (LTO enabled by default in release builds)
   ```
2. Check compiler version (GCC 5+, Clang 3.5+, MSVC 2015+)
3. Increase available memory (LTO is memory-intensive)

**Build Performance**

For faster builds:

1. Use external libraries: add `external` flag to setup.py command
2. Disable unused features
3. Use parallel compilation: (automatically handled by setup.py)
4. Enable build optimizations (see [Build Optimizations](#build-optimizations) section)

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
cd Scripts
python setup.py config.build.test.ninja.clang.debug
```

**For production releases:**
```bash
cd Scripts
python setup.py config.build.ninja.clang.release  # (LTO enabled by default in release builds)
```

**For CI/CD pipelines:**
```bash
cd Scripts
python setup.py config.build.test.ninja.clang.external
```

### Performance Optimization

1. **Enable high-performance libraries**: TBB, mimalloc
2. **Use appropriate vectorization**: AVX2 for modern CPUs
3. **Enable LTO**: For maximum optimization in release builds

### Resource-Constrained Environments

1. **Use minimal builds**: Disable optional features
2. **Disable vectorization**: For maximum compatibility
3. **Use external libraries**: Reduce build time and repository size

---

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
