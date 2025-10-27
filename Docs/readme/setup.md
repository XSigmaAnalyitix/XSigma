# XSigma Setup Guide

## Overview

The `Scripts/setup.py` script provides a convenient interface for configuring and building XSigma with various options. This guide explains how to use the script and documents all available CMake flags.

## Basic Usage

### Syntax

```bash
cd Scripts
python setup.py [builder].[compiler].[options].config.build.test
```

### Components

- **builder**: `ninja`, `vs22`, `xcode`, `cmake`
- **compiler**: `clang`, `gcc`, `msvc` (compiler-specific)
- **options**: Various feature flags (see below)
- **config**: Configure the build
- **build**: Build the project
- **test**: Run tests

### Examples

```bash
# Debug build with Clang
python setup.py ninja.clang.debug.config.build

# Release build with optimizations
python setup.py ninja.clang.release.lto.avx2.config.build

# With testing enabled
python setup.py ninja.clang.debug.test.config.build.test

# MSVC with coverage
python setup.py vs22.debug.coverage.config.build
```

## CMake Flags Reference

### Build Type Flags

| Flag | CMake Variable | Description | Default |
|------|----------------|-------------|---------|
| `debug` | `CMAKE_BUILD_TYPE` | Debug build with symbols | - |
| `release` | `CMAKE_BUILD_TYPE` | Release build with optimizations | - |
| `relwithdebinfo` | `CMAKE_BUILD_TYPE` | Release with debug symbols | - |
| `minsizerel` | `CMAKE_BUILD_TYPE` | Minimal size release | - |

### Compiler Flags

| Flag | CMake Variable | Description |
|------|----------------|-------------|
| `clang` | `CMAKE_CXX_COMPILER` | Use Clang compiler |
| `gcc` | `CMAKE_CXX_COMPILER` | Use GCC compiler |
| `msvc` | `CMAKE_CXX_COMPILER` | Use MSVC compiler |

### Optimization Flags

| Flag | CMake Variable | Description | Default |
|------|----------------|-------------|---------|
| `lto` | `XSIGMA_ENABLE_LTO` | Link-Time Optimization | ON |
| `avx2` | `XSIGMA_VECTORIZATION_TYPE` | AVX2 vectorization | - |
| `avx` | `XSIGMA_VECTORIZATION_TYPE` | AVX vectorization | - |
| `avx512` | `XSIGMA_VECTORIZATION_TYPE` | AVX-512 vectorization | - |
| `sse` | `XSIGMA_VECTORIZATION_TYPE` | SSE vectorization | - |

### Feature Flags

| Flag | CMake Variable | Description | Default |
|------|----------------|-------------|---------|
| `cuda` | `XSIGMA_ENABLE_CUDA` | GPU acceleration with CUDA | OFF |
| `tbb` | `XSIGMA_ENABLE_TBB` | Intel Threading Building Blocks | OFF |
| `mkl` | `XSIGMA_ENABLE_MKL` | Intel Math Kernel Library | OFF |
| `numa` | `XSIGMA_ENABLE_NUMA` | NUMA support | OFF |
| `memkind` | `XSIGMA_ENABLE_MEMKIND` | Memory kind support | OFF |
| `external` | `XSIGMA_ENABLE_EXTERNAL` | Use external system libraries | OFF |

### Testing Flags

| Flag | CMake Variable | Description | Default |
|------|----------------|-------------|---------|
| `test` | `XSIGMA_BUILD_TESTING` | Enable testing | ON |
| `gtest` | `XSIGMA_ENABLE_GTEST` | Google Test framework | ON |
| `benchmark` | `XSIGMA_ENABLE_BENCHMARK` | Benchmark library | OFF |

### Analysis Flags

| Flag | CMake Variable | Description | Default |
|------|----------------|-------------|---------|
| `coverage` | `XSIGMA_ENABLE_COVERAGE` | Code coverage analysis | OFF |
| `sanitizer` | `XSIGMA_ENABLE_SANITIZER` | Enable sanitizers | OFF |
| `sanitizer.address` | `XSIGMA_SANITIZER_TYPE` | Address Sanitizer | - |
| `sanitizer.thread` | `XSIGMA_SANITIZER_TYPE` | Thread Sanitizer | - |
| `sanitizer.undefined` | `XSIGMA_SANITIZER_TYPE` | Undefined Behavior Sanitizer | - |
| `sanitizer.memory` | `XSIGMA_SANITIZER_TYPE` | Memory Sanitizer | - |
| `valgrind` | `XSIGMA_ENABLE_VALGRIND` | Valgrind support | OFF |
| `clangtidy` | `XSIGMA_ENABLE_CLANGTIDY` | Clang-Tidy analysis | OFF |
| `iwyu` | `XSIGMA_ENABLE_IWYU` | Include-What-You-Use | OFF |
| `cppcheck` | `XSIGMA_ENABLE_CPPCHECK` | Cppcheck analysis | OFF |

### Logging Flags

| Flag | CMake Variable | Description | Default |
|------|----------------|-------------|---------|
| `loguru` | `XSIGMA_ENABLE_LOGURU` | Loguru logging backend | ON |
| `logging_backend=glog` | `XSIGMA_LOGGING_BACKEND` | Use GLOG backend | - |
| `logging_backend=native` | `XSIGMA_LOGGING_BACKEND` | Use native backend | - |

### Library Flags

| Flag | CMake Variable | Description | Default |
|------|----------------|-------------|---------|
| `magic_enum` | `XSIGMA_ENABLE_MAGICENUM` | Magic enum library | ON |
| `mimalloc` | `XSIGMA_ENABLE_MIMALLOC` | mimalloc allocator | ON |

### Caching Flags

| Flag | CMake Variable | Description | Default |
|------|----------------|-------------|---------|
| `ccache` | `XSIGMA_CACHE_TYPE` | Use ccache compiler cache | - |
| `sccache` | `XSIGMA_CACHE_TYPE` | Use sccache distributed cache | - |
| `buildcache` | `XSIGMA_CACHE_TYPE` | Use buildcache | - |
| `none` | `XSIGMA_CACHE_TYPE` | Disable caching | - |

### C++ Standard Flags

| Flag | CMake Variable | Description |
|------|----------------|-------------|
| `cxxstd=17` | `XSIGMA_CXX_STANDARD` | C++17 standard |
| `cxxstd=20` | `XSIGMA_CXX_STANDARD` | C++20 standard |
| `cxxstd=23` | `XSIGMA_CXX_STANDARD` | C++23 standard |

## Common Build Configurations

### Development Build

```bash
python setup.py ninja.clang.debug.test.config.build
```

### Release Build

```bash
python setup.py ninja.clang.release.lto.avx2.config.build
```

### With Coverage

```bash
python setup.py ninja.clang.debug.coverage.config.build
```

### With Sanitizers

```bash
python setup.py ninja.clang.debug.config.build.test --sanitizer.address
```

### Minimal Build

```bash
python setup.py ninja.clang.release.config.build
```

## Build Directory Naming

The build directory is automatically named based on the configuration:

- `build_ninja` - Ninja builder
- `build_ninja_coverage` - Ninja with coverage
- `build_vs22` - Visual Studio 2022
- `build_xcode` - Xcode

Suffixes are added for special configurations (e.g., `_coverage`, `_sanitizer`).

## Troubleshooting

### Build fails with missing dependencies

Ensure all submodules are initialized:
```bash
git submodule update --init --recursive
```

### Compiler not found

Verify the compiler is installed and in PATH:
```bash
which clang++  # Linux/macOS
where clang++  # Windows
```

### Cache not working

Verify cache tool is installed:
```bash
which ccache
which sccache
```

## Advanced Configuration

For more control, use CMake directly:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DXSIGMA_ENABLE_LTO=ON \
      -DXSIGMA_VECTORIZATION_TYPE=avx2 \
      ..
cmake --build . --parallel
```

