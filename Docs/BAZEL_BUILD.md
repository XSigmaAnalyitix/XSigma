# XSigma Bazel Build System

This document describes how to build XSigma using Bazel, as an alternative to CMake.

## Prerequisites

- Bazel 6.0 or later (install from https://bazel.build/)
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Platform-specific requirements (see below)

## Quick Start

### Basic Build

```bash
# Build all libraries
bazel build //...

# Build specific library
bazel build //Library/Core:Core
bazel build //Library/Security:Security
```

### Build with Optimizations

```bash
# Release build with optimizations
bazel build --config=release //...

# Release with debug info
bazel build --config=relwithdebinfo //...

# Debug build
bazel build --config=debug //...
```

### Build with Vectorization

```bash
# AVX2 (default)
bazel build --config=avx2 //...

# AVX512
bazel build --config=avx512 //...

# SSE
bazel build --config=sse //...

# No vectorization
bazel build //...
```

## Feature Configuration

### Enable Optional Features

Features are controlled via `--config` flags or `--define` flags:

```bash
# Enable mimalloc allocator
bazel build --config=mimalloc //...

# Enable magic_enum
bazel build --config=magic_enum //...

# Enable Kineto profiling
bazel build --config=kineto //...

# Enable TBB
bazel build --config=tbb //...

# Enable OpenMP
bazel build --config=openmp //...
```

### Logging Backend Selection

Choose one of the following logging backends:

```bash
# Google glog (requires glog to be added to WORKSPACE)
bazel build --config=logging_glog //...

# Loguru (requires loguru to be added to WORKSPACE)
bazel build --config=logging_loguru //...

# Native fmt-based logging (default)
bazel build --config=logging_native //...
```

### GPU Support

```bash
# CUDA support
bazel build --config=cuda //...

# HIP support (AMD ROCm)
bazel build --config=hip //...

# GPU allocation strategy
bazel build --config=gpu_alloc_pool_async //...  # Default
bazel build --config=gpu_alloc_async //...
bazel build --config=gpu_alloc_sync //...
```

### Algorithm Options

```bash
# Enable LU pivoting
bazel build --config=lu_pivoting //...

# Enable Sobol 1111 dimensions
bazel build --config=sobol_1111 //...
```

## Link Time Optimization (LTO)

```bash
# Enable LTO
bazel build --config=lto --config=release //...
```

## Sanitizers

```bash
# Address Sanitizer
bazel build --config=asan //...

# Thread Sanitizer
bazel build --config=tsan //...

# Undefined Behavior Sanitizer
bazel build --config=ubsan //...

# Memory Sanitizer
bazel build --config=msan //...
```

## Testing

```bash
# Build and run all tests
bazel test //...

# Run specific test
bazel test //Library/Core/Testing/Cxx:core_tests

# Run tests with Google Test
bazel test --config=gtest //...

# Run benchmarks
bazel test --config=benchmark //...
```

## Platform-Specific Builds

### macOS

```bash
bazel build --config=macos //...
```

### Linux

```bash
bazel build --config=linux //...
```

### Windows (MSVC)

```bash
bazel build --config=windows //...
```

## Combining Configurations

You can combine multiple configurations:

```bash
# Release build with AVX2, mimalloc, and magic_enum
bazel build --config=release --config=avx2 --config=mimalloc --config=magic_enum //...

# Debug build with ASan and native logging
bazel build --config=debug --config=asan --config=logging_native //...

# Release with LTO, TBB, and CUDA
bazel build --config=release --config=lto --config=tbb --config=cuda //...
```

## Custom Configuration

Create a `.bazelrc.user` file in the project root to customize your build settings:

```bash
# .bazelrc.user example
build --config=release
build --config=avx2
build --config=mimalloc
build --config=magic_enum
build --config=logging_native
```

Then simply run:
```bash
bazel build //...
```

## Shared vs Static Libraries

By default, XSigma builds static libraries. To build shared libraries:

```bash
bazel build --define=xsigma_build_shared_libs=true //...
```

## CMake to Bazel Feature Mapping

| CMake Option | Bazel Equivalent |
|-------------|------------------|
| `-DCMAKE_BUILD_TYPE=Release` | `--config=release` |
| `-DCMAKE_BUILD_TYPE=Debug` | `--config=debug` |
| `-DXSIGMA_ENABLE_LTO=ON` | `--config=lto` |
| `-DXSIGMA_VECTORIZATION_TYPE=avx2` | `--config=avx2` |
| `-DXSIGMA_ENABLE_MIMALLOC=ON` | `--config=mimalloc` |
| `-DXSIGMA_ENABLE_MAGICENUM=ON` | `--config=magic_enum` |
| `-DXSIGMA_ENABLE_KINETO=ON` | `--config=kineto` |
| `-DXSIGMA_ENABLE_TBB=ON` | `--config=tbb` |
| `-DXSIGMA_ENABLE_CUDA=ON` | `--config=cuda` |
| `-DXSIGMA_ENABLE_GTEST=ON` | `--config=gtest` |
| `-DXSIGMA_ENABLE_BENCHMARK=ON` | `--config=benchmark` |
| `-DXSIGMA_LU_PIVOTING=ON` | `--config=lu_pivoting` |
| `-DXSIGMA_SOBOL_1111=ON` | `--config=sobol_1111` |
| `-DXSIGMA_GPU_ALLOC=POOL_ASYNC` | `--config=gpu_alloc_pool_async` |

## Build Output

Build artifacts are located in:
- `bazel-bin/` - Compiled binaries and libraries
- `bazel-out/` - Build outputs
- `bazel-testlogs/` - Test logs

## Cleaning

```bash
# Clean build artifacts
bazel clean

# Clean everything including external dependencies
bazel clean --expunge
```

## Troubleshooting

### Missing Dependencies

If a third-party dependency is missing, check:
1. `WORKSPACE.bazel` - Ensure the dependency is declared
2. `third_party/*.BUILD` - Ensure the BUILD file exists
3. Network connectivity - Bazel downloads dependencies on first build

### Configuration Conflicts

Some configurations are mutually exclusive:
- Only one logging backend can be active at a time
- TBB and STDThread are mutually exclusive (TBB takes precedence)
- CUDA and HIP cannot both be enabled

### Platform-Specific Issues

**macOS:**
- Ensure Xcode Command Line Tools are installed: `xcode-select --install`

**Linux:**
- Install required development packages: `sudo apt-get install build-essential`

**Windows:**
- Ensure Visual Studio 2017 or later is installed
- Run builds from "Developer Command Prompt for VS"

## Advanced Usage

### Query Build Graph

```bash
# Show all targets
bazel query //...

# Show dependencies of Core library
bazel query 'deps(//Library/Core:Core)'

# Show reverse dependencies
bazel query 'rdeps(//..., //Library/Core:Core)'
```

### Build Analysis

```bash
# Profile build performance
bazel build --profile=profile.json //...

# Analyze profile
bazel analyze-profile profile.json
```

### Remote Caching

Configure remote caching for faster builds:

```bash
# .bazelrc.user
build --remote_cache=grpc://your-cache-server:9092
```

## Comparison with CMake Build

### Advantages of Bazel:
1. **Incremental builds**: Only rebuilds what changed
2. **Hermetic builds**: More reproducible across environments
3. **Remote caching**: Share build artifacts across team
4. **Parallel execution**: Better parallelization
5. **Cross-platform**: Unified build system

### When to use CMake:
1. Integration with CMake-based projects
2. IDE support (CLion, Visual Studio)
3. Existing CMake workflows
4. Package management with vcpkg/Conan

Both build systems are fully supported and produce equivalent binaries.

## Contributing

When adding new source files:
1. Update the appropriate `BUILD.bazel` file
2. Add any new dependencies to `WORKSPACE.bazel`
3. Create BUILD files for new third-party dependencies in `third_party/`
4. Update this documentation if adding new configuration options

## Support

For issues with the Bazel build:
- Check existing issues on GitHub
- Refer to CMake build as reference implementation
- Consult Bazel documentation: https://bazel.build/
