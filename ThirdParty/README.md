# Third-Party Dependencies

This directory contains third-party libraries used by XSigma. All libraries follow a strict conditional compilation pattern controlled by `XSIGMA_ENABLE_XXX` CMake options. Libraries can be included as Git submodules or found externally on the system.

## Conditional Compilation Pattern

Each third-party library is controlled by a corresponding `XSIGMA_ENABLE_XXX` option:

- **When `XSIGMA_ENABLE_XXX=ON`**: Include library, create `XSigma::xxx` target, add `XSIGMA_HAS_XXX` definition
- **When `XSIGMA_ENABLE_XXX=OFF`**: Skip library entirely, no targets or definitions created

This ensures only explicitly enabled dependencies are included in the build.

## Adding Git Submodules

To add the required third-party libraries as Git submodules, run the following commands from the project root:

### Core Libraries (Enabled by Default)

```bash
# fmt - Modern C++ formatting library (XSIGMA_ENABLE_FMT=ON)
git submodule add https://github.com/fmtlib/fmt.git ThirdParty/fmt

# cpuinfo - CPU feature detection library (XSIGMA_ENABLE_CPUINFO=ON)
git submodule add https://github.com/pytorch/cpuinfo.git ThirdParty/cpuinfo

# magic_enum - Static reflection for enums in C++ (XSIGMA_ENABLE_MAGICENUM=ON)
git submodule add https://github.com/Neargye/magic_enum.git ThirdParty/magic_enum

# loguru - Lightweight C++ logging library (XSIGMA_ENABLE_LOGURU=ON)
git submodule add https://github.com/emilk/loguru.git ThirdParty/loguru
```

### Optional Libraries (Disabled by Default)

```bash
# TBB - Intel Threading Building Blocks (when XSIGMA_ENABLE_TBB=ON)
git submodule add https://github.com/oneapi-src/oneTBB.git ThirdParty/tbb

# mimalloc - Microsoft's high-performance memory allocator (when XSIGMA_ENABLE_MIMALLOC=ON)
git submodule add https://github.com/microsoft/mimalloc.git ThirdParty/mimalloc

# Google Benchmark - Microbenchmarking library (when XSIGMA_ENABLE_BENCHMARK=ON)
git submodule add https://github.com/google/benchmark.git ThirdParty/benchmark

# Google Test - C++ testing framework (when XSIGMA_ENABLE_GTEST=ON)
git submodule add https://github.com/google/googletest.git ThirdParty/googletest
```

### Initialize and Update Submodules

After adding submodules, initialize and update them:

```bash
git submodule update --init --recursive
```

## Using External Libraries

If you prefer to use system-installed versions of these libraries, set the CMake option:

```bash
cmake -DXSIGMA_USE_EXTERNAL=ON ...
```

This will attempt to find external versions using `find_package()` before falling back to bundled versions.

## Library Information

### fmt
- **Purpose**: Modern C++ formatting library
- **License**: MIT
- **CMake Option**: `XSIGMA_ENABLE_FMT` (default: ON)
- **CMake Target**: `XSigma::fmt`
- **Compile Definition**: `XSIGMA_HAS_FMT`
- **Usage**: String formatting, logging

### cpuinfo
- **Purpose**: CPU feature detection
- **License**: BSD 2-Clause
- **CMake Option**: `XSIGMA_ENABLE_CPUINFO` (default: ON)
- **CMake Target**: `XSigma::cpuinfo`
- **Compile Definition**: `XSIGMA_HAS_CPUINFO`
- **Usage**: Runtime CPU feature detection for vectorization

### magic_enum
- **Purpose**: Static reflection for enums
- **License**: MIT
- **CMake Option**: `XSIGMA_ENABLE_MAGICENUM` (default: ON)
- **CMake Target**: `XSigma::magic_enum`
- **Compile Definition**: `XSIGMA_HAS_MAGIC_ENUM`
- **Usage**: Enum to string conversion, reflection

### loguru
- **Purpose**: Lightweight C++ logging
- **License**: Public Domain
- **CMake Option**: `XSIGMA_ENABLE_LOGURU` (default: ON)
- **CMake Target**: `XSigma::loguru`
- **Compile Definition**: `XSIGMA_HAS_LOGURU`
- **Usage**: Application logging

### TBB (Intel Threading Building Blocks)
- **Purpose**: Parallel programming library
- **License**: Apache 2.0
- **CMake Option**: `XSIGMA_ENABLE_TBB` (default: OFF)
- **CMake Target**: `XSigma::tbb`
- **Compile Definition**: `XSIGMA_HAS_TBB`
- **Usage**: Task-based parallelism (alternative to STDThread backend)

### mimalloc
- **Purpose**: High-performance memory allocator
- **License**: MIT
- **CMake Option**: `XSIGMA_ENABLE_MIMALLOC` (default: OFF)
- **CMake Target**: `XSigma::mimalloc`
- **Compile Definition**: `XSIGMA_HAS_MIMALLOC`
- **Usage**: Drop-in replacement for malloc/free

### Google Benchmark
- **Purpose**: Microbenchmarking library
- **License**: Apache 2.0
- **CMake Option**: `XSIGMA_ENABLE_BENCHMARK` (default: OFF)
- **CMake Target**: `XSigma::benchmark`
- **Compile Definition**: `XSIGMA_HAS_BENCHMARK`
- **Usage**: Performance benchmarking

### Google Test
- **Purpose**: C++ testing framework
- **License**: BSD 3-Clause
- **CMake Option**: `XSIGMA_ENABLE_GTEST` (default: OFF)
- **CMake Targets**: `XSigma::gtest`, `XSigma::gtest_main`
- **Compile Definition**: `XSIGMA_HAS_GTEST`
- **Usage**: Unit testing

## Build Configuration

The third-party libraries are configured to:
- Suppress warnings (using `-w` or `/w`)
- Use the same C++ standard as the main project
- Not interfere with the main project's build settings
- Provide consistent target names with `XSigma::` prefix

## Notes

- All libraries are optional and can be disabled by not adding their submodules
- The build system will warn if a required library is missing
- External libraries take precedence when `XSIGMA_ENABLE_EXTERNAL=ON`
- Some libraries may have additional dependencies that need to be satisfied
