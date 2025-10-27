# Third-Party Dependencies

XSigma uses a **conditional compilation pattern** where each library is controlled by `XSIGMA_ENABLE_XXX` options. This allows you to customize your build by including only the dependencies you need.

## Table of Contents

- [Dependency Categories](#dependency-categories)
- [Dependency Management Pattern](#dependency-management-pattern)
- [Setting Up Dependencies](#setting-up-dependencies)
- [Build Configuration](#build-configuration)
- [Using Dependencies in Code](#using-dependencies-in-code)
- [CMake Target Usage](#cmake-target-usage)

## Dependency Categories

### Mandatory Core Libraries (Always Included)

These libraries are always included in the build:

| Library | Description | Target Alias |
|---------|-------------|--------------|
| fmt | Modern C++ formatting | `XSigma::fmt` |
| cpuinfo | CPU feature detection | `XSigma::cpuinfo` |

### Optional Libraries (Enabled by Default)

These libraries are included by default but can be disabled:

| Library | Option | Description | Target Alias |
|---------|--------|-------------|--------------|
| magic_enum | `XSIGMA_ENABLE_MAGICENUM=ON` | Enum reflection | `XSigma::magic_enum` |
| loguru | `XSIGMA_ENABLE_LOGURU=ON` | Lightweight logging | `XSigma::loguru` |

### Optional Libraries (Disabled by Default)

These libraries must be explicitly enabled:

| Library | Option | Description | Target Alias |
|---------|--------|-------------|--------------|
| mimalloc | `XSIGMA_ENABLE_MIMALLOC=OFF` | High-performance allocator | `XSigma::mimalloc` |
| Google Test | `XSIGMA_ENABLE_GTEST=OFF` | Testing framework | `XSigma::gtest` |
| Benchmark | `XSIGMA_ENABLE_BENCHMARK=OFF` | Microbenchmarking | `XSigma::benchmark` |

## Dependency Management Pattern

### Mandatory Libraries (fmt, cpuinfo)

For mandatory libraries:
- Always included in the build
- Always create `XSigma::xxx` target aliases
- Always add `XSIGMA_HAS_XXX` compile definitions
- Always linked to Core target

### Optional Libraries (controlled by `XSIGMA_ENABLE_XXX`)

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

## Setting Up Dependencies

### Option 1: Git Submodules (Recommended)

Use Git submodules to include dependencies directly in your repository:

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

# Initialize all submodules
git submodule update --init --recursive
```

### Option 2: External Libraries

Use system-installed libraries instead of bundled submodules:

```bash
# Use system-installed libraries
cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON
```

**Benefits:**
- Faster build times (libraries already compiled)
- Smaller repository size
- Shared libraries across projects

**Requirements:**
- Libraries must be installed on the system
- May require additional system packages
- Consult upstream documentation if `find_package()` fails

## Build Configuration

### Enabling/Disabling Optional Libraries

```bash
# Enable high-performance allocator
cmake -B build -S . -DXSIGMA_ENABLE_MIMALLOC=ON

# Disable magic_enum
cmake -B build -S . -DXSIGMA_ENABLE_MAGICENUM=OFF

# Enable testing and benchmarking
cmake -B build -S . \
    -DXSIGMA_ENABLE_GTEST=ON \
    -DXSIGMA_ENABLE_BENCHMARK=ON
```

### Build Configuration for Third-Party Libraries

Third-party targets are configured to:
- Suppress warnings (using `-w` or `/w`)
- Use the same C++ standard as the main project
- Avoid altering the main project's compiler/linker settings
- Provide consistent target aliases with the `XSigma::` prefix

## Using Dependencies in Code

### Conditional Compilation

Use compile definitions to conditionally use libraries:

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

## CMake Target Usage

### Linking with XSigma Libraries

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

### Build Performance Issues

For faster builds:

1. **Use external libraries**:
   ```bash
   cmake -B build -S . -DXSIGMA_USE_EXTERNAL=ON
   ```

2. **Disable unused features**:
   ```bash
   cmake -B build -S . \
       -DXSIGMA_ENABLE_BENCHMARK=OFF \
       -DXSIGMA_GOOGLE_TEST=OFF
   ```

## Notes

- When `XSIGMA_USE_EXTERNAL=ON`, system-installed libraries are preferred over bundled submodules
- Some libraries may require additional system packages; consult their upstream documentation if `find_package()` fails
- All third-party libraries use the `XSigma::` namespace prefix for consistency

## Related Documentation

- [Build Configuration](build-configuration.md) - Build system configuration
- [Logging System](logging-system.md) - Logging backend selection
- [Usage Examples](usage-examples.md) - Example build configurations
