# XSigma SMP_NEW - Integration Guide

## Overview

This guide explains how to integrate the `smp_new` module into XSigma and how to use it alongside the existing `smp/` module.

## Directory Structure

```
XSigma/Library/Core/
├── smp/                    # Existing threading module (unchanged)
│   ├── Advanced/
│   ├── STDThread/
│   ├── TBB/
│   └── ...
└── smp_new/                # New PyTorch-compatible module
    ├── core/               # Core thread pool
    ├── parallel/           # Parallel APIs
    ├── native/             # Native backend
    ├── test/               # Unit tests
    ├── benchmark/          # Performance benchmarks
    ├── CMakeLists.txt
    ├── README.md
    ├── USAGE_GUIDE.md
    └── INTEGRATION_GUIDE.md
```

## Building the Module

### 1. Enable in CMake

Add to your CMakeLists.txt:

```cmake
# Enable smp_new module
set(XSIGMA_BUILD_SMP_NEW ON)

# Add the module
add_subdirectory(XSigma/Library/Core/smp_new)
```

### 2. Build

```bash
mkdir build && cd build
cmake -DXSIGMA_BUILD_SMP_NEW=ON ..
make
```

### 3. Run Tests

```bash
ctest -R smp_new
```

## Integration with Existing Code

### Coexistence with smp/

The `smp_new` module is completely independent and can coexist with the existing `smp/` module:

```cpp
// Both can be used in the same codebase
#include <xsigma/smp/tools.h>
#include <xsigma/smp_new/parallel/parallel_api.h>

// Use existing smp module
xsigma::tools::For(0, N, grain_size, functor);

// Use new smp_new module
xsigma::smp_new::parallel::parallel_for(0, N, grain_size, lambda);
```

### Migration Path

To migrate from `smp/` to `smp_new/`:

#### Step 1: Update Includes

```cpp
// Old
#include <xsigma/smp/tools.h>

// New
#include <xsigma/smp_new/parallel/parallel_api.h>
```

#### Step 2: Update API Calls

```cpp
// Old: Functor-based
struct MyFunctor {
    void operator()(int first, int last) { /* ... */ }
};
MyFunctor f;
xsigma::tools::For(0, N, grain_size, f);

// New: Lambda-based
xsigma::smp_new::parallel::parallel_for(0, N, grain_size, 
    [&](int64_t b, int64_t e) { /* ... */ });
```

#### Step 3: Update Reduction Code

```cpp
// Old: Manual thread-local storage
struct ReduceFunctor {
    void Initialize() { /* ... */ }
    void operator()(int first, int last) { /* ... */ }
    void Reduce() { /* ... */ }
};

// New: Built-in parallel_reduce
auto result = xsigma::smp_new::parallel::parallel_reduce(
    0, N, grain_size, identity,
    [&](int64_t b, int64_t e, T ident) { /* ... */ },
    [](T a, T b) { /* ... */ }
);
```

## Linking

### Static Linking

```cmake
target_link_libraries(my_target PRIVATE xsigma_smp_new)
```

### Dynamic Linking

```cmake
target_link_libraries(my_target PRIVATE xsigma_smp_new_shared)
```

## Dependencies

The `smp_new` module depends on:

- `xsigma_common` - Common utilities and macros
- `xsigma_memory` - Memory management and NUMA support
- `xsigma_util` - Utility functions
- `Threads::Threads` - Standard C++ threading library

## Configuration

### CMake Options

```cmake
# Enable/disable smp_new module
set(XSIGMA_BUILD_SMP_NEW ON)

# Enable/disable tests
set(XSIGMA_BUILD_TESTS ON)

# Enable/disable benchmarks
set(XSIGMA_BUILD_BENCHMARKS ON)

# Enable NUMA support
set(XSIGMA_ENABLE_NUMA ON)
```

### Runtime Configuration

```cpp
#include <xsigma/smp_new/parallel/parallel_api.h>

using namespace xsigma::smp_new::parallel;

// Configure thread counts
set_num_intraop_threads(4);
set_num_interop_threads(8);

// Query configuration
auto intraop = get_num_intraop_threads();
auto interop = get_num_interop_threads();
```

## Performance Considerations

### Thread Pool Overhead

- **Lazy Initialization**: Thread pools are created on first use
- **Master-Worker Pattern**: Reduces context switching overhead
- **Separate Pools**: Prevents thread pool exhaustion

### Optimization Tips

1. **Grain Size**: Choose based on work complexity
   - Small work: larger grain size
   - Large work: smaller grain size

2. **Thread Count**: Match system capabilities
   - Typically: number of CPU cores
   - NUMA systems: cores per NUMA node

3. **Data Locality**: Minimize cache misses
   - Use NUMA binding for large datasets
   - Align data structures appropriately

## Troubleshooting

### Issue: Compilation Errors

**Solution**: Ensure all dependencies are built:

```bash
cmake -DXSIGMA_BUILD_SMP_NEW=ON \
      -DXSIGMA_BUILD_COMMON=ON \
      -DXSIGMA_BUILD_MEMORY=ON \
      -DXSIGMA_BUILD_UTIL=ON ..
```

### Issue: Linking Errors

**Solution**: Check library order in CMakeLists.txt:

```cmake
target_link_libraries(my_target PRIVATE
    xsigma_smp_new
    xsigma_memory
    xsigma_common
    Threads::Threads
)
```

### Issue: Runtime Errors

**Solution**: Enable debug logging:

```cpp
#include <xsigma/smp_new/native/parallel_native.h>

auto info = xsigma::smp_new::native::GetNativeBackendInfo();
std::cout << info << std::endl;
```

## Testing Integration

### Unit Tests

```bash
ctest -R smp_new_tests
```

### Benchmarks

```bash
./benchmark_smp_new
```

### Custom Tests

```cpp
#include <xsigma/smp_new/parallel/parallel_api.h>

int main() {
    using namespace xsigma::smp_new::parallel;
    
    // Your test code
    parallel_for(0, 1000, 100, [](int64_t b, int64_t e) {
        // Process range
    });
    
    return 0;
}
```

## Documentation

- **README.md** - Module overview and features
- **USAGE_GUIDE.md** - Practical usage examples
- **INTEGRATION_GUIDE.md** - This file
- **API Headers** - Detailed API documentation in source files

## Support

For issues or questions:

1. Check the documentation
2. Review test cases for examples
3. Enable debug logging
4. Contact the development team

## Version Information

- **Module Version**: 1.0
- **XSigma Compatibility**: 1.0+
- **C++ Standard**: C++17 or later
- **Platforms**: Linux, Windows, macOS

## License

XSigma is dual-licensed under GPL-3.0-or-later (open-source) and a commercial license.

