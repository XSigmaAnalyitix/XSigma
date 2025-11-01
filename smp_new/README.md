# XSigma SMP_NEW Module - PyTorch-Compatible Threading Backend

## Overview

The `smp_new` module is a new threading backend for XSigma that replicates PyTorch's CPU threading logic while using XSigma's existing infrastructure. It provides modern, efficient parallel execution APIs that match PyTorch's threading capabilities.

## Features

- **PyTorch-Compatible API**: Implements `parallel_for()`, `parallel_reduce()`, `launch()`, and `intraop_launch()`
- **Modern C++ Design**: Lambda-based APIs for easy parallel programming
- **Separate Thread Pools**: Independent intra-op and inter-op thread pools
- **Robust Exception Handling**: Exceptions are captured and propagated to the caller
- **NUMA Support**: Automatic thread binding to NUMA nodes
- **Lazy Initialization**: Thread pools are created on first use
- **Master-Worker Pattern**: Efficient task execution with reduced overhead

## Architecture

```
smp_new/
├── core/
│   ├── thread_pool.h/cxx       - Core thread pool implementation
│   └── ...
├── parallel/
│   ├── parallel_api.h/cxx      - Public parallel APIs
│   ├── parallel_api.hxx        - Template implementations
│   └── ...
├── native/
│   ├── parallel_native.h/cxx   - Native backend implementation
│   └── ...
└── test/
    ├── test_thread_pool.cxx    - Thread pool tests
    ├── test_parallel_for.cxx   - parallel_for tests
    ├── test_parallel_reduce.cxx- parallel_reduce tests
    └── test_task_execution.cxx - Task execution tests
```

## API Reference

### Parallel Iteration

```cpp
#include <xsigma/smp_new/parallel/parallel_api.h>

using namespace xsigma::smp_new::parallel;

// Parallel for loop with lambda
parallel_for(0, 1000000, 1000, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
        data[i] = compute(data[i]);
    }
});
```

### Parallel Reduction

```cpp
// Parallel sum reduction
float sum = parallel_reduce(
    0, 1000000, 1000,
    0.0f,  // identity
    [&](int64_t b, int64_t e, float ident) {
        float s = ident;
        for (int64_t i = b; i < e; ++i) {
            s += data[i];
        }
        return s;
    },
    [](float a, float b) { return a + b; }  // combine
);
```

### Task Execution

```cpp
// Inter-op task (between operations)
launch([]() {
    process_data();
});

// Intra-op task (within operation)
intraop_launch([]() {
    process_chunk();
});
```

### Thread Configuration

```cpp
// Set thread counts
set_num_intraop_threads(4);
set_num_interop_threads(8);

// Get thread counts
auto intraop_threads = get_num_intraop_threads();
auto interop_threads = get_num_interop_threads();
```

## Usage Examples

### Example 1: Simple Parallel For

```cpp
#include <xsigma/smp_new/parallel/parallel_api.h>
#include <vector>

int main() {
    std::vector<float> data(1000000);
    
    xsigma::smp_new::parallel::parallel_for(
        0, data.size(), 1000,
        [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                data[i] = std::sqrt(data[i]);
            }
        }
    );
    
    return 0;
}
```

### Example 2: Parallel Reduction

```cpp
#include <xsigma/smp_new/parallel/parallel_api.h>
#include <vector>

int main() {
    std::vector<float> data(1000000);
    // ... fill data ...
    
    float sum = xsigma::smp_new::parallel::parallel_reduce(
        0, data.size(), 1000,
        0.0f,
        [&](int64_t b, int64_t e, float ident) {
            float s = ident;
            for (int64_t i = b; i < e; ++i) {
                s += data[i];
            }
            return s;
        },
        [](float a, float b) { return a + b; }
    );
    
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
```

### Example 3: Task-Based Parallelism

```cpp
#include <xsigma/smp_new/parallel/parallel_api.h>

int main() {
    using namespace xsigma::smp_new::parallel;
    
    // Launch independent tasks
    for (int i = 0; i < 10; ++i) {
        launch([i]() {
            process_task(i);
        });
    }
    
    return 0;
}
```

## Building

The module is built as part of XSigma's core library. To enable it:

```bash
cmake -DXSIGMA_BUILD_SMP_NEW=ON ..
make
```

## Testing

Run the comprehensive test suite:

```bash
ctest -R smp_new
```

Or run individual test suites:

```bash
./test_smp_new --gtest_filter=ThreadPoolTest.*
./test_smp_new --gtest_filter=ParallelForTest.*
./test_smp_new --gtest_filter=ParallelReduceTest.*
./test_smp_new --gtest_filter=TaskExecutionTest.*
```

## Performance Characteristics

- **Parallel For**: O(n/p) where p is number of threads
- **Parallel Reduce**: O(n/p + log(p)) for combining
- **Task Execution**: O(1) amortized per task
- **Thread Pool Overhead**: Minimal with lazy initialization

## Comparison with PyTorch

| Feature | XSigma smp_new | PyTorch |
|---------|---|---|
| parallel_for | ✅ | ✅ |
| parallel_reduce | ✅ | ✅ |
| launch | ✅ | ✅ |
| intraop_launch | ✅ | ✅ |
| Lambda support | ✅ | ✅ |
| Exception handling | ✅ | ✅ |
| NUMA support | ✅ | ✅ |
| Separate pools | ✅ | ✅ |

## Backward Compatibility

The `smp_new` module is completely independent of the existing `smp/` module. Both can coexist without conflicts. Existing code using `smp/` will continue to work unchanged.

## Future Enhancements

- [ ] OpenMP backend support
- [ ] Work-stealing queue for better load balancing
- [ ] Python bindings
- [ ] Performance profiling hooks
- [ ] Adaptive grain size tuning

## License

XSigma is dual-licensed under GPL-3.0-or-later (open-source) and a commercial license.

## Support

For issues, questions, or contributions, please refer to the XSigma documentation or contact the development team.

