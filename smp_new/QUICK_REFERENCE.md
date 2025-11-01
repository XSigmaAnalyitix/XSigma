# XSigma SMP_NEW - Quick Reference Card

## Include Header
```cpp
#include <xsigma/smp_new/parallel/parallel_api.h>
using namespace xsigma::smp_new::parallel;
```

## Parallel For Loop

### Basic Usage
```cpp
parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
        data[i] = compute(data[i]);
    }
});
```

### Auto Grain Size
```cpp
parallel_for(0, N, 0, [&](int64_t b, int64_t e) {
    // grain_size = 0 means auto-determine
});
```

### Grain Size Guidelines
- **Small work:** grain_size = N / (threads * 2)
- **Large work:** grain_size = N / (threads * 8)
- **Auto:** grain_size = 0

## Parallel Reduce

### Sum Reduction
```cpp
float sum = parallel_reduce(
    0, N, grain_size,
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

### Max Reduction
```cpp
int max_val = parallel_reduce(
    0, N, grain_size,
    INT_MIN,  // identity
    [&](int64_t b, int64_t e, int ident) {
        int m = ident;
        for (int64_t i = b; i < e; ++i) {
            m = std::max(m, data[i]);
        }
        return m;
    },
    [](int a, int b) { return std::max(a, b); }
);
```

### Min Reduction
```cpp
int min_val = parallel_reduce(
    0, N, grain_size,
    INT_MAX,  // identity
    [&](int64_t b, int64_t e, int ident) {
        int m = ident;
        for (int64_t i = b; i < e; ++i) {
            m = std::min(m, data[i]);
        }
        return m;
    },
    [](int a, int b) { return std::min(a, b); }
);
```

## Task Execution

### Inter-Op Task
```cpp
launch([]() {
    process_data();
});
```

### Intra-Op Task
```cpp
intraop_launch([]() {
    process_chunk();
});
```

### Nested Tasks
```cpp
parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    intraop_launch([&]() {
        // Nested task
    });
});
```

## Thread Configuration

### Set Thread Counts
```cpp
set_num_intraop_threads(4);
set_num_interop_threads(8);
```

### Get Thread Counts
```cpp
auto intraop = get_num_intraop_threads();
auto interop = get_num_interop_threads();
```

### Default Thread Count
```cpp
auto default_threads = xsigma::smp_new::core::TaskThreadPoolBase::DefaultNumThreads();
```

## Exception Handling

### Catch Exceptions
```cpp
try {
    parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
        if (error_condition) {
            throw std::runtime_error("Error");
        }
    });
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

## Common Patterns

### Pattern: Element-wise Operation
```cpp
parallel_for(0, data.size(), 10000, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
        data[i] = std::sqrt(data[i]);
    }
});
```

### Pattern: Accumulation
```cpp
T result = parallel_reduce(
    0, N, grain_size, identity,
    [&](int64_t b, int64_t e, T ident) {
        T acc = ident;
        for (int64_t i = b; i < e; ++i) {
            acc = combine(acc, data[i]);
        }
        return acc;
    },
    [](T a, T b) { return combine(a, b); }
);
```

### Pattern: Matrix Operation
```cpp
parallel_for(0, rows, 100, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = compute(matrix[i][j]);
        }
    }
});
```

### Pattern: Independent Tasks
```cpp
for (int i = 0; i < num_tasks; ++i) {
    launch([i]() {
        process_task(i);
    });
}
```

## Performance Tips

1. **Choose appropriate grain size**
   - Too small: overhead dominates
   - Too large: poor load balancing
   - Use 0 for auto-determination

2. **Minimize synchronization**
   - Avoid locks in parallel regions
   - Use thread-local storage if needed

3. **Use appropriate data types**
   - Prefer native types (int, float, double)
   - Avoid expensive copies

4. **Profile and benchmark**
   - Measure actual performance
   - Adjust grain size based on results

## Troubleshooting

### Parallel code slower than serial
**Solution:** Increase grain size
```cpp
parallel_for(0, N, N / (threads * 2), [&](int64_t b, int64_t e) {
    // ...
});
```

### Incorrect reduction results
**Solution:** Check identity value and combine function
```cpp
// Wrong: identity should be 0 for sum
float sum = parallel_reduce(0, N, grain_size, 1.0f, /* ... */);

// Correct: identity is 0 for sum
float sum = parallel_reduce(0, N, grain_size, 0.0f, /* ... */);
```

### Deadlock in nested parallelism
**Solution:** Use intraop_launch for nested tasks
```cpp
parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    intraop_launch([&]() {
        // Nested task
    });
});
```

## API Summary

| Function | Purpose | Returns |
|----------|---------|---------|
| `parallel_for()` | Parallel iteration | void |
| `parallel_reduce()` | Parallel reduction | T (template) |
| `launch()` | Inter-op task | void |
| `intraop_launch()` | Intra-op task | void |
| `set_num_intraop_threads()` | Configure threads | void |
| `get_num_intraop_threads()` | Query threads | size_t |
| `set_num_interop_threads()` | Configure threads | void |
| `get_num_interop_threads()` | Query threads | size_t |

## Building

```bash
cmake -DXSIGMA_BUILD_SMP_NEW=ON ..
make
ctest -R smp_new
```

## Documentation Links

- [README.md](README.md) - Overview
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed examples
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Integration
- [INDEX.md](INDEX.md) - Complete index

## Key Concepts

- **Grain Size:** Minimum work per thread
- **Intra-op:** Parallelism within a single operation
- **Inter-op:** Parallelism between independent operations
- **Identity:** Initial value for reduction
- **Combine:** Function to merge partial results

## Notes

- Thread pools are lazily initialized
- Exceptions are propagated to caller
- NUMA support is automatic
- Backward compatible with existing smp/ module
- Production-ready implementation

