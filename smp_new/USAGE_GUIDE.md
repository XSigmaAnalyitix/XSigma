# XSigma SMP_NEW - Usage Guide

## Quick Start

### 1. Include the Header

```cpp
#include <xsigma/smp_new/parallel/parallel_api.h>

using namespace xsigma::smp_new::parallel;
```

### 2. Use Parallel APIs

```cpp
// Parallel for loop
parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    // Process range [b, e)
});

// Parallel reduction
T result = parallel_reduce(
    0, N, grain_size,
    identity,
    [&](int64_t b, int64_t e, T ident) { /* reduce */ },
    [](T a, T b) { /* combine */ }
);
```

## Common Patterns

### Pattern 1: Element-wise Operations

```cpp
std::vector<float> data(1000000);

parallel_for(0, data.size(), 10000, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
        data[i] = std::sqrt(data[i]);
    }
});
```

### Pattern 2: Summation

```cpp
std::vector<float> data(1000000);

float sum = parallel_reduce(
    0, data.size(), 10000,
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
```

### Pattern 3: Finding Maximum

```cpp
std::vector<int> data(1000000);

int max_val = parallel_reduce(
    0, data.size(), 10000,
    INT_MIN,
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

### Pattern 4: Matrix Operations

```cpp
std::vector<std::vector<float>> matrix(N, std::vector<float>(M));

parallel_for(0, N, 100, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
        for (int j = 0; j < M; ++j) {
            matrix[i][j] = compute(matrix[i][j]);
        }
    }
});
```

### Pattern 5: Task-Based Parallelism

```cpp
// Launch independent tasks
for (int i = 0; i < num_tasks; ++i) {
    launch([i]() {
        process_task(i);
    });
}
```

## Configuration

### Setting Thread Counts

```cpp
// Set before first parallel operation
set_num_intraop_threads(4);
set_num_interop_threads(8);

// Query thread counts
auto intraop = get_num_intraop_threads();
auto interop = get_num_interop_threads();
```

### Environment Variables

```bash
# Set via environment (if supported)
export XSIGMA_INTRAOP_THREADS=4
export XSIGMA_INTEROP_THREADS=8
```

## Performance Tips

### 1. Choose Appropriate Grain Size

```cpp
// Too small: overhead from task creation
parallel_for(0, N, 1, [&](int64_t b, int64_t e) { /* ... */ });

// Too large: poor load balancing
parallel_for(0, N, N, [&](int64_t b, int64_t e) { /* ... */ });

// Good: balance between overhead and load balancing
parallel_for(0, N, N / (num_threads * 4), [&](int64_t b, int64_t e) { /* ... */ });

// Auto: let the library decide
parallel_for(0, N, 0, [&](int64_t b, int64_t e) { /* ... */ });
```

### 2. Minimize Synchronization

```cpp
// Good: minimal synchronization
parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
        data[i] = compute(data[i]);
    }
});

// Bad: synchronization inside loop
parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
        std::lock_guard<std::mutex> lock(mutex);  // Avoid!
        data[i] = compute(data[i]);
    }
});
```

### 3. Use Appropriate Data Types

```cpp
// Good: use native types
parallel_reduce(0, N, grain_size, 0.0f,
    [&](int64_t b, int64_t e, float ident) {
        float s = ident;
        for (int64_t i = b; i < e; ++i) {
            s += data[i];
        }
        return s;
    },
    [](float a, float b) { return a + b; }
);

// Avoid: expensive copies
parallel_reduce(0, N, grain_size, std::vector<float>(),
    [&](int64_t b, int64_t e, std::vector<float> ident) {
        // Expensive copy!
        return ident;
    },
    [](const std::vector<float>& a, const std::vector<float>& b) {
        // Expensive operation
        return a;
    }
);
```

## Error Handling

### Exception Handling

```cpp
try {
    parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
        if (error_condition) {
            throw std::runtime_error("Error in parallel region");
        }
    });
} catch (const std::exception& e) {
    std::cerr << "Parallel operation failed: " << e.what() << std::endl;
}
```

## Debugging

### Enable Logging

```cpp
// Set thread names for debugging
// (Automatically done by the library)

// Check backend info
auto info = xsigma::smp_new::native::GetNativeBackendInfo();
std::cout << info << std::endl;
```

### Verify Parallelism

```cpp
std::atomic<int> thread_count{0};

parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    thread_count++;
    // Process range
});

std::cout << "Executed on " << thread_count << " threads" << std::endl;
```

## Migration from PyTorch

If migrating from PyTorch's `at::parallel_for`:

```cpp
// PyTorch
at::parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    // ...
});

// XSigma smp_new (compatible API)
xsigma::smp_new::parallel::parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    // Same code!
});
```

## Troubleshooting

### Issue: Parallel code runs slower than serial

**Solution**: Check grain size. If too small, overhead dominates.

```cpp
// Increase grain size
parallel_for(0, N, N / (num_threads * 2), [&](int64_t b, int64_t e) {
    // ...
});
```

### Issue: Incorrect results from parallel_reduce

**Solution**: Ensure identity value is correct and combine function is associative.

```cpp
// Wrong: identity should be 0 for sum
float sum = parallel_reduce(0, N, grain_size, 1.0f, /* ... */);

// Correct: identity is 0 for sum
float sum = parallel_reduce(0, N, grain_size, 0.0f, /* ... */);
```

### Issue: Deadlock in nested parallelism

**Solution**: Use `intraop_launch` for nested tasks.

```cpp
// Good: use intraop_launch for nested tasks
parallel_for(0, N, grain_size, [&](int64_t b, int64_t e) {
    intraop_launch([&]() {
        // Nested task
    });
});
```

## Best Practices

1. **Use lambdas** for cleaner, more readable code
2. **Choose appropriate grain sizes** based on work complexity
3. **Minimize synchronization** in parallel regions
4. **Use parallel_reduce** for aggregation operations
5. **Handle exceptions** properly in parallel code
6. **Profile and benchmark** to find optimal configuration
7. **Test with different thread counts** to ensure scalability

## References

- [README.md](README.md) - Module overview
- [API Reference](../parallel/parallel_api.h) - Detailed API documentation
- [Examples](../test/) - Test cases with usage examples

