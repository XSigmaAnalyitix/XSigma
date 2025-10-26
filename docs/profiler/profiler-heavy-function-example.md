# Heavy Function Profiling Example - Comprehensive Guide

## Overview

The `TestProfilerHeavyFunction.cxx` demonstrates practical profiling usage with realistic computational workloads. This example shows how to instrument computationally intensive functions, analyze performance characteristics, and generate comprehensive profiling reports.

## Computational Functions Profiled

### 1. Matrix Multiplication (`matrix_multiply`)
**Purpose**: Dense matrix multiplication with O(n³) complexity
**Instrumentation**:
- Overall function timing
- Row-by-row computation timing
- Memory allocation tracking

**Code Example**:
```cpp
std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b)
{
    XSIGMA_PROFILE_SCOPE("matrix_multiply");

    // ... dimension validation ...

    {
        XSIGMA_PROFILE_SCOPE("matrix_multiply_computation");

        for (size_t i = 0; i < rows_a; ++i) {
            XSIGMA_PROFILE_SCOPE("matrix_row_computation");
            // Inner loops for multiplication
        }
    }

    return result;
}
```

**Performance Characteristics**:
- **Complexity**: O(n³) where n is matrix dimension
- **Memory Usage**: O(n²) for result matrix
- **Expected Time**: ~10-50ms for 100x100 matrices
- **Bottlenecks**: Cache misses, memory allocation

### 2. Merge Sort (`merge_sort`)
**Purpose**: Recursive divide-and-conquer sorting algorithm
**Instrumentation**:
- Depth-based profiling (`merge_sort_depth_N`)
- Left/right half separation
- Merge operation timing

**Code Example**:
```cpp
void merge_sort(std::vector<double>& arr, size_t left, size_t right, int depth = 0)
{
    XSIGMA_PROFILE_SCOPE("merge_sort_depth_" + std::to_string(depth));

    if (left >= right) return;

    size_t mid = left + (right - left) / 2;

    {
        XSIGMA_PROFILE_SCOPE("merge_sort_left_half");
        merge_sort(arr, left, mid, depth + 1);
    }

    {
        XSIGMA_PROFILE_SCOPE("merge_sort_right_half");
        merge_sort(arr, mid + 1, right, depth + 1);
    }

    {
        XSIGMA_PROFILE_SCOPE("merge_operation");
        // Merge logic
    }
}
```

**Performance Characteristics**:
- **Complexity**: O(n log n)
- **Memory Usage**: O(n) for temporary arrays
- **Expected Time**: ~5-20ms for 50,000 elements
- **Bottlenecks**: Memory allocation for temporary arrays

### 3. Monte Carlo Pi Estimation (`estimate_pi_monte_carlo`)
**Purpose**: Statistical approximation of π using random sampling
**Instrumentation**:
- Overall estimation timing
- Batch processing (100,000 samples per batch)
- Random number generation overhead

**Code Example**:
```cpp
double estimate_pi_monte_carlo(size_t num_samples)
{
    XSIGMA_PROFILE_SCOPE("monte_carlo_pi_estimation");

    // ... random number setup ...

    {
        XSIGMA_PROFILE_SCOPE("monte_carlo_sampling");

        for (size_t i = 0; i < num_samples; ++i) {
            if (i % 100000 == 0) {
                XSIGMA_PROFILE_SCOPE("monte_carlo_batch_" + std::to_string(i / 100000));
                // Process batch of 100,000 samples
            }
        }
    }

    return 4.0 * points_inside_circle / num_samples;
}
```

**Performance Characteristics**:
- **Complexity**: O(n) where n is number of samples
- **Memory Usage**: O(1) constant memory
- **Expected Time**: ~50-200ms for 1 million samples
- **Bottlenecks**: Random number generation, floating-point operations

### 4. FFT Simulation (`simulate_fft`)
**Purpose**: Frequency domain analysis simulation with O(n²) complexity
**Instrumentation**:
- Overall FFT computation
- Per-frequency bin processing
- Complex number arithmetic

**Code Example**:
```cpp
std::vector<std::complex<double>> simulate_fft(const std::vector<double>& signal)
{
    XSIGMA_PROFILE_SCOPE("simulate_fft");

    {
        XSIGMA_PROFILE_SCOPE("fft_computation");

        for (size_t k = 0; k < n; ++k) {
            XSIGMA_PROFILE_SCOPE("fft_frequency_bin");
            // Complex arithmetic for each frequency bin
        }
    }

    return result;
}
```

**Performance Characteristics**:
- **Complexity**: O(n²) (simplified implementation)
- **Memory Usage**: O(n) for result array
- **Expected Time**: ~20-100ms for 512 points
- **Bottlenecks**: Trigonometric function calls, complex arithmetic

## Profiling Configuration

### Session Setup
```cpp
auto session = profiler_session_builder()
    .with_timing(true)                    // Enable high-resolution timing
    .with_memory_tracking(true)           // Track memory allocations
    .with_statistical_analysis(true)     // Calculate statistics
    .with_thread_safety(true)            // Support multi-threading
    .with_output_format(profiler_options::output_format::json)
    .build();

statistical_analyzer analyzer;
session->add_observer(&analyzer);
```

### Multi-Format Output Generation
```cpp
// Generate reports in multiple formats
session->export_report("heavy_function_profile.json");
session->export_report("heavy_function_profile.csv");
session->export_report("heavy_function_profile.xml");
```

## Expected Profiling Output

### JSON Format Example
```json
{
  "profiling_session": {
    "start_time": "2024-01-15T10:30:00.000Z",
    "end_time": "2024-01-15T10:30:05.234Z",
    "total_duration_ms": 5234.567,
    "events": [
      {
        "name": "matrix_multiply",
        "start_time_ms": 0.123,
        "duration_ms": 45.678,
        "thread_id": 12345,
        "metadata": {
          "matrix_size": "100x100",
          "memory_allocated": 80000
        }
      },
      {
        "name": "merge_sort_depth_0",
        "start_time_ms": 50.234,
        "duration_ms": 12.345,
        "thread_id": 12345,
        "metadata": {
          "array_size": 50000,
          "recursion_depth": 0
        }
      },
      {
        "name": "monte_carlo_pi_estimation",
        "start_time_ms": 70.456,
        "duration_ms": 156.789,
        "thread_id": 12345,
        "metadata": {
          "num_samples": 1000000,
          "pi_estimate": 3.14159
        }
      }
    ],
    "statistics": {
      "matrix_multiply": {
        "count": 3,
        "total_time_ms": 137.034,
        "mean_time_ms": 45.678,
        "std_deviation_ms": 2.345,
        "min_time_ms": 43.123,
        "max_time_ms": 48.567
      }
    }
  }
}
```

### CSV Format Example
```csv
event_name,start_time_ms,duration_ms,thread_id,metadata
matrix_multiply,0.123,45.678,12345,"matrix_size=100x100;memory_allocated=80000"
matrix_row_computation,1.234,0.456,12345,"row_index=0"
matrix_row_computation,1.690,0.445,12345,"row_index=1"
merge_sort_depth_0,50.234,12.345,12345,"array_size=50000"
merge_sort_depth_1,51.123,5.678,12345,"array_size=25000"
monte_carlo_pi_estimation,70.456,156.789,12345,"num_samples=1000000"
```

### Console Output Example
```
=== Heavy Function Performance Analysis ===
Matrix Multiplication (100x100):
  Average time: 45.678 ms
  Total time: 137.034 ms
  Iterations: 3

Merge Sort (50,000 elements):
  Average time: 12.345 ms
  Total time: 12.345 ms

Monte Carlo Pi (1M samples):
  Average time: 156.789 ms
  Total time: 156.789 ms

FFT Simulation (512 points):
  Average time: 67.890 ms
  Total time: 67.890 ms
```

## Performance Analysis Insights

### Relative Performance Comparison
Based on typical execution times:

| Function | Complexity | Data Size | Expected Time | Relative Performance |
|----------|------------|-----------|---------------|---------------------|
| Matrix Multiply | O(n³) | 100×100 | ~45ms | Slowest (baseline) |
| FFT Simulation | O(n²) | 512 points | ~68ms | 1.5× slower |
| Monte Carlo | O(n) | 1M samples | ~157ms | 3.4× slower |
| Merge Sort | O(n log n) | 50K elements | ~12ms | 3.8× faster |

### Bottleneck Identification
1. **Matrix Multiplication**: Cache misses due to poor memory locality
2. **Monte Carlo**: Random number generation overhead
3. **FFT Simulation**: Trigonometric function calls
4. **Merge Sort**: Memory allocation for temporary arrays

### Optimization Opportunities
1. **Matrix Multiplication**:
   - Use blocked/tiled algorithms for better cache usage
   - Implement SIMD vectorization
   - Consider BLAS library integration

2. **Monte Carlo**:
   - Use faster pseudo-random number generators
   - Implement vectorized sampling
   - Parallelize across multiple threads

3. **FFT Simulation**:
   - Replace with actual FFT algorithm (O(n log n))
   - Pre-compute trigonometric values
   - Use lookup tables for common angles

## Multi-Threading Analysis

### Thread-Safe Profiling
```cpp
std::vector<std::thread> workers;
const int num_threads = 4;

for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back([i]() {
        XSIGMA_PROFILE_SCOPE("worker_thread_" + std::to_string(i));

        // Each thread performs independent work
        double pi_est = estimate_pi_monte_carlo(250000);
    });
}
```

### Expected Thread Performance
- **Thread Creation Overhead**: ~0.1-1ms per thread
- **Context Switching**: Minimal with CPU-bound tasks
- **Scalability**: Linear scaling up to CPU core count
- **Synchronization**: No contention (independent work)

## Integration with TensorBoard

### XPlane Export
```cpp
// Export profiling data in XPlane format for TensorBoard
session->export_report("heavy_function_profile.xplane");
```

### TensorBoard Visualization
```bash
# Launch TensorBoard with profiling data
tensorboard --logdir=./profiling_logs --port=6006
```

### Expected TensorBoard Views
1. **Timeline View**: Shows function execution timeline
2. **Memory View**: Displays memory allocation patterns
3. **Statistics View**: Provides performance metrics
4. **Trace Events**: Shows nested function calls

## Best Practices Demonstrated

### 1. Hierarchical Profiling
```cpp
XSIGMA_PROFILE_SCOPE("top_level_function");
{
    XSIGMA_PROFILE_SCOPE("sub_operation_1");
    // Work
}
{
    XSIGMA_PROFILE_SCOPE("sub_operation_2");
    // Work
}
```

### 2. Meaningful Scope Names
- Use descriptive names: `"matrix_multiply"` not `"func1"`
- Include parameters: `"merge_sort_depth_" + std::to_string(depth)`
- Indicate data sizes: `"monte_carlo_1M_samples"`

### 3. Balanced Granularity
- Profile significant operations (>1ms typically)
- Avoid over-instrumentation of trivial operations
- Group related operations when appropriate

### 4. Performance Validation
```cpp
// Verify profiling captured meaningful data
EXPECT_GT(matrix_stats.mean, 0);
EXPECT_GT(sort_stats.mean, 0);

// Verify performance expectations
EXPECT_GT(matrix_stats.mean, sort_stats.mean / 1000);
```

This comprehensive example demonstrates how to effectively profile computationally intensive functions, analyze performance characteristics, and generate actionable insights for optimization.
