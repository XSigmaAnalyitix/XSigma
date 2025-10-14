# XSigma Profiler Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Basic Usage Patterns](#basic-usage-patterns)
3. [Best Practices](#best-practices)
4. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
5. [Performance Impact](#performance-impact)
6. [Integration Examples](#integration-examples)

---

## Quick Start

Get started with XSigma profiling in 5 minutes:

```cpp
#include "experimental/profiler/session/profiler.h"

int main() {
    // 1. Configure profiler options
    xsigma::profiler_options opts;
    opts.enable_timing_ = true;
    opts.enable_memory_tracking_ = true;
    opts.output_format_ = xsigma::profiler_options::output_format_enum::JSON;
    
    // 2. Create profiler session
    xsigma::profiler_session session(opts);
    
    // 3. Start profiling
    session.start();
    
    // 4. Profile your code
    {
        XSIGMA_PROFILE_SCOPE("my_function");
        // Your code here
    }
    
    // 5. Stop and generate report
    session.stop();
    session.print_report();
    
    return 0;
}
```

---

## Basic Usage Patterns

### Pattern 1: Function-Level Profiling

Profile individual functions to measure execution time:

```cpp
void process_data(const std::vector<double>& data) {
    XSIGMA_PROFILE_SCOPE("process_data");
    
    // Function implementation
    for (const auto& value : data) {
        // Process each value
    }
}
```

### Pattern 2: Hierarchical Profiling

Profile nested scopes to understand call hierarchies:

```cpp
void complex_algorithm() {
    XSIGMA_PROFILE_SCOPE("complex_algorithm");
    
    {
        XSIGMA_PROFILE_SCOPE("initialization");
        // Initialization code
    }
    
    {
        XSIGMA_PROFILE_SCOPE("computation");
        // Main computation
        
        {
            XSIGMA_PROFILE_SCOPE("inner_loop");
            // Inner loop processing
        }
    }
    
    {
        XSIGMA_PROFILE_SCOPE("finalization");
        // Cleanup code
    }
}
```

### Pattern 3: Memory Profiling

Track memory allocations and deallocations:

```cpp
void memory_intensive_operation() {
    XSIGMA_PROFILE_SCOPE("memory_operation");
    
    // Enable memory tracking in profiler options
    xsigma::profiler_options opts;
    opts.enable_memory_tracking_ = true;
    
    xsigma::profiler_session session(opts);
    session.start();
    
    // Allocate memory
    std::vector<double> large_array(1000000);
    
    // Process data
    for (auto& val : large_array) {
        val = std::rand() / static_cast<double>(RAND_MAX);
    }
    
    session.stop();
    session.print_report();
}
```

### Pattern 4: Multi-Threaded Profiling

Profile concurrent operations safely:

```cpp
void parallel_computation() {
    xsigma::profiler_options opts;
    opts.enable_thread_safety_ = true;  // Enable thread-safe profiling
    opts.enable_timing_ = true;
    
    xsigma::profiler_session session(opts);
    session.start();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([i]() {
            XSIGMA_PROFILE_SCOPE("worker_thread_" + std::to_string(i));
            // Thread-specific work
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    session.stop();
    session.print_report();
}
```

### Pattern 5: Conditional Profiling

Enable profiling only when needed:

```cpp
void conditional_profiling(bool enable_profiling) {
    xsigma::profiler_options opts;
    opts.enable_timing_ = enable_profiling;
    opts.enable_memory_tracking_ = enable_profiling;
    
    xsigma::profiler_session session(opts);
    
    if (enable_profiling) {
        session.start();
    }
    
    // Your code here
    {
        XSIGMA_PROFILE_SCOPE("main_work");
        // Work that may or may not be profiled
    }
    
    if (enable_profiling) {
        session.stop();
        session.export_report("profile_output.json");
    }
}
```

---

## Best Practices

### 1. When to Use Profiling vs Tracing

**Use Profiling When:**
- You need to measure execution time and performance metrics
- You want to identify performance bottlenecks
- You need statistical analysis of function call patterns
- You want to track memory usage over time

**Use Tracing When:**
- You need detailed event logs for debugging
- You want to understand program flow and execution order
- You need to correlate events across multiple threads
- You want to export data to visualization tools (TensorBoard, Chrome Tracing)

**Use Both When:**
- You need comprehensive performance analysis
- You want to correlate timing data with execution events
- You're optimizing complex multi-threaded applications

### 2. Optimal Profiling Scope Granularity

**✅ Good Granularity:**
```cpp
void process_large_dataset() {
    XSIGMA_PROFILE_SCOPE("process_large_dataset");  // ✅ Profile entire function
    
    {
        XSIGMA_PROFILE_SCOPE("data_loading");  // ✅ Profile major phase
        load_data();
    }
    
    {
        XSIGMA_PROFILE_SCOPE("computation");  // ✅ Profile major phase
        compute_results();
    }
}
```

**❌ Too Fine-Grained (Avoid):**
```cpp
void process_data() {
    for (int i = 0; i < 1000000; ++i) {
        XSIGMA_PROFILE_SCOPE("single_iteration");  // ❌ Too much overhead!
        data[i] = compute(i);
    }
}
```

**Rule of Thumb:**
- Profile functions that take **> 1ms** to execute
- Profile major algorithmic phases (initialization, computation, finalization)
- Avoid profiling trivial operations (< 1μs)
- Limit profiling depth to 3-5 levels for hierarchical profiling

### 3. Thread-Safe Profiling Patterns

**Always enable thread safety for multi-threaded code:**

```cpp
xsigma::profiler_options opts;
opts.enable_thread_safety_ = true;  // ✅ Required for multi-threaded profiling
```

**Use thread-specific scope names:**

```cpp
void worker_thread(int thread_id) {
    std::string scope_name = "worker_" + std::to_string(thread_id);
    XSIGMA_PROFILE_SCOPE(scope_name.c_str());
    // Thread work
}
```

### 4. Minimizing Profiling Overhead in Production

**Development Build (Full Profiling):**
```cpp
#ifdef DEBUG
    xsigma::profiler_options opts;
    opts.enable_timing_ = true;
    opts.enable_memory_tracking_ = true;
    opts.enable_statistical_analysis_ = true;
#else
    xsigma::profiler_options opts;
    opts.enable_timing_ = false;  // Disable in production
    opts.enable_memory_tracking_ = false;
#endif
```

**Sampling-Based Profiling:**
```cpp
// Profile only 1% of executions
if (std::rand() % 100 == 0) {
    XSIGMA_PROFILE_SCOPE("sampled_function");
    expensive_operation();
}
```

### 5. Choosing Appropriate Output Formats

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| **JSON** | Machine-readable, integration with tools | Structured, easy to parse | Larger file size |
| **CSV** | Spreadsheet analysis, data science | Simple, widely supported | Limited structure |
| **XML** | Enterprise integration, legacy systems | Hierarchical, schema validation | Verbose |
| **XPlane** | TensorFlow/TensorBoard visualization | Rich metadata, timeline view | Requires TensorBoard |

**Example:**
```cpp
// For automated analysis
opts.output_format_ = xsigma::profiler_options::output_format_enum::JSON;
session.export_report("profile.json");

// For visualization
opts.output_format_ = xsigma::profiler_options::output_format_enum::XPLANE;
session.export_report("profile.xplane");
```

---

## Common Pitfalls to Avoid

### 1. ❌ Profiling Trivial Operations

**Problem:**
```cpp
for (int i = 0; i < 1000000; ++i) {
    XSIGMA_PROFILE_SCOPE("add_operation");  // ❌ Overhead > actual work!
    result += i;
}
```

**Solution:**
```cpp
{
    XSIGMA_PROFILE_SCOPE("sum_loop");  // ✅ Profile entire loop
    for (int i = 0; i < 1000000; ++i) {
        result += i;
    }
}
```

### 2. ❌ Forgetting to Stop Profiling Sessions

**Problem:**
```cpp
void bad_profiling() {
    xsigma::profiler_session session(opts);
    session.start();
    // ... work ...
    // ❌ Forgot to call session.stop()!
}  // Session destructor will stop, but report may be incomplete
```

**Solution:**
```cpp
void good_profiling() {
    xsigma::profiler_session session(opts);
    session.start();
    // ... work ...
    session.stop();  // ✅ Explicitly stop
    session.print_report();
}
```

### 3. ❌ Memory Leaks from Unclosed Profiling Scopes

**Problem:**
```cpp
void leaky_profiling() {
    XSIGMA_PROFILE_SCOPE("outer");
    
    if (error_condition) {
        return;  // ❌ Scope not properly closed!
    }
    
    // More work
}
```

**Solution:**
```cpp
void safe_profiling() {
    XSIGMA_PROFILE_SCOPE("outer");  // ✅ RAII ensures cleanup
    
    if (error_condition) {
        return;  // ✅ Scope automatically closed
    }
    
    // More work
}  // ✅ Scope automatically closed here too
```

### 4. ❌ Thread Safety Violations

**Problem:**
```cpp
xsigma::profiler_options opts;
opts.enable_thread_safety_ = false;  // ❌ Not thread-safe!

xsigma::profiler_session session(opts);
session.start();

std::thread t1([&]() { XSIGMA_PROFILE_SCOPE("thread1"); });
std::thread t2([&]() { XSIGMA_PROFILE_SCOPE("thread2"); });
// ❌ Race conditions!
```

**Solution:**
```cpp
xsigma::profiler_options opts;
opts.enable_thread_safety_ = true;  // ✅ Thread-safe!

xsigma::profiler_session session(opts);
session.start();

std::thread t1([&]() { XSIGMA_PROFILE_SCOPE("thread1"); });
std::thread t2([&]() { XSIGMA_PROFILE_SCOPE("thread2"); });
// ✅ Safe concurrent profiling
```

### 5. ❌ Incorrect Interpretation of Statistical Metrics

**Problem:**
```cpp
// Misunderstanding mean vs median
auto stats = analyzer.calculate_timing_stats("function");
std::cout << "Typical time: " << stats.mean << " ms\n";  // ❌ Mean affected by outliers!
```

**Solution:**
```cpp
// Use median for typical performance
auto stats = analyzer.calculate_timing_stats("function");
std::cout << "Typical time: " << stats.median << " ms\n";  // ✅ Median more robust
std::cout << "Average time: " << stats.mean << " ms\n";
std::cout << "Worst case: " << stats.max_value << " ms\n";
```

---

## Performance Impact

### Profiling Overhead by Granularity

| Profiling Level | Overhead | Use Case |
|----------------|----------|----------|
| **Disabled** | 0% | Production builds |
| **Coarse (function-level)** | < 1% | Always-on profiling |
| **Medium (major phases)** | 1-5% | Development profiling |
| **Fine (inner loops)** | 5-20% | Detailed analysis |
| **Very Fine (per-iteration)** | > 50% | ❌ Avoid! |

### Measured Overhead Examples

```cpp
// Baseline: No profiling
void baseline() {
    for (int i = 0; i < 1000000; ++i) {
        compute(i);
    }
}
// Time: 100ms

// Coarse profiling (1 scope)
void coarse_profiling() {
    XSIGMA_PROFILE_SCOPE("compute_loop");
    for (int i = 0; i < 1000000; ++i) {
        compute(i);
    }
}
// Time: 100.5ms (0.5% overhead)

// Fine profiling (1M scopes) - ❌ DON'T DO THIS!
void fine_profiling() {
    for (int i = 0; i < 1000000; ++i) {
        XSIGMA_PROFILE_SCOPE("single_compute");
        compute(i);
    }
}
// Time: 250ms (150% overhead!)
```

---

## Integration Examples

### Example 1: Integrating with Existing XSigma Applications

```cpp
#include "experimental/profiler/session/profiler.h"
#include "memory/backend/allocator_bfc.h"

class MyApplication {
public:
    MyApplication() {
        // Initialize profiler
        xsigma::profiler_options opts;
        opts.enable_timing_ = true;
        opts.enable_memory_tracking_ = true;
        profiler_ = std::make_unique<xsigma::profiler_session>(opts);
    }
    
    void run() {
        profiler_->start();
        
        {
            XSIGMA_PROFILE_SCOPE("application_initialization");
            initialize();
        }
        
        {
            XSIGMA_PROFILE_SCOPE("main_processing");
            process();
        }
        
        {
            XSIGMA_PROFILE_SCOPE("cleanup");
            cleanup();
        }
        
        profiler_->stop();
        profiler_->export_report("app_profile.json");
    }
    
private:
    std::unique_ptr<xsigma::profiler_session> profiler_;
    
    void initialize() { /* ... */ }
    void process() { /* ... */ }
    void cleanup() { /* ... */ }
};
```

### Example 2: Profiling with BFC Allocator

See `Library/Core/Testing/Cxx/TestAllocatorBFC.cxx` for complete examples:
- `ComprehensiveMemoryProfiling`: Tracks allocation patterns and fragmentation
- `AllocationHotspotsIdentification`: Identifies performance bottlenecks

### Example 3: Profiling Heavy Computational Workloads

See `Library/Core/Testing/Cxx/TestProfilerHeavyFunction.cxx` for complete examples:
- Matrix multiplication profiling
- Sorting algorithm profiling
- Monte Carlo simulation profiling
- FFT profiling
- Multi-threaded computation profiling

---

## Next Steps

- **Advanced Features**: See [Profiler System Documentation](profiler-system.md)
- **XPlane Format**: See [XPlane Format Guide](xplane-format-guide.md)
- **Enhancement Roadmap**: See [Enhancement Roadmap](profiler-enhancement-roadmap.md)
- **Third-Party Integration**: See [Third-Party Integration](profiler-third-party-integration.md)

