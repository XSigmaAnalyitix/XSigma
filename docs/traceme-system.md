# XSigma TraceMe System

## Overview

The XSigma TraceMe system is a high-performance, lightweight tracing infrastructure designed for profiling and performance analysis of C++ applications. It provides minimal-overhead instrumentation that enables developers to understand program execution flow, identify performance bottlenecks, and correlate CPU activities with GPU operations.

### Key Features

- **Zero-Cost When Disabled**: Compile-time and runtime optimizations eliminate overhead when tracing is inactive
- **RAII-Based Design**: Automatic timing with scope-based lifetime management
- **Thread-Safe**: Safe for concurrent use across multiple threads with lock-free recording
- **Hierarchical Filtering**: Support for filtering events by importance/verbosity levels
- **Rich Metadata**: Extensible metadata encoding for detailed contextual analysis
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Integration Ready**: Seamless integration with profiling tools and visualizers

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Overhead when disabled | ~1-2 CPU cycles |
| Overhead when enabled | ~50-100 nanoseconds per event |
| Memory usage | ~64 bytes per active trace event |
| Thread contention | Minimal (lock-free recording) |
| Timestamp resolution | Nanosecond precision |

## Architecture

The TraceMe system consists of three main components:

1. **`traceme`** - RAII trace objects for instrumenting code sections
2. **`traceme_recorder`** - Singleton event collector with thread-local storage
3. **`traceme_encode`** - Utilities for structured metadata encoding

### Data Flow

```
[traceme objects] → [thread-local buffers] → [traceme_recorder] → [profiling tools]
```

1. `traceme` objects record start/end times to thread-local buffers
2. `traceme_recorder` collects events from all threads during profiling sessions
3. Events are processed, paired (for split activities), and exported to profiling tools

## Quick Start

### Basic Usage

```cpp
#include "logging/tracing/traceme.h"

void example_function() {
    // Simple scoped tracing
    traceme trace("example_function");
    
    // ... your code here ...
    
    // Trace automatically ends when 'trace' goes out of scope
}
```

### With Metadata

```cpp
#include "logging/tracing/traceme.h"
#include "logging/tracing/traceme_encode.h"

void process_data(const std::vector<int>& data) {
    // Trace with rich metadata
    traceme trace([&]() {
        return traceme_encode("process_data", {
            {"size", data.size()},
            {"type", "vector<int>"},
            {"memory_mb", data.size() * sizeof(int) / 1024 / 1024}
        });
    });
    
    // ... process data ...
}
```

### Manual Activity Management

```cpp
// For cases where RAII scoping isn't suitable
auto activity_id = traceme::activity_start("async_operation");

// ... start asynchronous work ...

traceme::activity_end(activity_id);
```

## API Reference

### Core Classes

#### `traceme`

The main RAII tracing class for instrumenting code sections.

**Constructors:**
- `traceme(std::string_view name, int level = 1)` - Basic constructor
- `traceme(NameGenerator&& generator, int level = 1)` - Lazy name generation
- `traceme(const char* name, int level = 1)` - C-string constructor

**Methods:**
- `void stop()` - Manually stop tracing before destruction
- `void append_metadata(MetadataGenerator&& generator)` - Add runtime metadata

**Static Methods:**
- `static int64_t activity_start(name, level)` - Start manual activity
- `static void activity_end(int64_t id)` - End manual activity
- `static void instant_activity(name, level)` - Record point-in-time event
- `static bool active(int level = 1)` - Check if tracing is active

#### `traceme_recorder`

Singleton event collector and session manager.

**Static Methods:**
- `static bool start(int level)` - Begin tracing session
- `static Events stop()` - End session and collect events
- `static bool active(int level = 1)` - Fast activity check
- `static void record(Event&& event)` - Record event (internal use)

#### `traceme_encode`

Utilities for structured metadata encoding.

**Functions:**
- `traceme_encode(name, {args...})` - Encode name with metadata
- `traceme_encode({args...})` - Encode metadata only
- `traceme_op(op_name, op_type)` - Standard operation naming
- `traceme_op_override(op_name, op_type)` - TensorFlow operation override

### Trace Levels

| Level | Name | Purpose | Example Use Cases |
|-------|------|---------|-------------------|
| 1 | CRITICAL | User operations, major steps | Function entry/exit, algorithm phases |
| 2 | INFO | Significant internal operations | Expensive computations, I/O operations |
| 3+ | VERBOSE | Fine-grained internal details | Loop iterations, small utility functions |

## Usage Patterns

### 1. Function Profiling

```cpp
void expensive_computation() {
    traceme trace(__FUNCTION__);  // Uses function name
    
    // ... computation ...
}
```

### 2. Algorithm Phases

```cpp
void machine_learning_training() {
    traceme trace("ml_training");
    
    {
        traceme phase("data_loading", 2);
        load_training_data();
    }
    
    {
        traceme phase("model_training", 2);
        train_model();
    }
    
    {
        traceme phase("validation", 2);
        validate_model();
    }
}
```

### 3. Loop Instrumentation

```cpp
void process_batches(const std::vector<Batch>& batches) {
    traceme trace("process_batches");
    
    for (size_t i = 0; i < batches.size(); ++i) {
        // Only trace at verbose level to avoid noise
        traceme batch_trace([&]() {
            return traceme_encode("process_batch", {
                {"batch_id", i},
                {"batch_size", batches[i].size()}
            });
        }, 3);
        
        process_single_batch(batches[i]);
    }
}
```

### 4. Conditional Tracing

```cpp
void optimized_function() {
    // Only create trace if verbose tracing is enabled
    if (traceme::active(3)) {
        traceme trace([&]() {
            return traceme_encode("optimized_function", {
                {"optimization_level", get_optimization_level()},
                {"cache_hits", get_cache_hits()}
            });
        }, 3);
        
        // ... function implementation ...
    } else {
        // ... function implementation without tracing overhead ...
    }
}
```

### 5. Asynchronous Operations

```cpp
class AsyncProcessor {
    void start_processing() {
        auto id = traceme::activity_start("async_processing");
        
        // Store ID for later use
        processing_trace_id_ = id;
        
        // Start async work...
        std::async([this]() {
            do_work();
            
            // End trace when work completes
            traceme::activity_end(processing_trace_id_);
        });
    }
    
private:
    int64_t processing_trace_id_ = 0;
};
```

## Best Practices

### Performance Optimization

1. **Use Lambda Constructors for Expensive Names**
   ```cpp
   // DON'T: Always constructs string even when tracing disabled
   traceme trace(expensive_string_operation());
   
   // DO: Only constructs string when tracing enabled
   traceme trace([&]() { return expensive_string_operation(); });
   ```

2. **Check Activity for Expensive Metadata**
   ```cpp
   traceme trace("operation");
   
   if (traceme::active(2)) {
       trace.append_metadata([&]() {
           return traceme_encode({
               {"expensive_stat", compute_expensive_statistic()}
           });
       });
   }
   ```

3. **Use Appropriate Trace Levels**
   - Level 1: User-visible operations, major algorithm phases
   - Level 2: Internal operations, expensive computations
   - Level 3+: Fine-grained details, debugging information

### Memory Management

1. **Avoid Long-Running Traces**
   ```cpp
   // DON'T: Trace holds memory for entire application lifetime
   class Application {
       traceme app_trace_{"application_lifetime"};  // Bad
   };
   
   // DO: Trace specific operations
   void run_application() {
       traceme trace("application_run");
       // ... application logic ...
   }
   ```

2. **Prefer RAII Over Manual Management**
   ```cpp
   // Preferred: Automatic cleanup
   {
       traceme trace("operation");
       // ... work ...
   }  // Automatically cleaned up
   
   // Only when necessary: Manual management
   auto id = traceme::activity_start("cross_scope_operation");
   // ... work ...
   traceme::activity_end(id);
   ```

### Thread Safety

1. **Same-Thread Activity Management**
   ```cpp
   // CORRECT: Start and end from same thread
   void worker_thread() {
       auto id = traceme::activity_start("worker_task");
       // ... work ...
       traceme::activity_end(id);
   }
   
   // INCORRECT: Cross-thread activity management
   void main_thread() {
       auto id = traceme::activity_start("task");
       std::thread([id]() {
           // ... work ...
           traceme::activity_end(id);  // Wrong thread!
       }).join();
   }
   ```

2. **Thread-Local Tracing**
   ```cpp
   void multi_threaded_operation() {
       traceme main_trace("multi_threaded_op");
       
       std::vector<std::thread> threads;
       for (int i = 0; i < num_threads; ++i) {
           threads.emplace_back([i]() {
               // Each thread gets its own trace
               traceme thread_trace([i]() {
                   return traceme_encode("worker_thread", {{"thread_id", i}});
               });
               
               // ... thread work ...
           });
       }
       
       for (auto& t : threads) {
           t.join();
       }
   }
   ```

## Integration Examples

### With Memory Allocators

```cpp
class TracingAllocator {
public:
    void* allocate(size_t size) {
        traceme trace([&]() {
            return traceme_encode("memory_allocation", {
                {"size", size},
                {"allocator", name_}
            });
        });
        
        void* ptr = underlying_allocator_->allocate(size);
        
        // Add allocation result to trace
        trace.append_metadata([&]() {
            return traceme_encode({
                {"address", reinterpret_cast<uint64_t>(ptr)},
                {"success", ptr != nullptr}
            });
        });
        
        return ptr;
    }
    
private:
    std::string name_;
    std::unique_ptr<Allocator> underlying_allocator_;
};
```

### With GPU Operations

```cpp
void launch_gpu_kernel() {
    // Trace CPU-side preparation
    traceme cpu_trace("gpu_kernel_launch");
    
    // Prepare kernel parameters
    setup_kernel_parameters();
    
    // Record instant event for kernel launch
    traceme::instant_activity([&]() {
        return traceme_encode("gpu_kernel_start", {
            {"kernel_name", "matrix_multiply"},
            {"grid_size", grid_size},
            {"block_size", block_size}
        });
    });
    
    // Launch kernel (async)
    launch_kernel_async();
    
    // CPU trace ends here, GPU work continues asynchronously
}
```

### With Error Handling

```cpp
void risky_operation() {
    traceme trace("risky_operation");
    
    try {
        // ... operation that might fail ...
        
        trace.append_metadata([&]() {
            return traceme_encode({{"status", "success"}});
        });
    }
    catch (const std::exception& e) {
        trace.append_metadata([&]() {
            return traceme_encode({
                {"status", "error"},
                {"error_message", e.what()}
            });
        });
        throw;  // Re-throw after recording error
    }
}
```

## Profiling Integration

The TraceMe system is designed as the foundation for comprehensive profiling workflows in XSigma. It integrates seamlessly with the experimental profiler system and external profiling tools to provide detailed performance analysis.

### Profiling Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │    │   TraceMe        │    │   Profiler      │
│   Code          │───▶│   System         │───▶│   Tools         │
│                 │    │                  │    │                 │
│ • traceme()     │    │ • Event Collection│    │ • Analysis      │
│ • Metadata      │    │ • Thread Mgmt    │    │ • Visualization │
│ • Timing        │    │ • Data Processing│    │ • Export        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow Pipeline

1. **Event Generation**: `traceme` objects create timing events with metadata
2. **Thread-Local Collection**: Events stored in lock-free thread-local buffers
3. **Session Management**: `traceme_recorder` coordinates collection across threads
4. **Event Processing**: Start/end event pairing and cross-thread correlation
5. **Format Conversion**: Events converted to profiler-specific formats (XPlane, Chrome Tracing, etc.)
6. **Analysis & Visualization**: External tools process and display profiling data

### Integration with XSigma Profiler

The TraceMe system serves as the CPU tracing backend for XSigma's experimental profiler:

```cpp
// Host tracer integration (simplified)
class host_tracer : public profiler_interface {
public:
    bool start() override {
        start_timestamp_ns_ = get_current_time_nanos();
        recording_ = traceme_recorder::start(host_trace_level_);
        return recording_;
    }

    bool stop() override {
        events_ = traceme_recorder::stop();
        recording_ = false;
        return true;
    }

    bool collect_data(x_space* space) override {
        // Convert traceme events to XPlane format
        xplane* plane = find_or_add_mutable_plane_with_name(space, "Host Threads");
        convert_events_to_xplane(events_, plane);
        return true;
    }

private:
    traceme_recorder::Events events_;
    uint64_t start_timestamp_ns_;
    bool recording_ = false;
    int host_trace_level_;
};
```

### XPlane Format Integration

TraceMe events are converted to XPlane format for integration with TensorFlow-compatible profiling tools:

```cpp
void convert_events_to_xplane(const traceme_recorder::Events& events, xplane* plane) {
    for (const auto& thread_events : events) {
        // Create line for each thread
        xline* line = plane->GetOrCreateLine(thread_events.thread.tid);
        line->SetName(thread_events.thread.name);

        for (const auto& event : thread_events.events) {
            if (event.is_complete()) {
                // Create XEvent from traceme event
                xevent* xevent = line->AddEvent(
                    *plane->GetOrCreateEventMetadata(event.name));
                xevent->SetTimestampNs(event.start_time);
                xevent->SetDurationNs(event.end_time - event.start_time);

                // Parse and add metadata
                parse_and_add_metadata(event.name, xevent, plane);
            }
        }
    }
}
```

### Starting a Profiling Session

```cpp
#include "logging/tracing/traceme_recorder.h"

void profile_application() {
    // Start recording at INFO level (captures levels 1 and 2)
    bool started = traceme_recorder::start(2);
    if (!started) {
        std::cerr << "Tracing already active\n";
        return;
    }

    // Run the code you want to profile
    run_application_code();

    // Stop recording and collect events
    auto events = traceme_recorder::stop();

    // Process events for analysis
    analyze_trace_events(events);
}
```

### Concrete Profiling Examples

#### Memory Allocator Profiling

The TraceMe system is extensively used in XSigma's memory allocators:

```cpp
// From allocator_cpu.cxx
void* allocator_cpu::allocate_raw(size_t alignment, size_t num_bytes) {
    traceme trace([&]() {
        return traceme_encode("allocator_cpu::allocate_raw", {
            {"bytes_requested", num_bytes},
            {"alignment", alignment},
            {"allocator_name", name()}
        });
    });

    void* ptr = port::aligned_malloc(num_bytes, alignment);

    // Add allocation result to trace
    trace.append_metadata([&]() {
        return traceme_encode({
            {"ptr", reinterpret_cast<uint64_t>(ptr)},
            {"success", ptr != nullptr}
        });
    });

    return ptr;
}
```

This generates profiling data showing:
- Memory allocation patterns and frequency
- Allocation sizes and alignment requirements
- Success/failure rates
- Performance bottlenecks in memory management

#### GPU/CPU Correlation

```cpp
void launch_gpu_computation() {
    // CPU-side preparation
    traceme cpu_prep("gpu_launch_preparation");
    setup_gpu_parameters();
    cpu_prep.stop();

    // Record GPU kernel launch point
    traceme::instant_activity([&]() {
        return traceme_encode("gpu_kernel_launch", {
            {"kernel_name", "matrix_multiply"},
            {"grid_size", grid_dims.x * grid_dims.y},
            {"block_size", block_dims.x * block_dims.y},
            {"shared_memory", shared_mem_bytes}
        });
    });

    // Launch asynchronous GPU work
    launch_kernel_async();

    // CPU continues with other work
    traceme cpu_work("cpu_parallel_work");
    do_cpu_work_while_gpu_runs();
}
```

This enables analysis of:
- CPU/GPU synchronization points
- Parallel execution efficiency
- Resource utilization patterns
- Performance bottlenecks in heterogeneous computing

### Event Processing and Analysis

```cpp
void analyze_trace_events(const traceme_recorder::Events& events) {
    ProfileAnalyzer analyzer;

    for (const auto& thread_events : events) {
        std::cout << "Thread " << thread_events.thread.tid
                  << " (" << thread_events.thread.name << "):\n";

        for (const auto& event : thread_events.events) {
            if (event.is_complete()) {
                auto duration = event.end_time - event.start_time;
                std::cout << "  " << event.name
                          << ": " << duration << " ns\n";

                // Advanced analysis
                analyzer.process_event(event, thread_events.thread);
            }
        }
    }

    // Generate comprehensive analysis
    analyzer.print_summary();
}

class ProfileAnalyzer {
public:
    void process_event(const traceme_recorder::Event& event,
                      const traceme_recorder::ThreadInfo& thread) {
        // Extract metadata from event name
        auto metadata = parse_metadata(event.name);

        // Categorize events
        if (event.name.find("allocate") != std::string::npos) {
            analyze_memory_event(event, metadata);
        } else if (event.name.find("gpu") != std::string::npos) {
            analyze_gpu_event(event, metadata);
        } else {
            analyze_cpu_event(event, metadata);
        }

        // Track thread activity
        thread_stats_[thread.tid].total_time +=
            (event.end_time - event.start_time);
        thread_stats_[thread.tid].event_count++;
    }

    void print_summary() {
        std::cout << "\n=== Profiling Summary ===\n";

        // Memory allocation analysis
        std::cout << "Memory Allocations:\n";
        std::cout << "  Total: " << memory_stats_.allocation_count << "\n";
        std::cout << "  Total Bytes: " << memory_stats_.total_bytes << "\n";
        std::cout << "  Average Size: " <<
            (memory_stats_.total_bytes / memory_stats_.allocation_count) << "\n";

        // Thread utilization
        std::cout << "\nThread Utilization:\n";
        for (const auto& [tid, stats] : thread_stats_) {
            std::cout << "  Thread " << tid << ": "
                      << stats.event_count << " events, "
                      << (stats.total_time / 1000000) << " ms total\n";
        }

        // Top time consumers
        std::cout << "\nTop Time Consumers:\n";
        for (const auto& [name, duration] : get_top_events(10)) {
            std::cout << "  " << name << ": " << (duration / 1000000) << " ms\n";
        }
    }

private:
    struct MemoryStats {
        size_t allocation_count = 0;
        size_t total_bytes = 0;
        size_t peak_usage = 0;
    } memory_stats_;

    struct ThreadStats {
        size_t event_count = 0;
        uint64_t total_time = 0;
    };
    std::unordered_map<uint64_t, ThreadStats> thread_stats_;

    std::unordered_map<std::string, uint64_t> event_durations_;
};
```

### Export to External Tools

#### Chrome Tracing Format

```cpp
void export_to_chrome_tracing(const traceme_recorder::Events& events) {
    json trace_data;
    trace_data["traceEvents"] = json::array();
    trace_data["displayTimeUnit"] = "ns";

    for (const auto& thread_events : events) {
        for (const auto& event : thread_events.events) {
            if (event.is_complete()) {
                json trace_event;

                // Parse metadata from event name
                auto [base_name, metadata] = parse_event_name(event.name);
                trace_event["name"] = base_name;
                trace_event["cat"] = categorize_event(base_name);
                trace_event["ph"] = "X";  // Complete event
                trace_event["ts"] = event.start_time / 1000;  // Convert to microseconds
                trace_event["dur"] = (event.end_time - event.start_time) / 1000;
                trace_event["pid"] = 1;
                trace_event["tid"] = thread_events.thread.tid;

                // Add metadata as arguments
                if (!metadata.empty()) {
                    trace_event["args"] = metadata;
                }

                trace_data["traceEvents"].push_back(trace_event);
            }
        }
    }

    // Write to file for Chrome tracing (chrome://tracing)
    std::ofstream file("trace.json");
    file << trace_data.dump(2);
}

std::pair<std::string, json> parse_event_name(const std::string& name) {
    // Parse format: "base_name#key1=value1,key2=value2#"
    auto hash_pos = name.find('#');
    if (hash_pos == std::string::npos) {
        return {name, json::object()};
    }

    std::string base_name = name.substr(0, hash_pos);
    std::string metadata_str = name.substr(hash_pos + 1);

    // Remove trailing '#'
    if (!metadata_str.empty() && metadata_str.back() == '#') {
        metadata_str.pop_back();
    }

    json metadata = json::object();

    // Parse key=value pairs
    std::istringstream ss(metadata_str);
    std::string pair;
    while (std::getline(ss, pair, ',')) {
        auto eq_pos = pair.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = pair.substr(0, eq_pos);
            std::string value = pair.substr(eq_pos + 1);
            metadata[key] = value;
        }
    }

    return {base_name, metadata};
}
```

#### TensorBoard Profiler Integration

```cpp
void export_to_tensorboard(const traceme_recorder::Events& events,
                          const std::string& logdir) {
    // Create XSpace for TensorBoard profiler
    x_space space;
    xplane* host_plane = space.add_planes();
    host_plane->set_name("Host Threads");

    // Convert events to XPlane format
    for (const auto& thread_events : events) {
        xline* line = host_plane->GetOrCreateLine(thread_events.thread.tid);
        line->SetName(thread_events.thread.name);

        for (const auto& event : thread_events.events) {
            if (event.is_complete()) {
                // Create XEvent
                xevent* xevent = line->AddEvent(
                    *host_plane->GetOrCreateEventMetadata(event.name));
                xevent->SetTimestampNs(event.start_time);
                xevent->SetDurationNs(event.end_time - event.start_time);

                // Add TensorFlow-specific metadata
                add_tensorflow_metadata(event, xevent, host_plane);
            }
        }
    }

    // Write XSpace to TensorBoard log directory
    write_xspace_to_tensorboard(space, logdir);
}

void add_tensorflow_metadata(const traceme_recorder::Event& event,
                            xevent* xevent, xplane* plane) {
    // Parse metadata from event name
    auto metadata = parse_traceme_metadata(event.name);

    for (const auto& [key, value] : metadata) {
        // Map common keys to TensorFlow profiler schema
        if (key == "tf_op") {
            xevent->AddStatValue(*plane->GetOrCreateStatMetadata("tf_op"), value);
        } else if (key == "bytes") {
            xevent->AddStatValue(*plane->GetOrCreateStatMetadata("bytes_accessed"),
                               std::stoull(value));
        } else if (key == "flops") {
            xevent->AddStatValue(*plane->GetOrCreateStatMetadata("flops"),
                               std::stoull(value));
        } else {
            // Generic metadata
            xevent->AddStatValue(*plane->GetOrCreateStatMetadata(key), value);
        }
    }
}
```

#### Performance Monitoring Dashboard

```cpp
class PerformanceMonitor {
public:
    void start_continuous_profiling() {
        monitoring_thread_ = std::thread([this]() {
            while (monitoring_active_) {
                // Short profiling burst
                traceme_recorder::start(2);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                auto events = traceme_recorder::stop();

                // Process and store metrics
                update_performance_metrics(events);

                // Wait before next profiling session
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        });
    }

    void update_performance_metrics(const traceme_recorder::Events& events) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        for (const auto& thread_events : events) {
            for (const auto& event : thread_events.events) {
                if (event.is_complete()) {
                    auto duration = event.end_time - event.start_time;

                    // Update rolling averages
                    auto& stats = function_stats_[event.name];
                    stats.total_time += duration;
                    stats.call_count++;
                    stats.avg_time = stats.total_time / stats.call_count;
                    stats.max_time = std::max(stats.max_time, duration);

                    // Detect performance anomalies
                    if (duration > stats.avg_time * 3) {
                        performance_alerts_.push_back({
                            .function_name = event.name,
                            .duration = duration,
                            .expected_duration = stats.avg_time,
                            .timestamp = event.start_time
                        });
                    }
                }
            }
        }

        // Trim old alerts
        auto cutoff = get_current_time_nanos() - (60 * 1000000000ULL); // 1 minute
        performance_alerts_.erase(
            std::remove_if(performance_alerts_.begin(), performance_alerts_.end(),
                          [cutoff](const auto& alert) { return alert.timestamp < cutoff; }),
            performance_alerts_.end());
    }

    void print_dashboard() {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::cout << "\n=== Performance Dashboard ===\n";

        // Top functions by total time
        std::vector<std::pair<std::string, FunctionStats>> sorted_functions;
        for (const auto& [name, stats] : function_stats_) {
            sorted_functions.emplace_back(name, stats);
        }
        std::sort(sorted_functions.begin(), sorted_functions.end(),
                 [](const auto& a, const auto& b) {
                     return a.second.total_time > b.second.total_time;
                 });

        std::cout << "Top Functions by Total Time:\n";
        for (size_t i = 0; i < std::min(size_t(10), sorted_functions.size()); ++i) {
            const auto& [name, stats] = sorted_functions[i];
            std::cout << "  " << name << ": "
                      << (stats.total_time / 1000000) << " ms total, "
                      << stats.call_count << " calls, "
                      << (stats.avg_time / 1000) << " μs avg\n";
        }

        // Recent performance alerts
        if (!performance_alerts_.empty()) {
            std::cout << "\nRecent Performance Alerts:\n";
            for (const auto& alert : performance_alerts_) {
                std::cout << "  " << alert.function_name
                          << " took " << (alert.duration / 1000) << " μs "
                          << "(expected " << (alert.expected_duration / 1000) << " μs)\n";
            }
        }
    }

private:
    struct FunctionStats {
        uint64_t total_time = 0;
        uint64_t max_time = 0;
        uint64_t avg_time = 0;
        size_t call_count = 0;
    };

    struct PerformanceAlert {
        std::string function_name;
        uint64_t duration;
        uint64_t expected_duration;
        uint64_t timestamp;
    };

    std::unordered_map<std::string, FunctionStats> function_stats_;
    std::vector<PerformanceAlert> performance_alerts_;
    std::mutex metrics_mutex_;
    std::thread monitoring_thread_;
    std::atomic<bool> monitoring_active_{false};
};
```

### Suggested Profiling Enhancements

#### 1. Enhanced Metadata Collection

```cpp
// Current: Basic metadata
traceme trace([&]() {
    return traceme_encode("matrix_multiply", {
        {"rows", rows}, {"cols", cols}
    });
});

// Enhanced: Rich contextual metadata
traceme trace([&]() {
    return traceme_encode("matrix_multiply", {
        {"input_shape", format_shape(input.shape())},
        {"output_shape", format_shape(output.shape())},
        {"data_type", input.dtype_name()},
        {"memory_layout", input.layout_name()},
        {"compute_capability", get_device_capability()},
        {"thread_pool_size", get_thread_pool_size()},
        {"cache_hit_rate", get_cache_statistics().hit_rate},
        {"memory_bandwidth_utilization", get_memory_bandwidth()},
        {"estimated_flops", estimate_operation_flops(input, output)}
    });
});
```

#### 2. Hierarchical Event Correlation

```cpp
class HierarchicalTracer {
public:
    void begin_operation(const std::string& name) {
        auto trace_id = traceme::activity_start(name);
        trace_stack_.push({trace_id, name, get_current_time_nanos()});
    }

    void end_operation() {
        if (!trace_stack_.empty()) {
            auto trace_info = trace_stack_.top();
            trace_stack_.pop();

            traceme::activity_end(trace_info.id);

            // Add hierarchical metadata
            if (!trace_stack_.empty()) {
                auto parent_name = trace_stack_.top().name;
                traceme::instant_activity([&]() {
                    return traceme_encode("hierarchy_info", {
                        {"child", trace_info.name},
                        {"parent", parent_name},
                        {"nesting_level", trace_stack_.size()}
                    });
                });
            }
        }
    }

private:
    struct TraceInfo {
        int64_t id;
        std::string name;
        uint64_t start_time;
    };
    std::stack<TraceInfo> trace_stack_;
};
```

#### 3. Adaptive Sampling

```cpp
class AdaptiveProfiler {
public:
    void configure_adaptive_sampling() {
        // High-frequency sampling for critical operations
        set_sampling_rate("memory_allocation", 1.0);  // 100%
        set_sampling_rate("gpu_kernel_launch", 1.0);  // 100%

        // Medium-frequency sampling for important operations
        set_sampling_rate("matrix_operations", 0.1);  // 10%
        set_sampling_rate("data_loading", 0.05);      // 5%

        // Low-frequency sampling for routine operations
        set_sampling_rate("utility_functions", 0.01); // 1%
    }

    bool should_trace(const std::string& operation_type) {
        auto it = sampling_rates_.find(operation_type);
        if (it == sampling_rates_.end()) {
            return false;  // Unknown operations not traced by default
        }

        return random_generator_() < it->second;
    }

private:
    std::unordered_map<std::string, double> sampling_rates_;
    std::uniform_real_distribution<double> random_generator_{0.0, 1.0};
};
```

#### 4. Real-Time Performance Alerts

```cpp
class RealTimeProfiler {
public:
    void setup_performance_thresholds() {
        // Configure performance thresholds
        thresholds_["memory_allocation"] = 1000000;  // 1ms
        thresholds_["gpu_kernel_launch"] = 5000000;  // 5ms
        thresholds_["matrix_multiply"] = 10000000;   // 10ms

        // Start background monitoring
        start_monitoring_thread();
    }

    void check_performance_threshold(const traceme_recorder::Event& event) {
        auto duration = event.end_time - event.start_time;
        auto base_name = extract_base_name(event.name);

        auto it = thresholds_.find(base_name);
        if (it != thresholds_.end() && duration > it->second) {
            // Performance threshold exceeded
            trigger_alert({
                .operation = base_name,
                .duration = duration,
                .threshold = it->second,
                .metadata = extract_metadata(event.name),
                .timestamp = event.start_time,
                .thread_id = get_current_thread_id()
            });
        }
    }

    void trigger_alert(const PerformanceAlert& alert) {
        // Log alert
        XSIGMA_LOG_WARNING("Performance threshold exceeded: {} took {}μs (threshold: {}μs)",
                          alert.operation, alert.duration / 1000, alert.threshold / 1000);

        // Send to monitoring system
        send_to_monitoring_system(alert);

        // Optionally trigger detailed profiling
        if (alert.duration > alert.threshold * 5) {
            trigger_detailed_profiling(alert.operation);
        }
    }

private:
    std::unordered_map<std::string, uint64_t> thresholds_;
    std::queue<PerformanceAlert> recent_alerts_;
};
```

#### 5. Cross-Platform Profiling Integration

```cpp
class CrossPlatformProfiler {
public:
    void export_to_multiple_formats(const traceme_recorder::Events& events) {
        // Export to Chrome Tracing
        export_to_chrome_tracing(events, "trace.json");

        // Export to TensorBoard
        export_to_tensorboard(events, "./tensorboard_logs");

        // Export to Intel VTune format
        export_to_vtune(events, "vtune_trace.itt");

        // Export to NVIDIA Nsight format
        export_to_nsight(events, "nsight_trace.nvvp");

        // Export to custom analysis format
        export_to_custom_format(events, "analysis.csv");
    }

    void export_to_custom_format(const traceme_recorder::Events& events,
                                const std::string& filename) {
        std::ofstream file(filename);
        file << "thread_id,event_name,start_time,duration,metadata\n";

        for (const auto& thread_events : events) {
            for (const auto& event : thread_events.events) {
                if (event.is_complete()) {
                    auto duration = event.end_time - event.start_time;
                    auto metadata = extract_metadata_as_json(event.name);

                    file << thread_events.thread.tid << ","
                         << "\"" << event.name << "\","
                         << event.start_time << ","
                         << duration << ","
                         << "\"" << metadata << "\"\n";
                }
            }
        }
    }
};
```

### Integration Best Practices

1. **Minimize Profiling Overhead**
   - Use appropriate trace levels to control event volume
   - Implement adaptive sampling for high-frequency operations
   - Leverage lambda constructors to defer expensive name generation

2. **Structured Metadata Design**
   - Use consistent key naming conventions across the codebase
   - Include relevant contextual information (sizes, types, configurations)
   - Add performance-relevant metrics (FLOPS, memory bandwidth, cache hits)

3. **Profiling Session Management**
   - Keep profiling sessions short to minimize memory usage
   - Implement automatic session rotation for long-running applications
   - Provide clear start/stop APIs for integration with external tools

4. **Data Analysis Pipeline**
   - Process events immediately after collection to reduce memory usage
   - Implement streaming analysis for real-time monitoring
   - Store aggregated metrics rather than raw events for long-term analysis

5. **Cross-Component Integration**
   - Coordinate profiling across CPU, GPU, and I/O subsystems
   - Implement unified timeline correlation for heterogeneous systems
   - Provide consistent metadata schemas across different components
```

## Common Pitfalls and Solutions

### 1. Expensive Name Generation

**Problem**: Creating trace names even when tracing is disabled
```cpp
// BAD: Always creates expensive string
traceme trace(format_complex_name(data));
```

**Solution**: Use lambda constructors
```cpp
// GOOD: Only creates string when tracing is active
traceme trace([&]() { return format_complex_name(data); });
```

### 2. Cross-Thread Activity Management

**Problem**: Starting activity in one thread, ending in another
```cpp
// BAD: Cross-thread activity
auto id = traceme::activity_start("task");
std::async([id]() {
    // ... work ...
    traceme::activity_end(id);  // Wrong thread!
});
```

**Solution**: Use separate traces per thread
```cpp
// GOOD: Each thread has its own trace
std::async([]() {
    traceme trace("async_task");
    // ... work ...
});
```

### 3. Forgetting to Stop Manual Activities

**Problem**: Memory leaks from unmatched activity_start calls
```cpp
// BAD: No matching activity_end
auto id = traceme::activity_start("operation");
if (error_condition) {
    return;  // Leaked activity!
}
traceme::activity_end(id);
```

**Solution**: Use RAII or ensure all paths call activity_end
```cpp
// GOOD: RAII ensures cleanup
traceme trace("operation");
if (error_condition) {
    return;  // Automatically cleaned up
}
```

### 4. Inappropriate Trace Levels

**Problem**: Using wrong levels causes noise or missing important events
```cpp
// BAD: Important operation at verbose level
traceme trace("critical_algorithm", 3);  // Might be filtered out

// BAD: Trivial operation at critical level
traceme trace("increment_counter", 1);   // Creates noise
```

**Solution**: Use appropriate levels for event importance
```cpp
// GOOD: Match level to importance
traceme trace("critical_algorithm", 1);  // Always captured
traceme trace("increment_counter", 3);   // Only in verbose mode
```

## Troubleshooting

### No Events Recorded

1. **Check if tracing was started**: Ensure `traceme_recorder::start()` was called
2. **Verify trace levels**: Events above the started level are filtered out
3. **Confirm platform support**: Mobile platforms may have tracing disabled

### Performance Issues

1. **Reduce trace frequency**: Use higher levels for frequent operations
2. **Optimize name generation**: Use lambda constructors for expensive names
3. **Check metadata complexity**: Avoid expensive computations in metadata

### Memory Usage

1. **Monitor active traces**: Long-running traces consume memory
2. **Regular collection**: Call `traceme_recorder::stop()` periodically
3. **Appropriate levels**: Higher levels generate more events

### Thread Safety Issues

1. **Same-thread activities**: Ensure activity_start/end from same thread
2. **Trace object ownership**: Don't share trace objects between threads
3. **Metadata thread safety**: Ensure captured variables are thread-safe

## Advanced Topics

### Custom Event Processing

```cpp
class CustomEventProcessor {
public:
    void process_events(const traceme_recorder::Events& events) {
        for (const auto& thread_events : events) {
            process_thread_events(thread_events);
        }
    }
    
private:
    void process_thread_events(const traceme_recorder::ThreadEvents& thread_events) {
        // Build call stack from nested events
        std::stack<const traceme_recorder::Event*> call_stack;
        
        for (const auto& event : thread_events.events) {
            if (event.is_complete()) {
                // Process complete event
                analyze_event_duration(event);
                
                // Check for nested events
                update_call_hierarchy(event, call_stack);
            }
        }
    }
    
    void analyze_event_duration(const traceme_recorder::Event& event) {
        auto duration_ns = event.end_time - event.start_time;
        
        // Collect statistics
        event_stats_[event.name].total_time += duration_ns;
        event_stats_[event.name].call_count++;
        event_stats_[event.name].max_time = 
            std::max(event_stats_[event.name].max_time, duration_ns);
    }
    
    struct EventStats {
        int64_t total_time = 0;
        int64_t max_time = 0;
        size_t call_count = 0;
    };
    
    std::unordered_map<std::string, EventStats> event_stats_;
};
```

### Integration with Logging System

The TraceMe system is designed to complement, not replace, the existing logging infrastructure. See the [Logger Integration](#logger-integration) section for details on how these systems work together.

### Performance Monitoring

```cpp
class PerformanceMonitor {
public:
    void start_monitoring() {
        monitoring_thread_ = std::thread([this]() {
            while (monitoring_active_) {
                // Start short profiling session
                traceme_recorder::start(2);
                
                std::this_thread::sleep_for(std::chrono::seconds(1));
                
                auto events = traceme_recorder::stop();
                analyze_performance(events);
                
                std::this_thread::sleep_for(std::chrono::seconds(9));
            }
        });
    }
    
    void stop_monitoring() {
        monitoring_active_ = false;
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }
    
private:
    void analyze_performance(const traceme_recorder::Events& events) {
        // Detect performance anomalies
        // Update performance metrics
        // Trigger alerts if needed
    }
    
    std::atomic<bool> monitoring_active_{false};
    std::thread monitoring_thread_;
};
```

## Conclusion

The XSigma TraceMe system provides a powerful, efficient foundation for application profiling and performance analysis. Its zero-cost design when disabled, combined with rich metadata support and thread-safe operation, makes it suitable for both development debugging and production monitoring.

Key takeaways:
- Use RAII-based tracing for most scenarios
- Leverage lambda constructors for expensive operations
- Choose appropriate trace levels for your use case
- Integrate with existing profiling and monitoring tools
- Follow thread safety guidelines for reliable operation

## Logger Integration

The TraceMe system is designed to work alongside XSigma's logging infrastructure, providing complementary functionality while maintaining clear separation of concerns.

### Architectural Separation

| System | Purpose | Use Cases | Output Format |
|--------|---------|-----------|---------------|
| **TraceMe** | Performance profiling, timing analysis | Algorithm optimization, bottleneck identification, flow analysis | Structured events for profilers |
| **Logger** | Error reporting, debug messages, application state | Debugging, monitoring, error tracking | Human-readable log messages |

### Design Principles

1. **Separation of Concerns**: TraceMe focuses on performance, Logger focuses on diagnostics
2. **Optional Integration**: Integration is opt-in to avoid performance impact
3. **Consistent APIs**: Both systems use similar patterns (RAII, levels, metadata)
4. **Cross-Platform**: Works with all logging backends (LOGURU, GLOG, NATIVE)

### Integration Architecture

```cpp
// Optional bridge between traceme_recorder and logger
class TracemeLoggerBridge {
public:
    // Enable/disable trace event logging
    static void enable_trace_logging(bool enabled, logger_verbosity_enum level = VERBOSITY_INFO);

    // Callback for trace events (called by traceme_recorder)
    static void on_trace_event(const traceme_recorder::Event& event);

    // Configure which events to log
    static void set_event_filter(std::function<bool(const traceme_recorder::Event&)> filter);

private:
    static std::atomic<bool> logging_enabled_;
    static logger_verbosity_enum log_level_;
    static std::function<bool(const traceme_recorder::Event&)> event_filter_;
};
```

### Implementation Design

#### 1. Optional Callback Integration

```cpp
// In traceme_recorder.h - add optional logging callback
class traceme_recorder {
public:
    // Existing methods...

    /**
     * @brief Sets optional callback for trace event logging.
     * @param callback Function to call for each recorded event (nullptr to disable)
     * @param level Minimum trace level to trigger logging
     */
    static void set_logging_callback(
        std::function<void(const Event&)> callback,
        int level = 1
    );

private:
    static std::function<void(const Event&)> logging_callback_;
    static int logging_callback_level_;
};

// In traceme_recorder.cxx - implementation
std::function<void(const traceme_recorder::Event&)> traceme_recorder::logging_callback_;
int traceme_recorder::logging_callback_level_ = 1;

void traceme_recorder::set_logging_callback(
    std::function<void(const Event&)> callback,
    int level) {
    logging_callback_ = std::move(callback);
    logging_callback_level_ = level;
}

void traceme_recorder::record(Event&& event) {
    // Existing recording logic
    PerThread<ThreadLocalRecorder>::Get().Record(std::move(event));

    // Optional logging callback
    if (logging_callback_ && event.level <= logging_callback_level_) {
        logging_callback_(event);
    }
}
```

#### 2. Bridge Implementation

```cpp
// traceme_logger_bridge.h
#pragma once
#include "logging/logger.h"
#include "logging/tracing/traceme_recorder.h"

namespace xsigma {

/**
 * @brief Optional bridge between TraceMe and Logger systems.
 *
 * Provides integration between performance tracing and diagnostic logging
 * while maintaining separation of concerns. Integration is opt-in and
 * configurable to minimize performance impact.
 */
class XSIGMA_VISIBILITY TracemeLoggerBridge {
public:
    /**
     * @brief Enables logging of trace events to the logger system.
     * @param enabled Whether to enable trace event logging
     * @param level Logger verbosity level for trace events
     * @param trace_level Minimum trace level to log (1=CRITICAL, 2=INFO, 3+=VERBOSE)
     */
    static void enable_trace_logging(
        bool enabled,
        logger_verbosity_enum level = logger_verbosity_enum::VERBOSITY_INFO,
        int trace_level = 2
    );

    /**
     * @brief Sets a custom filter for which trace events to log.
     * @param filter Function returning true for events that should be logged
     */
    static void set_event_filter(std::function<bool(const traceme_recorder::Event&)> filter);

    /**
     * @brief Configures trace event logging format.
     * @param include_metadata Whether to include parsed metadata in log messages
     * @param include_timing Whether to include timing information
     * @param include_thread_info Whether to include thread information
     */
    static void configure_logging_format(
        bool include_metadata = true,
        bool include_timing = true,
        bool include_thread_info = false
    );

private:
    // Callback function for traceme_recorder
    static void on_trace_event(const traceme_recorder::Event& event);

    // Parse metadata from trace event name
    static std::unordered_map<std::string, std::string> parse_metadata(const std::string& name);

    // Format trace event for logging
    static std::string format_trace_event(const traceme_recorder::Event& event);

    static std::atomic<bool> logging_enabled_;
    static logger_verbosity_enum log_level_;
    static int trace_level_;
    static std::function<bool(const traceme_recorder::Event&)> event_filter_;
    static bool include_metadata_;
    static bool include_timing_;
    static bool include_thread_info_;
};

} // namespace xsigma
```

#### 3. Bridge Implementation

```cpp
// traceme_logger_bridge.cxx
#include "logging/tracing/traceme_logger_bridge.h"
#include "logging/logger.h"
#include <sstream>
#include <regex>

namespace xsigma {

// Static member definitions
std::atomic<bool> TracemeLoggerBridge::logging_enabled_{false};
logger_verbosity_enum TracemeLoggerBridge::log_level_{logger_verbosity_enum::VERBOSITY_INFO};
int TracemeLoggerBridge::trace_level_{2};
std::function<bool(const traceme_recorder::Event&)> TracemeLoggerBridge::event_filter_;
bool TracemeLoggerBridge::include_metadata_{true};
bool TracemeLoggerBridge::include_timing_{true};
bool TracemeLoggerBridge::include_thread_info_{false};

void TracemeLoggerBridge::enable_trace_logging(
    bool enabled,
    logger_verbosity_enum level,
    int trace_level) {

    logging_enabled_ = enabled;
    log_level_ = level;
    trace_level_ = trace_level;

    if (enabled) {
        // Register callback with traceme_recorder
        traceme_recorder::set_logging_callback(
            [](const traceme_recorder::Event& event) {
                on_trace_event(event);
            },
            trace_level
        );

        XSIGMA_LOG_INFO("TracemeLoggerBridge enabled (level={}, trace_level={})",
                       static_cast<int>(level), trace_level);
    } else {
        // Disable callback
        traceme_recorder::set_logging_callback(nullptr);
        XSIGMA_LOG_INFO("TracemeLoggerBridge disabled");
    }
}

void TracemeLoggerBridge::set_event_filter(
    std::function<bool(const traceme_recorder::Event&)> filter) {
    event_filter_ = std::move(filter);
}

void TracemeLoggerBridge::configure_logging_format(
    bool include_metadata,
    bool include_timing,
    bool include_thread_info) {

    include_metadata_ = include_metadata;
    include_timing_ = include_timing;
    include_thread_info_ = include_thread_info;
}

void TracemeLoggerBridge::on_trace_event(const traceme_recorder::Event& event) {
    if (!logging_enabled_) {
        return;
    }

    // Apply custom filter if set
    if (event_filter_ && !event_filter_(event)) {
        return;
    }

    // Format and log the event
    std::string formatted_event = format_trace_event(event);
    logger::Log(log_level_, __FILE__, __LINE__, formatted_event.c_str());
}

std::unordered_map<std::string, std::string> TracemeLoggerBridge::parse_metadata(
    const std::string& name) {

    std::unordered_map<std::string, std::string> metadata;

    // Parse format: "base_name#key1=value1,key2=value2#"
    auto hash_pos = name.find('#');
    if (hash_pos == std::string::npos) {
        return metadata; // No metadata
    }

    std::string metadata_str = name.substr(hash_pos + 1);

    // Remove trailing '#'
    if (!metadata_str.empty() && metadata_str.back() == '#') {
        metadata_str.pop_back();
    }

    // Parse key=value pairs
    std::istringstream ss(metadata_str);
    std::string pair;
    while (std::getline(ss, pair, ',')) {
        auto eq_pos = pair.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = pair.substr(0, eq_pos);
            std::string value = pair.substr(eq_pos + 1);
            metadata[key] = value;
        }
    }

    return metadata;
}

std::string TracemeLoggerBridge::format_trace_event(const traceme_recorder::Event& event) {
    std::ostringstream oss;

    // Extract base name and metadata
    auto hash_pos = event.name.find('#');
    std::string base_name = (hash_pos != std::string::npos) ?
        event.name.substr(0, hash_pos) : event.name;

    oss << "[TRACE] " << base_name;

    // Add timing information
    if (include_timing_ && event.is_complete()) {
        auto duration_us = (event.end_time - event.start_time) / 1000;
        oss << " (" << duration_us << "μs)";
    }

    // Add thread information
    if (include_thread_info_) {
        oss << " [tid=" << std::this_thread::get_id() << "]";
    }

    // Add metadata
    if (include_metadata_ && hash_pos != std::string::npos) {
        auto metadata = parse_metadata(event.name);
        if (!metadata.empty()) {
            oss << " {";
            bool first = true;
            for (const auto& [key, value] : metadata) {
                if (!first) oss << ", ";
                oss << key << "=" << value;
                first = false;
            }
            oss << "}";
        }
    }

    return oss.str();
}

} // namespace xsigma
```

### Usage Examples

#### 1. Basic Integration

```cpp
#include "logging/tracing/traceme_logger_bridge.h"

int main() {
    // Initialize logging
    xsigma::logger::Init(argc, argv);

    // Enable trace event logging at INFO level for trace levels 1-2
    TracemeLoggerBridge::enable_trace_logging(true, logger_verbosity_enum::VERBOSITY_INFO, 2);

    // Start profiling session
    traceme_recorder::start(2);

    // Your application code with traceme instrumentation
    run_application();

    // Stop profiling
    auto events = traceme_recorder::stop();

    // Disable trace logging
    TracemeLoggerBridge::enable_trace_logging(false);

    return 0;
}
```

#### 2. Filtered Integration

```cpp
void setup_selective_trace_logging() {
    // Only log memory allocation and GPU-related traces
    TracemeLoggerBridge::set_event_filter([](const traceme_recorder::Event& event) {
        return event.name.find("allocate") != std::string::npos ||
               event.name.find("gpu") != std::string::npos ||
               event.name.find("cuda") != std::string::npos;
    });

    // Configure minimal logging format for performance
    TracemeLoggerBridge::configure_logging_format(
        true,   // include_metadata
        true,   // include_timing
        false   // include_thread_info (expensive)
    );

    TracemeLoggerBridge::enable_trace_logging(true, logger_verbosity_enum::VERBOSITY_WARNING);
}
```

#### 3. Debug Integration

```cpp
void debug_with_integrated_tracing() {
    traceme trace("debug_operation");

    XSIGMA_LOG_INFO("Starting debug operation");

    try {
        // ... operation logic ...

        trace.append_metadata([&]() {
            return traceme_encode({
                {"status", "success"},
                {"items_processed", item_count}
            });
        });

        XSIGMA_LOG_INFO("Debug operation completed successfully");
    }
    catch (const std::exception& e) {
        trace.append_metadata([&]() {
            return traceme_encode({
                {"status", "error"},
                {"error_type", typeid(e).name()},
                {"error_message", e.what()}
            });
        });

        XSIGMA_LOG_ERROR("Debug operation failed: {}", e.what());
        throw;
    }
}

// With TracemeLoggerBridge enabled, this produces logs like:
// [INFO] Starting debug operation
// [INFO] [TRACE] debug_operation (1250μs) {status=success, items_processed=42}
// [INFO] Debug operation completed successfully
```

#### 4. Performance Monitoring Integration

```cpp
class PerformanceLogger {
public:
    void setup_performance_monitoring() {
        // Filter for performance-critical operations only
        TracemeLoggerBridge::set_event_filter([](const traceme_recorder::Event& event) {
            if (!event.is_complete()) return false;

            auto duration_ms = (event.end_time - event.start_time) / 1000000;

            // Log operations that take longer than thresholds
            if (event.name.find("memory_allocation") != std::string::npos) {
                return duration_ms > 1; // 1ms threshold for allocations
            }
            if (event.name.find("gpu_kernel") != std::string::npos) {
                return duration_ms > 5; // 5ms threshold for GPU kernels
            }
            if (event.name.find("matrix_") != std::string::npos) {
                return duration_ms > 10; // 10ms threshold for matrix operations
            }

            return duration_ms > 100; // 100ms threshold for other operations
        });

        TracemeLoggerBridge::enable_trace_logging(
            true,
            logger_verbosity_enum::VERBOSITY_WARNING  // Use WARNING for performance issues
        );
    }
};
```

### Integration Benefits

1. **Unified Debugging**: See both diagnostic logs and performance traces in one stream
2. **Performance Alerting**: Automatically log slow operations for investigation
3. **Contextual Information**: Correlate performance issues with application state
4. **Flexible Filtering**: Choose which trace events are worth logging
5. **Cross-Platform**: Works with all XSigma logging backends

### Performance Considerations

1. **Opt-In Design**: Integration is disabled by default to avoid overhead
2. **Efficient Filtering**: Event filtering happens before expensive formatting
3. **Minimal Allocation**: String formatting only occurs for events that pass filters
4. **Configurable Detail**: Control metadata inclusion to balance detail vs. performance
5. **Async Logging**: Leverage logger's async capabilities where available (LOGURU)

## Comparison with back_trace

The XSigma codebase includes both TraceMe and back_trace systems, each serving different purposes:

### Functional Differences

| Aspect | TraceMe | back_trace |
|--------|---------|------------|
| **Primary Purpose** | Performance profiling, event tracing | Error diagnostics, stack unwinding |
| **Use Case** | Understanding program flow and timing | Debugging crashes and errors |
| **Data Collected** | Timing, metadata, hierarchical events | Call stack, function names, addresses |
| **Performance Impact** | Minimal (designed for production) | Higher (detailed stack analysis) |
| **Activation** | Session-based (start/stop) | On-demand (error conditions) |
| **Thread Safety** | Lock-free, thread-local storage | Thread-safe but may block |
| **Output Format** | Structured events for profilers | Human-readable stack traces |

### When to Use Each System

#### Use TraceMe For:
- **Performance Analysis**: Understanding where time is spent
- **Algorithm Profiling**: Measuring computational efficiency
- **Flow Tracing**: Following execution paths through complex systems
- **Production Monitoring**: Lightweight performance tracking
- **GPU/CPU Correlation**: Understanding synchronization points

```cpp
// TraceMe: Performance analysis
void matrix_multiply(const Matrix& a, const Matrix& b) {
    traceme trace([&]() {
        return traceme_encode("matrix_multiply", {
            {"rows_a", a.rows()}, {"cols_a", a.cols()},
            {"rows_b", b.rows()}, {"cols_b", b.cols()}
        });
    });

    // ... matrix multiplication logic ...
}
```

#### Use back_trace For:
- **Error Diagnostics**: Understanding how errors occurred
- **Crash Analysis**: Debugging segmentation faults and exceptions
- **Call Stack Analysis**: Seeing the sequence of function calls
- **Debug Builds**: Detailed debugging information
- **Exception Handling**: Providing context in error messages

```cpp
// back_trace: Error diagnostics
void risky_operation() {
    try {
        // ... potentially failing operation ...
    }
    catch (const std::exception& e) {
        auto stack_trace = back_trace::print();
        XSIGMA_LOG_ERROR("Operation failed: {}\nStack trace:\n{}",
                         e.what(), stack_trace);
        throw;
    }
}
```

### Complementary Usage

Both systems can be used together for comprehensive analysis:

```cpp
void comprehensive_error_handling() {
    traceme trace("comprehensive_operation");

    try {
        // ... complex operation ...

        trace.append_metadata([&]() {
            return traceme_encode({{"status", "success"}});
        });
    }
    catch (const std::exception& e) {
        // Add error info to performance trace
        trace.append_metadata([&]() {
            return traceme_encode({
                {"status", "error"},
                {"error_message", e.what()}
            });
        });

        // Get detailed stack trace for debugging
        auto stack_trace = back_trace::print();
        XSIGMA_LOG_ERROR("Operation failed with stack trace:\n{}", stack_trace);

        throw;
    }
}
```

### Performance Comparison

| Metric | TraceMe | back_trace |
|--------|---------|------------|
| **Overhead (disabled)** | ~1-2 CPU cycles | N/A (on-demand) |
| **Overhead (enabled)** | ~50-100 ns | ~1-10 ms |
| **Memory per event** | ~64 bytes | ~1-5 KB |
| **Thread contention** | None (lock-free) | Minimal (symbol resolution) |
| **Production suitability** | High | Low (debug builds only) |

## Migration Guide

### From Manual Timing

```cpp
// OLD: Manual timing
auto start = std::chrono::high_resolution_clock::now();
perform_operation();
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
std::cout << "Operation took: " << duration.count() << " ns\n";

// NEW: TraceMe
traceme trace("perform_operation");
perform_operation();
// Timing is automatic, integrated with profiling tools
```

### From Printf Debugging

```cpp
// OLD: Printf debugging
printf("Entering function with param=%d\n", param);
result = complex_function(param);
printf("Function returned result=%d\n", result);

// NEW: TraceMe with metadata
traceme trace([&]() {
    return traceme_encode("complex_function", {{"param", param}});
});
result = complex_function(param);
trace.append_metadata([&]() {
    return traceme_encode({{"result", result}});
});
```

### From Custom Profiling

```cpp
// OLD: Custom profiling
class CustomProfiler {
    void start_timer(const std::string& name);
    void end_timer(const std::string& name);
    void print_results();
};

// NEW: TraceMe (standardized, more efficient)
// Just use traceme objects - collection and analysis handled automatically
traceme trace("operation_name");
```

## References and Further Reading

- **Source Code**:
  - `Library/Core/logging/tracing/traceme.h` - Main RAII tracing class
  - `Library/Core/logging/tracing/traceme_recorder.h` - Event collection system
  - `Library/Core/logging/tracing/traceme_encode.h` - Metadata encoding utilities

- **Test Examples**: `Library/Core/Testing/Cxx/TestTraceme.cxx`
- **Integration Examples**:
  - `Library/Core/memory/cpu/allocator_cpu.cxx` - Memory allocator tracing
  - `Library/Core/experimental/profiler/cpu/host_tracer.cxx` - Profiler integration

- **Related Documentation**:
  - `docs/logging-system.md` - XSigma logging infrastructure
  - `docs/usage-examples.md` - General XSigma usage patterns
  - `Library/Core/experimental/profiler/README.md` - Enhanced profiler system

For more advanced usage patterns and integration examples, see the source code in `Library/Core/Testing/Cxx/TestTraceme.cxx` and the memory allocator integration in `Library/Core/memory/cpu/allocator_cpu.cxx`.
