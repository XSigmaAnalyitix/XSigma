# XSigma Profiler System - Comprehensive Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Classification](#component-classification)
3. [Design Patterns](#design-patterns)
4. [Data Flow](#data-flow)
5. [Integration with TraceMe](#integration-with-traceme)
6. [XPlane Format](#xplane-format)
7. [Usage Examples](#usage-examples)
8. [Performance Characteristics](#performance-characteristics)
9. [Platform Support](#platform-support)
10. [API Reference](#api-reference)

## Architecture Overview

The XSigma Profiler System is a comprehensive, modular performance analysis framework designed for high-performance applications. It provides multi-layered profiling capabilities with minimal runtime overhead.

### Core Architecture Principles

1. **Modular Design**: Components are organized into required (core) and optional (plugin) modules
2. **Zero-Cost Abstraction**: Profiling overhead is eliminated when disabled at compile time
3. **Thread Safety**: All components support concurrent profiling across multiple threads
4. **Cross-Platform**: Unified API with platform-specific optimizations
5. **Extensible**: Plugin architecture allows custom profilers and exporters

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (User Code with XSIGMA_PROFILE_SCOPE macros)             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Session Management                        │
│  profiler_session, profiler_scope, profiler_report        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Core Framework                          │
│  profiler_interface, profiler_controller, profiler_factory │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Plugin Implementations                    │
│     CPU Profiler  │  Memory Profiler  │  Custom Profilers │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Data Export & Analysis                    │
│  XPlane │ JSON │ CSV │ XML │ Statistical Analysis │ Visualization │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Platform Abstraction                     │
│        Windows │ Linux │ macOS │ Time APIs │ Memory APIs    │
└─────────────────────────────────────────────────────────────┘
```

## Component Classification

### Required Components (Core)
Essential components for basic profiling functionality:

#### Core Framework (`core/`)
- **profiler_interface.h**: Abstract profiler interface
- **profiler_controller.***: Profiler lifecycle management
- **profiler_factory.***: Factory pattern for profiler creation
- **profiler_collection.***: Managing multiple profilers
- **profiler_options.h**: Configuration options
- **profiler_lock.***: Thread synchronization primitives
- **timespan.h**: Time span utilities

#### Session Management (`session/`)
- **profiler.***: Main profiler session class
- **profiler_report.***: Basic report generation

#### Platform Abstraction (`platform/`)
- **env_time.***: Cross-platform high-resolution timing
- **env_time_win.cxx**: Windows-specific time implementation
- **env_var.***: Environment variable handling

### Optional Components (Extensions)
Additional features that can be disabled to reduce binary size:

#### CPU Profiling (`cpu/`)
- **host_tracer.***: Host CPU activity tracing
- **python_tracer.***: Python integration support
- **metadata_collector.***: Profiling metadata collection

#### Memory Profiling (`memory/`)
- **memory_tracker.***: Memory allocation tracking
- **scoped_memory_debug_annotation.***: Memory debugging annotations

#### Statistical Analysis (`analysis/`)
- **statistical_analyzer.***: Comprehensive statistical analysis
- **stats_calculator.***: Statistics computation algorithms

#### Data Export (`exporters/`)
- **xplane/**: XPlane format for TensorFlow integration
- **json_exporter.***: JSON format export
- **csv_exporter.***: CSV format export
- **xml_exporter.***: XML format export

#### Visualization (`visualization/`)
- **ascii_visualizer.***: ASCII-based charts and graphs
- **timeline_visualizer.***: Timeline visualization
- **flamegraph_exporter.***: Flamegraph generation

## Design Patterns

### 1. Factory Pattern
The profiler system uses the Factory pattern to create profiler instances:

```cpp
// Register profiler factories
register_profiler_factory([](const profile_options& options) {
    if (options.enable_cpu_profiling()) {
        return std::make_unique<host_tracer>(options.cpu_trace_level());
    }
    return nullptr;
});

// Create profilers based on options
auto profilers = create_profilers(options);
```

### 2. Builder Pattern
Configuration uses the Builder pattern for flexible setup:

```cpp
auto session = profiler_session_builder()
    .with_timing(true)
    .with_memory_tracking(true)
    .with_statistical_analysis(true)
    .with_output_format(profiler_options::output_format::json)
    .build();
```

### 3. RAII Pattern
Automatic scope-based profiling using RAII:

```cpp
{
    XSIGMA_PROFILE_SCOPE("critical_section");
    // Code is automatically profiled until scope ends
    perform_critical_work();
} // Profiling data automatically collected here
```

### 4. Observer Pattern
Statistical analysis components observe profiling events:

```cpp
statistical_analyzer analyzer;
session.add_observer(&analyzer);
// Analyzer automatically receives profiling events
```

### 5. Strategy Pattern
Different export formats implement a common interface:

```cpp
class exporter_interface {
public:
    virtual void export_data(const profiling_data& data, 
                           const std::string& filename) = 0;
};

class json_exporter : public exporter_interface { /* ... */ };
class csv_exporter : public exporter_interface { /* ... */ };
```

## Data Flow

### 1. Profiling Data Collection Flow

```
User Code
    │
    ▼
XSIGMA_PROFILE_SCOPE("name")
    │
    ▼
profiler_scope constructor
    │
    ▼
profiler_session::start_scope()
    │
    ▼
profiler_interface::start()
    │
    ▼
Platform-specific timing (env_time)
    │
    ▼
Thread-local data collection
    │
    ▼
profiler_scope destructor
    │
    ▼
profiler_session::end_scope()
    │
    ▼
profiler_interface::stop()
    │
    ▼
Data aggregation and storage
```

### 2. Report Generation Flow

```
profiler_session::generate_report()
    │
    ▼
profiler_interface::collect_data()
    │
    ▼
Data aggregation from all profilers
    │
    ▼
statistical_analyzer::analyze()
    │
    ▼
profiler_report::format_data()
    │
    ▼
Export to selected format(s)
    │
    ▼
Output files (JSON/CSV/XML/XPlane)
```

## Integration with TraceMe

The profiler system builds upon and extends the existing TraceMe infrastructure:

### Relationship
- **TraceMe**: Low-level, high-performance event recording
- **Profiler**: High-level session management and analysis built on TraceMe
- **Integration**: Profiler uses TraceMe for actual event collection

### Data Flow Integration
```cpp
// TraceMe provides the low-level event recording
{
    traceme trace("operation_name");
    // TraceMe records start time
    
    // Profiler adds higher-level session management
    XSIGMA_PROFILE_SCOPE("operation_name");
    
    perform_operation();
    
    // TraceMe records end time
    // Profiler aggregates data for analysis
}
```

### Benefits of Integration
1. **Consistency**: Same underlying timing mechanisms
2. **Performance**: Leverages TraceMe's optimized recording
3. **Compatibility**: Works with existing TraceMe instrumentation
4. **Flexibility**: Can use either system independently or together

## XPlane Format

### What is XPlane?
XPlane is a structured data format originally developed for TensorFlow profiling tools. It provides a standardized way to represent profiling data that can be consumed by various analysis and visualization tools.

### Why XPlane is Needed
1. **Standardization**: Common format across different profiling tools
2. **Tool Integration**: Compatible with TensorFlow Profiler, TensorBoard
3. **Rich Metadata**: Supports complex profiling metadata and relationships
4. **Scalability**: Efficient representation of large profiling datasets
5. **Extensibility**: Schema-based approach allows custom metadata types

### XPlane Data Structure
```
x_space (Root container)
├── xplane (Per-device or logical grouping)
│   ├── xline (Per-thread or execution context)
│   │   ├── xevent (Individual profiling events)
│   │   │   ├── metadata_id (Event type identifier)
│   │   │   ├── timestamp_ps (Start time in picoseconds)
│   │   │   ├── duration_ps (Duration in picoseconds)
│   │   │   └── stats (Key-value metadata)
│   │   └── ...
│   ├── event_metadata (Event type definitions)
│   └── stat_metadata (Metadata type definitions)
└── ...
```

### XPlane Components

#### xplane_builder.h/cxx
**Purpose**: Construction API for building XPlane data structures
**Key Classes**:
- `xplane_builder`: Main builder for XPlane objects
- `xline_builder`: Builder for timeline data
- `xevent_builder`: Builder for individual events

**Usage Example**:
```cpp
xplane_builder builder(&plane);
auto line = builder.get_or_create_line(thread_id);
auto event_metadata = builder.get_or_create_event_metadata("function_call");
auto event = line.add_event(*event_metadata);
event.SetTimestampPs(start_time);
event.SetDurationPs(duration);
```

#### xplane_visitor.h/cxx
**Purpose**: Traversal and reading API for XPlane data
**Key Classes**:
- `xplane_visitor`: Read-only access to XPlane data
- `xline_visitor`: Read-only access to timeline data
- `xevent_visitor`: Read-only access to individual events

**Usage Example**:
```cpp
xplane_visitor visitor(&plane);
visitor.for_each_line([](const xline_visitor& line) {
    line.for_each_event([](const xevent_visitor& event) {
        auto name = event.name();
        auto duration = event.duration_ps();
        // Process event data
    });
});
```

#### xplane_schema.h/cxx
**Purpose**: Schema definitions and metadata types
**Key Components**:
- `StatType`: Enumeration of standard metadata types
- `EventType`: Enumeration of standard event types
- Schema lookup functions for TensorFlow compatibility

**Standard Metadata Types**:
```cpp
enum StatType {
    kStepId,           // Training step identifier
    kDeviceOrdinal,    // Device number
    kBytesAllocated,   // Memory allocation size
    kTensorShapes,     // Tensor dimension information
    kFlops,            // Floating point operations count
    // ... many more standard types
};
```

#### xplane_utils.h/cxx
**Purpose**: Utility functions for XPlane manipulation
**Key Functions**:
- `NormalizeTimestamps()`: Adjust timestamps to relative values
- `MergePlanes()`: Combine multiple XPlane objects
- `GetSortedEvents()`: Extract and sort events by timestamp
- `AddFlowsToXplane()`: Add flow relationships between events

#### tf_xplane_visitor.h
**Purpose**: TensorFlow-specific XPlane integration
**Function**: Creates XPlane visitors configured for TensorFlow profiling data

### Data Flow: TraceMe → XPlane
```cpp
// 1. Collect TraceMe events
traceme_recorder::start(1);
{
    traceme trace("my_operation");
    perform_work();
}
auto events = traceme_recorder::stop();

// 2. Convert to XPlane format
x_space space;
xplane* host_plane = space.add_planes();
host_plane->set_name("Host");

xplane_builder builder(host_plane);
for (const auto& thread_events : events) {
    auto line = builder.get_or_create_line(thread_events.thread.tid);
    line.SetName(thread_events.thread.name);

    for (const auto& event : thread_events.events) {
        if (event.is_complete()) {
            auto* event_metadata = builder.get_or_create_event_metadata(event.name);
            auto xevent = line.add_event(*event_metadata);
            xevent.SetTimestampPs(event.start_time * 1000);  // ns to ps
            xevent.SetDurationPs((event.end_time - event.start_time) * 1000);
        }
    }
}

// 3. Export XPlane data
write_xplane_to_file(space, "profile.xplane");
```

### XPlane vs Other Formats

| Feature | XPlane | JSON | CSV | XML |
|---------|--------|------|-----|-----|
| **Metadata Support** | Rich, typed | Limited | Minimal | Good |
| **Tool Integration** | TensorFlow ecosystem | Universal | Spreadsheets | Various |
| **File Size** | Compact | Large | Medium | Large |
| **Human Readable** | No | Yes | Yes | Yes |
| **Streaming Support** | Yes | Limited | Yes | Limited |
| **Schema Validation** | Built-in | External | None | DTD/XSD |

### When to Use XPlane
- **TensorFlow Integration**: When using TensorFlow profiling tools
- **Large Datasets**: For efficient storage of extensive profiling data
- **Tool Interoperability**: When sharing data between different profiling tools
- **Rich Metadata**: When profiling data includes complex relationships and metadata

### Alternatives to XPlane
1. **JSON**: Human-readable, universally supported, larger file sizes
2. **CSV**: Simple, spreadsheet-compatible, limited metadata support
3. **Protocol Buffers**: Efficient binary format, requires schema definition
4. **Custom Binary**: Maximum efficiency, requires custom tooling

## Usage Examples

### Basic Profiling Session
```cpp
#include "experimental/profiler/session/profiler.h"

int main() {
    // Create and configure profiler session
    auto session = profiler_session_builder()
        .with_timing(true)
        .with_memory_tracking(false)
        .with_output_format(profiler_options::output_format::json)
        .build();

    // Start profiling
    session->start();

    // Profile application code
    {
        XSIGMA_PROFILE_SCOPE("main_computation");

        // Nested profiling
        {
            XSIGMA_PROFILE_SCOPE("data_loading");
            load_data();
        }

        {
            XSIGMA_PROFILE_SCOPE("processing");
            process_data();
        }

        {
            XSIGMA_PROFILE_SCOPE("output_generation");
            generate_output();
        }
    }

    // Stop profiling and generate report
    session->stop();
    session->export_report("profile_results.json");
    session->print_report();

    return 0;
}
```

### Advanced Profiling with Memory Tracking
```cpp
#include "experimental/profiler/session/profiler.h"
#include "experimental/profiler/memory/memory_tracker.h"
#include "experimental/profiler/analysis/statistical_analyzer.h"

void advanced_profiling_example() {
    // Configure comprehensive profiling
    auto session = profiler_session_builder()
        .with_timing(true)
        .with_memory_tracking(true)
        .with_statistical_analysis(true)
        .with_thread_safety(true)
        .with_output_format(profiler_options::output_format::json)
        .build();

    // Add statistical analyzer
    statistical_analyzer analyzer;
    session->add_observer(&analyzer);

    session->start();

    // Profile memory-intensive operations
    {
        XSIGMA_PROFILE_SCOPE("memory_intensive_work");

        std::vector<std::vector<double>> large_data;
        for (int i = 0; i < 1000; ++i) {
            XSIGMA_PROFILE_SCOPE("allocation_batch");
            large_data.emplace_back(10000, 3.14159);
        }

        // Process data
        {
            XSIGMA_PROFILE_SCOPE("data_processing");
            for (auto& batch : large_data) {
                std::sort(batch.begin(), batch.end());
            }
        }
    }

    session->stop();

    // Generate comprehensive reports
    session->export_report("detailed_profile.json");

    // Export statistical analysis
    auto timing_stats = analyzer.calculate_timing_stats("memory_intensive_work");
    std::cout << "Mean execution time: " << timing_stats.mean << " ms\n";
    std::cout << "Standard deviation: " << timing_stats.std_deviation << " ms\n";

    auto memory_stats = analyzer.calculate_memory_stats("allocation_batch");
    std::cout << "Peak memory usage: " << memory_stats.max_value << " bytes\n";
}
```

### Multi-threaded Profiling
```cpp
#include "experimental/profiler/session/profiler.h"
#include <thread>
#include <vector>

void multithreaded_profiling_example() {
    auto session = profiler_session_builder()
        .with_timing(true)
        .with_thread_safety(true)
        .build();

    session->start();

    // Launch multiple worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back([i]() {
            XSIGMA_PROFILE_SCOPE("worker_thread_" + std::to_string(i));

            // Simulate work
            for (int j = 0; j < 100; ++j) {
                XSIGMA_PROFILE_SCOPE("work_item");
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    // Wait for all threads to complete
    for (auto& worker : workers) {
        worker.join();
    }

    session->stop();
    session->export_report("multithreaded_profile.json");
}
```

## Performance Characteristics

### Timing Overhead
- **Disabled profiling**: 0 nanoseconds (compile-time elimination)
- **Enabled profiling**: < 100 nanoseconds per scope
- **High-resolution timing**: Platform-optimized (RDTSC on x86, clock_gettime on Unix)

### Memory Overhead
- **Per active scope**: < 1KB
- **Thread-local storage**: Minimal contention
- **Statistical data**: On-demand calculation

### Scalability
- **Thread count**: Tested up to 64 concurrent threads
- **Event count**: Handles millions of events efficiently
- **Memory usage**: Linear growth with active scope count

### Platform-Specific Optimizations

#### Windows
- Uses `QueryPerformanceCounter()` for high-resolution timing
- `GetProcessMemoryInfo()` for memory usage queries
- Thread-local storage via `__declspec(thread)`

#### Linux/Unix
- Uses `clock_gettime(CLOCK_MONOTONIC)` for timing
- `/proc/self/status` for memory usage queries
- POSIX thread-local storage

#### macOS
- Uses `mach_absolute_time()` for timing
- `task_info()` for memory usage queries
- POSIX thread-local storage

## Platform Support

### Supported Platforms
- **Windows**: Windows 10/11, Windows Server 2016+
- **Linux**: Ubuntu 18.04+, CentOS 7+, RHEL 7+
- **macOS**: macOS 10.14+

### Compiler Support
- **Clang**: 10.0+
- **GCC**: 9.0+
- **MSVC**: Visual Studio 2019+

### Dependencies
- **Required**: C++17 standard library
- **Optional**: TBB (for enhanced threading), MKL (for mathematical operations)

## API Reference

### Core Classes

#### profiler_session
Main profiler session management class.

**Key Methods**:
```cpp
class profiler_session {
public:
    static profiler_session_builder builder();

    bool start();
    bool stop();
    bool is_active() const;

    void export_report(const std::string& filename);
    void print_report() const;

    void add_observer(statistical_analyzer* analyzer);
    void remove_observer(statistical_analyzer* analyzer);
};
```

#### profiler_scope
RAII-based scope profiling class.

**Key Methods**:
```cpp
class profiler_scope {
public:
    explicit profiler_scope(const std::string& name);
    explicit profiler_scope(std::string_view name);
    template<typename NameGenerator>
    explicit profiler_scope(NameGenerator&& generator);

    ~profiler_scope();

    void stop();
    void append_metadata(const std::string& key, const std::string& value);
};
```

#### statistical_analyzer
Statistical analysis of profiling data.

**Key Methods**:
```cpp
class statistical_analyzer {
public:
    void start_analysis();
    void stop_analysis();

    statistical_metrics calculate_timing_stats(const std::string& name) const;
    statistical_metrics calculate_memory_stats(const std::string& name) const;

    void add_timing_sample(const std::string& name, double time_ms);
    void add_memory_sample(const std::string& name, size_t memory_bytes);
};
```

### Profiling Macros

```cpp
// Profile current scope with automatic name generation
XSIGMA_PROFILE_FUNCTION();

// Profile current scope with custom name
XSIGMA_PROFILE_SCOPE("custom_name");

// Profile a block of code
XSIGMA_PROFILE_BLOCK("block_name") {
    // Code to profile
}
```

### Configuration Options

```cpp
enum class output_format {
    console,    // Human-readable console output
    json,       // JSON format for programmatic analysis
    csv,        // CSV format for spreadsheet analysis
    xml,        // XML format for structured data
    xplane      // XPlane format for TensorFlow integration
};

struct profiler_options {
    bool enable_timing = true;
    bool enable_memory_tracking = false;
    bool enable_statistical_analysis = false;
    bool enable_thread_safety = true;
    output_format format = output_format::console;
    int trace_level = 1;
};
```

This comprehensive documentation provides developers with everything needed to understand, integrate, and extend the XSigma profiler system.
```
