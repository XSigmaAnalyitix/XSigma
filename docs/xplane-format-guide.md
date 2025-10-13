# XPlane Format - Comprehensive Technical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Why XPlane is Needed](#why-xplane-is-needed)
3. [XPlane Data Model](#xplane-data-model)
4. [Core Components](#core-components)
5. [Data Flow Examples](#data-flow-examples)
6. [TensorFlow Integration](#tensorflow-integration)
7. [Performance Considerations](#performance-considerations)
8. [Alternatives Comparison](#alternatives-comparison)

## Introduction

XPlane is a structured profiling data format originally developed as part of the TensorFlow ecosystem for representing execution traces and profiling information. The XSigma profiler system adopts and extends this format to provide compatibility with TensorFlow profiling tools while maintaining flexibility for custom profiling scenarios.

### Key Characteristics
- **Hierarchical Structure**: Organized as spaces → planes → lines → events
- **Rich Metadata**: Supports typed metadata with schema validation
- **Efficient Storage**: Compact binary representation with optional compression
- **Tool Interoperability**: Compatible with TensorBoard and other TensorFlow profiling tools
- **Extensible Schema**: Supports custom event types and metadata

## Why XPlane is Needed

### 1. Standardization Across Tools
Different profiling tools often use incompatible data formats, making it difficult to:
- Share profiling data between tools
- Build unified analysis pipelines
- Compare results from different profilers

XPlane provides a common format that enables:
```cpp
// Data from different sources can be merged
x_space combined_space;
merge_xplane_data(cpu_profiler_data, &combined_space);
merge_xplane_data(gpu_profiler_data, &combined_space);
merge_xplane_data(custom_profiler_data, &combined_space);
```

### 2. Rich Metadata Support
Traditional formats like CSV or simple JSON lack structured metadata support:

**CSV Limitations**:
```csv
timestamp,duration,name
1000000,5000,function_a
1005000,3000,function_b
```

**XPlane Advantages**:
```cpp
// Rich metadata with typed values
xevent_builder event = line.add_event(*metadata);
event.SetTimestampPs(1000000000);  // Picosecond precision
event.SetDurationPs(5000000);
event.add_stat(*plane.get_or_create_stat_metadata("tensor_shape"), "1024x768x3");
event.add_stat(*plane.get_or_create_stat_metadata("memory_usage"), 1048576);
event.add_stat(*plane.get_or_create_stat_metadata("device_id"), 0);
```

### 3. TensorFlow Ecosystem Integration
XPlane enables seamless integration with TensorFlow profiling infrastructure:
- **TensorBoard**: Direct visualization of profiling data
- **TensorFlow Profiler**: Analysis of ML workloads
- **Performance Analysis Tools**: Bottleneck identification and optimization

### 4. Scalability for Large Datasets
XPlane is designed to handle large-scale profiling data efficiently:
- **Streaming Support**: Process data without loading entire datasets into memory
- **Compression**: Optional compression reduces storage requirements
- **Indexing**: Fast lookup of specific events or time ranges

## XPlane Data Model

### Hierarchical Structure
```
x_space (Root Container)
├── xplane (Device/Process/Logical Grouping)
│   ├── id: unique identifier
│   ├── name: human-readable name
│   ├── xline[] (Execution Context/Thread)
│   │   ├── id: unique identifier (often thread ID)
│   │   ├── name: human-readable name
│   │   ├── timestamp_ns: line start time
│   │   ├── duration_ps: line duration
│   │   └── xevent[] (Individual Events)
│   │       ├── metadata_id: reference to event_metadata
│   │       ├── offset_ps: offset from line start
│   │       ├── duration_ps: event duration
│   │       └── stats[] (Key-Value Metadata)
│   │           ├── metadata_id: reference to stat_metadata
│   │           └── value: typed value (int64, double, string)
│   ├── event_metadata{} (Event Type Definitions)
│   │   ├── id: unique identifier
│   │   ├── name: event type name
│   │   └── display_name: human-readable name
│   └── stat_metadata{} (Metadata Type Definitions)
│       ├── id: unique identifier
│       ├── name: metadata key name
│       └── description: human-readable description
└── ...
```

### Data Types and Precision
- **Time Precision**: Picoseconds (10^-12 seconds) for maximum accuracy
- **Metadata Values**: int64, double, string, bytes
- **IDs**: 64-bit integers for unique identification
- **Names**: UTF-8 strings for human-readable labels

## Core Components

### xplane_builder.h/cxx - Construction API

#### Purpose
Provides a fluent API for constructing XPlane data structures with automatic ID management and validation.

#### Key Classes and Methods
```cpp
class xplane_builder {
public:
    // Plane management
    void SetId(int64_t id);
    void SetName(std::string_view name);
    
    // Line creation
    xline_builder get_or_create_line(int64_t line_id);
    
    // Metadata management
    xevent_metadata* get_or_create_event_metadata(std::string_view name);
    x_stat_metadata* get_or_create_stat_metadata(std::string_view name);
    
    // Statistics
    void add_stat(const x_stat_metadata& metadata, int64_t value);
    void add_stat(const x_stat_metadata& metadata, double value);
    void add_stat(const x_stat_metadata& metadata, std::string_view value);
};

class xline_builder {
public:
    // Line properties
    void SetName(std::string_view name);
    void SetTimestampNs(int64_t timestamp);
    void SetDurationPs(int64_t duration);
    
    // Event creation
    xevent_builder add_event(const xevent_metadata& metadata);
    xevent_builder add_event(const timespan& span, const xevent_metadata& metadata);
};

class xevent_builder {
public:
    // Event timing
    void SetOffsetPs(int64_t offset);
    void SetDurationPs(int64_t duration);
    void SetTimestampPs(int64_t timestamp);
    
    // Event metadata
    void add_stat(const x_stat_metadata& metadata, int64_t value);
    void add_stat(const x_stat_metadata& metadata, double value);
    void add_stat(const x_stat_metadata& metadata, std::string_view value);
};
```

#### Usage Example
```cpp
// Create XPlane data structure
x_space space;
xplane* plane = space.add_planes();
xplane_builder builder(plane);

// Configure plane
builder.SetId(0);
builder.SetName("Host CPU");

// Create event metadata
auto* func_call_metadata = builder.get_or_create_event_metadata("function_call");
auto* tensor_shape_metadata = builder.get_or_create_stat_metadata("tensor_shape");
auto* memory_usage_metadata = builder.get_or_create_stat_metadata("memory_usage");

// Create line for thread
auto line = builder.get_or_create_line(12345);  // Thread ID
line.SetName("Main Thread");

// Add events
auto event = line.add_event(*func_call_metadata);
event.SetTimestampPs(1000000000000);  // 1 second in picoseconds
event.SetDurationPs(5000000000);     // 5 milliseconds in picoseconds
event.add_stat(*tensor_shape_metadata, "1024x768x3");
event.add_stat(*memory_usage_metadata, 1048576);
```

### xplane_visitor.h/cxx - Traversal API

#### Purpose
Provides read-only access to XPlane data with efficient traversal patterns and type-safe metadata access.

#### Key Classes and Methods
```cpp
class xplane_visitor {
public:
    // Plane properties
    int64_t id() const;
    std::string_view name() const;
    size_t num_lines() const;
    
    // Line traversal
    template<typename ForEachLineFunc>
    void for_each_line(ForEachLineFunc&& func) const;
    
    // Metadata access
    const xevent_metadata* get_event_metadata(int64_t id) const;
    const x_stat_metadata* get_stat_metadata(int64_t id) const;
    
    // Statistics
    template<typename ForEachStatFunc>
    void for_each_stat(ForEachStatFunc&& func) const;
};

class xline_visitor {
public:
    // Line properties
    int64_t id() const;
    std::string_view name() const;
    int64_t timestamp_ns() const;
    int64_t duration_ps() const;
    
    // Event traversal
    template<typename ForEachEventFunc>
    void for_each_event(ForEachEventFunc&& func) const;
    
    size_t num_events() const;
};

class xevent_visitor {
public:
    // Event properties
    int64_t metadata_id() const;
    std::string_view name() const;
    int64_t offset_ps() const;
    int64_t duration_ps() const;
    int64_t timestamp_ps() const;
    
    // Metadata access
    template<typename ForEachStatFunc>
    void for_each_stat(ForEachStatFunc&& func) const;
    
    std::optional<x_stat_visitor> get_stat(int64_t stat_type) const;
};
```

#### Usage Example
```cpp
// Read XPlane data
xplane_visitor visitor(&plane);

std::cout << "Plane: " << visitor.name() << " (ID: " << visitor.id() << ")\n";

visitor.for_each_line([](const xline_visitor& line) {
    std::cout << "  Line: " << line.name() << " (ID: " << line.id() << ")\n";
    
    line.for_each_event([](const xevent_visitor& event) {
        std::cout << "    Event: " << event.name() 
                  << " Duration: " << event.duration_ps() / 1000000.0 << " ms\n";
        
        // Access metadata
        event.for_each_stat([](const x_stat_visitor& stat) {
            std::cout << "      " << stat.name() << ": " << stat.ToString() << "\n";
        });
    });
});
```

### xplane_schema.h/cxx - Schema Definitions

#### Purpose
Defines standard metadata types and event types for compatibility with TensorFlow profiling tools.

#### Standard Event Types
```cpp
enum EventType {
    kUnknownHostEventType = 0,
    kTraceContext,
    kSessionRun,
    kFunctionRun,
    kRunGraph,
    kEagerKernelExecute,
    kExecutorStateProcess,
    kExecutorDoneCallback,
    kMemoryAllocation,
    kMemoryDeallocation,
    // ... many more standard types
};
```

#### Standard Metadata Types
```cpp
enum StatType {
    kUnknownStatType = 0,
    // TraceMe arguments
    kStepId,
    kDeviceOrdinal,
    kChipOrdinal,
    kNodeOrdinal,
    kModelId,
    kQueueId,
    kRequestId,
    kRunId,
    kReplicaId,
    kGraphType,
    kStepNum,
    kIterNum,
    // Memory-related
    kBytesReserved,
    kBytesAllocated,
    kBytesAvailable,
    kFragmentation,
    kPeakBytesInUse,
    kRequestedBytes,
    kAllocationBytes,
    kAddress,
    // Tensor-related
    kTensorShapes,
    kTensorLayout,
    kDataType,
    // Performance metrics
    kFlops,
    kModelFlops,
    kBytesAccessed,
    kBytesTransferred,
    // ... many more standard types
};
```

#### Schema Lookup Functions
```cpp
// Find standard event types
const xevent_metadata* FindHostEventType(std::string_view event_name);
const xevent_metadata* FindTfOpEventType(std::string_view op_name);

// Find standard metadata types
const x_stat_metadata* FindStatType(StatType stat_type);

// Create TensorFlow-compatible visitor
xplane_visitor CreateTfXPlaneVisitor(const xplane* plane);
```

### xplane_utils.h/cxx - Utility Functions

#### Purpose
Provides utility functions for common XPlane operations like merging, sorting, and timestamp normalization.

#### Key Functions
```cpp
// Timestamp operations
void NormalizeTimestamps(xplane* plane, uint64_t start_time_ns);
void NormalizeTimestamps(x_space* space, uint64_t start_time_ns);

// Plane operations
void MergePlanes(const xplane& src_plane, xplane* dst_plane);
void RemoveEmptyLines(xplane* plane);

// Event operations
template<typename Event, typename Plane>
std::vector<Event> GetSortedEvents(Plane& plane, bool include_derived_events = false);

// Flow operations (for representing data dependencies)
void AddFlowsToXplane(int32_t host_id, bool is_host_plane, bool connect_traceme, xplane* plane);

// Analysis utilities
bool IsXSpaceGrouped(const x_space& space);
void GroupEvents(xplane* plane);
```

#### Usage Examples
```cpp
// Normalize timestamps to start from zero
NormalizeTimestamps(&plane, earliest_timestamp);

// Merge profiling data from multiple sources
xplane combined_plane;
MergePlanes(cpu_plane, &combined_plane);
MergePlanes(gpu_plane, &combined_plane);

// Get all events sorted by timestamp
auto events = GetSortedEvents<xevent_visitor>(xplane_visitor(&plane));
for (const auto& event : events) {
    std::cout << event.name() << " at " << event.timestamp_ps() << "\n";
}
```

## Data Flow Examples

### 1. TraceMe to XPlane Conversion
```cpp
#include "logging/tracing/traceme.h"
#include "logging/tracing/traceme_recorder.h"
#include "experimental/profiler/exporters/xplane/xplane_builder.h"

void convert_traceme_to_xplane() {
    // 1. Collect TraceMe events
    traceme_recorder::start(1);

    {
        traceme trace("matrix_multiplication");
        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        {
            traceme inner_trace("memory_allocation");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    auto events = traceme_recorder::stop();

    // 2. Create XPlane structure
    x_space space;
    xplane* host_plane = space.add_planes();
    xplane_builder builder(host_plane);

    builder.SetId(0);
    builder.SetName("Host CPU");

    // 3. Convert events to XPlane format
    for (const auto& thread_events : events) {
        auto line = builder.get_or_create_line(thread_events.thread.tid);
        line.SetName(thread_events.thread.name);

        for (const auto& event : thread_events.events) {
            if (event.is_complete()) {
                // Create event metadata
                auto* event_metadata = builder.get_or_create_event_metadata(event.name);

                // Add event
                auto xevent = line.add_event(*event_metadata);
                xevent.SetTimestampPs(event.start_time * 1000);  // ns to ps
                xevent.SetDurationPs((event.end_time - event.start_time) * 1000);

                // Add custom metadata if present
                if (event.name.find("#") != std::string::npos) {
                    parse_and_add_metadata(event.name, xevent, &builder);
                }
            }
        }
    }

    // 4. Export to file
    write_xplane_to_file(space, "profile.xplane");
}

void parse_and_add_metadata(const std::string& name, xevent_builder& event, xplane_builder* builder) {
    // Parse TraceMe metadata format: "name#key1=value1#key2=value2#"
    size_t pos = name.find('#');
    if (pos == std::string::npos) return;

    std::string metadata_str = name.substr(pos + 1);
    std::istringstream stream(metadata_str);
    std::string pair;

    while (std::getline(stream, pair, '#')) {
        if (pair.empty()) continue;

        size_t eq_pos = pair.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = pair.substr(0, eq_pos);
            std::string value = pair.substr(eq_pos + 1);

            auto* stat_metadata = builder->get_or_create_stat_metadata(key);

            // Try to parse as number, otherwise store as string
            try {
                if (value.find('.') != std::string::npos) {
                    event.add_stat(*stat_metadata, std::stod(value));
                } else {
                    event.add_stat(*stat_metadata, std::stoll(value));
                }
            } catch (...) {
                event.add_stat(*stat_metadata, value);
            }
        }
    }
}
```

### 2. XPlane Analysis and Visualization
```cpp
#include "experimental/profiler/exporters/xplane/xplane_visitor.h"

void analyze_xplane_data(const x_space& space) {
    for (const auto& plane : space.planes()) {
        xplane_visitor visitor(&plane);

        std::cout << "=== Plane: " << visitor.name() << " ===\n";

        // Collect timing statistics
        std::map<std::string, std::vector<double>> event_durations;

        visitor.for_each_line([&](const xline_visitor& line) {
            std::cout << "Thread: " << line.name() << " (ID: " << line.id() << ")\n";

            line.for_each_event([&](const xevent_visitor& event) {
                double duration_ms = event.duration_ps() / 1e9;  // ps to ms
                event_durations[std::string(event.name())].push_back(duration_ms);

                std::cout << "  " << event.name()
                          << ": " << duration_ms << " ms\n";

                // Print metadata
                event.for_each_stat([](const x_stat_visitor& stat) {
                    std::cout << "    " << stat.name() << ": " << stat.ToString() << "\n";
                });
            });
        });

        // Print summary statistics
        std::cout << "\n=== Summary Statistics ===\n";
        for (const auto& [event_name, durations] : event_durations) {
            if (durations.empty()) continue;

            double total = std::accumulate(durations.begin(), durations.end(), 0.0);
            double mean = total / durations.size();

            auto minmax = std::minmax_element(durations.begin(), durations.end());

            std::cout << event_name << ":\n";
            std::cout << "  Count: " << durations.size() << "\n";
            std::cout << "  Total: " << total << " ms\n";
            std::cout << "  Mean: " << mean << " ms\n";
            std::cout << "  Min: " << *minmax.first << " ms\n";
            std::cout << "  Max: " << *minmax.second << " ms\n\n";
        }
    }
}
```

## TensorFlow Integration

### TensorBoard Visualization
XPlane data can be directly loaded into TensorBoard for visualization:

```python
# Python script to load XPlane data in TensorBoard
import tensorflow as tf
from tensorboard.plugins.profile import profile_plugin

# Load XPlane data
with open('profile.xplane', 'rb') as f:
    xplane_data = f.read()

# Create TensorBoard log directory
logdir = './tensorboard_logs'
os.makedirs(logdir, exist_ok=True)

# Write XPlane data in TensorBoard format
with tf.io.gfile.GFile(os.path.join(logdir, 'plugins/profile/2024_01_01_12_00_00/host.xplane.pb'), 'wb') as f:
    f.write(xplane_data)

# Launch TensorBoard
# tensorboard --logdir=./tensorboard_logs
```

### TensorFlow Profiler Integration
```cpp
// Integration with TensorFlow Profiler API
#include "experimental/profiler/exporters/xplane/tf_xplane_visitor.h"

void integrate_with_tensorflow_profiler(const x_space& space) {
    for (const auto& plane : space.planes()) {
        // Create TensorFlow-compatible visitor
        auto tf_visitor = CreateTfXPlaneVisitor(&plane);

        // Use TensorFlow-specific analysis
        tf_visitor.for_each_line([](const xline_visitor& line) {
            line.for_each_event([](const xevent_visitor& event) {
                // Check for TensorFlow-specific metadata
                auto step_id_stat = event.get_stat(StatType::kStepId);
                if (step_id_stat) {
                    std::cout << "TensorFlow step: " << step_id_stat->int64_value() << "\n";
                }

                auto tensor_shape_stat = event.get_stat(StatType::kTensorShapes);
                if (tensor_shape_stat) {
                    std::cout << "Tensor shape: " << tensor_shape_stat->str_value() << "\n";
                }

                auto flops_stat = event.get_stat(StatType::kFlops);
                if (flops_stat) {
                    std::cout << "FLOPS: " << flops_stat->int64_value() << "\n";
                }
            });
        });
    }
}
```

## Performance Considerations

### Memory Usage
- **Metadata Deduplication**: Event and stat metadata are stored once and referenced by ID
- **String Interning**: Common strings are deduplicated automatically
- **Lazy Loading**: Large XPlane files can be processed without loading everything into memory

### Time Precision
- **Picosecond Precision**: Supports sub-nanosecond timing accuracy
- **Overflow Protection**: 64-bit timestamps support ~292 years of runtime
- **Relative Timestamps**: Events use offsets from line start time for efficiency

### Serialization Performance
```cpp
// Efficient serialization patterns
void efficient_xplane_creation() {
    x_space space;
    xplane* plane = space.add_planes();
    xplane_builder builder(plane);

    // Pre-create metadata to avoid repeated lookups
    auto* func_metadata = builder.get_or_create_event_metadata("function_call");
    auto* duration_metadata = builder.get_or_create_stat_metadata("duration_ms");

    // Reserve space for known number of events
    auto line = builder.get_or_create_line(thread_id);
    // line.ReserveEvents(expected_event_count);  // If available

    // Batch event creation
    for (const auto& trace_event : trace_events) {
        auto event = line.add_event(*func_metadata);
        event.SetTimestampPs(trace_event.timestamp);
        event.SetDurationPs(trace_event.duration);
        event.add_stat(*duration_metadata, trace_event.duration / 1e6);
    }
}
```

## Alternatives Comparison

### XPlane vs JSON
| Aspect | XPlane | JSON |
|--------|--------|------|
| **File Size** | Compact (binary) | Large (text) |
| **Parse Speed** | Fast | Moderate |
| **Human Readable** | No | Yes |
| **Schema Validation** | Built-in | External |
| **Metadata Support** | Rich, typed | Limited |
| **Tool Support** | TensorFlow ecosystem | Universal |
| **Streaming** | Yes | Limited |

### XPlane vs Protocol Buffers
| Aspect | XPlane | Protocol Buffers |
|--------|--------|------------------|
| **Schema Evolution** | Built-in | Excellent |
| **Tool Integration** | TensorFlow-specific | Universal |
| **Performance** | Optimized for profiling | General purpose |
| **Metadata Types** | Profiling-specific | Generic |
| **Learning Curve** | Moderate | Steep |

### XPlane vs CSV
| Aspect | XPlane | CSV |
|--------|--------|-----|
| **Metadata Support** | Rich | Minimal |
| **Hierarchical Data** | Native | Flattened |
| **Tool Support** | Specialized | Universal |
| **File Size** | Compact | Large |
| **Processing Speed** | Fast | Fast |
| **Human Readable** | No | Yes |

### When to Choose XPlane
✅ **Use XPlane when:**
- Integrating with TensorFlow/TensorBoard
- Need rich metadata support
- Working with large profiling datasets
- Require precise timing information
- Building profiling tool ecosystems

❌ **Avoid XPlane when:**
- Need human-readable output
- Working with simple profiling data
- Require universal tool compatibility
- Building one-off analysis scripts
- Working in resource-constrained environments

This comprehensive guide provides everything needed to understand, implement, and extend XPlane format support in the XSigma profiler system.
```
