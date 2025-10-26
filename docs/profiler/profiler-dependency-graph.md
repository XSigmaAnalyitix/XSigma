# XSigma Profiler System - Dependency Graph and Component Classification

## Component Classification Summary

### Required Components (CORE)
These components are essential for basic profiling functionality and cannot be disabled:

| Component | Location | Purpose | Dependencies |
|-----------|----------|---------|--------------|
| `profiler_interface.h` | `core/` | Abstract profiler interface | None |
| `profiler_controller.*` | `core/` | Profiler lifecycle management | profiler_interface |
| `profiler_factory.*` | `core/` | Factory pattern implementation | profiler_interface, profiler_options |
| `profiler_collection.*` | `core/` | Multiple profiler management | profiler_interface |
| `profiler_options.h` | `core/` | Configuration options | None |
| `profiler_lock.*` | `core/` | Thread synchronization | None |
| `timespan.h` | `core/` | Time span utilities | None |
| `profiler.*` | `session/` | Main profiler session | profiler_interface, profiler_options |
| `profiler_report.*` | `session/` | Basic report generation | profiler session |
| `env_time.*` | `platform/` | Cross-platform timing | Platform APIs |
| `env_var.*` | `platform/` | Environment variables | Platform APIs |

### Optional Components (PLUGINS)
These components provide additional features but can be disabled to reduce binary size:

| Component | Location | Purpose | Dependencies |
|-----------|----------|---------|--------------|
| `host_tracer.*` | `cpu/` | Host CPU tracing | profiler_interface, traceme |
| `python_tracer.*` | `cpu/` | Python integration | profiler_interface |
| `metadata_collector.*` | `cpu/` | Metadata collection | profiler_interface |
| `memory_tracker.*` | `memory/` | Memory allocation tracking | profiler session |
| `scoped_memory_debug_annotation.*` | `memory/` | Memory debugging | memory_tracker |
| `statistical_analyzer.*` | `analysis/` | Statistical analysis | profiler session |
| `stats_calculator.*` | `analysis/` | Statistics computation | statistical_analyzer |
| `xplane/*` | `exporters/xplane/` | XPlane format export | None (self-contained) |
| `json_exporter.*` | `exporters/` | JSON format export | profiler_report |
| `csv_exporter.*` | `exporters/` | CSV format export | profiler_report |
| `xml_exporter.*` | `exporters/` | XML format export | profiler_report |
| `ascii_visualizer.*` | `visualization/` | ASCII charts | statistical_analyzer |

## Dependency Graph

### Core Dependencies (Required)
```
profiler_interface.h (ROOT)
    ├── profiler_controller.*
    ├── profiler_factory.*
    │   └── profiler_options.h
    ├── profiler_collection.*
    └── profiler.* (session/)
        ├── profiler_report.* (session/)
        ├── profiler_lock.*
        ├── timespan.h
        └── platform/
            ├── env_time.*
            └── env_var.*
```

### Optional Plugin Dependencies
```
CPU Profiling (OPTIONAL)
├── host_tracer.*
│   ├── profiler_interface.h (REQUIRED)
│   ├── traceme system (EXTERNAL)
│   └── xplane export (OPTIONAL)
├── python_tracer.*
│   └── profiler_interface.h (REQUIRED)
└── metadata_collector.*
    └── profiler_interface.h (REQUIRED)

Memory Profiling (OPTIONAL)
├── memory_tracker.*
│   └── profiler.* (REQUIRED)
└── scoped_memory_debug_annotation.*
    └── memory_tracker.*

Statistical Analysis (OPTIONAL)
├── statistical_analyzer.*
│   └── profiler.* (REQUIRED)
└── stats_calculator.*
    └── statistical_analyzer.*

Data Export (OPTIONAL)
├── xplane/* (Self-contained)
├── json_exporter.*
│   └── profiler_report.* (REQUIRED)
├── csv_exporter.*
│   └── profiler_report.* (REQUIRED)
└── xml_exporter.*
    └── profiler_report.* (REQUIRED)

Visualization (OPTIONAL)
├── ascii_visualizer.*
│   └── statistical_analyzer.* (OPTIONAL)
├── timeline_visualizer.*
│   └── profiler_report.* (REQUIRED)
└── flamegraph_exporter.*
    └── profiler_report.* (REQUIRED)
```

## External Dependencies

### System Dependencies
- **Windows**: `windows.h`, `psapi.h` (for memory tracking)
- **Unix/Linux**: `sys/resource.h`, `unistd.h` (for memory tracking)
- **Cross-platform**: `<chrono>`, `<thread>`, `<mutex>`

### XSigma Internal Dependencies
- **TraceMe System**: `logging/tracing/traceme.h` (for CPU profiling)
- **Logger**: `logging/logger.h` (for error reporting)
- **Exception Handling**: `util/exception.h` (for error management)
- **Hash Utilities**: `util/flat_hash.h` (for data structures)

### Optional External Dependencies
- **TBB**: Intel Threading Building Blocks (for enhanced threading)
- **MKL**: Intel Math Kernel Library (for mathematical operations)

## Component Interaction Patterns

### 1. Core Framework Pattern
```
Application Code
    ↓
XSIGMA_PROFILE_SCOPE macro
    ↓
profiler_scope (RAII)
    ↓
profiler_session
    ↓
profiler_factory
    ↓
profiler_interface implementations
```

### 2. Plugin Registration Pattern
```
Static Initialization
    ↓
register_profiler_factory()
    ↓
Factory Registry (Singleton)
    ↓
create_profilers() (Runtime)
    ↓
profiler_collection
```

### 3. Data Flow Pattern
```
Raw Profiling Data
    ↓
profiler_interface::collect_data()
    ↓
profiler_report
    ↓
Export Format Plugins
    ↓
Output Files (JSON/CSV/XML/XPlane)
```

## Build System Integration

### CMake Configuration
```cmake
# Required components (always included)
set(PROFILER_CORE_SOURCES
    core/profiler_interface.h
    core/profiler_controller.cxx
    core/profiler_factory.cxx
    core/profiler_collection.cxx
    core/profiler_lock.cxx
    session/profiler.cxx
    session/profiler_report.cxx
    platform/env_time.cxx
    platform/env_var.cxx
)

# Optional components (conditionally included)
if(XSIGMA_ENABLE_CPU_PROFILING)
    list(APPEND PROFILER_SOURCES cpu/host_tracer.cxx)
endif()

if(XSIGMA_ENABLE_MEMORY_PROFILING)
    list(APPEND PROFILER_SOURCES memory/memory_tracker.cxx)
endif()

if(XSIGMA_ENABLE_STATISTICAL_ANALYSIS)
    list(APPEND PROFILER_SOURCES analysis/statistical_analyzer.cxx)
endif()

if(XSIGMA_ENABLE_XPLANE_EXPORT)
    list(APPEND PROFILER_SOURCES exporters/xplane/xplane_builder.cxx)
endif()
```

### Preprocessor Configuration
```cpp
// Conditional compilation for optional components
#ifdef XSIGMA_ENABLE_CPU_PROFILING
    #include "experimental/profiler/cpu/host_tracer.h"
#endif

#ifdef XSIGMA_ENABLE_MEMORY_PROFILING
    #include "experimental/profiler/memory/memory_tracker.h"
#endif

#ifdef XSIGMA_ENABLE_STATISTICAL_ANALYSIS
    #include "experimental/profiler/analysis/statistical_analyzer.h"
#endif
```

## Runtime Component Discovery

### Plugin Discovery Mechanism
```cpp
// Automatic plugin registration via static initialization
namespace {
    auto register_cpu_profiler = []() {
        #ifdef XSIGMA_ENABLE_CPU_PROFILING
        register_profiler_factory(&create_host_tracer);
        #endif
        return 0;
    }();
}
```

### Configuration-Based Enabling
```cpp
profiler_options options;
options.enable_cpu_profiling = true;      // Requires CPU plugin
options.enable_memory_tracking = false;   // Memory plugin not needed
options.enable_statistical_analysis = true; // Requires analysis plugin

// Only enabled plugins will be instantiated
auto profilers = create_profilers(options);
```

## Binary Size Impact

### Minimal Configuration (Core Only)
- **Size**: ~50KB additional binary size
- **Components**: Core framework + basic reporting
- **Functionality**: Basic timing profiling only

### Full Configuration (All Plugins)
- **Size**: ~200KB additional binary size
- **Components**: All profiling, analysis, and export capabilities
- **Functionality**: Complete profiling suite

### Recommended Configurations

#### Development Build
```cpp
#define XSIGMA_ENABLE_CPU_PROFILING 1
#define XSIGMA_ENABLE_MEMORY_PROFILING 1
#define XSIGMA_ENABLE_STATISTICAL_ANALYSIS 1
#define XSIGMA_ENABLE_ALL_EXPORTERS 1
// Full functionality for development and debugging
```

#### Production Build
```cpp
#define XSIGMA_ENABLE_CPU_PROFILING 1
#define XSIGMA_ENABLE_MEMORY_PROFILING 0
#define XSIGMA_ENABLE_STATISTICAL_ANALYSIS 0
#define XSIGMA_ENABLE_JSON_EXPORT 1
// Minimal overhead for production profiling
```

#### Embedded/Resource-Constrained Build
```cpp
#define XSIGMA_ENABLE_CPU_PROFILING 0
#define XSIGMA_ENABLE_MEMORY_PROFILING 0
#define XSIGMA_ENABLE_STATISTICAL_ANALYSIS 0
#define XSIGMA_ENABLE_ALL_EXPORTERS 0
// Core framework only, profiling can be enabled at runtime
```

This dependency graph and classification system enables fine-grained control over which profiling components are included in the final binary, allowing optimization for different deployment scenarios.
