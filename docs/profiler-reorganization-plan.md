# XSigma Profiler System Reorganization Plan

## Current Architecture Analysis

### Current File Structure
```
Library/Core/experimental/profiler/
├── README.md                                    # Documentation
├── cpu/                                         # CPU-specific profiling
│   ├── host_tracer.cxx/.h                      # Host CPU tracing
│   ├── python_tracer.cxx/.h                    # Python integration
│   ├── python_tracer_factory.cxx               # Python tracer factory
│   ├── host_tracer_factory.cxx                 # Host tracer factory
│   ├── metadata_collector.cxx                  # Metadata collection
│   └── metadata_utils.h                        # Metadata utilities
├── xplane/                                      # XPlane format support
│   ├── xplane.h                                # Core XPlane data structures
│   ├── xplane_builder.cxx/.h                   # XPlane construction API
│   ├── xplane_visitor.cxx/.h                   # XPlane traversal API
│   ├── xplane_schema.cxx/.h                    # Schema definitions
│   ├── xplane_utils.cxx/.h                     # Utility functions
│   ├── xplane_mutators.h                       # Mutation APIs
│   └── tf_xplane_visitor.h                     # TensorFlow integration
├── profiler.cxx/.h                             # Main profiler interface
├── profiler_interface.h                        # Abstract profiler interface
├── profiler_controller.cxx/.h                  # Profiler lifecycle management
├── profiler_factory.cxx/.h                     # Profiler factory pattern
├── profiler_collection.cxx/.h                  # Multiple profiler management
├── profiler_options.h                          # Configuration options
├── profiler_report.cxx/.h                      # Report generation
├── profiler_lock.cxx/.h                        # Thread synchronization
├── memory_tracker.cxx/.h                       # Memory profiling
├── statistical_analyzer.cxx/.h                 # Statistical analysis
├── stats_calculator.cxx/.h                     # Statistics calculations
├── stat_summarizer_options.h                   # Statistics options
├── scoped_memory_debug_annotation.cxx/.h       # Memory debugging
├── env_time.cxx/.h                             # Time utilities
├── env_time_win.cxx                            # Windows time implementation
├── env_var.cxx/.h                              # Environment variables
└── timespan.h                                  # Time span utilities
```

### Current Dependencies
- **Core Dependencies**: traceme system, logger, exception handling
- **Platform Dependencies**: Windows (psapi.h), Unix (sys/resource.h)
- **Build Integration**: CMakeLists.txt includes all profiler files via GLOB_RECURSE
- **External Integration**: TensorFlow profiling tools via XPlane format

## Proposed New Structure

### Logical Organization Principles
1. **Separation of Concerns**: Group related functionality together
2. **Required vs Optional**: Clear distinction between core and optional components
3. **Platform Abstraction**: Isolate platform-specific code
4. **Export Formats**: Separate different output format implementations
5. **Visualization**: Dedicated space for visualization components

### New Directory Structure
```
Library/Core/experimental/profiler/
├── README.md                                    # Updated documentation
├── core/                                        # Required core components
│   ├── profiler_interface.h                    # Abstract interface (REQUIRED)
│   ├── profiler_controller.cxx/.h              # Lifecycle management (REQUIRED)
│   ├── profiler_factory.cxx/.h                 # Factory pattern (REQUIRED)
│   ├── profiler_collection.cxx/.h              # Multi-profiler support (REQUIRED)
│   ├── profiler_options.h                      # Configuration (REQUIRED)
│   ├── profiler_lock.cxx/.h                    # Thread sync (REQUIRED)
│   └── timespan.h                              # Time utilities (REQUIRED)
├── session/                                     # Session management
│   ├── profiler.cxx/.h                         # Main session class (REQUIRED)
│   └── profiler_report.cxx/.h                  # Report generation (REQUIRED)
├── cpu/                                         # CPU profiling components
│   ├── host_tracer.cxx/.h                      # Host CPU tracing (OPTIONAL)
│   ├── host_tracer_factory.cxx                 # Host tracer factory (OPTIONAL)
│   ├── python_tracer.cxx/.h                    # Python integration (OPTIONAL)
│   ├── python_tracer_factory.cxx               # Python tracer factory (OPTIONAL)
│   ├── metadata_collector.cxx                  # Metadata collection (OPTIONAL)
│   └── metadata_utils.h                        # Metadata utilities (OPTIONAL)
├── memory/                                      # Memory profiling components
│   ├── memory_tracker.cxx/.h                   # Memory tracking (OPTIONAL)
│   └── scoped_memory_debug_annotation.cxx/.h   # Memory debugging (OPTIONAL)
├── analysis/                                    # Statistical analysis
│   ├── statistical_analyzer.cxx/.h             # Statistical analysis (OPTIONAL)
│   ├── stats_calculator.cxx/.h                 # Statistics calculations (OPTIONAL)
│   └── stat_summarizer_options.h               # Statistics options (OPTIONAL)
├── exporters/                                   # Data export formats
│   ├── xplane/                                 # XPlane format (OPTIONAL)
│   │   ├── xplane.h                            # Core data structures
│   │   ├── xplane_builder.cxx/.h               # Construction API
│   │   ├── xplane_visitor.cxx/.h               # Traversal API
│   │   ├── xplane_schema.cxx/.h                # Schema definitions
│   │   ├── xplane_utils.cxx/.h                 # Utility functions
│   │   ├── xplane_mutators.h                   # Mutation APIs
│   │   └── tf_xplane_visitor.h                 # TensorFlow integration
│   ├── json_exporter.cxx/.h                    # JSON export (OPTIONAL)
│   ├── csv_exporter.cxx/.h                     # CSV export (OPTIONAL)
│   └── xml_exporter.cxx/.h                     # XML export (OPTIONAL)
├── visualization/                               # Visualization components
│   ├── ascii_visualizer.cxx/.h                 # ASCII charts (OPTIONAL)
│   ├── timeline_visualizer.cxx/.h              # Timeline visualization (OPTIONAL)
│   └── flamegraph_exporter.cxx/.h              # Flamegraph export (OPTIONAL)
└── platform/                                   # Platform-specific code
    ├── env_time.cxx/.h                         # Cross-platform time (REQUIRED)
    ├── env_time_win.cxx                        # Windows implementation (REQUIRED)
    ├── env_time_unix.cxx                       # Unix implementation (REQUIRED)
    └── env_var.cxx/.h                          # Environment variables (REQUIRED)
```

## Component Classification

### Required Components (Core Functionality)
These components are essential for basic profiling functionality:

- `core/profiler_interface.h` - Abstract profiler interface
- `core/profiler_controller.*` - Profiler lifecycle management
- `core/profiler_factory.*` - Factory pattern for profiler creation
- `core/profiler_collection.*` - Managing multiple profilers
- `core/profiler_options.h` - Configuration options
- `core/profiler_lock.*` - Thread synchronization
- `core/timespan.h` - Time span utilities
- `session/profiler.*` - Main profiler session class
- `session/profiler_report.*` - Basic report generation
- `platform/env_time.*` - Cross-platform time utilities
- `platform/env_var.*` - Environment variable handling

### Optional Components (Extensions/Plugins)
These components provide additional features but are not required for basic operation:

- `cpu/*` - CPU-specific profiling implementations
- `memory/*` - Memory profiling and tracking
- `analysis/*` - Statistical analysis and metrics
- `exporters/*` - Various export format implementations
- `visualization/*` - Visualization and charting components

## Migration Strategy

### Phase 1: Create New Directory Structure
1. Create new subdirectories
2. Copy files to new locations
3. Update include paths in source files
4. Update CMakeLists.txt to reflect new structure

### Phase 2: Update Include Paths
1. Update all `#include` statements to use new paths
2. Maintain backward compatibility with deprecated path warnings
3. Update documentation and examples

### Phase 3: Add Component Classification
1. Add comments to each file indicating REQUIRED/OPTIONAL status
2. Create dependency graph documentation
3. Update build system to support optional component selection

### Phase 4: Enhance Modularity
1. Implement proper plugin interfaces for optional components
2. Add runtime component discovery
3. Create configuration system for enabling/disabling components

## Backward Compatibility

### Include Path Compatibility
- Maintain old include paths with deprecation warnings for one release cycle
- Provide compatibility headers that forward to new locations
- Update all internal XSigma code to use new paths immediately

### API Compatibility
- All public APIs remain unchanged
- Internal reorganization should be transparent to users
- Existing profiling macros continue to work without modification

## Benefits of New Structure

1. **Improved Discoverability**: Related functionality is grouped together
2. **Clear Dependencies**: Required vs optional components are clearly separated
3. **Modular Design**: Optional components can be disabled to reduce binary size
4. **Platform Abstraction**: Platform-specific code is isolated
5. **Extensibility**: New exporters and visualizers can be easily added
6. **Maintainability**: Logical organization makes code easier to maintain
