# Kineto and ITTAPI Comparison Summary

## Quick Reference

### What is Kineto?
**PyTorch's profiling library** for comprehensive performance analysis
- Supports CPU and GPU profiling (NVIDIA, AMD, Intel XPU)
- Collects detailed activity traces
- Integrates with TensorBoard and Chrome trace viewer
- Used for performance debugging and optimization

### What is ITTAPI?
**Intel's Instrumentation and Tracing Technology API** for VTune integration
- Lightweight task/frame annotations
- Minimal overhead (~1-2%)
- Real-time visualization in VTune
- Used for performance monitoring and analysis

---

## Key Differences at a Glance

| Feature | PyTorch | XSigma |
|---------|---------|--------|
| **Kineto Default** | Enabled | Enabled |
| **Kineto Approach** | Direct libkineto usage | Wrapper with graceful degradation |
| **ITT Default** | Enabled | Disabled |
| **ITT Library Type** | Static | Shared (DLL on Windows) |
| **ITT Python Support** | Yes | No |
| **GPU Support** | CUPTI, ROCtracer, XPUPTI, AIUPTI | Same (inherited) |
| **Wrapper Mode** | Limited | Explicit |
| **Documentation** | Code comments | Comprehensive markdown |

---

## Integration Comparison

### PyTorch Integration

**Kineto:**
- Direct integration via `libkineto::api()`
- Automatic initialization in profiler
- Supports multiple GPU backends
- Comprehensive activity tracking

**ITT:**
- Global domain per process
- Python-level bindings
- Observer-based automatic instrumentation
- Thread-local state management

### XSigma Integration

**Kineto:**
- Wrapper class `xsigma::kineto_profiler`
- Graceful degradation without full dependencies
- Thread-safe initialization
- Documented manual setup requirement

**ITT:**
- Direct ITT API usage
- User-managed domain creation
- Shared library requirement
- VTune integration documentation

---

## Build System Comparison

### PyTorch CMake

```cmake
# Kineto
if(XSIGMA_ENABLE_KINETO)
    set(LIBKINETO_NOCUPTI OFF)  # Based on XSIGMA_ENABLE_CUDA
    set(LIBKINETO_NOROCTRACER OFF)  # Based on USE_ROCM
    set(LIBKINETO_NOXPUPTI OFF)  # Based on USE_XPU
    add_subdirectory(third_party/kineto/libkineto)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_KINETO")
endif()

# ITT
if(XSIGMA_ENABLE_ITT)
    find_package(ITT)
    include_directories(${ITT_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${ITT_LIBRARIES})
endif()
```

### XSigma CMake

```cmake
# Kineto
if(XSIGMA_ENABLE_KINETO)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kineto/libkineto/CMakeLists.txt")
        # Wrapper interface enabled
        # Graceful degradation if not fully available
    endif()
endif()

# ITT
if(XSIGMA_ENABLE_ITTAPI)
    set(BUILD_SHARED_LIBS ON)  # Force shared
    add_subdirectory(ittapi)
    set(BUILD_SHARED_LIBS ${_saved_build_shared_libs})
endif()
```

---

## Initialization Comparison

### PyTorch Kineto Init
```cpp
libkineto_init(false, true);  // GPU, logOnError
libkineto::api().initProfilerIfRegistered();
libkineto::api().activityProfiler().prepareTrace(activities);
```

### XSigma Kineto Init
```cpp
auto profiler = xsigma::kineto_profiler::create();
if (profiler) {
    profiler->start_profiling();
    // ... work ...
    profiler->stop_profiling();
}
```

### PyTorch ITT Init
```cpp
__itt_domain* domain = __itt_domain_create("PyTorch");
__itt_task_begin(domain, __itt_null, __itt_null, handle);
// ... work ...
__itt_task_end(domain);
```

### XSigma ITT Init
```cpp
#ifdef XSIGMA_HAS_ITT
__itt_domain* domain = __itt_domain_create("XSigmaApp");
__itt_task_begin(domain, __itt_null, __itt_null, handle);
// ... work ...
__itt_task_end(domain);
#endif
```

---

## GPU Backend Support

Both PyTorch and XSigma support:
- **NVIDIA CUDA**: CUPTI (CUDA Profiling Tools Interface)
- **AMD ROCm**: ROCtracer
- **Intel XPU**: XPUPTI
- **AI Accelerators**: AIUPTI

Configuration via CMake flags:
- `LIBKINETO_NOCUPTI` - Disable CUPTI
- `LIBKINETO_NOROCTRACER` - Disable ROCtracer
- `LIBKINETO_NOXPUPTI` - Disable XPUPTI
- `LIBKINETO_NOAIUPTI` - Disable AIUPTI

---

## Compile Definitions

### PyTorch
```
-DUSE_KINETO              # Kineto enabled
-DUSE_ITT                 # ITT API enabled
-DLIBKINETO_NOCUPTI       # CUPTI disabled
-DLIBKINETO_NOROCTRACER   # ROCtracer disabled
-DLIBKINETO_NOXPUPTI      # XPUPTI disabled
-DKINETO_NAMESPACE=libkineto
-DENABLE_IPC_FABRIC
```

### XSigma
```
-DXSIGMA_HAS_KINETO       # Kineto available
-DXSIGMA_HAS_ITTAPI       # ITT API available
```

---

## File Organization

### PyTorch Profiler Files
```
torch/csrc/profiler/
├── kineto_shim.cpp/h
├── kineto_client_interface.cpp
├── stubs/itt.cpp
├── standalone/itt_observer.cpp
└── python/init.cpp

torch/csrc/
├── itt_wrapper.cpp/h
└── itt.cpp
```

### XSigma Profiler Files
```
XSigma/Library/Core/profiler/
├── kineto_profiler.cxx/h
└── session/profiler.h

XSigma/ThirdParty/
├── kineto/libkineto/
└── ittapi/
```

---

## Environment Variables

### Kineto
- `KINETO_LOG_LEVEL` - Logging level (0=VERBOSE)
- `CUDA_INJECTION64_PATH` - Enable injection mode
- Daemon-related variables (Linux)

### ITT
- `INTEL_LIBITTNOTIFY64` - ITT library path
- `INTEL_VTUNE_PROFILER` - Enable VTune collection

---

## Profiling Output

### Kineto Output
- JSON trace format (Chrome trace viewer compatible)
- TensorBoard trace format
- Custom trace handlers

### ITT Output
- VTune event collection
- Real-time visualization
- Performance analysis reports

---

## Performance Overhead

| Tool | Overhead | Notes |
|------|----------|-------|
| Kineto CPU | 5-10% | Depends on activity types |
| Kineto GPU | 2-5% | Minimal GPU impact |
| Kineto Memory | 10-20% | Significant overhead |
| ITT API | 1-2% | Minimal overhead |

---

## When to Use What

### Use Kineto When:
- Need comprehensive performance analysis
- Profiling GPU operations
- Analyzing memory usage
- Debugging performance bottlenecks
- Comparing different implementations

### Use ITT API When:
- Need lightweight annotations
- Integrating with VTune
- Minimal overhead required
- Real-time monitoring needed
- Task-level profiling sufficient

### Use Both When:
- Comprehensive analysis with VTune integration
- Detailed GPU profiling with task annotations
- Multi-level performance analysis

---

## Troubleshooting Quick Guide

| Issue | PyTorch | XSigma |
|-------|---------|--------|
| Kineto not found | Check `XSIGMA_ENABLE_KINETO=1` | Check `XSIGMA_ENABLE_KINETO=ON` |
| CUPTI not found | Install CUDA toolkit | Install CUDA toolkit |
| ITT not available | Check `XSIGMA_ENABLE_ITT=1` | Check `XSIGMA_ENABLE_ITTAPI=ON` |
| VTune not collecting | Install VTune | Install VTune |
| Shared library missing | N/A | Check ITT shared library |

---

## Related Documentation

1. **KINETO_ITTAPI_COMPARISON.md** - Detailed feature comparison
2. **KINETO_ITTAPI_TECHNICAL_REFERENCE.md** - Code examples and CMake patterns
3. **KINETO_ITTAPI_ARCHITECTURE.md** - Architecture and integration points
4. **KINETO_ITTAPI_USAGE_GUIDE.md** - Practical usage and troubleshooting

---

## Key Takeaways

1. **Kineto** is PyTorch's comprehensive profiling solution with GPU support
2. **ITTAPI** is Intel's lightweight annotation API for VTune integration
3. **PyTorch** uses direct libkineto integration with full feature support
4. **XSigma** uses wrapper-based integration with graceful degradation
5. **Both** support multiple GPU backends (NVIDIA, AMD, Intel XPU)
6. **ITT** is disabled by default in XSigma but enabled in PyTorch
7. **XSigma** forces ITT as shared library (important for Windows DLL distribution)
8. **Overhead** is minimal for ITT (~1-2%) but higher for Kineto memory profiling

---

## References

- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- Kineto: https://github.com/pytorch/kineto
- Intel ITT API: https://github.com/intel/ittapi
- Intel VTune: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html
