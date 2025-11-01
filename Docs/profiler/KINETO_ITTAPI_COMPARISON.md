# Kineto and ITTAPI Implementation Comparison: PyTorch vs XSigma

## Executive Summary

This document provides a detailed comparison of how Kineto (PyTorch profiling library) and ITTAPI (Intel Instrumentation and Tracing Technology API) are integrated and configured in both the PyTorch and XSigma codebases.

---

## 1. KINETO INTEGRATION

### 1.1 PyTorch Implementation

#### Build System Integration
- **CMake Flag**: `XSIGMA_ENABLE_KINETO` (boolean)
- **Location**: `cmake/Dependencies.cmake` (lines 1604-1707)
- **Library Type**: Static (default)
- **Configuration**:
  ```cmake
  set(KINETO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/kineto/libkineto")
  set(KINETO_BUILD_TESTS OFF)
  set(KINETO_LIBRARY_TYPE "static")
  ```

#### GPU Support Configuration
PyTorch uses conditional compilation flags based on available GPU backends:
- **CUPTI** (NVIDIA): `LIBKINETO_NOCUPTI` - Enabled if `XSIGMA_ENABLE_CUDA=ON`
- **ROCtracer** (AMD): `LIBKINETO_NOROCTRACER` - Enabled if `USE_ROCM=ON`
- **XPUPTI** (Intel XPU): `LIBKINETO_NOXPUPTI` - Enabled if `USE_XPU=ON` AND `XPU_ENABLE_KINETO=ON`
- **CPU-only fallback**: Automatically used if no GPU backends available

#### Compile Definitions
```cmake
-DUSE_KINETO
-DLIBKINETO_NOCUPTI (if CUPTI disabled)
-DLIBKINETO_NOROCTRACER (if ROCtracer disabled)
-DLIBKINETO_NOXPUPTI=ON/OFF
-DKINETO_NAMESPACE=libkineto
-DENABLE_IPC_FABRIC
```

#### Initialization Pattern
- **Function**: `libkineto_init(bool cpuOnly, bool logOnError)`
- **Location**: `third_party/kineto/libkineto/src/init.cpp`
- **Initialization Steps**:
  1. Set log level from `KINETO_LOG_LEVEL` environment variable
  2. Register daemon config loader (Linux only)
  3. Setup CUPTI initialization callback (if HAS_CUPTI)
  4. Initialize CUPTI Range Profiler (if supported)
  5. Register activity profilers (XPUPTI, AIUPTI)
  6. Initialize profilers if daemon mode enabled

#### Usage Pattern (PyTorch)
```cpp
// In torch/csrc/profiler/kineto_shim.cpp
libkineto::api().resetKinetoTLS();
if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(cpuOnly, true);
    libkineto::api().suppressLogMessages();
}
if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
}
libkineto::api().activityProfiler().prepareTrace(k_activities, configStr);
```

#### Linking
- Added to `Caffe2_DEPENDENCY_LIBS`
- Linked in `TorchConfig.cmake.in` when `XSIGMA_ENABLE_KINETO=ON`

---

### 1.2 XSigma Implementation

#### Build System Integration
- **CMake Flag**: `XSIGMA_ENABLE_KINETO` (boolean, default: ON)
- **Location**: `XSigma/CMakeLists.txt` (line 152)
- **Wrapper Mode**: Graceful degradation support
- **Configuration**:
  ```cmake
  if(XSIGMA_ENABLE_KINETO)
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kineto/libkineto/CMakeLists.txt")
          message(STATUS "PyTorch Kineto found - wrapper interface enabled")
      endif()
  endif()
  ```

#### GPU Support Configuration
- Same as PyTorch (inherits from third_party/kineto)
- Supports CUPTI, ROCtracer, XPUPTI, AIUPTI

#### Compile Definitions
- `XSIGMA_HAS_KINETO` (when enabled and target available)
- Inherits all PyTorch Kineto definitions

#### Initialization Pattern
- **Wrapper Class**: `xsigma::kineto_profiler`
- **Location**: `XSigma/Library/Core/profiler/kineto_profiler.cxx`
- **Key Feature**: Graceful degradation (commented-out full initialization)
- **Current Implementation**:
  ```cpp
  bool kineto_profiler::initialize(bool cpu_only) {
      // Currently returns true without full initialization
      // Full Kineto integration requires manual dependency setup
      initialized_ = true;
      return true;
  }
  ```

#### Usage Pattern (XSigma)
```cpp
#ifdef XSIGMA_HAS_KINETO
auto profiler = xsigma::kineto_profiler::create();
if (profiler) {
    if (profiler->start_profiling()) {
        // Your code
        profiler->stop_profiling();
    }
}
#endif
```

#### Linking
- Conditionally linked to Core library if `XSIGMA_ENABLE_KINETO=ON`
- Target: `XSigma::kineto`
- Graceful fallback if target not available

#### Key Differences from PyTorch
1. **Wrapper Interface**: XSigma provides abstraction layer
2. **Graceful Degradation**: Wrapper mode allows compilation without full Kineto
3. **Conditional Linking**: Only links if target available
4. **Documentation**: Explicit notes about manual dependency setup

---

## 2. ITTAPI INTEGRATION

### 2.1 PyTorch Implementation

#### Build System Integration
- **CMake Flag**: `XSIGMA_ENABLE_ITT` (boolean)
- **Location**: `cmake/Dependencies.cmake` (lines 730-741)
- **Find Module**: `cmake/Modules/FindITT.cmake`
- **Library Type**: Static (ittnotify)
- **Configuration**:
  ```cmake
  find_package(ITT)
  if(ITT_FOUND)
      include_directories(SYSTEM ${ITT_INCLUDE_DIR})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${ITT_LIBRARIES})
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ITT_LIBRARIES})
  endif()
  ```

#### Source Files Included
- `torch/csrc/itt_wrapper.cpp` - C++ wrapper functions
- `torch/csrc/profiler/stubs/itt.cpp` - Profiler stub implementation
- `torch/csrc/itt.cpp` - Python bindings

#### Compile Definitions
- `XSIGMA_ENABLE_ITT` (added to TORCH_PYTHON_COMPILE_DEFINITIONS)

#### Initialization Pattern
- **Domain Creation**: `__itt_domain_create("PyTorch")`
- **Location**: `torch/csrc/itt_wrapper.cpp`
- **Static Domain**: Single global domain for all PyTorch operations
- **Functions Exposed**:
  ```cpp
  bool itt_is_available()
  void itt_range_push(const char* msg)
  void itt_range_pop()
  void itt_mark(const char* msg)
  ```

#### Usage Pattern (PyTorch)
```cpp
// C++ level
__itt_domain* domain = __itt_domain_create("PyTorch");
__itt_string_handle* hsMsg = __itt_string_handle_create(msg);
__itt_task_begin(domain, __itt_null, __itt_null, hsMsg);
// ... work ...
__itt_task_end(domain);

// Python level (via torch.profiler._itt)
torch.profiler._itt.rangePush("task_name")
torch.profiler._itt.rangePop()
torch.profiler._itt.mark("marker_name")
```

#### Linking
- Added to `Caffe2_DEPENDENCY_LIBS`
- Added to `TORCH_PYTHON_LINK_LIBRARIES`
- Linked in `TorchConfig.cmake.in` when `XSIGMA_ENABLE_ITT=ON`

#### Observer Integration
- **ITT Observer**: `torch/csrc/profiler/standalone/itt_observer.cpp`
- **Thread-Local State**: `ITTThreadLocalState` for thread-safe operation
- **Integration**: Registered as profiler observer for automatic instrumentation

---

### 2.2 XSigma Implementation

#### Build System Integration
- **CMake Flag**: `XSIGMA_ENABLE_ITTAPI` (boolean, default: OFF)
- **Location**: `XSigma/CMakeLists.txt` (line 157)
- **Library Type**: Shared (forced)
- **Configuration**:
  ```cmake
  if(XSIGMA_ENABLE_ITTAPI)
      set(_saved_build_shared_libs ${BUILD_SHARED_LIBS})
      set(BUILD_SHARED_LIBS ON)  # Force shared library
      add_subdirectory(ittapi ${CMAKE_CURRENT_BINARY_DIR}/ittapi_build)
      set(BUILD_SHARED_LIBS ${_saved_build_shared_libs})
  endif()
  ```

#### Targets Created
- `ittnotify` - Main ITT API library
- `jitprofiling` - JIT profiling support

#### Compile Definitions
- `XSIGMA_HAS_ITT` (when enabled)

#### Initialization Pattern
- **Domain Creation**: Same as PyTorch
- **Location**: User code (not pre-initialized)
- **Example**:
  ```cpp
  #ifdef XSIGMA_HAS_ITT
  __itt_domain* domain = __itt_domain_create("XSigmaProfiler");
  auto handle = __itt_string_handle_create("ProfiledTask");
  __itt_task_begin(domain, __itt_null, __itt_null, handle);
  #endif
  ```

#### Usage Pattern (XSigma)
```cpp
#ifdef XSIGMA_HAS_ITT
__itt_domain* domain = __itt_domain_create("XSigmaProfiler");
auto handle = __itt_string_handle_create("ProfiledTask");
__itt_task_begin(domain, __itt_null, __itt_null, handle);
// ... work ...
__itt_task_end(domain);
#endif
```

#### Linking
- Conditionally linked to Core library if `XSIGMA_ENABLE_ITTAPI=ON`
- Target: `XSigma::ittapi`
- Shared library requirement (DLL on Windows)

#### Key Differences from PyTorch
1. **Shared Library**: XSigma forces shared library build (vs static in PyTorch)
2. **Default State**: Disabled by default (vs enabled in PyTorch)
3. **No Python Bindings**: XSigma doesn't expose ITT API to Python
4. **Manual Domain Management**: User responsible for domain creation
5. **Documentation**: Includes VTune integration guide

---

## 3. KEY DIFFERENCES SUMMARY

| Aspect | PyTorch | XSigma |
|--------|---------|--------|
| **Kineto Default** | Enabled (if XSIGMA_ENABLE_KINETO=ON) | Enabled (XSIGMA_ENABLE_KINETO=ON) |
| **Kineto Library Type** | Static | Static (inherited) |
| **Kineto Wrapper** | Direct libkineto usage | Abstraction wrapper with graceful degradation |
| **ITT Default** | Enabled (if XSIGMA_ENABLE_ITT=ON) | Disabled (XSIGMA_ENABLE_ITTAPI=OFF) |
| **ITT Library Type** | Static | Shared (forced) |
| **ITT Python Bindings** | Yes (torch.profiler._itt) | No |
| **ITT Domain Management** | Pre-initialized global domain | User-managed domains |
| **GPU Backend Support** | CUPTI, ROCtracer, XPUPTI, AIUPTI | Same (inherited) |
| **Graceful Degradation** | Limited | Explicit wrapper mode |
| **Documentation** | Inline code comments | Comprehensive markdown docs |

---

## 4. BUILD FLAGS AND DEPENDENCIES

### PyTorch Kineto Flags
- `XSIGMA_ENABLE_KINETO` - Enable Kineto profiling
- `XSIGMA_ENABLE_CUDA` - Enable CUPTI support
- `USE_ROCM` - Enable ROCtracer support
- `USE_XPU` - Enable XPUPTI support
- `XPU_ENABLE_KINETO` - Enable Kineto for XPU
- `USE_CUPTI_SO` - Use CUPTI as shared object

### PyTorch ITT Flags
- `XSIGMA_ENABLE_ITT` - Enable ITT API support

### XSigma Kineto Flags
- `XSIGMA_ENABLE_KINETO` - Enable Kineto (default: ON)

### XSigma ITT Flags
- `XSIGMA_ENABLE_ITTAPI` - Enable ITT API (default: OFF)

---

## 5. CONDITIONAL COMPILATION

### PyTorch
```cpp
#ifdef XSIGMA_ENABLE_KINETO
    // Kineto code
#endif

#ifdef XSIGMA_ENABLE_ITT
    // ITT code
#endif
```

### XSigma
```cpp
#ifdef XSIGMA_HAS_KINETO
    // Kineto code
#endif

#ifdef XSIGMA_HAS_ITT
    // ITT code
#endif
```

---

## 6. NOTABLE IMPLEMENTATION DETAILS

### PyTorch Kineto
- Automatic CUPTI initialization callback setup
- Range Profiler support (fbcode only)
- Daemon mode support (Linux)
- Injection mode via `CUDA_INJECTION64_PATH`
- Comprehensive error handling

### XSigma Kineto
- Wrapper-based abstraction
- Graceful degradation without full Kineto
- Thread-safe initialization with mutex
- Documented manual dependency setup requirement

### PyTorch ITT
- Global domain per process
- Python-level profiler integration
- Observer-based automatic instrumentation
- Thread-local state management

### XSigma ITT
- Shared library requirement (Windows DLL)
- User-managed domain creation
- VTune integration documentation
- Minimal overhead (~1-2%)

---

## 7. RECOMMENDATIONS

1. **For PyTorch Users**: Use `XSIGMA_ENABLE_KINETO=1` and `XSIGMA_ENABLE_ITT=1` for full profiling support
2. **For XSigma Users**: Enable `XSIGMA_ENABLE_KINETO=ON` for profiling; enable `XSIGMA_ENABLE_ITTAPI=ON` for VTune integration
3. **GPU Profiling**: Ensure CUDA/ROCm/XPU toolkits are available for GPU-specific profiling
4. **Windows Deployment**: Note XSigma's ITT API shared library requirement for DLL distribution
