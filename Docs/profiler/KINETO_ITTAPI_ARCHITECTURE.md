# Kineto and ITTAPI Architecture and Integration Points

## 1. PYTORCH ARCHITECTURE

### Kineto Integration Points

```
PyTorch Application
    ↓
torch.profiler (Python API)
    ↓
torch/csrc/profiler/python/init.cpp
    ↓
torch/csrc/profiler/kineto_shim.cpp (Kineto wrapper)
    ↓
libkineto::api() (Kineto C++ API)
    ↓
├─ ActivityProfilerProxy (CPU profiling)
├─ CuptiActivityProfiler (NVIDIA GPU)
├─ RoctracerActivityProfiler (AMD GPU)
├─ XPUActivityProfiler (Intel XPU)
└─ AIUActivityProfiler (AI Accelerator)
    ↓
GPU Drivers / Runtime APIs
```

### ITT Integration Points

```
PyTorch Application
    ↓
torch.profiler (Python API)
    ↓
torch/csrc/profiler/standalone/itt_observer.cpp
    ↓
torch/csrc/itt_wrapper.cpp
    ↓
libittnotify (ITT API)
    ↓
Intel VTune Profiler
```

### File Organization (PyTorch)

```
torch/csrc/profiler/
├── kineto_shim.cpp/h          # Kineto wrapper
├── kineto_client_interface.cpp # Kineto client
├── stubs/
│   ├── base.cpp/h              # Profiler stub base
│   └── itt.cpp                 # ITT stub implementation
├── standalone/
│   ├── itt_observer.cpp/h      # ITT observer
│   ├── nvtx_observer.cpp/h     # NVIDIA NVTX observer
│   └── execution_trace_observer.cpp/h
├── python/
│   └── init.cpp/h              # Python bindings
└── orchestration/
    └── observer.cpp/h          # Observer orchestration

torch/csrc/
├── itt_wrapper.cpp/h           # ITT C++ wrapper
└── itt.cpp                     # ITT Python bindings
```

---

## 2. XSIGMA ARCHITECTURE

### Kineto Integration Points

```
XSigma Application
    ↓
XSigma::kineto_profiler (Wrapper)
    ↓
libkineto::api() (if available)
    ↓
├─ ActivityProfilerProxy (CPU)
├─ CuptiActivityProfiler (NVIDIA)
├─ RoctracerActivityProfiler (AMD)
├─ XPUActivityProfiler (Intel XPU)
└─ AIUActivityProfiler (AI Accelerator)
    ↓
GPU Drivers / Runtime APIs

Note: Graceful degradation if Kineto not fully available
```

### ITT Integration Points

```
XSigma Application
    ↓
User Code (Direct ITT API calls)
    ↓
libittnotify (Shared Library)
    ↓
Intel VTune Profiler
```

### File Organization (XSigma)

```
XSigma/Library/Core/profiler/
├── kineto_profiler.cxx/h       # Kineto wrapper
├── session/
│   └── profiler.h              # Session-based profiler
└── ...

XSigma/ThirdParty/
├── kineto/
│   └── libkineto/              # PyTorch Kineto (submodule)
└── ittapi/                     # Intel ITT API (submodule)
```

---

## 3. BUILD SYSTEM INTEGRATION

### PyTorch Build Flow

```
CMake Configuration
    ↓
cmake/Dependencies.cmake
    ├─ XSIGMA_ENABLE_KINETO flag
    │   ├─ Check XSIGMA_ENABLE_CUDA → LIBKINETO_NOCUPTI
    │   ├─ Check USE_ROCM → LIBKINETO_NOROCTRACER
    │   ├─ Check USE_XPU → LIBKINETO_NOXPUPTI
    │   └─ Add third_party/kineto/libkineto
    │
    └─ XSIGMA_ENABLE_ITT flag
        ├─ find_package(ITT)
        ├─ Include ITT headers
        └─ Link ITT libraries
    ↓
caffe2/CMakeLists.txt
    ├─ Add ITT source files (if XSIGMA_ENABLE_ITT)
    └─ Add Kineto to dependencies (if XSIGMA_ENABLE_KINETO)
    ↓
torch/CMakeLists.txt
    ├─ Add ITT Python bindings (if XSIGMA_ENABLE_ITT)
    └─ Link Kineto library (if XSIGMA_ENABLE_KINETO)
    ↓
Compilation with -DUSE_KINETO and/or -DUSE_ITT
```

### XSigma Build Flow

```
CMake Configuration
    ↓
XSigma/CMakeLists.txt
    ├─ XSIGMA_ENABLE_KINETO (default: ON)
    └─ XSIGMA_ENABLE_ITTAPI (default: OFF)
    ↓
XSigma/ThirdParty/CMakeLists.txt
    ├─ Kineto configuration
    │   └─ Check if kineto/libkineto/CMakeLists.txt exists
    │
    └─ ITT API configuration
        ├─ Force BUILD_SHARED_LIBS=ON
        ├─ Add ittapi subdirectory
        └─ Restore BUILD_SHARED_LIBS state
    ↓
XSigma/Library/Core/CMakeLists.txt
    ├─ Link XSigma::kineto (if available)
    │   └─ Define XSIGMA_HAS_KINETO
    │
    └─ Link XSigma::ittapi (if available)
        └─ Define XSIGMA_HAS_ITT
    ↓
Compilation with -DXSIGMA_HAS_KINETO and/or -DXSIGMA_HAS_ITTAPI
```

---

## 4. INITIALIZATION SEQUENCE

### PyTorch Kineto Initialization

```
1. Application starts
2. torch.profiler.profile() called
3. kineto_shim::prepareTrace()
   a. libkineto::api().resetKinetoTLS()
   b. if (!isProfilerRegistered()) → libkineto_init()
      - Set log level from KINETO_LOG_LEVEL env var
      - Register daemon config loader (Linux)
      - Setup CUPTI callback (if HAS_CUPTI)
      - Initialize Range Profiler (if supported)
      - Register activity profilers
   c. if (!isProfilerInitialized()) → initProfilerIfRegistered()
   d. Prepare trace with activity types
4. Profiling starts
5. Activities collected
6. Trace processed and saved
```

### PyTorch ITT Initialization

```
1. Application starts
2. torch.profiler.profile() called with ITT observer
3. itt_observer::enterITT() called for each operation
   a. Check ITTThreadLocalState
   b. Call itt_range_push(operation_name)
      - Create string handle
      - Call __itt_task_begin()
4. Operation executes
5. itt_observer::exitITT() called
   a. Call itt_range_pop()
      - Call __itt_task_end()
6. VTune collects ITT events
```

### XSigma Kineto Initialization

```
1. Application starts
2. xsigma::kineto_profiler::create() called
3. kineto_profiler::initialize()
   a. Lock mutex
   b. Check if already initialized
   c. Return true (graceful degradation)
      - Full Kineto init commented out
      - Allows compilation without full dependencies
4. Profiler instance created
5. start_profiling() called
6. Profiling executes (if Kineto available)
7. stop_profiling() called
```

### XSigma ITT Initialization

```
1. Application starts
2. User code includes <ittnotify.h>
3. User creates domain: __itt_domain_create("DomainName")
4. User creates string handle: __itt_string_handle_create("TaskName")
5. User calls __itt_task_begin(domain, ...)
6. Work executes
7. User calls __itt_task_end(domain)
8. VTune collects ITT events
```

---

## 5. DEPENDENCY GRAPH

### PyTorch Dependencies

```
PyTorch Core
    ├─ Kineto (optional)
    │   ├─ CUPTI (optional, if XSIGMA_ENABLE_CUDA)
    │   ├─ ROCtracer (optional, if USE_ROCM)
    │   ├─ XPUPTI (optional, if USE_XPU)
    │   ├─ AIUPTI (optional)
    │   └─ fmt (header-only)
    │
    └─ ITT API (optional)
        └─ ittnotify (static library)
```

### XSigma Dependencies

```
XSigma Core
    ├─ Kineto (optional, XSIGMA_ENABLE_KINETO)
    │   ├─ CUPTI (optional)
    │   ├─ ROCtracer (optional)
    │   ├─ XPUPTI (optional)
    │   ├─ AIUPTI (optional)
    │   └─ fmt (header-only)
    │
    └─ ITT API (optional, XSIGMA_ENABLE_ITTAPI)
        ├─ ittnotify (shared library)
        └─ jitprofiling (shared library)
```

---

## 6. CONDITIONAL COMPILATION SYMBOLS

### PyTorch Symbols

```
XSIGMA_ENABLE_KINETO                    # Kineto enabled
LIBKINETO_NOCUPTI            # CUPTI disabled
LIBKINETO_NOROCTRACER        # ROCtracer disabled
LIBKINETO_NOXPUPTI           # XPUPTI disabled
LIBKINETO_NOAIUPTI           # AIUPTI disabled
HAS_CUPTI                    # CUPTI available
HAS_XPUPTI                   # XPUPTI available
HAS_AIUPTI                   # AIUPTI available
USE_CUPTI_RANGE_PROFILER     # Range Profiler enabled
KINETO_NAMESPACE=libkineto   # Namespace definition
ENABLE_IPC_FABRIC            # IPC fabric enabled
XSIGMA_ENABLE_ITT                      # ITT API enabled
```

### XSigma Symbols

```
XSIGMA_HAS_KINETO            # Kineto available
XSIGMA_HAS_ITT            # ITT API available
XSIGMA_ENABLE_KINETO         # Kineto enabled (CMake)
XSIGMA_ENABLE_ITTAPI         # ITT API enabled (CMake)
```

---

## 7. PROFILER OBSERVER CHAIN

### PyTorch Observer Registration

```
torch::profiler::impl::ProfilerStateBase
    ├─ CUDAProfilerState (CUDA profiling)
    ├─ ITTThreadLocalState (ITT profiling)
    ├─ NVTXThreadLocalState (NVIDIA NVTX)
    └─ PrivateUse1ThreadLocalState (Custom device)

Observer callbacks:
    ├─ enterITT() → itt_range_push()
    ├─ exitITT() → itt_range_pop()
    ├─ enterNVTX() → nvtxRangePush()
    └─ exitNVTX() → nvtxRangePop()
```

---

## 8. PROFILING OUTPUT FORMATS

### Kineto Output
- JSON trace format (Chrome trace format)
- Tensorboard trace format
- Custom trace handlers

### ITT Output
- VTune event collection
- Real-time visualization in VTune GUI
- Performance analysis reports

---

## 9. THREAD SAFETY

### PyTorch
- Thread-local state management via `ProfilerStateBase::get()`
- Mutex protection in Kineto initialization
- Thread-safe activity collection

### XSigma
- Mutex-protected initialization in `kineto_profiler`
- User responsible for thread safety in ITT API usage
- Shared library ITT API handles thread safety internally
