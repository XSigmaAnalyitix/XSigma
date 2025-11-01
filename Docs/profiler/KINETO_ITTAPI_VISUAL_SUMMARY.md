# Kineto and ITTAPI Visual Summary

## 1. INTEGRATION ARCHITECTURE COMPARISON

### PyTorch Integration Model
```
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch Application                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐         ┌──────────────────────┐  │
│  │  torch.profiler      │         │  torch.profiler._itt │  │
│  │  (Python API)        │         │  (Python API)        │  │
│  └──────────┬───────────┘         └──────────┬───────────┘  │
│             │                                 │               │
│  ┌──────────▼──────────────────────────────────▼───────────┐ │
│  │         torch/csrc/profiler/                            │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │  kineto_shim.cpp (Kineto wrapper)              │   │ │
│  │  │  itt_observer.cpp (ITT observer)               │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  └──────────┬──────────────────────────────────────┬───────┘ │
│             │                                      │          │
│  ┌──────────▼──────────────┐      ┌───────────────▼────────┐ │
│  │  libkineto::api()       │      │  libittnotify          │ │
│  │  ├─ CPU Profiler       │      │  ├─ Domain API        │ │
│  │  ├─ CUDA Profiler      │      │  ├─ Task API          │ │
│  │  ├─ ROCm Profiler      │      │  └─ String Handles    │ │
│  │  └─ XPU Profiler       │      └────────────────────────┘ │
│  └────────────────────────┘                                  │
│             │                                                 │
│  ┌──────────▼──────────────────────────────────────────────┐ │
│  │  GPU Drivers / Runtime APIs                            │ │
│  │  (CUDA, ROCm, oneAPI)                                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### XSigma Integration Model
```
┌─────────────────────────────────────────────────────────────┐
│                    XSigma Application                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐         ┌──────────────────────┐  │
│  │  xsigma::            │         │  Direct ITT API      │  │
│  │  kineto_profiler     │         │  (User Code)         │  │
│  │  (C++ Wrapper)       │         │                      │  │
│  └──────────┬───────────┘         └──────────┬───────────┘  │
│             │                                 │               │
│  ┌──────────▼──────────────────────────────────▼───────────┐ │
│  │         XSigma/Library/Core/profiler/                   │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │  kineto_profiler.cxx (Wrapper)                 │   │ │
│  │  │  Graceful degradation support                  │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  └──────────┬──────────────────────────────────────┬───────┘ │
│             │                                      │          │
│  ┌──────────▼──────────────┐      ┌───────────────▼────────┐ │
│  │  libkineto::api()       │      │  libittnotify (shared) │ │
│  │  (if available)         │      │  ├─ Domain API        │ │
│  │  ├─ CPU Profiler       │      │  ├─ Task API          │ │
│  │  ├─ CUDA Profiler      │      │  └─ String Handles    │ │
│  │  ├─ ROCm Profiler      │      └────────────────────────┘ │
│  │  └─ XPU Profiler       │                                  │
│  └────────────────────────┘                                  │
│             │                                                 │
│  ┌──────────▼──────────────────────────────────────────────┐ │
│  │  GPU Drivers / Runtime APIs                            │ │
│  │  (CUDA, ROCm, oneAPI)                                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. BUILD SYSTEM FLOW

### PyTorch Build Flow
```
CMake Configuration
    │
    ├─ XSIGMA_ENABLE_KINETO flag
    │   ├─ Check XSIGMA_ENABLE_CUDA → LIBKINETO_NOCUPTI
    │   ├─ Check USE_ROCM → LIBKINETO_NOROCTRACER
    │   ├─ Check USE_XPU → LIBKINETO_NOXPUPTI
    │   └─ add_subdirectory(third_party/kineto)
    │
    └─ XSIGMA_ENABLE_ITT flag
        ├─ find_package(ITT)
        ├─ include_directories(${ITT_INCLUDE_DIR})
        └─ link_libraries(${ITT_LIBRARIES})
    │
    ▼
Compilation
    ├─ -DUSE_KINETO
    ├─ -DUSE_ITT
    └─ GPU-specific flags
    │
    ▼
Linking
    ├─ libkineto (static)
    └─ libittnotify (static)
```

### XSigma Build Flow
```
CMake Configuration
    │
    ├─ XSIGMA_ENABLE_KINETO (default: ON)
    │   └─ Check if kineto/libkineto/CMakeLists.txt exists
    │
    └─ XSIGMA_ENABLE_ITTAPI (default: OFF)
        ├─ Force BUILD_SHARED_LIBS=ON
        ├─ add_subdirectory(ittapi)
        └─ Restore BUILD_SHARED_LIBS
    │
    ▼
Compilation
    ├─ -DXSIGMA_HAS_KINETO (if available)
    └─ -DXSIGMA_HAS_ITTAPI (if available)
    │
    ▼
Linking
    ├─ libkineto (static, inherited)
    └─ libittnotify (shared, forced)
```

---

## 3. FEATURE COMPARISON MATRIX

```
┌──────────────────────┬──────────────┬──────────────┐
│ Feature              │ PyTorch      │ XSigma       │
├──────────────────────┼──────────────┼──────────────┤
│ Kineto Default       │ ✓ ON         │ ✓ ON         │
│ Kineto Wrapper       │ Direct       │ Abstraction  │
│ Graceful Degrad.     │ Limited      │ ✓ Explicit   │
│ ITT Default          │ ✓ ON         │ ✗ OFF        │
│ ITT Library Type     │ Static       │ Shared (DLL) │
│ ITT Python Bindings  │ ✓ Yes        │ ✗ No         │
│ GPU Support          │ ✓ Full       │ ✓ Full       │
│ CUPTI Support        │ ✓ Yes        │ ✓ Yes        │
│ ROCtracer Support    │ ✓ Yes        │ ✓ Yes        │
│ XPUPTI Support       │ ✓ Yes        │ ✓ Yes        │
│ AIUPTI Support       │ ✓ Yes        │ ✓ Yes        │
│ Thread-safe          │ ✓ Yes        │ ✓ Yes        │
│ Documentation        │ Code         │ Markdown     │
└──────────────────────┴──────────────┴──────────────┘
```

---

## 4. INITIALIZATION SEQUENCE

### PyTorch Kineto Init
```
Application Start
    │
    ▼
torch.profiler.profile()
    │
    ▼
kineto_shim::prepareTrace()
    │
    ├─ libkineto::api().resetKinetoTLS()
    │
    ├─ if (!isProfilerRegistered())
    │   └─ libkineto_init(cpuOnly, logOnError)
    │       ├─ Set log level
    │       ├─ Register daemon (Linux)
    │       ├─ Setup CUPTI callback
    │       ├─ Initialize Range Profiler
    │       └─ Register profilers
    │
    ├─ if (!isProfilerInitialized())
    │   └─ initProfilerIfRegistered()
    │
    └─ activityProfiler().prepareTrace(activities)
    │
    ▼
Profiling Active
```

### XSigma Kineto Init
```
Application Start
    │
    ▼
xsigma::kineto_profiler::create()
    │
    ▼
kineto_profiler::initialize()
    │
    ├─ Lock mutex
    │
    ├─ Check if already initialized
    │
    └─ Return true (graceful degradation)
       └─ Full Kineto init commented out
    │
    ▼
Profiler Instance Created
    │
    ▼
start_profiling()
    │
    ▼
Profiling Active (if Kineto available)
```

---

## 5. PERFORMANCE OVERHEAD COMPARISON

```
Operation                PyTorch    XSigma     Notes
─────────────────────────────────────────────────────
CPU Profiling            5-10%      5-10%      Same (via Kineto)
GPU Profiling            2-5%       2-5%       Same (via Kineto)
Memory Profiling         10-20%     10-20%     Same (via Kineto)
ITT Annotations          1-2%       1-2%       Minimal
No Profiling             0%         0%         No overhead
```

---

## 6. GPU BACKEND SUPPORT

```
┌─────────────────────────────────────────────────────┐
│              GPU Backend Support                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  NVIDIA CUDA                                        │
│  ├─ PyTorch: ✓ CUPTI (via libkineto)              │
│  └─ XSigma:  ✓ CUPTI (via libkineto)              │
│                                                     │
│  AMD ROCm                                           │
│  ├─ PyTorch: ✓ ROCtracer (via libkineto)          │
│  └─ XSigma:  ✓ ROCtracer (via libkineto)          │
│                                                     │
│  Intel XPU                                          │
│  ├─ PyTorch: ✓ XPUPTI (via libkineto)             │
│  └─ XSigma:  ✓ XPUPTI (via libkineto)             │
│                                                     │
│  AI Accelerators                                    │
│  ├─ PyTorch: ✓ AIUPTI (via libkineto)             │
│  └─ XSigma:  ✓ AIUPTI (via libkineto)             │
│                                                     │
│  CPU-only Fallback                                  │
│  ├─ PyTorch: ✓ Yes                                 │
│  └─ XSigma:  ✓ Yes                                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 7. DECISION TREE

```
Do you need profiling?
    │
    ├─ NO → No profiling needed
    │
    └─ YES
        │
        ├─ Need comprehensive GPU profiling?
        │   │
        │   ├─ YES → Use Kineto
        │   │   ├─ PyTorch: XSIGMA_ENABLE_KINETO=1
        │   │   └─ XSigma: XSIGMA_ENABLE_KINETO=ON
        │   │
        │   └─ NO → Continue
        │
        ├─ Need lightweight annotations?
        │   │
        │   ├─ YES → Use ITT API
        │   │   ├─ PyTorch: XSIGMA_ENABLE_ITT=1
        │   │   └─ XSigma: XSIGMA_ENABLE_ITTAPI=ON
        │   │
        │   └─ NO → Continue
        │
        └─ Need both?
            │
            ├─ YES → Enable both
            │   ├─ PyTorch: XSIGMA_ENABLE_KINETO=1 XSIGMA_ENABLE_ITT=1
            │   └─ XSigma: Both flags ON
            │
            └─ NO → Choose based on needs
```

---

## 8. FILE ORGANIZATION

### PyTorch
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

### XSigma
```
XSigma/Library/Core/profiler/
├── kineto_profiler.cxx/h
└── session/profiler.h

XSigma/ThirdParty/
├── kineto/libkineto/
└── ittapi/
```

---

## 9. QUICK REFERENCE COMMANDS

### PyTorch Build
```bash
# Enable Kineto and ITT
XSIGMA_ENABLE_KINETO=1 XSIGMA_ENABLE_ITT=1 python setup.py install

# With CUDA
XSIGMA_ENABLE_KINETO=1 XSIGMA_ENABLE_ITT=1 XSIGMA_ENABLE_CUDA=1 python setup.py install
```

### XSigma Build
```bash
# Enable both
cmake -DXSIGMA_ENABLE_KINETO=ON -DXSIGMA_ENABLE_ITTAPI=ON ..

# Kineto only
cmake -DXSIGMA_ENABLE_KINETO=ON ..

# ITT only
cmake -DXSIGMA_ENABLE_ITTAPI=ON ..
```

---

## 10. TROUBLESHOOTING FLOWCHART

```
Profiling not working?
    │
    ├─ Kineto issue?
    │   ├─ Check: cmake -LA | grep XSIGMA_ENABLE_KINETO
    │   ├─ Check: find /usr -name "libkineto*"
    │   └─ Solution: Rebuild with XSIGMA_ENABLE_KINETO=1
    │
    ├─ ITT issue?
    │   ├─ Check: cmake -LA | grep XSIGMA_ENABLE_ITT
    │   ├─ Check: find /usr -name "libittnotify*"
    │   └─ Solution: Rebuild with XSIGMA_ENABLE_ITT=1
    │
    ├─ CUDA/GPU issue?
    │   ├─ Check: CUDA toolkit installed
    │   ├─ Check: CUPTI available
    │   └─ Solution: Install CUDA toolkit
    │
    └─ VTune not collecting?
        ├─ Check: VTune installed
        ├─ Check: ITT library path
        └─ Solution: Run with vtune -collect hotspots
```

---

**Visual Summary Complete**
All diagrams and flowcharts provided for quick reference.
