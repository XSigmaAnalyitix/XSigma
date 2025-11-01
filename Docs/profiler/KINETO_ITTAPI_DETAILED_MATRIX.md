# Kineto and ITTAPI Detailed Comparison Matrix

## 1. BUILD SYSTEM INTEGRATION

| Aspect | PyTorch | XSigma |
|--------|---------|--------|
| **Kineto CMake Flag** | `XSIGMA_ENABLE_KINETO` | `XSIGMA_ENABLE_KINETO` |
| **Kineto Default** | ON (if dependencies available) | ON |
| **Kineto Library Type** | Static | Static (inherited) |
| **Kineto Location** | `third_party/kineto/libkineto` | `XSigma/ThirdParty/kineto/libkineto` |
| **ITT CMake Flag** | `XSIGMA_ENABLE_ITT` | `XSIGMA_ENABLE_ITTAPI` |
| **ITT Default** | ON (if found) | OFF |
| **ITT Library Type** | Static | Shared (forced) |
| **ITT Location** | `third_party/ittapi` | `XSigma/ThirdParty/ittapi` |
| **ITT Build Control** | `find_package(ITT)` | `add_subdirectory(ittapi)` |
| **ITT Shared Lib Requirement** | No | Yes (Windows DLL) |

---

## 2. INITIALIZATION AND CONFIGURATION

| Aspect | PyTorch | XSigma |
|--------|---------|--------|
| **Kineto Init Function** | `libkineto_init(cpuOnly, logOnError)` | `kineto_profiler::initialize(cpu_only)` |
| **Kineto Init Location** | `third_party/kineto/libkineto/src/init.cpp` | `XSigma/Library/Core/profiler/kineto_profiler.cxx` |
| **Kineto Wrapper** | Direct libkineto usage | Abstraction wrapper |
| **Kineto Graceful Degradation** | Limited | Explicit (wrapper mode) |
| **Kineto Thread Safety** | Mutex in libkineto | Mutex in wrapper |
| **ITT Domain Creation** | Pre-initialized global | User-managed |
| **ITT Domain Name** | "PyTorch" | User-defined |
| **ITT Python Bindings** | Yes (`torch.profiler._itt`) | No |
| **ITT Initialization** | Automatic | Manual |

---

## 3. GPU BACKEND SUPPORT

| Backend | PyTorch | XSigma | Flag |
|---------|---------|--------|------|
| **NVIDIA CUDA** | ✓ CUPTI | ✓ CUPTI | `LIBKINETO_NOCUPTI` |
| **AMD ROCm** | ✓ ROCtracer | ✓ ROCtracer | `LIBKINETO_NOROCTRACER` |
| **Intel XPU** | ✓ XPUPTI | ✓ XPUPTI | `LIBKINETO_NOXPUPTI` |
| **AI Accelerators** | ✓ AIUPTI | ✓ AIUPTI | `LIBKINETO_NOAIUPTI` |
| **CPU-only Fallback** | ✓ Yes | ✓ Yes | All disabled |

---

## 4. COMPILE DEFINITIONS

| Definition | PyTorch | XSigma | Purpose |
|-----------|---------|--------|---------|
| `XSIGMA_ENABLE_KINETO` | ✓ | - | Enable Kineto |
| `XSIGMA_HAS_KINETO` | - | ✓ | Kineto available |
| `LIBKINETO_NOCUPTI` | ✓ | ✓ | Disable CUPTI |
| `LIBKINETO_NOROCTRACER` | ✓ | ✓ | Disable ROCtracer |
| `LIBKINETO_NOXPUPTI` | ✓ | ✓ | Disable XPUPTI |
| `LIBKINETO_NOAIUPTI` | ✓ | ✓ | Disable AIUPTI |
| `HAS_CUPTI` | ✓ | ✓ | CUPTI available |
| `HAS_XPUPTI` | ✓ | ✓ | XPUPTI available |
| `HAS_AIUPTI` | ✓ | ✓ | AIUPTI available |
| `USE_CUPTI_RANGE_PROFILER` | ✓ | ✓ | Range profiler |
| `KINETO_NAMESPACE=libkineto` | ✓ | ✓ | Namespace |
| `ENABLE_IPC_FABRIC` | ✓ | ✓ | IPC support |
| `XSIGMA_ENABLE_ITT` | ✓ | - | Enable ITT |
| `XSIGMA_HAS_ITT` | - | ✓ | ITT available |

---

## 5. SOURCE FILES INCLUDED

### PyTorch Kineto Files
- `torch/csrc/profiler/kineto_shim.cpp` - Kineto wrapper
- `torch/csrc/profiler/kineto_client_interface.cpp` - Client interface
- `torch/csrc/profiler/kineto_shim.h` - Header

### PyTorch ITT Files
- `torch/csrc/itt_wrapper.cpp` - C++ wrapper
- `torch/csrc/itt.cpp` - Python bindings
- `torch/csrc/profiler/stubs/itt.cpp` - Profiler stub
- `torch/csrc/itt_wrapper.h` - Header

### XSigma Kineto Files
- `XSigma/Library/Core/profiler/kineto_profiler.cxx` - Wrapper
- `XSigma/Library/Core/profiler/kineto_profiler.h` - Header

### XSigma ITT Files
- User includes `<ittnotify.h>` directly
- No wrapper layer

---

## 6. PROFILING CAPABILITIES

| Capability | PyTorch | XSigma |
|-----------|---------|--------|
| **CPU Profiling** | ✓ Full | ✓ Full (via Kineto) |
| **GPU Profiling** | ✓ Full | ✓ Full (via Kineto) |
| **Memory Profiling** | ✓ Yes | ✓ Yes (via Kineto) |
| **Activity Tracking** | ✓ Comprehensive | ✓ Comprehensive (via Kineto) |
| **Task Annotations** | ✓ ITT API | ✓ ITT API |
| **Frame Profiling** | ✓ ITT API | ✓ ITT API |
| **Domain Organization** | ✓ ITT API | ✓ ITT API |
| **Thread-safe** | ✓ Yes | ✓ Yes |
| **Real-time Monitoring** | ✓ ITT/VTune | ✓ ITT/VTune |

---

## 7. OUTPUT FORMATS

| Format | PyTorch | XSigma |
|--------|---------|--------|
| **JSON Trace** | ✓ Chrome format | ✓ Via Kineto |
| **TensorBoard** | ✓ Yes | ✓ Via Kineto |
| **VTune Events** | ✓ ITT API | ✓ ITT API |
| **Custom Handlers** | ✓ Yes | ✓ Via Kineto |

---

## 8. ENVIRONMENT VARIABLES

| Variable | PyTorch | XSigma | Purpose |
|----------|---------|--------|---------|
| `KINETO_LOG_LEVEL` | ✓ | ✓ | Log level |
| `CUDA_INJECTION64_PATH` | ✓ | ✓ | Injection mode |
| `ROCM_SOURCE_DIR` | ✓ | ✓ | ROCm path |
| `INTEL_LIBITTNOTIFY64` | ✓ | ✓ | ITT library |
| `INTEL_VTUNE_PROFILER` | ✓ | ✓ | VTune enable |

---

## 9. PERFORMANCE OVERHEAD

| Operation | PyTorch | XSigma | Notes |
|-----------|---------|--------|-------|
| **CPU Profiling** | 5-10% | 5-10% | Same (via Kineto) |
| **GPU Profiling** | 2-5% | 2-5% | Same (via Kineto) |
| **Memory Profiling** | 10-20% | 10-20% | Same (via Kineto) |
| **ITT Annotations** | 1-2% | 1-2% | Minimal |
| **No Profiling** | 0% | 0% | No overhead |

---

## 10. PYTHON INTEGRATION

| Feature | PyTorch | XSigma |
|---------|---------|--------|
| **torch.profiler** | ✓ Full | N/A |
| **torch.profiler._itt** | ✓ Yes | N/A |
| **Python Bindings** | ✓ Yes | No |
| **Decorator Support** | ✓ Yes | N/A |
| **Context Manager** | ✓ Yes | N/A |

---

## 11. C++ API USAGE

| API | PyTorch | XSigma |
|-----|---------|--------|
| **libkineto::api()** | ✓ Direct | ✓ Via wrapper |
| **libkineto_init()** | ✓ Direct | ✓ Wrapped |
| **__itt_domain_create()** | ✓ Global | ✓ User-managed |
| **__itt_task_begin()** | ✓ Yes | ✓ Yes |
| **__itt_task_end()** | ✓ Yes | ✓ Yes |
| **__itt_mark()** | ✓ Yes | ✓ Yes |

---

## 12. DOCUMENTATION

| Aspect | PyTorch | XSigma |
|--------|---------|--------|
| **Inline Comments** | ✓ Extensive | ✓ Moderate |
| **Markdown Docs** | ✓ Limited | ✓ Comprehensive |
| **CMake Comments** | ✓ Yes | ✓ Yes |
| **Usage Examples** | ✓ Code samples | ✓ Code samples |
| **Troubleshooting** | ✓ Limited | ✓ Detailed |
| **VTune Integration** | ✓ Limited | ✓ Documented |

---

## 13. ERROR HANDLING

| Aspect | PyTorch | XSigma |
|--------|---------|--------|
| **Kineto Errors** | Exceptions | Return values |
| **ITT Errors** | Silent | Silent |
| **Graceful Degradation** | Limited | Explicit |
| **Fallback Behavior** | CPU-only | Wrapper mode |
| **Error Logging** | Via KINETO_LOG_LEVEL | Via wrapper |

---

## 14. PLATFORM SUPPORT

| Platform | PyTorch | XSigma |
|----------|---------|--------|
| **Linux** | ✓ Full | ✓ Full |
| **Windows** | ✓ Full | ✓ Full (DLL) |
| **macOS** | ✓ Limited | ✓ Limited |
| **Android** | ✓ Limited | N/A |
| **iOS** | ✓ Limited | N/A |

---

## 15. DEPENDENCY REQUIREMENTS

### PyTorch Kineto
- fmt (header-only)
- CUDA toolkit (optional, for CUPTI)
- ROCm (optional, for ROCtracer)
- Intel oneAPI (optional, for XPUPTI)

### PyTorch ITT
- ittapi (third_party)

### XSigma Kineto
- Same as PyTorch (inherited)

### XSigma ITT
- ittapi (third_party, built as shared library)

---

## 16. LINKING STRATEGY

| Aspect | PyTorch | XSigma |
|--------|---------|--------|
| **Kineto Linking** | Static | Static (inherited) |
| **ITT Linking** | Static | Shared |
| **Dependency Libs** | Caffe2_DEPENDENCY_LIBS | Core library |
| **Python Linking** | TORCH_PYTHON_LINK_LIBRARIES | N/A |
| **Whole Archive** | No | No |

---

## 17. CONDITIONAL COMPILATION

| Condition | PyTorch | XSigma |
|-----------|---------|--------|
| **XSIGMA_ENABLE_KINETO** | ✓ | - |
| **XSIGMA_ENABLE_KINETO** | - | ✓ |
| **XSIGMA_ENABLE_ITT** | ✓ | - |
| **XSIGMA_ENABLE_ITTAPI** | - | ✓ |
| **XSIGMA_ENABLE_CUDA** | ✓ | ✓ |
| **USE_ROCM** | ✓ | ✓ |
| **USE_XPU** | ✓ | ✓ |

---

## 18. TESTING AND VALIDATION

| Aspect | PyTorch | XSigma |
|--------|---------|--------|
| **Unit Tests** | ✓ Yes | ✓ Yes |
| **Integration Tests** | ✓ Yes | ✓ Yes |
| **Sample Programs** | ✓ Yes | ✓ Yes |
| **Profiler Tests** | ✓ Comprehensive | ✓ Moderate |
| **GPU Tests** | ✓ Yes | ✓ Yes |

---

## 19. MAINTENANCE AND UPDATES

| Aspect | PyTorch | XSigma |
|--------|---------|--------|
| **Active Development** | ✓ Yes | ✓ Yes |
| **Community Support** | ✓ Large | ✓ Moderate |
| **Issue Tracking** | ✓ GitHub | ✓ Internal |
| **Release Cycle** | Regular | Regular |
| **Backward Compatibility** | ✓ Maintained | ✓ Maintained |

---

## 20. RECOMMENDED USAGE SCENARIOS

### Use PyTorch When:
- Need full PyTorch integration
- Python-level profiling required
- Comprehensive GPU profiling needed
- TensorBoard visualization desired

### Use XSigma When:
- Need wrapper-based abstraction
- Graceful degradation important
- Windows DLL distribution required
- VTune integration primary goal
- Lightweight profiling sufficient
