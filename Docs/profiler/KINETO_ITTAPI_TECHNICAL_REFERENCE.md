# Kineto and ITTAPI Technical Reference

## 1. KINETO INITIALIZATION FLOW

### PyTorch Initialization Sequence

```cpp
// torch/csrc/profiler/kineto_shim.cpp - prepareTrace()
void prepareTrace(const bool cpuOnly, const ActivitySet& activities, ...) {
#ifdef XSIGMA_ENABLE_KINETO
    // Step 1: Reset thread-local storage
    libkineto::api().resetKinetoTLS();
    
    // Step 2: Initialize if not already registered
    if (!libkineto::api().isProfilerRegistered()) {
        libkineto_init(cpuOnly, true);  // Initialize Kineto
        libkineto::api().suppressLogMessages();
    }
    
    // Step 3: Initialize profiler if registered
    if (!libkineto::api().isProfilerInitialized()) {
        libkineto::api().initProfilerIfRegistered();
    }
    
    // Step 4: Prepare trace with activities
    std::set<libkineto::ActivityType> k_activities;
    // ... populate activities ...
    libkineto::api().activityProfiler().prepareTrace(k_activities, configStr);
#endif
}
```

### libkineto_init() Implementation

```cpp
// third_party/kineto/libkineto/src/init.cpp
void libkineto_init(bool cpuOnly, bool logOnError) {
    // Set log level from environment
    const char* logLevelEnv = getenv("KINETO_LOG_LEVEL");
    if (logLevelEnv) {
        SET_LOG_SEVERITY_LEVEL(atoi(logLevelEnv));
    }
    
    // Register daemon config loader (Linux)
#if __linux__
    if (libkineto::isDaemonEnvVarSet()) {
        DaemonConfigLoader::registerFactory();
    }
#endif
    
    // Initialize CUPTI if available
#ifdef HAS_CUPTI
    if (!cpuOnly && !libkineto::isDaemonEnvVarSet()) {
        bool success = setupCuptiInitCallback(logOnError);
        cpuOnly = !success;
    }
    
    if (!cpuOnly && initRangeProfiler) {
        rangeProfilerInit = std::make_unique<CuptiRangeProfilerInit>();
    }
    
    if (!cpuOnly && shouldPreloadCuptiInstrumentation()) {
        CuptiActivityApi::forceLoadCupti();
    }
#endif
    
    // Register profilers
    ConfigLoader& config_loader = libkineto::api().configLoader();
    libkineto::api().registerProfiler(
        std::make_unique<ActivityProfilerProxy>(cpuOnly, config_loader));
    
    // Register GPU-specific profilers
#ifdef HAS_XPUPTI
    libkineto::api().registerProfilerFactory(
        []() -> std::unique_ptr<IActivityProfiler> {
            return std::make_unique<XPUActivityProfiler>();
        });
#endif
    
#ifdef HAS_AIUPTI
    libkineto::api().registerProfilerFactory(
        []() -> std::unique_ptr<IActivityProfiler> {
            return std::make_unique<AIUActivityProfiler>();
        });
#endif
}
```

### XSigma Wrapper Implementation

```cpp
// XSigma/Library/Core/profiler/kineto_profiler.cxx
class kineto_profiler {
private:
    static std::mutex init_mutex_;
    static bool initialized_;
    
public:
    static bool initialize(bool cpu_only) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        
        if (initialized_) {
            return true;  // Already initialized
        }
        
        // Graceful degradation: wrapper mode
        // Full Kineto integration commented out for manual setup
        /*
#ifdef XSIGMA_HAS_KINETO
        try {
            libkineto_init(cpu_only, false);
            initialized_ = true;
            return true;
        } catch (...) {
            return false;
        }
#endif
        */
        
        initialized_ = true;
        return true;
    }
    
    static std::unique_ptr<kineto_profiler> create() {
        if (!initialize(true)) {
            return nullptr;
        }
        return std::make_unique<kineto_profiler>();
    }
};
```

---

## 2. ITTAPI INITIALIZATION FLOW

### PyTorch ITT Wrapper

```cpp
// torch/csrc/itt_wrapper.cpp
namespace torch::profiler {
    static __itt_domain* _itt_domain = __itt_domain_create("PyTorch");
    
    bool itt_is_available() {
        return torch::profiler::impl::ittStubs()->enabled();
    }
    
    void itt_range_push(const char* msg) {
        __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
        __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
    }
    
    void itt_range_pop() {
        __itt_task_end(_itt_domain);
    }
    
    void itt_mark(const char* msg) {
        __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
        __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
        __itt_task_end(_itt_domain);
    }
}
```

### PyTorch ITT Observer

```cpp
// torch/csrc/profiler/standalone/itt_observer.cpp
template <bool report_input_shapes>
static std::unique_ptr<at::ObserverContext> enterITT(
    const at::RecordFunction& fn) {
    if (ITTThreadLocalState::getTLS() != nullptr) {
        torch::profiler::impl::ittStubs()->rangePush(fn.name());
    }
    return nullptr;
}

static void exitITT(at::RecordFunction& fn, at::ObserverContext* ctx) {
    if (ITTThreadLocalState::getTLS() != nullptr) {
        torch::profiler::impl::ittStubs()->rangePop();
    }
}
```

### XSigma ITT Usage

```cpp
// User code in XSigma
#ifdef XSIGMA_HAS_ITTAPI
#include <ittnotify.h>

void profile_with_itt() {
    // Create domain
    __itt_domain* domain = __itt_domain_create("XSigmaProfiler");
    
    // Create string handle
    auto handle = __itt_string_handle_create("ProfiledTask");
    
    // Begin task
    __itt_task_begin(domain, __itt_null, __itt_null, handle);
    
    // ... work ...
    
    // End task
    __itt_task_end(domain);
}
#endif
```

---

## 3. CMAKE CONFIGURATION PATTERNS

### PyTorch Kineto Configuration

```cmake
# cmake/Dependencies.cmake
if(XSIGMA_ENABLE_KINETO)
    # Determine GPU backend support
    if(NOT XSIGMA_ENABLE_CUDA)
        set(LIBKINETO_NOCUPTI ON CACHE STRING "" FORCE)
    else()
        set(LIBKINETO_NOCUPTI OFF CACHE STRING "")
        message(STATUS "Using Kineto with CUPTI support")
    endif()
    
    if(NOT USE_ROCM)
        set(LIBKINETO_NOROCTRACER ON CACHE STRING "" FORCE)
    else()
        set(LIBKINETO_NOROCTRACER OFF CACHE STRING "")
        message(STATUS "Using Kineto with Roctracer support")
    endif()
    
    if((NOT USE_XPU) OR (NOT XPU_ENABLE_KINETO))
        set(LIBKINETO_NOXPUPTI ON CACHE STRING "" FORCE)
    else()
        set(LIBKINETO_NOXPUPTI OFF CACHE STRING "")
        message(STATUS "Using Kineto with XPUPTI support")
    endif()
    
    # Configure Kineto
    set(KINETO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/kineto/libkineto")
    set(KINETO_BUILD_TESTS OFF)
    set(KINETO_LIBRARY_TYPE "static")
    
    # Add subdirectory
    if(NOT TARGET kineto)
        add_subdirectory("${KINETO_SOURCE_DIR}")
        set_property(TARGET kineto PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()
    
    # Link and define
    list(APPEND Caffe2_DEPENDENCY_LIBS kineto)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_KINETO")
    
    # Add GPU-specific flags
    if(LIBKINETO_NOCUPTI)
        string(APPEND CMAKE_CXX_FLAGS " -DLIBKINETO_NOCUPTI")
    endif()
    if(LIBKINETO_NOROCTRACER)
        string(APPEND CMAKE_CXX_FLAGS " -DLIBKINETO_NOROCTRACER")
    endif()
    if(LIBKINETO_NOXPUPTI)
        string(APPEND CMAKE_CXX_FLAGS " -DLIBKINETO_NOXPUPTI=ON")
    else()
        string(APPEND CMAKE_CXX_FLAGS " -DLIBKINETO_NOXPUPTI=OFF")
    endif()
endif()
```

### PyTorch ITT Configuration

```cmake
# cmake/Dependencies.cmake
if(XSIGMA_ENABLE_ITT)
    find_package(ITT)
    if(ITT_FOUND)
        include_directories(SYSTEM ${ITT_INCLUDE_DIR})
        list(APPEND Caffe2_DEPENDENCY_LIBS ${ITT_LIBRARIES})
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ITT_LIBRARIES})
    else()
        message(WARNING "Not compiling with ITT. Suppress with -DUSE_ITT=OFF")
        set(XSIGMA_ENABLE_ITT OFF CACHE BOOL "" FORCE)
    endif()
endif()

# caffe2/CMakeLists.txt
if(${XSIGMA_ENABLE_ITT})
    list(APPEND TORCH_SRCS
        ${TORCH_SRC_DIR}/csrc/itt_wrapper.cpp
        ${TORCH_SRC_DIR}/csrc/profiler/stubs/itt.cpp
    )
endif()
```

### XSigma ITT Configuration

```cmake
# XSigma/ThirdParty/CMakeLists.txt
if(XSIGMA_ENABLE_ITTAPI)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ittapi/CMakeLists.txt")
        message(STATUS "Intel ITT API found - building as shared library")
        
        # Save and force shared library
        set(_saved_build_shared_libs ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS ON)
        
        # Add subdirectory
        add_subdirectory(ittapi ${CMAKE_CURRENT_BINARY_DIR}/ittapi_build)
        
        # Restore original state
        set(BUILD_SHARED_LIBS ${_saved_build_shared_libs})
        
        message(STATUS "✓ Intel ITT API targets: ittnotify, jitprofiling")
    else()
        message(WARNING "Intel ITT API not found")
    endif()
endif()
```

---

## 4. ENVIRONMENT VARIABLES

### Kineto Environment Variables
- `KINETO_LOG_LEVEL` - Set logging level (0=VERBOSE, higher=less verbose)
- `CUDA_INJECTION64_PATH` - Enable injection mode for CUDA profiling
- Daemon-related variables (Linux only)

### ITT Environment Variables
- VTune-specific environment variables for collection control
- Platform-specific library loading paths

---

## 5. ACTIVITY TYPES (Kineto)

```cpp
// Common activity types
libkineto::ActivityType::CPU_OP
libkineto::ActivityType::CPU_INSTANT_EVENT
libkineto::ActivityType::CUDA_RUNTIME
libkineto::ActivityType::CUDA_DRIVER
libkineto::ActivityType::CUDA_KERNEL
libkineto::ActivityType::CUDA_MEMCPY
libkineto::ActivityType::CUDA_MEMSET
libkineto::ActivityType::COLLECTIVE_COMM
libkineto::ActivityType::EXTERNAL_CORRELATION
libkineto::ActivityType::PYTHON_FUNCTION
```

---

## 6. PROFILER STATES (Kineto)

```cpp
// Finite state machine for on-demand profiling
enum class ProfilerState {
    WaitForRequest,   // Waiting for profiling request
    Warmup,           // Warmup phase
    CollectTrace,     // Active trace collection
    ProcessTrace      // Post-processing
};
```

