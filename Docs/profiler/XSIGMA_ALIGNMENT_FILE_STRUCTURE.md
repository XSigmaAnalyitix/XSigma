# XSigma Kineto/ITTAPI Alignment - File Structure

## 📁 Complete File Structure

### Modified Files

#### 1. XSigma/CMakeLists.txt
```
Lines 150-160: CMake Flags
├── XSIGMA_ENABLE_KINETO (default: ON)
├── XSIGMA_ENABLE_ITT (default: ON)
└── mark_as_advanced() for both
```

**Changes**:
- Renamed `XSIGMA_ENABLE_KINETO` → `XSIGMA_ENABLE_KINETO`
- Renamed `XSIGMA_ENABLE_ITTAPI` → `XSIGMA_ENABLE_ITT`
- Changed ITT default from OFF to ON

---

#### 2. XSigma/ThirdParty/CMakeLists.txt
```
Lines 284-334: Kineto Configuration
├── GPU Backend Configuration
│   ├── LIBKINETO_NOCUPTI (NVIDIA CUDA)
│   ├── LIBKINETO_NOROCTRACER (AMD ROCm)
│   ├── LIBKINETO_NOXPUPTI (Intel XPU)
│   └── LIBKINETO_NOAIUPTI (AI Accelerators)
├── Kineto Library Configuration
│   ├── CAFFE2_THIRD_PARTY_ROOT
│   ├── KINETO_SOURCE_DIR
│   ├── KINETO_BUILD_TESTS
│   └── KINETO_LIBRARY_TYPE (static)
└── add_subdirectory(kineto)

Lines 336-357: ITT Configuration
├── find_path(ITT_INCLUDE_DIR)
├── add_subdirectory(ittapi)
└── Error handling
```

**Changes**:
- Implemented GPU backend configuration
- Changed ITT from shared to static library
- Implemented find_package pattern
- Added error handling

---

#### 3. XSigma/Library/Core/CMakeLists.txt
```
Lines 236-246: Kineto Linking
├── if(XSIGMA_ENABLE_KINETO)
├── target_link_libraries(Core PUBLIC kineto)
└── target_compile_definitions(Core PUBLIC XSIGMA_ENABLE_KINETO)

Lines 248-259: ITT Linking
├── if(XSIGMA_ENABLE_ITT)
├── target_link_libraries(Core PUBLIC ittnotify)
└── target_compile_definitions(Core PUBLIC XSIGMA_ENABLE_ITT)

Lines 261-273: Source File Inclusion
├── if(XSIGMA_ENABLE_KINETO)
│   └── target_sources(Core PRIVATE kineto_shim.cpp)
└── if(XSIGMA_ENABLE_ITT)
    └── target_sources(Core PRIVATE itt_wrapper.cpp)
```

**Changes**:
- Updated compile definitions
- Updated target linking
- Added conditional source file inclusion

---

### New Files Created

#### 1. XSigma/Library/Core/profiler/kineto_shim.h
```
~150 lines

Header Structure:
├── Include Guards & Includes
├── Mobile Check
├── Forward Declarations
├── Namespace: xsigma::profiler
├── Availability Check (kKinetoAvailable)
├── Initialization Functions
│   ├── kineto_init()
│   ├── kineto_is_profiler_registered()
│   ├── kineto_is_profiler_initialized()
│   ├── kineto_prepare_trace()
│   ├── kineto_start_trace()
│   ├── kineto_stop_trace()
│   └── kineto_reset_tls()
├── Activity Type Definitions
└── Stub Implementations (non-XSIGMA_ENABLE_KINETO)
```

**Key Features**:
- Direct libkineto::api() interface
- Thread-safe initialization
- Automatic profiler registration
- Stub implementations for graceful degradation

---

#### 2. XSigma/Library/Core/profiler/kineto_shim.cpp
```
~80 lines

Implementation Structure:
├── Includes
├── Namespace: xsigma::profiler
├── Thread-Local State
│   ├── g_kineto_initialized
│   └── g_kineto_init_mutex
├── Function Implementations
│   ├── kineto_init()
│   ├── kineto_is_profiler_registered()
│   ├── kineto_is_profiler_initialized()
│   ├── kineto_prepare_trace()
│   ├── kineto_start_trace()
│   ├── kineto_stop_trace()
│   └── kineto_reset_tls()
└── Namespace Closing
```

**Key Features**:
- Mutex-protected initialization
- Automatic profiler registration
- Activity type configuration
- Trace management

---

#### 3. XSigma/Library/Core/profiler/itt_wrapper.h
```
~100 lines

Header Structure:
├── Include Guards & Includes
├── Namespace: xsigma::profiler
├── Availability Check (kITTAvailable)
├── Initialization Functions
│   ├── itt_init()
│   ├── itt_range_push()
│   ├── itt_range_pop()
│   ├── itt_mark()
│   └── itt_get_domain()
├── Stub Implementations (non-XSIGMA_ENABLE_ITT)
└── Namespace Closing
```

**Key Features**:
- Global ITT domain interface
- Task range annotations
- Event markers
- Stub implementations for graceful degradation

---

#### 4. XSigma/Library/Core/profiler/itt_wrapper.cpp
```
~60 lines

Implementation Structure:
├── Includes
├── Namespace: xsigma::profiler
├── Global State
│   ├── g_itt_domain
│   ├── g_itt_init_mutex
│   └── g_string_handles (thread-local)
├── Function Implementations
│   ├── itt_init()
│   ├── itt_range_push()
│   ├── itt_range_pop()
│   ├── itt_mark()
│   └── itt_get_domain()
└── Namespace Closing
```

**Key Features**:
- Global "XSigma" domain
- Mutex-protected initialization
- String handle management
- Thread-local caching

---

## 📊 File Statistics

### Modified Files
| File | Lines Changed | Type |
|------|---------------|------|
| XSigma/CMakeLists.txt | 10 | CMake |
| XSigma/ThirdParty/CMakeLists.txt | 50+ | CMake |
| XSigma/Library/Core/CMakeLists.txt | 30+ | CMake |
| **Total** | **90+** | |

### New Files
| File | Lines | Type |
|------|-------|------|
| kineto_shim.h | ~150 | Header |
| kineto_shim.cpp | ~80 | Implementation |
| itt_wrapper.h | ~100 | Header |
| itt_wrapper.cpp | ~60 | Implementation |
| **Total** | **~390** | |

### Documentation Files
| File | Lines | Purpose |
|------|-------|---------|
| XSIGMA_ALIGNMENT_QUICK_REFERENCE.md | ~200 | Quick reference |
| XSIGMA_PYTORCH_ALIGNMENT_SUMMARY.md | ~300 | Detailed summary |
| XSIGMA_ALIGNMENT_CODE_CHANGES.md | ~250 | Code changes |
| XSIGMA_ALIGNMENT_INDEX.md | ~300 | Navigation index |
| XSIGMA_ALIGNMENT_COMPLETION_REPORT.md | ~250 | Completion report |
| XSIGMA_ALIGNMENT_FILE_STRUCTURE.md | ~300 | This file |
| **Total** | **~1,600** | |

---

## 🔗 File Dependencies

### CMake Dependencies
```
XSigma/CMakeLists.txt
├── Defines: XSIGMA_ENABLE_KINETO, XSIGMA_ENABLE_ITT
└── Includes: XSigma/ThirdParty/CMakeLists.txt

XSigma/ThirdParty/CMakeLists.txt
├── Reads: XSIGMA_ENABLE_KINETO, XSIGMA_ENABLE_ITT
├── Configures: Kineto, ITT
└── Creates: kineto, ittnotify targets

XSigma/Library/Core/CMakeLists.txt
├── Reads: XSIGMA_ENABLE_KINETO, XSIGMA_ENABLE_ITT
├── Links: kineto, ittnotify targets
├── Includes: kineto_shim.cpp, itt_wrapper.cpp
└── Defines: XSIGMA_ENABLE_KINETO, XSIGMA_ENABLE_ITT compile definitions
```

### Source Dependencies
```
kineto_shim.h
├── Includes: libkineto.h (when XSIGMA_ENABLE_KINETO)
└── Provides: Direct libkineto interface

kineto_shim.cpp
├── Includes: kineto_shim.h
├── Includes: libkineto.h
└── Implements: Kineto initialization

itt_wrapper.h
├── Includes: ittnotify.h (when XSIGMA_ENABLE_ITT)
└── Provides: Global ITT domain interface

itt_wrapper.cpp
├── Includes: itt_wrapper.h
├── Includes: ittnotify.h
└── Implements: Global ITT domain
```

---

## 📋 Compilation Flow

### When XSIGMA_ENABLE_KINETO=ON
```
CMake Configuration
├── XSigma/CMakeLists.txt (XSIGMA_ENABLE_KINETO=ON)
├── XSigma/ThirdParty/CMakeLists.txt
│   ├── Configure GPU backends
│   ├── add_subdirectory(kineto)
│   └── Create kineto target
├── XSigma/Library/Core/CMakeLists.txt
│   ├── target_link_libraries(kineto)
│   ├── target_compile_definitions(XSIGMA_ENABLE_KINETO)
│   └── target_sources(kineto_shim.cpp)
└── Compilation
    ├── Compile kineto_shim.cpp
    ├── Link kineto library
    └── Define XSIGMA_ENABLE_KINETO
```

### When XSIGMA_ENABLE_ITT=ON
```
CMake Configuration
├── XSigma/CMakeLists.txt (XSIGMA_ENABLE_ITT=ON)
├── XSigma/ThirdParty/CMakeLists.txt
│   ├── find_path(ittnotify.h)
│   ├── add_subdirectory(ittapi)
│   └── Create ittnotify target
├── XSigma/Library/Core/CMakeLists.txt
│   ├── target_link_libraries(ittnotify)
│   ├── target_compile_definitions(XSIGMA_ENABLE_ITT)
│   └── target_sources(itt_wrapper.cpp)
└── Compilation
    ├── Compile itt_wrapper.cpp
    ├── Link ittnotify library
    └── Define XSIGMA_ENABLE_ITT
```

---

## 🎯 Integration Points

### Kineto Integration
```
Application Code
├── #include "profiler/kineto_shim.h"
├── xsigma::profiler::kineto_init()
├── xsigma::profiler::kineto_start_trace()
├── ... code to profile ...
├── xsigma::profiler::kineto_stop_trace()
└── trace->save("trace.json")
    ↓
kineto_shim.cpp
├── libkineto::api().initProfilerIfRegistered()
├── libkineto::api().activityProfiler().startTrace()
├── libkineto::api().activityProfiler().stopTrace()
└── libkineto library
```

### ITTAPI Integration
```
Application Code
├── #include "profiler/itt_wrapper.h"
├── xsigma::profiler::itt_range_push("op")
├── ... code to profile ...
├── xsigma::profiler::itt_range_pop()
└── VTune profiler
    ↓
itt_wrapper.cpp
├── __itt_domain_create("XSigma")
├── __itt_task_begin()
├── __itt_task_end()
└── ittnotify library
```

---

## ✅ Verification Checklist

- [x] All CMake files updated
- [x] All new files created
- [x] All dependencies configured
- [x] All compilation flows verified
- [x] All integration points documented
- [x] File structure complete

---

**Last Updated**: 2025-10-30
**Status**: ✅ COMPLETE
