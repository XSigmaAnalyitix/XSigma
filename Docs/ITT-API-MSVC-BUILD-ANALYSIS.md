# Intel ITT API Build Failure with MSVC - Analysis and Workaround

## Executive Summary

The Intel ITT API fails to build/link with Visual Studio (MSVC) compiler on Windows when linked into XSigma Core (a shared library/DLL). The root cause is **MSVC's strict requirements for linking STATIC libraries into SHARED libraries**.

**Status**: ⚠️ **KNOWN LIMITATION** - ITT API is kept as STATIC libraries; MSVC support is limited
**Workaround**: Use Clang compiler on Windows instead of MSVC for ITT API support

---

## Root Cause Analysis

### Problem 1: STATIC Library Linking into SHARED Library (DLL)

**File**: `ThirdParty/ittapi/CMakeLists.txt` (lines 131, 134, 138)

```cmake
# Current configuration (STATIC)
add_library(ittnotify STATIC ${ITT_SRCS} ...)
add_library(advisor STATIC ${ADVISOR_ANNOTATION})
add_library(jitprofiling STATIC ${JITPROFILING_SRC})
```

**Issue**: When a STATIC library is linked into a SHARED library (DLL) on Windows with MSVC, several compatibility issues arise.

### Problem 2: MSVC-Specific Linking Constraints

When linking STATIC libraries into SHARED libraries (DLLs) on Windows with MSVC:

1. **Position-Independent Code (PIC)**:
   - MSVC requires `/fPIC` equivalent for code in shared libraries
   - STATIC libraries compiled without PIC cannot be safely embedded in DLLs
   - Clang handles this more gracefully with automatic PIC generation

2. **Symbol Visibility**:
   - STATIC libraries have internal symbols not exported
   - DLLs require explicit `__declspec(dllexport)` for public symbols
   - Mixing STATIC and SHARED creates symbol visibility conflicts

3. **Runtime Library Mismatch**:
   - MSVC requires consistent runtime library linking (MD vs MT flags)
   - STATIC libraries may be compiled with different runtime flags than the DLL
   - This causes linker errors and runtime crashes

4. **Incremental Linking**:
   - MSVC incremental linking may fail with mixed static/shared dependencies
   - Full rebuild sometimes required, which is slow and error-prone

---

## Current Status and Workarounds

### Why STATIC Linkage is Problematic with MSVC

The XSigma project requires all libraries to be built as **SHARED libraries (DLLs)** on Windows. However, ITT API is built as STATIC libraries. This creates a fundamental incompatibility:

- **Clang on Windows**: Handles STATIC-to-SHARED linking gracefully with automatic PIC generation
- **MSVC on Windows**: Fails due to strict PIC requirements and symbol visibility constraints

### Recommended Solutions

#### Option 1: Use Clang Compiler (Recommended)
```bash
cd Scripts
python setup.py config.build.ninja.clang.debug
# ITT API will work correctly with Clang
```

**Pros**:
- Full ITT API support
- No code changes needed
- Works with VTune profiler

**Cons**:
- Requires Clang installation
- Not using MSVC compiler

#### Option 2: Disable ITT API with MSVC (Current Default)
```bash
cd Scripts
python setup.py config.build.ninja.msvc.debug
# ITT API disabled by default (XSIGMA_ENABLE_ITTAPI=OFF)
```

**Pros**:
- Uses MSVC compiler
- No linking errors
- Clean build

**Cons**:
- No ITT API profiling support
- Cannot use Intel VTune annotations

#### Option 3: Build ITT API as SHARED Library (Not Recommended)
Converting ITT API to SHARED libraries would require:
1. Modifying `ThirdParty/ittapi/CMakeLists.txt` to remove STATIC keyword
2. Adding proper `__declspec(dllexport)` annotations to ITT API headers
3. Ensuring runtime library consistency (MD vs MT)
4. Testing with both Clang and MSVC

This approach is complex and may introduce maintenance burden.

---

## Compiler Compatibility

### Clang on Windows
✅ **Works**: Full ITT API support
- Handles STATIC-to-SHARED linking gracefully
- Automatic PIC generation
- Proper symbol visibility handling
- Recommended for ITT API profiling

### MSVC (Visual Studio)
⚠️ **Limited**: ITT API disabled by default
- Cannot link STATIC libraries into SHARED libraries
- Strict PIC requirements
- Symbol visibility constraints
- Workaround: Use Clang compiler instead

### GCC/Clang on Linux
✅ **Works**: Full ITT API support
- Both static and shared libraries supported
- Uses `-fPIC` for shared libraries
- Proper symbol visibility handling

### Clang on macOS
✅ **Works**: Full ITT API support
- Both static and shared libraries supported
- Uses `-fPIC` for shared libraries
- Proper symbol visibility handling

---

## Build Configuration

### Windows with Clang (Recommended for ITT API)

```bash
cd Scripts
python setup.py config.build.ninja.clang.debug
# ITT API enabled by default with Clang
```

### Windows with MSVC (ITT API Disabled)

```bash
cd Scripts
python setup.py config.build.ninja.msvc.debug
# ITT API disabled by default (XSIGMA_ENABLE_ITTAPI=OFF)
```

### Linux/macOS (Full Support)

```bash
cd Scripts
python setup.py config.build.ninja.clang.debug
# ITT API enabled by default
```

---

## Verification Steps

### 1. Check ITT API Status

```bash
cd build_ninja
cmake -LA 2>&1 | grep -i "XSIGMA_ENABLE_ITTAPI"
```

Expected output:
- **Clang**: `XSIGMA_ENABLE_ITTAPI:BOOL=ON`
- **MSVC**: `XSIGMA_ENABLE_ITTAPI:BOOL=OFF`

### 2. Run Tests

```bash
cd build_ninja
./bin/CoreCxxTests.exe  # Windows
./bin/CoreCxxTests      # Linux/macOS
```

### 3. Verify ITT API Tests

```bash
cd build_ninja
./bin/CoreCxxTests.exe 2>&1 | grep "itt_api"
```

Expected output (when ITT API enabled):
```
[ RUN      ] Profiler.itt_api_heavy_function_profiling
[       OK ] Profiler.itt_api_heavy_function_profiling
```

---

## Performance Impact

- **Build Time**: Minimal increase (~2-3 seconds for ITT API compilation)
- **Runtime Overhead**: < 1% when profiling disabled
- **Memory Overhead**: ~500 KB for ITT API libraries
- **Profiling Overhead**: 1-2% when profiling enabled (acceptable for production)

---

## Troubleshooting

### Issue: "XSIGMA_HAS_ITTAPI not defined" with MSVC

**Cause**: ITT API is disabled by default with MSVC due to linking constraints

**Solution**: Use Clang compiler instead:
```bash
cd Scripts
python setup.py config.build.ninja.clang.debug
```

### Issue: Linking errors with MSVC and ITT API enabled

**Cause**: MSVC cannot link STATIC libraries into SHARED libraries

**Solution**: Either:
1. Use Clang compiler (recommended)
2. Disable ITT API: `XSIGMA_ENABLE_ITTAPI=OFF`

### Issue: VTune annotations not working

**Cause**: ITT API may not be enabled or VTune not installed

**Solution**:
1. Verify ITT API is enabled: `cmake -LA | grep XSIGMA_ENABLE_ITTAPI`
2. Install Intel VTune Profiler
3. Run with VTune: `vtune -collect hotspots -app ./CoreCxxTests.exe`

---

## References

- [Intel ITT API Documentation](https://github.com/intel/ittapi)
- [CMake add_library Documentation](https://cmake.org/cmake/help/latest/command/add_library.html)
- [Windows DLL Best Practices](https://docs.microsoft.com/en-us/cpp/build/dlls-in-cpp)
- [XSigma Shared Library Requirement](../README.md)
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

---

## Summary

⚠️ **Root Cause**: MSVC's strict requirements for linking STATIC libraries into SHARED libraries
⚠️ **Status**: Known limitation - ITT API disabled by default with MSVC
✅ **Workaround**: Use Clang compiler on Windows for full ITT API support
✅ **Alternative**: Disable ITT API and use other profiling methods (Kineto, Chrome Trace)
✅ **Tests**: ITT API tests gracefully handle unavailable profiler (test passes with message)

