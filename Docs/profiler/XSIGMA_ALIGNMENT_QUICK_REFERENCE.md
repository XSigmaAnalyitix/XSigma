# XSigma Kineto/ITTAPI Alignment - Quick Reference

## 🎯 What Changed?

XSigma's Kineto and ITTAPI implementation has been aligned with PyTorch's for feature parity.

---

## 📝 CMake Flags

### Old Flags (Deprecated)
```cmake
option(XSIGMA_ENABLE_KINETO "..." ON)
option(XSIGMA_ENABLE_ITTAPI "..." OFF)
```

### New Flags (PyTorch-Compatible)
```cmake
option(XSIGMA_ENABLE_KINETO "..." ON)
option(XSIGMA_ENABLE_ITT "..." ON)
```

### Build Command
```bash
# Old way (still works but deprecated)
cmake -DXSIGMA_ENABLE_KINETO=ON -DXSIGMA_ENABLE_ITTAPI=ON ..

# New way (PyTorch-compatible)
cmake -DUSE_KINETO=ON -DUSE_ITT=ON ..
```

---

## 🔧 Compile Definitions

### Old Definitions (Deprecated)
```cpp
#ifdef XSIGMA_HAS_KINETO
#ifdef XSIGMA_HAS_ITT
```

### New Definitions (PyTorch-Compatible)
```cpp
#ifdef XSIGMA_ENABLE_KINETO
#ifdef XSIGMA_ENABLE_ITT
```

---

## 📚 New API

### Kineto Shim (Direct libkineto Integration)
```cpp
#include "profiler/kineto_shim.h"

// Initialize
xsigma::profiler::kineto_init(false, true);

// Check status
bool registered = xsigma::profiler::kineto_is_profiler_registered();
bool initialized = xsigma::profiler::kineto_is_profiler_initialized();

// Prepare and run
std::set<libkineto::ActivityType> activities;
activities.insert(libkineto::ActivityType::CPU_OP);
xsigma::profiler::kineto_prepare_trace(activities);
xsigma::profiler::kineto_start_trace();
// ... code ...
auto trace = xsigma::profiler::kineto_stop_trace();
trace->save("trace.json");

// Cleanup
xsigma::profiler::kineto_reset_tls();
```

### ITT Wrapper (Global Domain)
```cpp
#include "profiler/itt_wrapper.h"

// Initialize (automatic on first use)
xsigma::profiler::itt_init();

// Annotate tasks
xsigma::profiler::itt_range_push("operation_name");
// ... code ...
xsigma::profiler::itt_range_pop();

// Mark events
xsigma::profiler::itt_mark("checkpoint");

// Get domain
__itt_domain* domain = xsigma::profiler::itt_get_domain();
```

---

## 🗂️ New Files

### Created
- `XSigma/Library/Core/profiler/kineto_shim.h` - Kineto interface
- `XSigma/Library/Core/profiler/kineto_shim.cpp` - Kineto implementation
- `XSigma/Library/Core/profiler/itt_wrapper.h` - ITT interface
- `XSigma/Library/Core/profiler/itt_wrapper.cpp` - ITT implementation

### Modified
- `XSigma/CMakeLists.txt` - Updated flags
- `XSigma/ThirdParty/CMakeLists.txt` - Updated Kineto/ITT config
- `XSigma/Library/Core/CMakeLists.txt` - Updated linking and definitions

### Deprecated (Not Removed)
- `XSigma/Library/Core/profiler/kineto_profiler.h` - Old wrapper
- `XSigma/Library/Core/profiler/kineto_profiler.cxx` - Old wrapper

---

## 🔄 Migration Guide

### If You Were Using Old Wrapper
```cpp
// OLD CODE (deprecated)
#ifdef XSIGMA_HAS_KINETO
auto profiler = xsigma::kineto_profiler::create();
profiler->start_profiling();
// ...
profiler->stop_profiling();
#endif

// NEW CODE (PyTorch-compatible)
#ifdef XSIGMA_ENABLE_KINETO
#include "profiler/kineto_shim.h"
xsigma::profiler::kineto_init();
xsigma::profiler::kineto_start_trace();
// ...
auto trace = xsigma::profiler::kineto_stop_trace();
#endif
```

### If You Were Using ITT Directly
```cpp
// OLD CODE (manual domain management)
#ifdef XSIGMA_HAS_ITT
__itt_domain* domain = __itt_domain_create("MyDomain");
__itt_task_begin(domain, ...);
// ...
__itt_task_end(domain);
#endif

// NEW CODE (global domain)
#ifdef XSIGMA_ENABLE_ITT
#include "profiler/itt_wrapper.h"
xsigma::profiler::itt_range_push("operation");
// ...
xsigma::profiler::itt_range_pop();
#endif
```

---

## 🎯 GPU Backend Support

### Automatic Configuration
```cmake
# CUDA support (if XSIGMA_ENABLE_CUDA=ON)
cmake -DUSE_KINETO=ON -DUSE_CUDA=ON ..

# ROCm support (if USE_ROCM=ON)
cmake -DUSE_KINETO=ON -DUSE_ROCM=ON ..

# Intel XPU support (if USE_XPU=ON)
cmake -DUSE_KINETO=ON -DUSE_XPU=ON ..

# CPU-only (default)
cmake -DUSE_KINETO=ON ..
```

### Manual Configuration
```cmake
# Disable specific backends
cmake -DUSE_KINETO=ON \
      -DLIBKINETO_NOCUPTI=ON \
      -DLIBKINETO_NOROCTRACER=ON \
      ..
```

---

## ✅ Verification Checklist

- [ ] CMake configuration uses `XSIGMA_ENABLE_KINETO` and `XSIGMA_ENABLE_ITT`
- [ ] Compile definitions use `XSIGMA_ENABLE_KINETO` and `XSIGMA_ENABLE_ITT`
- [ ] Code includes `profiler/kineto_shim.h` for Kineto
- [ ] Code includes `profiler/itt_wrapper.h` for ITT
- [ ] Build succeeds with both flags ON
- [ ] Build succeeds with both flags OFF
- [ ] Profiling traces are generated correctly
- [ ] VTune shows ITT annotations correctly

---

## 🔗 Related Files

- `XSIGMA_PYTORCH_ALIGNMENT_SUMMARY.md` - Detailed alignment summary
- `KINETO_ITTAPI_COMPARISON.md` - Detailed comparison with PyTorch
- `KINETO_ITTAPI_TECHNICAL_REFERENCE.md` - Code examples and patterns

---

## ❓ FAQ

**Q: Do I need to update my code?**
A: Only if you were using the old `xsigma::kineto_profiler` wrapper or manual ITT domain management. New code should use the new APIs.

**Q: Are the old APIs still supported?**
A: The old wrapper files are still present but deprecated. They will be removed in a future version.

**Q: What about backward compatibility?**
A: The old CMake flags still work but are deprecated. Use the new `XSIGMA_ENABLE_KINETO` and `XSIGMA_ENABLE_ITT` flags.

**Q: How do I disable Kineto/ITT?**
A: Use `cmake -DUSE_KINETO=OFF -DUSE_ITT=OFF ..`

**Q: What if I only want Kineto without ITT?**
A: Use `cmake -DUSE_KINETO=ON -DUSE_ITT=OFF ..`

---

## 📞 Support

For detailed information, see:
- `XSIGMA_PYTORCH_ALIGNMENT_SUMMARY.md` - Complete alignment details
- `KINETO_ITTAPI_USAGE_GUIDE.md` - Practical usage examples
- `KINETO_ITTAPI_TECHNICAL_REFERENCE.md` - Technical deep dive
