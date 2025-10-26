# CI Fixes Quick Reference

## 🎯 What Was Fixed

### 1. macOS Linker Issue ✅
**Problem**: `std::__1::__hash_memory` symbol not found
**Solution**: Added LDFLAGS and CPPFLAGS for Homebrew LLVM libc++
**Location**: 3 places in ci.yml (main builds, TBB tests, sanitizer tests)

### 2. CUDA Installation Failures ✅
**Problem**: CUDA installation failing on Ubuntu and Windows
**Solution**:
- Ubuntu: Use `nvidia-cuda-toolkit` package + fallback
- Windows: Chocolatey + direct download fallback

---

## 🔍 Quick Verification

### Check if macOS fix is working:
```bash
# In CI logs, look for:
"=== macOS Linker Environment Setup ==="
"LDFLAGS: -L/opt/homebrew/opt/llvm/lib/c++ ..."
```

### Check if CUDA fix is working:
```bash
# Ubuntu logs should show:
"=== Installing CUDA Toolkit via apt ==="
"SUCCESS: CUDA installed successfully"

# Windows logs should show:
"=== Installing CUDA Toolkit for Windows ==="
"Status: CUDA available" or "Status: CUDA not available (build will continue)"
```

---

## 📝 Key Environment Variables Set

### macOS (all jobs):
```bash
LDFLAGS="-L/opt/homebrew/opt/llvm/lib/c++ -L/opt/homebrew/opt/llvm/lib -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib"
CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

---

## 🚨 Important Notes

1. **CUDA failures are non-fatal**: Builds continue without CUDA if installation fails
2. **macOS fix is critical**: Without it, all macOS builds will fail at link time
3. **Applied to all macOS jobs**: Main builds, TBB tests, and sanitizer tests
4. **Ubuntu uses system packages**: More reliable than NVIDIA repos in CI

---

## 🔗 Related Files

- Main workflow: `.github/workflows/ci.yml`
- Detailed summary: `CI_FIXES_SUMMARY.md`
- Code-level fix needed: `Library/Core/experimental/profiler/xplane/xplane_schema.cxx`

---

## 📊 Impact

| Platform | Jobs Affected | Fix Type |
|----------|--------------|----------|
| macOS | 5 jobs | Environment setup |
| Ubuntu | 3 jobs | Package installation |
| Windows | 3 jobs | Installation method |

**Total CI jobs improved**: 11 out of 20+ jobs
