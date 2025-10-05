# Benchmark Flag - Quick Reference

**Status**: ✅ Implemented and Pushed  
**Commit**: `6c4c9f6`

---

## TL;DR

```bash
# Enable benchmark for local development
cd Scripts
python setup.py ninja.clang.release.benchmark.config.build

# Benchmark is disabled by default (matches CI behavior)
python setup.py ninja.clang.release.config.build  # No benchmark
```

---

## What Changed

### Before
- ❌ Benchmark default was `ON` in CMakeLists.txt
- ❌ setup.py had inverted logic (specifying `benchmark` would disable it)
- ❌ No clear way to enable benchmark for local testing

### After
- ✅ Benchmark default is `OFF` in CMakeLists.txt (matches CI)
- ✅ setup.py correctly enables benchmark when flag is specified
- ✅ Build directory includes `_benchmark` suffix for clarity
- ✅ Help documentation includes benchmark examples

---

## Usage Examples

### Basic
```bash
# Enable benchmark in Release build
python setup.py ninja.clang.release.benchmark.config.build
```

### With Optimizations
```bash
# With LTO
python setup.py ninja.clang.release.lto.benchmark.config.build

# With AVX2
python setup.py ninja.clang.release.avx2.benchmark.config.build

# With both
python setup.py ninja.clang.release.lto.avx2.benchmark.config.build
```

### With Other Features
```bash
# With TBB
python setup.py ninja.clang.release.benchmark.tbb.config.build

# With custom C++ standard
python setup.py ninja.clang.release.cxx20.benchmark.config.build
```

---

## Build Directory Naming

| Command | Build Directory |
|---------|----------------|
| `python setup.py ninja.clang.config` | `build_ninja_python/` |
| `python setup.py ninja.clang.benchmark.config` | `build_ninja_python_benchmark/` |

---

## Verification

### Check Help
```bash
cd Scripts
python setup.py --help | grep -A 5 "Benchmark examples"
```

### Check CMake Configuration
```bash
cd Scripts
python setup.py ninja.clang.release.benchmark.config 2>&1 | grep BENCHMARK
```

**Expected output**:
```
-- XSIGMA_ENABLE_BENCHMARK: ON
```

---

## CI Behavior

CI explicitly sets `-DXSIGMA_ENABLE_BENCHMARK=OFF` in all jobs, which **overrides** the CMake default. This is intentional to avoid regex detection issues on macOS.

---

## Files Modified

1. **Scripts/setup.py**
   - Lines 302-322: Updated default values
   - Lines 370-383: Fixed inverse logic and added suffix
   - Lines 1098-1107: Added documentation

2. **CMakeLists.txt**
   - Line 61: Changed default from `ON` to `OFF`

3. **BENCHMARK_FLAG_IMPLEMENTATION.md**
   - Comprehensive documentation (new file)

---

## Related Commits

- `6c4c9f6` - Add benchmark flag to setup.py for local development
- `124db6f` - Fix CI/CD pipeline failures: Disable benchmark and fix sanitizers

---

## For More Details

See `BENCHMARK_FLAG_IMPLEMENTATION.md` for:
- Detailed explanation of changes
- Troubleshooting guide
- Best practices
- Future work (re-enabling in CI)

---

**Quick Start**: `python setup.py ninja.clang.release.benchmark.config.build`

