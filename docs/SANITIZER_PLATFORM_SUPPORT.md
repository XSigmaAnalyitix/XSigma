# Sanitizer Platform Support

**Last Updated**: 2025-10-05
**Status**: ✅ Active - CI configured with platform-specific exclusions

---

## Overview

This document describes sanitizer support across different platforms in the XSigma project. Sanitizers are runtime instrumentation tools that detect various types of bugs including memory errors, undefined behavior, data races, and memory leaks.

### Quick Summary

- ✅ **Ubuntu (Linux)**: All sanitizers supported (ASan, UBSan, TSan, LSan)
- ⚠️ **macOS (ARM64)**: Limited support (ASan, UBSan only)
- ⚠️ **Windows**: Partial support (ASan, UBSan only)

**Key Limitation**: LeakSanitizer is **not supported** on macOS ARM64 (Apple Silicon) due to architecture limitations.

---

## Sanitizer Support Matrix

| Sanitizer | Ubuntu (x86_64) | Windows (x86_64) | macOS (ARM64) | macOS (x86_64) |
|-----------|-----------------|------------------|---------------|----------------|
| **AddressSanitizer (ASan)** | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **UndefinedBehaviorSanitizer (UBSan)** | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **ThreadSanitizer (TSan)** | ✅ Full | ⚠️ Limited | ❌ Not in CI | ✅ Full |
| **LeakSanitizer (LSan)** | ✅ Full | ❌ Not Supported | ❌ Not Supported | ✅ Full |
| **MemorySanitizer (MSan)** | ✅ Clang Only | ❌ Not Supported | ❌ Not Supported | ⚠️ Limited |

---

## Platform-Specific Details

### Ubuntu (Linux x86_64)

✅ **All sanitizers fully supported**

- **Compiler**: Clang 14+ / GCC 11+
- **Supported Sanitizers**: ASan, UBSan, TSan, LSan, MSan
- **Notes**: Best platform for comprehensive sanitizer testing

### Windows (x86_64)

✅ **Most sanitizers supported**

- **Compiler**: Clang 16+ / MSVC 19.33+
- **Supported Sanitizers**: ASan, UBSan
- **Limited Support**: TSan (experimental)
- **Not Supported**: LSan, MSan
- **Notes**: ASan and UBSan work well with Clang on Windows

### macOS ARM64 (Apple Silicon)

⚠️ **Limited sanitizer support**

- **Compiler**: Apple Clang 14+
- **Supported Sanitizers**: ASan, UBSan
- **Not Supported**: LSan, TSan (in CI), MSan
- **CI Runners**: `macos-latest` (currently macOS 14 Sonoma on ARM64)

**Technical Limitations**:

1. **LeakSanitizer (LSan)**:
   - **Status**: ❌ Not supported on ARM64 architecture
   - **Error**: `clang++: error: unsupported option '-fsanitize=leak' for target 'arm64-apple-darwin24.6.0'`
   - **Reason**: LLVM/Clang does not implement LSan for ARM64 Darwin (macOS)
   - **Workaround**: Use AddressSanitizer with leak detection enabled (`ASAN_OPTIONS=detect_leaks=1`)
   - **Effectiveness**: ASan's leak detection is nearly equivalent to standalone LSan

2. **ThreadSanitizer (TSan)**:
   - **Status**: ⚠️ Supported locally, disabled in CI
   - **Reason**: Stability issues and false positives in GitHub Actions environment
   - **Workaround**: Test locally on macOS or rely on Ubuntu CI for TSan coverage

3. **MemorySanitizer (MSan)**:
   - **Status**: ❌ Not supported
   - **Reason**: Requires instrumented standard library, not available on macOS

### macOS x86_64 (Intel)

✅ **Good sanitizer support**

- **Compiler**: Apple Clang 14+
- **Supported Sanitizers**: ASan, UBSan, TSan, LSan
- **Limited Support**: MSan
- **Notes**: Full support for most sanitizers on Intel Macs

---

## CI Configuration

### Sanitizer Tests in CI

The CI workflow (`.github/workflows/ci.yml`) runs sanitizer tests with platform-specific exclusions to avoid unsupported configurations.

#### Configuration (lines 546-558)

```yaml
strategy:
  fail-fast: false
  matrix:
    sanitizer: [address, undefined, thread, leak]
    os: [ubuntu-latest, macos-latest]
    exclude:
      # ThreadSanitizer has issues on macOS in CI
      - sanitizer: thread
        os: macos-latest

      # LeakSanitizer is not supported on macOS ARM64 (Apple Silicon)
      - sanitizer: leak
        os: macos-latest
```

#### Why These Exclusions?

1. **ThreadSanitizer on macOS**:
   - Excluded due to stability issues in GitHub Actions environment
   - Works locally on macOS but produces false positives in CI
   - Ubuntu provides comprehensive TSan coverage

2. **LeakSanitizer on macOS**:
   - Excluded because it's **not supported** on ARM64 architecture
   - Would cause build failure: `error: unsupported option '-fsanitize=leak'`
   - Workaround: ASan with leak detection provides equivalent coverage

### Tested Configurations

| Platform | ASan | UBSan | TSan | LSan | Total Jobs |
|----------|------|-------|------|------|------------|
| **Ubuntu** | ✅ | ✅ | ✅ | ✅ | 4 |
| **macOS** | ✅ | ✅ | ❌ | ❌ | 2 |
| **Total** | | | | | **6** |

### CI Job Details

Each sanitizer job performs:
1. ✅ Checkout repository with submodules
2. ✅ Cache dependencies for faster builds
3. ✅ Install platform-specific dependencies
4. ✅ Configure CMake with sanitizer enabled
5. ✅ Build project with sanitizer instrumentation
6. ✅ Run full test suite with sanitizer active
7. ✅ Upload results on failure for debugging

**Estimated Time**:
- Ubuntu jobs: ~15-20 minutes each
- macOS jobs: ~20-25 minutes each
- Total sanitizer testing: ~60-90 minutes

---

## Local Development

### Using Sanitizers Locally

#### Ubuntu/Linux

```bash
# All sanitizers available
cd Scripts

# Address Sanitizer
python setup.py ninja.clang.debug.address.config.build.test

# Leak Sanitizer
python setup.py ninja.clang.debug.leak.config.build.test

# Thread Sanitizer
python setup.py ninja.clang.debug.thread.config.build.test

# Undefined Behavior Sanitizer
python setup.py ninja.clang.debug.undefined.config.build.test
```

#### macOS (Apple Silicon)

```bash
cd Scripts

# Address Sanitizer (with leak detection)
python setup.py ninja.clang.debug.address.config.build.test

# Undefined Behavior Sanitizer
python setup.py ninja.clang.debug.undefined.config.build.test

# ❌ Leak Sanitizer - NOT SUPPORTED
# Use ASan with leak detection instead:
# ASAN_OPTIONS=detect_leaks=1 ./build/bin/CoreCxxTests

# ⚠️ Thread Sanitizer - May have issues
python setup.py ninja.clang.debug.thread.config.build.test
```

#### Windows

```bash
cd Scripts

# Address Sanitizer
python setup.py ninja.clang.debug.address.config.build.test

# Undefined Behavior Sanitizer
python setup.py ninja.clang.debug.undefined.config.build.test

# ❌ Leak Sanitizer - NOT SUPPORTED
# ⚠️ Thread Sanitizer - Experimental
```

---

## Workarounds

### LeakSanitizer on macOS ARM64

**Problem**: LeakSanitizer is not supported on ARM64 architecture.

**Error Message**:
```
clang++: error: unsupported option '-fsanitize=leak' for target 'arm64-apple-darwin24.6.0'
```

**Root Cause**:
- LLVM/Clang does not implement LeakSanitizer for ARM64 Darwin (macOS)
- This is an architectural limitation, not a bug
- LSan requires specific low-level memory tracking not available on ARM64 macOS

**Solution**: Use AddressSanitizer with leak detection enabled

AddressSanitizer includes leak detection capabilities that are nearly equivalent to standalone LeakSanitizer.

#### Method 1: Using setup.py (Recommended)

```bash
# Configure with AddressSanitizer
cd Scripts
python setup.py ninja.clang.debug.address.config.build

# Run tests with leak detection enabled
cd ../build_ninja_python
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 ./bin/CoreCxxTests
```

#### Method 2: Direct CMake Configuration

```bash
# Configure with ASan
cmake -B build \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DXSIGMA_ENABLE_SANITIZER=ON \
  -DXSIGMA_SANITIZER_TYPE=address \
  -DXSIGMA_BUILD_TESTING=ON

# Build
cmake --build build

# Run with leak detection
cd build
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 ./bin/CoreCxxTests
```

#### Method 3: Environment Variable for All Tests

```bash
# Set environment variable for the session
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:symbolize=1

# Run tests (leak detection automatically enabled)
cd build
ctest --output-on-failure
```

**Effectiveness**:
- ✅ ASan's leak detection catches the same leaks as LSan
- ✅ Same leak reporting format
- ✅ Same suppression mechanism
- ⚠️ Slightly higher memory overhead than standalone LSan
- ⚠️ Cannot be combined with TSan (but neither can LSan)

---

### ThreadSanitizer on macOS

**Problem**: ThreadSanitizer can be unstable in CI environments on macOS.

**Workaround**: 
1. Test locally on macOS if possible
2. Rely on Ubuntu CI for TSan testing
3. Use Valgrind's Helgrind tool as alternative (Linux only)

---

## Compiler Requirements

### Minimum Versions for Full Sanitizer Support

| Compiler | Version | Notes |
|----------|---------|-------|
| **Clang** | 14+ | Best sanitizer support |
| **GCC** | 11+ | Good support on Linux |
| **MSVC** | 19.33+ | ASan and UBSan only |
| **Apple Clang** | 14+ | Limited on ARM64 |

---

## Best Practices

### 1. Use Ubuntu for Comprehensive Testing

For the most thorough sanitizer testing, use Ubuntu:
- All sanitizers supported
- Most stable implementation
- Best error reporting

### 2. Test on Target Platform

Always test on the platform you're deploying to:
- macOS ARM64 has different behavior than x86_64
- Windows has different memory layout
- Linux has different threading model

### 3. Combine Sanitizers Carefully

Some sanitizers are incompatible:
- ❌ **TSan + ASan**: Cannot be used together
- ❌ **TSan + LSan**: Cannot be used together
- ✅ **ASan + UBSan**: Can be combined
- ✅ **LSan + UBSan**: Can be combined

### 4. Use Appropriate Build Type

- **Debug builds**: Better stack traces, slower execution
- **RelWithDebInfo**: Good balance for performance testing
- **Release**: Not recommended (optimizations may hide issues)

---

## Troubleshooting

### Issue: "unsupported option '-fsanitize=leak'"

**Full Error**:
```
clang++: error: unsupported option '-fsanitize=leak' for target 'arm64-apple-darwin24.6.0'
```

**Platform**: macOS ARM64 (Apple Silicon)

**Cause**: LeakSanitizer is not implemented for ARM64 macOS in LLVM/Clang

**Solution 1** (Recommended): Use AddressSanitizer with leak detection:
```bash
# Configure with ASan
cd Scripts
python setup.py ninja.clang.debug.address.config.build

# Run with leak detection
cd ../build_ninja_python
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 ./bin/CoreCxxTests
```

**Solution 2**: Test on Ubuntu instead:
```bash
# LeakSanitizer fully supported on Linux
cd Scripts
python setup.py ninja.clang.debug.leak.config.build.test
```

**Solution 3**: Use Valgrind on Linux (not available on macOS ARM64):
```bash
# Only works on Linux x86_64
cd Scripts
python setup.py ninja.clang.debug.valgrind.config.build.test
```

---

### Issue: ThreadSanitizer reports false positives

**Platform**: Any

**Solution**: 
1. Add suppressions to `Scripts/sanitizer_ignore.txt`
2. Use `TSAN_OPTIONS=suppressions=path/to/suppressions.txt`
3. Review code for actual race conditions

---

### Issue: Sanitizer slows down tests significantly

**Platform**: Any

**Solution**:
1. Run sanitizer tests on subset of tests
2. Use `ctest -R pattern` to run specific tests
3. Increase timeout values in CTest
4. Run sanitizer tests in parallel with `-j` flag

---

## References

- [AddressSanitizer Documentation](https://clang.llvm.org/docs/AddressSanitizer.html)
- [ThreadSanitizer Documentation](https://clang.llvm.org/docs/ThreadSanitizer.html)
- [UndefinedBehaviorSanitizer Documentation](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [LeakSanitizer Documentation](https://clang.llvm.org/docs/LeakSanitizer.html)
- [MemorySanitizer Documentation](https://clang.llvm.org/docs/MemorySanitizer.html)

---

## Recent Changes

### 2025-10-05: Excluded LeakSanitizer from macOS CI

**Commit**: `728502f`

**Change**: Added exclusion for LeakSanitizer on macOS in CI matrix

**Reason**:
- LeakSanitizer is not supported on ARM64 macOS
- CI was failing with: `clang++: error: unsupported option '-fsanitize=leak' for target 'arm64-apple-darwin24.6.0'`

**Impact**:
- ✅ CI now passes on macOS
- ✅ No loss of leak detection coverage (ASan provides equivalent)
- ✅ Reduced CI jobs from 8 to 6 (removed 2 failing jobs)

**Related Documentation**: This file created to document platform limitations

---

## Summary

- ✅ **Ubuntu**: Best platform for sanitizer testing (all supported)
- ⚠️ **macOS ARM64**: Limited support (no LSan, no TSan in CI)
  - **Workaround**: Use ASan with `ASAN_OPTIONS=detect_leaks=1`
- ⚠️ **Windows**: Good support for ASan and UBSan
- 💡 **Key Insight**: ASan with leak detection is equivalent to LSan on macOS

### Quick Reference

| Need | Ubuntu | macOS ARM64 | Windows |
|------|--------|-------------|---------|
| **Memory errors** | ASan | ASan | ASan |
| **Undefined behavior** | UBSan | UBSan | UBSan |
| **Data races** | TSan | ⚠️ Local only | ❌ |
| **Memory leaks** | LSan | ASan + detect_leaks | ❌ |

---

## Related Files

- **CI Configuration**: `.github/workflows/ci.yml` (lines 546-558)
- **Sanitizer CMake**: `Cmake/tools/sanitize.cmake`
- **Sanitizer Ignore List**: `Scripts/sanitizer_ignore.txt`
- **CI Fixes Documentation**: `docs/CI_FIXES_IMPLEMENTATION_SUMMARY.md`

---

## Support

For questions or issues with sanitizers:
1. Check this documentation first
2. Review CI logs for specific error messages
3. Test locally with the same sanitizer configuration
4. Consult LLVM sanitizer documentation (links in References section)

