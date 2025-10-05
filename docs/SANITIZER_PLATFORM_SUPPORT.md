# Sanitizer Platform Support

**Last Updated**: 2025-10-05

---

## Overview

This document describes sanitizer support across different platforms in the XSigma project.

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
- **Reason**: 
  - LeakSanitizer is not supported on ARM64 architecture
  - ThreadSanitizer has stability issues in CI environment
- **Workaround**: Use AddressSanitizer with leak detection enabled (`ASAN_OPTIONS=detect_leaks=1`)

### macOS x86_64 (Intel)

✅ **Good sanitizer support**

- **Compiler**: Apple Clang 14+
- **Supported Sanitizers**: ASan, UBSan, TSan, LSan
- **Limited Support**: MSan
- **Notes**: Full support for most sanitizers on Intel Macs

---

## CI Configuration

### Sanitizer Tests in CI

The CI workflow (`.github/workflows/ci.yml`) runs sanitizer tests with the following exclusions:

```yaml
exclude:
  # ThreadSanitizer has issues on macOS in CI
  - sanitizer: thread
    os: macos-latest
  
  # LeakSanitizer is not supported on macOS ARM64 (Apple Silicon)
  - sanitizer: leak
    os: macos-latest
```

### Tested Configurations

| Platform | ASan | UBSan | TSan | LSan |
|----------|------|-------|------|------|
| **Ubuntu** | ✅ | ✅ | ✅ | ✅ |
| **macOS** | ✅ | ✅ | ❌ | ❌ |

**Total Sanitizer Jobs**: 6 (4 on Ubuntu + 2 on macOS)

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

**Error**:
```
clang++: error: unsupported option '-fsanitize=leak' for target 'arm64-apple-darwin24.6.0'
```

**Workaround**: Use AddressSanitizer with leak detection enabled:

```bash
# Configure with AddressSanitizer
cd Scripts
python setup.py ninja.clang.debug.address.config.build

# Run tests with leak detection
cd ../build_ninja_python
ASAN_OPTIONS=detect_leaks=1 ./bin/CoreCxxTests
```

**Note**: ASan's leak detection is nearly as effective as standalone LSan.

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

**Platform**: macOS ARM64

**Solution**: Use AddressSanitizer with leak detection instead:
```bash
ASAN_OPTIONS=detect_leaks=1 ./bin/CoreCxxTests
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

## Summary

- ✅ **Ubuntu**: Best platform for sanitizer testing (all supported)
- ⚠️ **macOS ARM64**: Limited support (no LSan, no TSan in CI)
- ⚠️ **Windows**: Good support for ASan and UBSan
- 💡 **Workaround**: Use ASan with leak detection on macOS ARM64

---

**For CI configuration details, see**: `.github/workflows/ci.yml` (lines 546-558)

