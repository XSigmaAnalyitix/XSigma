# Sanitizers

XSigma provides comprehensive sanitizer support for memory debugging and analysis with all modern sanitizers across multiple compilers. Sanitizers are powerful tools that detect memory errors, undefined behavior, data races, and other runtime issues.

## Table of Contents

- [Supported Sanitizers](#supported-sanitizers)
- [Quick Start](#quick-start)
- [Sanitizer Descriptions](#sanitizer-descriptions)
- [Customizing Sanitizer Behavior](#customizing-sanitizer-behavior)
- [Platform-Specific Considerations](#platform-specific-considerations)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Supported Sanitizers

| Sanitizer | Purpose | GCC | Clang | Apple Clang | MSVC |
|-----------|---------|-----|-------|-------------|------|
| **AddressSanitizer** | Memory errors, buffer overflows | ✅ | ✅ | ✅ | ✅ |
| **UndefinedBehaviorSanitizer** | Undefined behavior detection | ✅ | ✅ | ✅ | ❌ |
| **ThreadSanitizer** | Data race detection | ✅ | ✅ | ✅ | ❌ |
| **MemorySanitizer** | Uninitialized memory reads | ❌ | ✅ | ✅ | ❌ |
| **LeakSanitizer** | Memory leak detection | ✅ | ✅ | ✅ | ❌ |

## Quick Start

### Using setup.py (Recommended)

```bash
cd Scripts

# AddressSanitizer - detects memory errors and buffer overflows
python setup.py ninja.clang.test.config --sanitizer.address

# UndefinedBehaviorSanitizer - detects undefined behavior
python setup.py ninja.clang.test.config --sanitizer.undefined

# ThreadSanitizer - detects data races and threading issues
python setup.py ninja.clang.test.config --sanitizer.thread

# MemorySanitizer - detects uninitialized memory reads (Clang only)
python setup.py ninja.clang.test.config --sanitizer.memory

# LeakSanitizer - detects memory leaks
python setup.py ninja.clang.test.config --sanitizer.leak

# Alternative syntax
python setup.py vs22.test.build --sanitizer-type=address
```

### Using CMake Directly

```bash
# AddressSanitizer
cmake -B build -S . \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=address

# UndefinedBehaviorSanitizer
cmake -B build -S . \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=undefined

# ThreadSanitizer
cmake -B build -S . \
    -DXSIGMA_ENABLE_SANITIZER=ON \
    -DXSIGMA_SANITIZER_TYPE=thread

# Build
cmake --build build
```

## Sanitizer Descriptions

### AddressSanitizer (ASan)

**Purpose**: Detects buffer overflows, use-after-free, double-free, memory leaks

**Performance**: ~2x slowdown, ~2-3x memory usage

**Best for**: General memory debugging, CI/CD pipelines

**Platforms**: All supported (Windows, Linux, macOS)

**Example Issues Detected**:
- Heap buffer overflow
- Stack buffer overflow
- Use after free
- Use after return
- Double free
- Memory leaks (when combined with LeakSanitizer)

### UndefinedBehaviorSanitizer (UBSan)

**Purpose**: Detects undefined behavior like integer overflow, null pointer dereference

**Performance**: ~20% slowdown, minimal memory overhead

**Best for**: Code quality assurance, detecting subtle bugs

**Platforms**: GCC, Clang (not MSVC)

**Example Issues Detected**:
- Integer overflow
- Division by zero
- Null pointer dereference
- Misaligned pointer access
- Invalid type casts

### ThreadSanitizer (TSan)

**Purpose**: Detects data races, deadlocks, thread safety issues

**Performance**: ~5-15x slowdown, ~5-10x memory usage

**Best for**: Multithreaded code debugging

**Platforms**: GCC, Clang (not MSVC)

**Note**: Cannot be used with AddressSanitizer simultaneously

**Example Issues Detected**:
- Data races
- Deadlocks
- Thread leaks
- Improper synchronization

### MemorySanitizer (MSan)

**Purpose**: Detects reads of uninitialized memory

**Performance**: ~3x slowdown, ~3x memory usage

**Best for**: Finding uninitialized variable bugs

**Platforms**: Clang only

**Note**: Requires rebuilding all dependencies with MSan

**Example Issues Detected**:
- Use of uninitialized memory
- Uninitialized function arguments
- Uninitialized struct fields

### LeakSanitizer (LSan)

**Purpose**: Detects memory leaks

**Performance**: Minimal runtime overhead

**Best for**: Memory leak detection in long-running applications

**Platforms**: GCC, Clang (not MSVC)

**Note**: Can be used standalone or with AddressSanitizer

**Example Issues Detected**:
- Memory leaks
- Resource leaks
- Unreachable memory

## Customizing Sanitizer Behavior

### Sanitizer Ignore File

XSigma uses `Scripts/sanitizer_ignore.txt` to exclude files, functions, or types from sanitizer checks:

```bash
# Edit the ignore file to customize sanitizer behavior
vim Scripts/sanitizer_ignore.txt
```

**Common ignore patterns:**
```
# Ignore third-party libraries
src:*/ThirdParty/*
src:*/external/*

# Ignore specific functions
fun:*test*
fun:benchmark_*

# Ignore specific types
type:std::*
type:boost::*

# Sanitizer-specific exclusions
# AddressSanitizer only
src:*/performance_critical.cpp

# ThreadSanitizer only
fun:*lockfree*
```

### Environment Variables

Control sanitizer behavior at runtime:

```bash
# AddressSanitizer options
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:check_initialization_order=1"

# UndefinedBehaviorSanitizer options
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"

# ThreadSanitizer options
export TSAN_OPTIONS="detect_deadlocks=1:second_deadlock_stack=1"

# MemorySanitizer options
export MSAN_OPTIONS="print_stats=1:halt_on_error=1"

# LeakSanitizer options
export LSAN_OPTIONS="suppressions=leak_suppressions.txt"
```

### Debugging Sanitizer Output

```bash
# Get detailed stack traces
export ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer
export MSAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer

# Save sanitizer output to file
export ASAN_OPTIONS="log_path=./asan_log"
export TSAN_OPTIONS="log_path=./tsan_log"

# Enable additional debugging
export ASAN_OPTIONS="verbosity=1:debug=1"
```

## Platform-Specific Considerations

### Windows (MSVC)

- Only AddressSanitizer is supported
- Requires Visual Studio 2019 16.9+ or Visual Studio 2022
- May require `/MD` runtime library flag
- Debug information (`/Zi`) recommended for better stack traces

### Linux (GCC/Clang)

- All sanitizers supported (MSan only with Clang)
- Runtime libraries automatically detected and preloaded
- Best performance with debug symbols (`-g`)
- Consider using `gold` linker for faster linking

### macOS (Apple Clang)

- All sanitizers supported except MemorySanitizer
- System Integrity Protection (SIP) may interfere with some sanitizers
- Use `DYLD_INSERT_LIBRARIES` for runtime library preloading
- Xcode integration available through build schemes

## Best Practices

### Development Workflow

1. **Start with AddressSanitizer** - catches most common memory errors
2. **Add UndefinedBehaviorSanitizer** - minimal overhead, catches subtle bugs
3. **Use ThreadSanitizer** for multithreaded code - run separately from ASan
4. **Apply MemorySanitizer** for critical code paths - requires clean build
5. **Enable LeakSanitizer** for long-running applications

### CI/CD Integration

```bash
# Separate CI jobs for different sanitizers
- name: "AddressSanitizer Tests"
  run: python setup.py ninja.clang.test.build --sanitizer.address

- name: "UndefinedBehavior Tests"
  run: python setup.py ninja.clang.test.build --sanitizer.undefined

- name: "Thread Safety Tests"
  run: python setup.py ninja.clang.test.build --sanitizer.thread
```

### Performance Testing

- Use sanitizers in debug/testing builds only
- Disable sanitizers for performance benchmarks
- Consider sanitizer overhead when setting test timeouts
- Use sanitizer-specific optimization flags when needed

## Troubleshooting

### Common Problems and Solutions

**1. "Sanitizer runtime library not found"**
```bash
# Install sanitizer runtime libraries
# Ubuntu/Debian
sudo apt-get install libc6-dbg gcc-multilib

# CentOS/RHEL
sudo yum install glibc-debuginfo

# macOS
xcode-select --install
```

**2. "Cannot combine AddressSanitizer with ThreadSanitizer"**
- These sanitizers are mutually exclusive
- Run separate builds for each sanitizer
- Use different build directories

**3. "MemorySanitizer requires rebuilding dependencies"**
```bash
# Build all dependencies with MemorySanitizer
export CC=clang
export CXX=clang++
export CFLAGS="-fsanitize=memory -fsanitize-memory-track-origins=2"
export CXXFLAGS="-fsanitize=memory -fsanitize-memory-track-origins=2"
```

**4. "Sanitizer reports false positives"**
- Add patterns to `Scripts/sanitizer_ignore.txt`
- Use sanitizer-specific suppressions
- Check for third-party library issues

**5. "Slow build times with sanitizers"**
```bash
# Use faster linker
export LDFLAGS="-fuse-ld=gold"  # Linux
export LDFLAGS="-fuse-ld=lld"   # Clang

# Reduce optimization level
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
```

**6. "Out of memory during sanitizer build"**
- Increase system swap space
- Use distributed compilation
- Build with fewer parallel jobs: `cmake --build build -j2`

## Related Documentation

- [Build Configuration](build-configuration.md) - Build system configuration
- [Code Coverage](code-coverage.md) - Test coverage analysis
- [Static Analysis](static-analysis.md) - IWYU and Cppcheck tools

