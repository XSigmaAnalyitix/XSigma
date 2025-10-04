# XSigma CI/CD Pipeline Documentation

## Overview

The XSigma project uses a comprehensive CI/CD pipeline implemented with GitHub Actions to ensure code quality, correctness, and performance across multiple platforms and configurations.

## Pipeline Architecture

The CI/CD pipeline consists of the following major components:

### 1. Build Configuration Matrix
- **Platforms**: Ubuntu, Windows, macOS
- **Build Types**: Debug, Release
- **C++ Standards**: C++17, C++20
- **Logging Backends**: NATIVE, LOGURU, GLOG
- **Dependencies**: Intel TBB enabled by default

### 2. Memory Testing
- **Valgrind** (Linux only): Full memory leak detection and error checking
- **Platform Support**: Ubuntu latest
- **Configuration**: Debug build with comprehensive Valgrind options

### 3. Sanitizer Testing
Comprehensive sanitizer coverage across multiple platforms:
- **AddressSanitizer (ASan)**: Detects memory errors, buffer overflows, use-after-free
- **UndefinedBehaviorSanitizer (UBSan)**: Detects undefined behavior
- **ThreadSanitizer (TSan)**: Detects data races and threading issues
- **LeakSanitizer (LSan)**: Detects memory leaks
- **Platforms**: Ubuntu and macOS (where supported)

### 4. Code Quality Checks
- **Static Analysis**: clang-tidy
- **Code Checking**: cppcheck
- **Build Type**: Debug for maximum diagnostic information

### 5. Code Coverage Analysis
- **Minimum Requirement**: 98% code coverage
- **Tool**: LLVM coverage tools (llvm-cov, llvm-profdata)
- **Reporting**: Automatic analysis with threshold checking
- **Integration**: Codecov upload for visualization

### 6. Optimization Testing
Tests with different optimization levels:
- **-O0**: No optimization (Debug)
- **-O2**: Standard optimization (Release)
- **-O3**: Aggressive optimization (Release)

### 7. Performance Benchmarks
- **Build Type**: Release with LTO enabled
- **Framework**: Google Benchmark
- **Purpose**: Performance regression detection

## Job Details

### Build Matrix Job
```yaml
Job: build-matrix
Runs on: ubuntu-latest, windows-latest, macos-latest
Matrix dimensions:
  - OS: 3 platforms
  - Build Type: 2 types (Debug, Release)
  - C++ Standard: 2 versions (17, 20)
  - Logging Backend: 3 backends (NATIVE, LOGURU, GLOG)
Total combinations: ~30 (with exclusions for optimization)
```

**Features**:
- Dependency caching for faster builds
- Intel MKL and TBB installation
- Parallel test execution
- Artifact upload on failure

### Valgrind Memory Check Job
```yaml
Job: valgrind-memory-check
Runs on: ubuntu-latest
Build Type: Debug
```

**Checks**:
- Memory leaks (definitely lost, indirectly lost)
- Invalid memory access (reads/writes)
- Use of uninitialized memory
- Memory errors with detailed stack traces

**Configuration**:
- Full leak checking
- Origin tracking for uninitialized values
- Suppression file support
- Error exit codes for CI failure

### Sanitizer Tests Job
```yaml
Job: sanitizer-tests
Runs on: ubuntu-latest, macos-latest
Sanitizers: address, undefined, thread, leak
```

**Environment Options**:
- ASan: `detect_leaks=1:abort_on_error=1:symbolize=1`
- UBSan: `print_stacktrace=1:halt_on_error=1:symbolize=1`
- TSan: `halt_on_error=1:second_deadlock_stack=1`
- LSan: Custom suppression file support

### Coverage Test Job
```yaml
Job: coverage-test
Runs on: ubuntu-latest
Build Type: Debug
Minimum Coverage: 98%
```

**Process**:
1. Build with coverage instrumentation
2. Run all tests
3. Generate coverage data
4. Analyze with `analyze_coverage.py`
5. Upload to Codecov
6. Fail if below 98% threshold

### Optimization Flags Test Job
```yaml
Job: optimization-flags-test
Runs on: ubuntu-latest
Optimization Levels: -O0, -O2, -O3
```

**Purpose**: Ensure code correctness across all optimization levels

### Benchmark Tests Job
```yaml
Job: benchmark-tests
Runs on: ubuntu-latest
Build Type: Release
LTO: Enabled
```

**Purpose**: Performance regression detection and baseline establishment

## Dependency Management

### Intel MKL Installation

#### Ubuntu
```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-oneapi-mkl-devel intel-oneapi-tbb-devel
```

#### macOS
Uses system Accelerate framework as BLAS/LAPACK alternative.

#### Windows
Placeholder for Intel installer (requires manual setup or custom action).

### Intel TBB Installation

#### Ubuntu
```bash
sudo apt-get install -y libtbb-dev
```

#### macOS
```bash
brew install tbb
```

#### Windows
Included with Intel oneAPI or via vcpkg.

### Dependency Caching

All jobs use GitHub Actions cache to speed up builds:
```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache
      build/ThirdParty
    key: ${{ runner.os }}-deps-${{ env.CACHE_VERSION }}-${{ hashFiles('**/CMakeLists.txt') }}
```

**Benefits**:
- Faster CI runs (5-10x speedup for dependencies)
- Reduced network usage
- Consistent dependency versions

## Running Tests Locally

### Full Build Matrix Test
```bash
# Test with different configurations
cd Scripts
python3 setup.py config.ninja.clang.build.test
python3 setup.py config.ninja.clang.build.test.release
python3 setup.py config.ninja.clang.build.test --logging.backend=GLOG
```

### Valgrind Memory Check
```bash
cd Scripts
python3 setup.py config.ninja.clang.valgrind.build.test
# Or manually:
./valgrind_ctest.sh ../build_ninja_valgrind
```

### Sanitizer Tests
```bash
cd Scripts
python3 setup.py config.ninja.clang.build.test --sanitizer.address
python3 setup.py config.ninja.clang.build.test --sanitizer.undefined
python3 setup.py config.ninja.clang.build.test --sanitizer.thread
python3 setup.py config.ninja.clang.build.test --sanitizer.leak
```

### Coverage Analysis
```bash
cd Scripts
python3 setup.py ninja.clang.config.build.test.coverage
# Coverage analysis runs automatically
# Or re-analyze existing data:
python3 setup.py analyze
```

### Optimization Testing
```bash
cmake -B build -DCMAKE_CXX_FLAGS="-O3" ...
cmake --build build
cd build && ctest
```

## CI Success Criteria

All jobs must pass for CI to succeed:

1. ✅ **Build Matrix**: All platform/configuration combinations build and test successfully
2. ✅ **Valgrind**: No memory leaks or errors detected
3. ✅ **Sanitizers**: All sanitizer tests pass without errors
4. ✅ **Code Quality**: Static analysis passes without critical issues
5. ✅ **Coverage**: Minimum 98% code coverage achieved
6. ✅ **Optimization**: Tests pass at all optimization levels
7. ⚠️  **Benchmarks**: Non-blocking (informational only)

## Troubleshooting

### Valgrind Failures
- Check `Testing/Temporary/MemoryChecker.*.log` for details
- Add suppressions to `Cmake/xsigmaValgrindSuppression.txt` for false positives
- Ensure tests don't have actual memory leaks

### Sanitizer Failures
- Review sanitizer output in test logs
- Check `Scripts/sanitizer_ignore.txt` for suppressions
- Verify thread safety for TSan failures

### Coverage Failures
- Run `python3 Scripts/analyze_coverage.py --verbose` locally
- Identify untested code paths
- Add tests to increase coverage

### Build Failures
- Check compiler compatibility
- Verify all dependencies are installed
- Review CMake configuration output

## Performance Optimization

### Parallel Execution
- Tests run with `-j 2` for parallel execution
- Build uses all available cores
- Matrix jobs run in parallel

### Caching Strategy
- Dependencies cached per OS and configuration
- Cache invalidated on CMakeLists.txt changes
- Separate caches for different job types

### Matrix Optimization
- Strategic exclusions to reduce redundant combinations
- Focus on critical configurations
- Balance between coverage and CI time

## Future Enhancements

- [ ] Add Docker-based testing for reproducibility
- [ ] Implement nightly builds with extended test suites
- [ ] Add performance regression tracking
- [ ] Integrate with additional code quality tools
- [ ] Add Windows-specific memory testing tools
- [ ] Implement automatic benchmark comparison
- [ ] Add integration tests with real-world scenarios

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Valgrind Documentation](https://valgrind.org/docs/)
- [Clang Sanitizers](https://clang.llvm.org/docs/index.html)
- [CMake Testing](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)

