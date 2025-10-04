# CI/CD Pipeline Quick Start Guide

## Overview

This guide provides quick commands to test your code locally before pushing to CI.

## Prerequisites

### Ubuntu/Linux
```bash
sudo apt-get update
sudo apt-get install -y \
  ninja-build \
  clang \
  llvm \
  cmake \
  python3 \
  python3-pip \
  valgrind \
  libtbb-dev \
  lcov

pip install psutil colorama
```

### macOS
```bash
brew install ninja llvm cmake python3 tbb

# Valgrind (Intel Macs only)
brew install valgrind  # Will fail on Apple Silicon

pip3 install psutil colorama
```

### Windows
```powershell
choco install ninja llvm cmake python3 -y
pip install psutil colorama
```

## Quick Test Commands

### 1. Basic Build and Test
```bash
cd Scripts
python3 setup.py config.ninja.clang.build.test
```

### 2. Memory Check with Valgrind (Linux/Intel Mac only)
```bash
cd Scripts
python3 setup.py config.ninja.clang.valgrind.build.test
```

### 3. AddressSanitizer Test
```bash
cd Scripts
python3 setup.py config.ninja.clang.build.test --sanitizer.address
```

### 4. UndefinedBehaviorSanitizer Test
```bash
cd Scripts
python3 setup.py config.ninja.clang.build.test --sanitizer.undefined
```

### 5. ThreadSanitizer Test
```bash
cd Scripts
python3 setup.py config.ninja.clang.build.test --sanitizer.thread
```

### 6. Code Coverage Test (98% minimum)
```bash
cd Scripts
python3 setup.py ninja.clang.config.build.test.coverage
# Coverage analysis runs automatically
```

### 7. Test Different C++ Standards
```bash
cd Scripts
# C++17
python3 setup.py config.ninja.clang.build.test cxx17

# C++20
python3 setup.py config.ninja.clang.build.test cxx20
```

### 8. Test Different Logging Backends
```bash
cd Scripts
# LOGURU (default)
python3 setup.py config.ninja.clang.build.test

# GLOG
python3 setup.py config.ninja.clang.build.test --logging.backend=GLOG

# NATIVE
python3 setup.py config.ninja.clang.build.test --logging.backend=NATIVE
```

### 9. Debug vs Release Builds
```bash
cd Scripts
# Debug
python3 setup.py config.ninja.clang.build.test debug

# Release
python3 setup.py config.ninja.clang.build.test release
```

### 10. Run Benchmarks
```bash
cd Scripts
python3 setup.py config.ninja.clang.build.test release
cd ../build_ninja
ctest -R ".*[Bb]enchmark.*" --verbose
```

## Pre-Push Checklist

Before pushing your code, run these tests locally:

### Minimum Tests (Required)
```bash
# 1. Basic build and test
cd Scripts
python3 setup.py config.ninja.clang.build.test

# 2. Memory check (Linux/Intel Mac)
python3 setup.py config.ninja.clang.valgrind.build.test

# 3. AddressSanitizer
python3 setup.py config.ninja.clang.build.test --sanitizer.address

# 4. Code coverage
python3 setup.py ninja.clang.config.build.test.coverage
```

### Recommended Tests
```bash
# 5. UndefinedBehaviorSanitizer
cd Scripts
python3 setup.py config.ninja.clang.build.test --sanitizer.undefined

# 6. Different logging backend
python3 setup.py config.ninja.clang.build.test --logging.backend=GLOG

# 7. Release build
python3 setup.py config.ninja.clang.build.test release
```

## Common Issues and Solutions

### Issue: Valgrind not found
**Solution**: Install Valgrind or skip this test on Apple Silicon
```bash
# Ubuntu
sudo apt-get install valgrind

# macOS (Intel only)
brew install valgrind

# Apple Silicon - use sanitizers instead
cd Scripts
python3 setup.py config.ninja.clang.build.test --sanitizer.address
```

### Issue: Coverage below 98%
**Solution**: Add tests for uncovered code
```bash
# Identify uncovered files
cd Scripts
python3 analyze_coverage.py --build-dir ../build_ninja_coverage --verbose

# Add tests for the identified files
# Re-run coverage
python3 setup.py ninja.clang.config.build.test.coverage
```

### Issue: Sanitizer errors
**Solution**: Fix the reported issues
```bash
# Run with verbose output
cd build_ninja_sanitizer_address
ctest --verbose --output-on-failure

# Check sanitizer logs
cat Testing/Temporary/LastTest.log
```

### Issue: Build fails on Windows
**Solution**: Ensure all dependencies are installed
```powershell
# Reinstall dependencies
choco install ninja llvm cmake python3 -y --force

# Try building again
cd Scripts
python setup.py config.ninja.clang.build.test
```

## CI Pipeline Simulation

To simulate the full CI pipeline locally:

```bash
#!/bin/bash
# Save as: test_all.sh

set -e

echo "=== Running Full CI Simulation ==="

cd Scripts

# 1. Build Matrix (subset)
echo "1. Testing Debug build..."
python3 setup.py config.ninja.clang.build.test debug

echo "2. Testing Release build..."
python3 setup.py config.ninja.clang.build.test release

echo "3. Testing C++20..."
python3 setup.py config.ninja.clang.build.test cxx20

# 2. Memory Testing
echo "4. Running Valgrind..."
python3 setup.py config.ninja.clang.valgrind.build.test || echo "Valgrind skipped"

# 3. Sanitizers
echo "5. Running AddressSanitizer..."
python3 setup.py config.ninja.clang.build.test --sanitizer.address

echo "6. Running UndefinedBehaviorSanitizer..."
python3 setup.py config.ninja.clang.build.test --sanitizer.undefined

# 4. Coverage
echo "7. Running coverage analysis..."
python3 setup.py ninja.clang.config.build.test.coverage

echo ""
echo "=== All Tests Passed! ==="
```

Make it executable and run:
```bash
chmod +x test_all.sh
./test_all.sh
```

## Viewing CI Results

### GitHub Actions
1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select your workflow run
4. View individual job results

### Artifacts
Failed jobs upload artifacts for debugging:
- Test logs: `build/Testing/Temporary/LastTest.log`
- Valgrind logs: `build/Testing/Temporary/MemoryChecker.*.log`
- Coverage reports: `build/coverage_report/`

### Coverage Reports
View coverage on Codecov (if configured):
1. Go to https://codecov.io/gh/YOUR_ORG/XSigma
2. View coverage by file
3. Identify uncovered lines

## Performance Tips

### Speed Up Local Testing
```bash
# Use ccache for faster rebuilds
sudo apt-get install ccache  # Ubuntu
brew install ccache          # macOS

# Configure CMake to use ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_C_COMPILER_LAUNCHER=ccache
```

### Parallel Testing
```bash
# Run tests in parallel
cd build_ninja
ctest -j $(nproc)  # Linux
ctest -j $(sysctl -n hw.ncpu)  # macOS
```

### Incremental Builds
```bash
# Don't clean between builds
cd Scripts
python3 setup.py build.test  # Skip 'config' to reuse existing configuration
```

## Getting Help

### Documentation
- Full CI/CD docs: `docs/CI_CD_PIPELINE.md`
- Valgrind setup: `docs/VALGRIND_SETUP.md`
- Implementation summary: `docs/CI_IMPLEMENTATION_SUMMARY.md`

### Debugging
```bash
# Verbose CMake output
cd Scripts
python3 setup.py config.ninja.clang.build.test v

# Very verbose output
python3 setup.py config.ninja.clang.build.test vv

# Verbose test output
cd build_ninja
ctest --verbose --output-on-failure
```

### Common Commands
```bash
# Clean build
rm -rf build_ninja*

# View test list
cd build_ninja
ctest -N

# Run specific test
ctest -R TestName --verbose

# Re-run failed tests
ctest --rerun-failed --output-on-failure
```

## Summary

**Minimum local testing before push:**
1. ✅ Basic build and test
2. ✅ Memory check (Valgrind or ASan)
3. ✅ Code coverage (98% minimum)

**Recommended for major changes:**
4. ✅ Multiple sanitizers
5. ✅ Different build types
6. ✅ Different C++ standards
7. ✅ Different logging backends

**CI will automatically test:**
- All platform combinations (Linux, macOS, Windows)
- All build configurations
- All sanitizers
- Code quality checks
- Performance benchmarks

---

For more details, see `docs/CI_CD_PIPELINE.md`

