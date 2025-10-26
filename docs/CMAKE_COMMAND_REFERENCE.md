# CMake Command Reference - setup.py Equivalents

## Overview

This document provides a complete reference of CMake commands that replace `setup.py` invocations in the CI workflow.

---

## Build Matrix Commands

### Ubuntu C++17 Debug - LOGURU - TBB:ON

**setup.py**:
```bash
python setup.py ninja clang debug test cxx17 loguru tbb
```

**Raw CMake**:
```bash
cmake -B build_ninja_tbb \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DXSIGMA_LOGGING_BACKEND=LOGURU \
  -DXSIGMA_BUILD_TESTING=ON \
  -DXSIGMA_ENABLE_GTEST=ON \
  -DXSIGMA_ENABLE_TBB=ON \
  -DXSIGMA_ENABLE_CUDA=OFF \
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja_tbb -j 2
ctest --test-dir build_ninja_tbb --output-on-failure -j 2
```

### Ubuntu C++17 Debug - NATIVE - TBB:OFF

**setup.py**:
```bash
python setup.py ninja clang debug test cxx17 native
```

**Raw CMake**:
```bash
cmake -B build_ninja \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DXSIGMA_LOGGING_BACKEND=NATIVE \
  -DXSIGMA_BUILD_TESTING=ON \
  -DXSIGMA_ENABLE_GTEST=ON \
  -DXSIGMA_ENABLE_TBB=OFF \
  -DXSIGMA_ENABLE_CUDA=OFF \
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja -j 2
ctest --test-dir build_ninja --output-on-failure -j 2
```

### Ubuntu C++17 Release - GLOG - TBB:ON

**setup.py**:
```bash
python setup.py ninja clang release test cxx17 glog tbb
```

**Raw CMake**:
```bash
cmake -B build_ninja_tbb \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DXSIGMA_LOGGING_BACKEND=GLOG \
  -DXSIGMA_BUILD_TESTING=ON \
  -DXSIGMA_ENABLE_GTEST=ON \
  -DXSIGMA_ENABLE_TBB=ON \
  -DXSIGMA_ENABLE_CUDA=OFF \
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja_tbb -j 2
ctest --test-dir build_ninja_tbb --output-on-failure -j 2
```

### Windows C++17 Release - NATIVE - CUDA:OFF

**setup.py**:
```powershell
python setup.py ninja clang release test cxx17 native
```

**Raw CMake**:
```powershell
cmake -B build_ninja `
  -S . `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang `
  -DCMAKE_CXX_COMPILER=clang++ `
  -DCMAKE_CXX_STANDARD=17 `
  -DXSIGMA_LOGGING_BACKEND=NATIVE `
  -DXSIGMA_BUILD_TESTING=ON `
  -DXSIGMA_ENABLE_GTEST=ON `
  -DXSIGMA_ENABLE_TBB=OFF `
  -DXSIGMA_ENABLE_CUDA=OFF `
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja -j 2
ctest --test-dir build_ninja --output-on-failure -j 2
```

### Ubuntu C++20 Release - TBB:ON

**setup.py**:
```bash
python setup.py ninja clang release test cxx20 loguru tbb
```

**Raw CMake**:
```bash
cmake -B build_ninja_tbb_cxx20 \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=20 \
  -DXSIGMA_LOGGING_BACKEND=LOGURU \
  -DXSIGMA_BUILD_TESTING=ON \
  -DXSIGMA_ENABLE_GTEST=ON \
  -DXSIGMA_ENABLE_TBB=ON \
  -DXSIGMA_ENABLE_CUDA=OFF \
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja_tbb_cxx20 -j 2
ctest --test-dir build_ninja_tbb_cxx20 --output-on-failure -j 2
```

### Ubuntu C++23 Release - TBB:ON

**setup.py**:
```bash
python setup.py ninja clang release test cxx23 glog tbb
```

**Raw CMake**:
```bash
cmake -B build_ninja_tbb_cxx23 \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=23 \
  -DXSIGMA_LOGGING_BACKEND=GLOG \
  -DXSIGMA_BUILD_TESTING=ON \
  -DXSIGMA_ENABLE_GTEST=ON \
  -DXSIGMA_ENABLE_TBB=ON \
  -DXSIGMA_ENABLE_CUDA=OFF \
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja_tbb_cxx23 -j 2
ctest --test-dir build_ninja_tbb_cxx23 --output-on-failure -j 2
```

---

## Coverage Build Command

### Clang Coverage - C++17

**setup.py**:
```bash
cd Scripts
python setup.py config.build.ninja.clang.TEST.coverage
```

**Raw CMake**:
```bash
cmake -B build_ninja_coverage \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DXSIGMA_LOGGING_BACKEND=NATIVE \
  -DXSIGMA_BUILD_TESTING=ON \
  -DXSIGMA_ENABLE_GTEST=ON \
  -DXSIGMA_ENABLE_COVERAGE=ON \
  -DXSIGMA_ENABLE_TBB=OFF \
  -DXSIGMA_ENABLE_CUDA=OFF \
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja_coverage -j 2
ctest --test-dir build_ninja_coverage --output-on-failure -j 2

# Coverage analysis (Python script)
cd Scripts
python run_coverage.py
```

---

## CMake Flag Definitions

### Build Type
- `-DCMAKE_BUILD_TYPE=Debug` - Debug build with symbols
- `-DCMAKE_BUILD_TYPE=Release` - Optimized release build
- `-DCMAKE_BUILD_TYPE=RelWithDebInfo` - Release with debug info

### Compiler Selection
- `-DCMAKE_C_COMPILER=clang` - C compiler
- `-DCMAKE_CXX_COMPILER=clang++` - C++ compiler
- `-DCMAKE_C_COMPILER=gcc` - GCC C compiler
- `-DCMAKE_CXX_COMPILER=g++` - GCC C++ compiler

### C++ Standard
- `-DCMAKE_CXX_STANDARD=17` - C++17
- `-DCMAKE_CXX_STANDARD=20` - C++20
- `-DCMAKE_CXX_STANDARD=23` - C++23

### XSigma Features
- `-DXSIGMA_LOGGING_BACKEND=LOGURU` - Loguru logging
- `-DXSIGMA_LOGGING_BACKEND=GLOG` - Google logging
- `-DXSIGMA_LOGGING_BACKEND=NATIVE` - Native logging
- `-DXSIGMA_BUILD_TESTING=ON` - Enable testing
- `-DXSIGMA_ENABLE_GTEST=ON` - Enable Google Test
- `-DXSIGMA_ENABLE_TBB=ON` - Enable Intel TBB
- `-DXSIGMA_ENABLE_TBB=OFF` - Disable Intel TBB
- `-DXSIGMA_ENABLE_CUDA=ON` - Enable CUDA
- `-DXSIGMA_ENABLE_CUDA=OFF` - Disable CUDA
- `-DXSIGMA_ENABLE_COVERAGE=ON` - Enable code coverage
- `-DXSIGMA_ENABLE_LTO=ON` - Enable Link Time Optimization
- `-DXSIGMA_ENABLE_LTO=OFF` - Disable LTO

### Build Commands
- `cmake --build <dir> -j 2` - Build with 2 parallel jobs
- `ctest --test-dir <dir> --output-on-failure -j 2` - Run tests

---

## Build Directory Naming Convention

| Configuration | Directory Name |
|---------------|----------------|
| Base Ninja | `build_ninja` |
| Ninja + TBB | `build_ninja_tbb` |
| Ninja + C++20 | `build_ninja_cxx20` |
| Ninja + TBB + C++20 | `build_ninja_tbb_cxx20` |
| Ninja + C++23 | `build_ninja_cxx23` |
| Ninja + TBB + C++23 | `build_ninja_tbb_cxx23` |
| Coverage | `build_ninja_coverage` |

---

## Notes

1. **Parallel Jobs**: `-j 2` limits parallel jobs for CI stability
2. **Test Output**: `--output-on-failure` shows test output only on failure
3. **Build Directory**: `-B` specifies build directory, `-S` specifies source
4. **Generator**: `-G Ninja` specifies Ninja as build generator
5. **Coverage**: Requires `-DXSIGMA_ENABLE_COVERAGE=ON` and Debug build type

