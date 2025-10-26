# Quick Reference: CMake CI Commands

## One-Liner Equivalents

### Basic Build (Unix/macOS)
```bash
# setup.py
python setup.py ninja clang debug test cxx17 loguru

# CMake
cmake -B build_ninja -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DXSIGMA_LOGGING_BACKEND=LOGURU -DXSIGMA_BUILD_TESTING=ON -DXSIGMA_ENABLE_GTEST=ON -DXSIGMA_ENABLE_TBB=OFF -DXSIGMA_ENABLE_CUDA=OFF -DXSIGMA_ENABLE_LTO=OFF && cmake --build build_ninja -j 2 && ctest --test-dir build_ninja --output-on-failure -j 2
```

### With TBB
```bash
# setup.py
python setup.py ninja clang debug test cxx17 loguru tbb

# CMake
cmake -B build_ninja_tbb -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DXSIGMA_LOGGING_BACKEND=LOGURU -DXSIGMA_BUILD_TESTING=ON -DXSIGMA_ENABLE_GTEST=ON -DXSIGMA_ENABLE_TBB=ON -DXSIGMA_ENABLE_CUDA=OFF -DXSIGMA_ENABLE_LTO=OFF && cmake --build build_ninja_tbb -j 2 && ctest --test-dir build_ninja_tbb --output-on-failure -j 2
```

### Coverage Build
```bash
# setup.py
python setup.py config.build.ninja.clang.TEST.coverage

# CMake
cmake -B build_ninja_coverage -S . -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DXSIGMA_LOGGING_BACKEND=NATIVE -DXSIGMA_BUILD_TESTING=ON -DXSIGMA_ENABLE_GTEST=ON -DXSIGMA_ENABLE_COVERAGE=ON -DXSIGMA_ENABLE_TBB=OFF -DXSIGMA_ENABLE_CUDA=OFF -DXSIGMA_ENABLE_LTO=OFF && cmake --build build_ninja_coverage -j 2 && ctest --test-dir build_ninja_coverage --output-on-failure -j 2
```

---

## Multi-Line Format (Recommended)

### Basic Build
```bash
cmake -B build_ninja \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DXSIGMA_LOGGING_BACKEND=LOGURU \
  -DXSIGMA_BUILD_TESTING=ON \
  -DXSIGMA_ENABLE_GTEST=ON \
  -DXSIGMA_ENABLE_TBB=OFF \
  -DXSIGMA_ENABLE_CUDA=OFF \
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja -j 2
ctest --test-dir build_ninja --output-on-failure -j 2
```

### With TBB
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

### Release Build
```bash
cmake -B build_ninja_release \
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

cmake --build build_ninja_release -j 2
ctest --test-dir build_ninja_release --output-on-failure -j 2
```

### C++20 Build
```bash
cmake -B build_ninja_cxx20 \
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

cmake --build build_ninja_cxx20 -j 2
ctest --test-dir build_ninja_cxx20 --output-on-failure -j 2
```

### Coverage Build
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

# Coverage analysis
cd Scripts
python run_coverage.py
```

---

## Windows PowerShell Format

### Basic Build
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

---

## Common Patterns

### Change Build Type
```bash
# Debug
-DCMAKE_BUILD_TYPE=Debug

# Release
-DCMAKE_BUILD_TYPE=Release

# Release with Debug Info
-DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Change C++ Standard
```bash
# C++17
-DCMAKE_CXX_STANDARD=17

# C++20
-DCMAKE_CXX_STANDARD=20

# C++23
-DCMAKE_CXX_STANDARD=23
```

### Change Logging Backend
```bash
# Loguru (default)
-DXSIGMA_LOGGING_BACKEND=LOGURU

# Google Log
-DXSIGMA_LOGGING_BACKEND=GLOG

# Native
-DXSIGMA_LOGGING_BACKEND=NATIVE
```

### Enable/Disable Features
```bash
# TBB
-DXSIGMA_ENABLE_TBB=ON    # Enable
-DXSIGMA_ENABLE_TBB=OFF   # Disable

# CUDA
-DXSIGMA_ENABLE_CUDA=ON   # Enable
-DXSIGMA_ENABLE_CUDA=OFF  # Disable

# Coverage
-DXSIGMA_ENABLE_COVERAGE=ON   # Enable
-DXSIGMA_ENABLE_COVERAGE=OFF  # Disable

# LTO
-DXSIGMA_ENABLE_LTO=ON    # Enable
-DXSIGMA_ENABLE_LTO=OFF   # Disable
```

---

## Build Directory Naming

| Configuration | Directory |
|---------------|-----------|
| Base | `build_ninja` |
| + TBB | `build_ninja_tbb` |
| + C++20 | `build_ninja_cxx20` |
| + TBB + C++20 | `build_ninja_tbb_cxx20` |
| + C++23 | `build_ninja_cxx23` |
| + TBB + C++23 | `build_ninja_tbb_cxx23` |
| Coverage | `build_ninja_coverage` |

---

## Useful Commands

### Clean Build
```bash
rm -rf build_ninja*
```

### Rebuild
```bash
cmake --build build_ninja --clean-first -j 2
```

### Run Specific Test
```bash
ctest --test-dir build_ninja -R "test_name" --output-on-failure
```

### Verbose Build
```bash
cmake --build build_ninja --verbose -j 2
```

### Verbose Tests
```bash
ctest --test-dir build_ninja --verbose
```

---

## Files Modified

- `.github/workflows/ci.yml` - Replaced setup.py calls with raw CMake

---

## Documentation

- `CI_CMAKE_MIGRATION_GUIDE.md` - Full migration guide
- `CMAKE_COMMAND_REFERENCE.md` - Detailed command reference
- `CI_MIGRATION_COMPLETE.md` - Migration summary
- `QUICK_REFERENCE_CMAKE_CI.md` - This file

