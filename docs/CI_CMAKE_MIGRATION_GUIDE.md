# CI Workflow: setup.py to Raw CMake Migration

## Overview

The `.github/workflows/ci.yml` file has been updated to replace all `setup.py` calls with equivalent raw CMake commands. This eliminates the Python wrapper layer and provides more direct control over the build process.

---

## Changes Made

### 1. Main Build Matrix (Lines 355-430)

#### Before (setup.py)
```bash
# Unix/macOS
python setup.py ninja clang debug test cxx17 loguru tbb

# Windows
python setup.py ninja clang debug test cxx17 loguru
```

#### After (Raw CMake)

**Unix/macOS**:
```bash
# Determine compiler paths
if [ "${{ matrix.compiler_cxx }}" = "clang" ]; then
  CC_COMPILER="clang"
  CXX_COMPILER="clang++"
else
  CC_COMPILER="${{ matrix.compiler_cxx }}"
  CXX_COMPILER="g++"
fi

# Configure CMake
cmake -B "$BUILD_DIR" \
  -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} \
  -DCMAKE_C_COMPILER=$CC_COMPILER \
  -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
  -DCMAKE_CXX_STANDARD=${{ matrix.cxx_std }} \
  -DXSIGMA_LOGGING_BACKEND=${{ matrix.logging_backend }} \
  -DXSIGMA_BUILD_TESTING=ON \
  -DXSIGMA_ENABLE_GTEST=ON \
  -DXSIGMA_ENABLE_TBB=${{ matrix.tbb_enabled }} \
  -DXSIGMA_ENABLE_CUDA=${{ matrix.cuda_enabled }} \
  -DXSIGMA_ENABLE_LTO=OFF

# Build
cmake --build "$BUILD_DIR" -j 2

# Run tests
ctest --test-dir "$BUILD_DIR" --output-on-failure -j 2
```

**Windows**:
```powershell
# Configure CMake
cmake -B "$buildDir" `
  -S . `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} `
  -DCMAKE_C_COMPILER=clang `
  -DCMAKE_CXX_COMPILER=clang++ `
  -DCMAKE_CXX_STANDARD=${{ matrix.cxx_std }} `
  -DXSIGMA_LOGGING_BACKEND=${{ matrix.logging_backend }} `
  -DXSIGMA_BUILD_TESTING=ON `
  -DXSIGMA_ENABLE_GTEST=ON `
  -DXSIGMA_ENABLE_TBB=${{ matrix.tbb_enabled }} `
  -DXSIGMA_ENABLE_CUDA=${{ matrix.cuda_enabled }} `
  -DXSIGMA_ENABLE_LTO=OFF

# Build
cmake --build "$buildDir" -j 2

# Run tests
ctest --test-dir "$buildDir" --output-on-failure -j 2
```

### 2. Coverage Build (Lines 1355-1382)

#### Before (setup.py)
```bash
cd Scripts
python setup.py config.build.ninja.clang.TEST.coverage
```

#### After (Raw CMake)
```bash
# Configure CMake with coverage enabled
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

# Build
cmake --build build_ninja_coverage -j 2

# Run tests
ctest --test-dir build_ninja_coverage --output-on-failure -j 2

# Run coverage analysis (Python script)
cd Scripts
python run_coverage.py
```

---

## Command Mapping

### setup.py Arguments → CMake Flags

| setup.py Arg | CMake Flag | Purpose |
|--------------|-----------|---------|
| `ninja` | `-G Ninja` | Build generator |
| `clang` | `-DCMAKE_CXX_COMPILER=clang++` | C++ compiler |
| `gcc` | `-DCMAKE_CXX_COMPILER=g++` | C++ compiler |
| `debug` | `-DCMAKE_BUILD_TYPE=Debug` | Build type |
| `release` | `-DCMAKE_BUILD_TYPE=Release` | Build type |
| `cxx17` | `-DCMAKE_CXX_STANDARD=17` | C++ standard |
| `cxx20` | `-DCMAKE_CXX_STANDARD=20` | C++ standard |
| `cxx23` | `-DCMAKE_CXX_STANDARD=23` | C++ standard |
| `loguru` | `-DXSIGMA_LOGGING_BACKEND=LOGURU` | Logging backend |
| `glog` | `-DXSIGMA_LOGGING_BACKEND=GLOG` | Logging backend |
| `native` | `-DXSIGMA_LOGGING_BACKEND=NATIVE` | Logging backend |
| `tbb` | `-DXSIGMA_ENABLE_TBB=ON` | TBB support |
| `test` | `-DXSIGMA_BUILD_TESTING=ON` | Enable testing |
| `coverage` | `-DXSIGMA_ENABLE_COVERAGE=ON` | Enable coverage |

---

## Benefits

✅ **Direct CMake Control**: No Python wrapper layer

✅ **Transparency**: Exact CMake commands visible in CI logs

✅ **Faster Execution**: Eliminates Python parsing overhead

✅ **Easier Debugging**: Direct CMake error messages

✅ **Better CI Integration**: Standard CMake commands

✅ **Reduced Dependencies**: No need for setup.py in CI

---

## Build Directory Naming

The build directories are now named consistently:

- **Base**: `build_ninja`
- **With TBB**: `build_ninja_tbb`
- **With C++20**: `build_ninja_cxx20`
- **With C++23**: `build_ninja_cxx23`
- **Coverage**: `build_ninja_coverage`

---

## Testing the Changes

### Local Testing

To verify the changes work locally, run:

```bash
# Unix/macOS
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
  -DXSIGMA_ENABLE_TBB=ON \
  -DXSIGMA_ENABLE_CUDA=OFF \
  -DXSIGMA_ENABLE_LTO=OFF

cmake --build build_ninja -j 2
ctest --test-dir build_ninja --output-on-failure -j 2
```

### CI Testing

The changes are automatically tested in the CI pipeline:
- All matrix configurations use raw CMake
- Coverage build uses raw CMake + Python coverage script
- All other jobs already use raw CMake

---

## Backward Compatibility

⚠️ **Breaking Changes**:
- `setup.py` is no longer used in CI
- Local developers can still use `setup.py` for convenience
- CI now requires CMake to be installed

✅ **Preserved**:
- Same build outputs
- Same test execution
- Same coverage analysis
- Same build directory structure

---

## Migration Notes

### For CI Maintainers

1. **No more setup.py in CI**: All builds use raw CMake
2. **Compiler detection**: Handled in shell/PowerShell scripts
3. **Build directory naming**: Consistent across all jobs
4. **Coverage analysis**: Still uses Python scripts (run_coverage.py)

### For Local Development

1. **setup.py still works**: Use for local builds
2. **Raw CMake also works**: Use for direct control
3. **Both produce same results**: Interchangeable

---

## Troubleshooting

### CMake not found
```bash
# Install CMake
# Ubuntu: sudo apt-get install cmake
# macOS: brew install cmake
# Windows: choco install cmake
```

### Compiler not found
```bash
# Ensure compiler is in PATH
# Ubuntu: sudo apt-get install clang
# macOS: brew install llvm
# Windows: Already installed via install-deps-windows.ps1
```

### Build directory conflicts
```bash
# Clean old build directories
rm -rf build_ninja*
```

---

## Files Modified

- `.github/workflows/ci.yml` - Replaced setup.py calls with raw CMake commands

---

## Status

✅ **Migration Complete**

All `setup.py` calls in CI have been replaced with equivalent raw CMake commands. The CI pipeline now has direct control over the build process without Python wrapper overhead.

