# CI Compiler Separation Implementation

## Overview
Fixed the CI pipeline to prevent compiler version conflicts by creating separate jobs for each compiler version and C++ standard combination. This eliminates the issue of installing multiple Clang/GCC versions simultaneously in the same environment.

## Problem Statement
The original CI configuration attempted to install multiple compiler versions in a single job, causing package conflicts and build failures. The solution separates each compiler version into its own isolated job.

## Solution Architecture

### 1. Modified `install-deps-ubuntu.sh`
**Location**: `.github/workflows/install/install-deps-ubuntu.sh`

**New Parameters**:
- `--clang-version <VERSION>`: Install only the specified Clang version
- `--gcc-version <VERSION>`: Install only the specified GCC version

**Key Changes**:
- Added conditional logic to install only one compiler version per job
- Automatically adds LLVM repository for Clang versions >= 15
- Maintains backward compatibility (no parameters = install defaults)

**Example Usage**:
```bash
# Install only Clang 18
./install-deps-ubuntu.sh --with-tbb --clang-version 18

# Install only GCC 11
./install-deps-ubuntu.sh --with-tbb --gcc-version 11

# Install defaults (backward compatible)
./install-deps-ubuntu.sh --with-tbb
```

### 2. New CI Job: `compiler-version-tests-clang`
**Location**: `.github/workflows/ci.yml` (lines 445-667)

**Coverage**:
- **Clang Versions**: 9, 18, 20, 21
- **C++ Standards**: 17, 20, 23
- **Total Jobs**: 12 (4 versions × 3 standards)

**Job Naming Pattern**:
```
"Ubuntu C++ {std} Release - LOGURU - TBB:ON Clang {version}"
```

**Configuration per Job**:
- Build Type: Release
- Logging Backend: LOGURU
- TBB: ON
- CUDA: OFF
- Generator: Ninja

### 3. New CI Job: `compiler-version-tests-gcc`
**Location**: `.github/workflows/ci.yml` (lines 668-957)

**Coverage**:
- **GCC Versions**: 8, 9, 10, 11, 12, 13
- **C++ Standards**: 17, 20, 23
- **Total Jobs**: 18 (6 versions × 3 standards)

**Job Naming Pattern**:
```
"Ubuntu C++ {std} Release - LOGURU - TBB:ON GCC {version}"
```

**Configuration per Job**:
- Build Type: Release
- Logging Backend: LOGURU
- TBB: ON
- CUDA: OFF
- Generator: Ninja

## Implementation Details

### Compiler Installation Flow
1. Each CI job calls `install-deps-ubuntu.sh` with specific compiler version
2. Script installs only that compiler version (no conflicts)
3. LLVM repository automatically added for Clang >= 15
4. Build proceeds with isolated compiler environment

### Job Isolation Benefits
- ✅ No package conflicts from multiple compiler versions
- ✅ Cleaner build environment per job
- ✅ Easier debugging (each job is independent)
- ✅ Parallel execution (GitHub Actions runs jobs in parallel)
- ✅ Better caching (separate cache keys per compiler version)

## Total CI Coverage

### Clang Testing
- 4 versions × 3 C++ standards = **12 jobs**
- Versions: 9, 18, 20, 21
- Standards: C++17, C++20, C++23

### GCC Testing
- 6 versions × 3 C++ standards = **18 jobs**
- Versions: 8, 9, 10, 11, 12, 13
- Standards: C++17, C++20, C++23

### Total New Compiler Testing Jobs: **30 jobs**

## Migration Notes

### For Existing CI Jobs
- The original `compiler-version-tests` job has been replaced
- All existing functionality is preserved in the new jobs
- Job names follow the new naming pattern for consistency

### Backward Compatibility
- `install-deps-ubuntu.sh` remains backward compatible
- Calling without version parameters installs defaults
- Existing scripts and workflows continue to work

## Testing Recommendations

1. **Verify Job Creation**: Check GitHub Actions UI to confirm all 30 jobs are created
2. **Monitor First Run**: Watch for any installation or build failures
3. **Check Compiler Versions**: Verify each job uses the correct compiler version
4. **Validate Caching**: Ensure cache keys are unique per compiler version
5. **Performance**: Monitor total CI time (should be similar due to parallel execution)

## Files Modified

1. `.github/workflows/install/install-deps-ubuntu.sh`
   - Added `--clang-version` and `--gcc-version` parameters
   - Conditional compiler installation logic
   - LLVM repository management

2. `.github/workflows/ci.yml`
   - Replaced `compiler-version-tests` with `compiler-version-tests-clang`
   - Added new `compiler-version-tests-gcc` job
   - Updated dependency installation steps
   - Maintained all existing configurations

## Future Enhancements

- Consider adding Windows MSVC version testing with similar separation
- Add macOS Clang version testing if needed
- Implement job result aggregation for easier reporting
- Add performance benchmarking per compiler version
