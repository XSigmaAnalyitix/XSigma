# CI Compiler Separation - Quick Reference

## What Changed?

### Problem
CI pipeline was installing multiple compiler versions in a single job, causing package conflicts and build failures.

### Solution
Created separate CI jobs for each compiler version and C++ standard combination.

## New CI Jobs

### Clang Testing (12 jobs)
```
compiler-version-tests-clang:
  - Clang 9 with C++17, C++20, C++23
  - Clang 18 with C++17, C++20, C++23
  - Clang 20 with C++17, C++20, C++23
  - Clang 21 with C++17, C++20, C++23
```

### GCC Testing (18 jobs)
```
compiler-version-tests-gcc:
  - GCC 8 with C++17, C++20, C++23
  - GCC 9 with C++17, C++20, C++23
  - GCC 10 with C++17, C++20, C++23
  - GCC 11 with C++17, C++20, C++23
  - GCC 12 with C++17, C++20, C++23
  - GCC 13 with C++17, C++20, C++23
```

## Job Naming Pattern

```
"Ubuntu C++ {std} Release - LOGURU - TBB:ON {Compiler} {version}"
```

**Examples**:
- `Ubuntu C++ 17 Release - LOGURU - TBB:ON Clang 18`
- `Ubuntu C++ 20 Release - LOGURU - TBB:ON GCC 11`
- `Ubuntu C++ 23 Release - LOGURU - TBB:ON Clang 21`

## Modified Files

### 1. `.github/workflows/install/install-deps-ubuntu.sh`

**New Parameters**:
```bash
--clang-version <VERSION>    # Install specific Clang version
--gcc-version <VERSION>      # Install specific GCC version
```

**Usage Examples**:
```bash
# Install Clang 18 with TBB
./install-deps-ubuntu.sh --with-tbb --clang-version 18

# Install GCC 11 with TBB
./install-deps-ubuntu.sh --with-tbb --gcc-version 11

# Install defaults (backward compatible)
./install-deps-ubuntu.sh --with-tbb
```

### 2. `.github/workflows/ci.yml`

**Changes**:
- Replaced `compiler-version-tests` job with two new jobs:
  - `compiler-version-tests-clang` (lines 445-667)
  - `compiler-version-tests-gcc` (lines 668-957)
- Updated dependency installation to use version parameters
- Maintained all existing configurations

## Key Features

✅ **No Package Conflicts**: Each job installs only one compiler version
✅ **Parallel Execution**: GitHub Actions runs jobs in parallel
✅ **Isolated Environments**: Clean build environment per job
✅ **Better Caching**: Separate cache keys per compiler version
✅ **Backward Compatible**: Old scripts still work
✅ **Automatic LLVM Repo**: Clang >= 15 automatically adds LLVM repository

## Configuration per Job

All new compiler testing jobs use:
- **Build Type**: Release
- **Logging Backend**: LOGURU
- **TBB**: ON
- **CUDA**: OFF
- **Generator**: Ninja

## Total CI Impact

| Category | Count |
|----------|-------|
| Clang Jobs | 12 |
| GCC Jobs | 18 |
| **Total New Jobs** | **30** |

## Verification Checklist

- [ ] All 30 new jobs appear in GitHub Actions
- [ ] Each job installs only one compiler version
- [ ] No package conflicts during installation
- [ ] All jobs complete successfully
- [ ] Compiler versions match job names
- [ ] Build artifacts are generated correctly
- [ ] Test results are uploaded on failure

## Troubleshooting

### Job Not Running
- Check GitHub Actions UI for job status
- Verify job name matches the pattern
- Check if job is filtered by branch/tag

### Compiler Installation Fails
- Check Ubuntu package availability
- Verify LLVM repository is added for Clang >= 15
- Check network connectivity in CI environment

### Build Fails
- Check compiler version in job output
- Verify C++ standard is correctly set
- Check build logs for specific errors

## Related Documentation

- Full Implementation: `docs/CI_COMPILER_SEPARATION_IMPLEMENTATION.md`
- Build Process: `docs/BUILD_AND_TEST_SUMMARY.md`
- CI Pipeline: `docs/CI_CD_PIPELINE.md`

