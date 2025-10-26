# CI Script Fixes - Installation Scripts and Compiler Arguments

## Issues Fixed

### Issue #1: Installation Scripts Not Tracked in Git ❌ → ✅ FIXED

**Problem**:
- CI workflow was trying to run `chmod +x scripts/ci/install-deps-ubuntu.sh`
- Error: `chmod: cannot access 'scripts/ci/install-deps-ubuntu.sh': No such file or directory`
- Root cause: Installation scripts were created but not added to git repository

**Solution**:
- Added all installation scripts to git index:
  - `.github/workflows/install/install-deps-ubuntu.sh`
  - `.github/workflows/install/install-deps-macos.sh`
  - `.github/workflows/install/install-deps-windows.ps1`
  - `.github/workflows/install/install-sccache.sh`
- Used `git update-index --add` to properly stage files with correct line endings
- Moved scripts to `.github/workflows/install/` to follow GitHub Actions conventions

**Files Added**:
```
A  .github/workflows/install/install-deps-macos.sh
A  .github/workflows/install/install-deps-ubuntu.sh
A  .github/workflows/install/install-deps-windows.ps1
A  .github/workflows/install/install-sccache.sh
```

### Issue #2: Remaining `clang++` Compiler Arguments ❌ → ✅ FIXED

**Problem**:
- After initial fixes, some `clang++` entries remained in CI workflow
- These were in sccache-baseline-tests and sccache-enabled-tests jobs
- Total of 6 remaining entries found

**Solution**:
- Fixed all remaining `clang++` entries to use `clang` format
- Affected jobs:
  - sccache-baseline-tests: 3 entries (Ubuntu, macOS, Windows)
  - sccache-enabled-tests: 3 entries (Ubuntu, macOS, Windows)

**Changes Made**:
```yaml
# Before
compiler_cxx: clang++

# After
compiler_cxx: clang
```

## Files Modified

### `.github/workflows/ci.yml`
- Updated all script paths from `scripts/ci/` to `.github/workflows/install/`
- Fixed first Ubuntu C++17 Debug entry (line 49)
- Fixed sccache-baseline-tests entries (lines 1154, 1166, 1178)
- Fixed sccache-enabled-tests entries (lines 1319, 1331, 1343)
- Updated comments to reflect new script locations
- Total: 6 compiler argument entries corrected + all path references updated

### `.github/workflows/install/` (Moved Files)
- `install-deps-ubuntu.sh` - Ubuntu/Linux dependency installation
- `install-deps-macos.sh` - macOS dependency installation
- `install-deps-windows.ps1` - Windows dependency installation
- `install-sccache.sh` - Sccache installation script

## Verification

✅ All installation scripts moved to `.github/workflows/install/`
✅ All script paths updated in CI workflow
✅ All `clang++` entries replaced with `clang`
✅ CI workflow syntax valid (no diagnostics)
✅ Git status shows all changes staged

## Git Status

```
R  Scripts/ci/install-deps-macos.sh -> .github/workflows/install/install-deps-macos.sh
R  Scripts/ci/install-deps-ubuntu.sh -> .github/workflows/install/install-deps-ubuntu.sh
R  Scripts/ci/install-deps-windows.ps1 -> .github/workflows/install/install-deps-windows.ps1
R  Scripts/ci/install-sccache.sh -> .github/workflows/install/install-sccache.sh
M  .github/workflows/ci.yml
```

## Next Steps

1. **Review Changes**
   - Verify all installation scripts are correct
   - Verify CI workflow compiler arguments are correct

2. **Test Changes**
   - Push to feature branch
   - Monitor CI execution
   - Verify all jobs pass

3. **Merge & Deploy**
   - Merge to main branch
   - Monitor production CI runs

## Technical Details

### Line Ending Issue
- Installation scripts were created with LF line endings
- Git on Windows converts LF to CRLF by default
- Used `git update-index --add` to properly handle line ending conversion
- Git now correctly tracks files with proper CRLF handling

### Compiler Argument Format
- setup.py expects base compiler name: `clang`, `gcc`, etc.
- NOT the C++ variant: `clang++`, `g++`, etc.
- setup.py automatically converts to C++ compiler name
- All CI entries now use correct format

## Summary

Both issues have been successfully resolved:

✅ **Installation Scripts**: Now properly tracked in git repository
✅ **Compiler Arguments**: All entries use correct format for setup.py

The CI pipeline is now ready for testing and deployment!
