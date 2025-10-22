# CI Compiler Installation - Quick Reference

## What Changed?

The XSigma CI pipeline now uses **on-demand compiler installation** to avoid package conflicts when testing multiple compiler versions.

## Files Modified

| File | Change | Reason |
|------|--------|--------|
| `.github/workflows/ci.yml` | Added per-matrix compiler installation steps | Install only required compiler per test |
| `.github/workflows/install/install-deps-ubuntu.sh` | Removed version-specific Clang packages | Avoid conflicts in shared dependency script |
| `.github/workflows/install/install-clang-version.sh` | **NEW** - On-demand Clang installer | Install specific Clang versions without conflicts |

## How It Works

### Before (Broken)
```
install-deps-ubuntu.sh
  ├─ Install clang (default)
  ├─ Install clang-16 ❌ CONFLICT
  ├─ Install clang-17 ❌ CONFLICT
  └─ Install clang-18 ❌ CONFLICT
```

### After (Fixed)
```
install-deps-ubuntu.sh
  └─ Install clang (default only)

Per-Matrix Installation (runs separately for each test)
  ├─ Clang 15 test → install-clang-version.sh 15
  ├─ Clang 16 test → install-clang-version.sh 16
  ├─ Clang 17 test → install-clang-version.sh 17
  └─ GCC 13 test → apt-get install gcc-13
```

## Key Features

✅ **No Conflicts** - Each test installs only its compiler  
✅ **Parallel Testing** - Multiple compiler versions tested simultaneously  
✅ **Automatic** - No manual intervention needed  
✅ **Extensible** - Easy to add new compiler versions  
✅ **Robust** - Graceful fallback on failures  

## Using the New Script

### Manual Installation (for local testing)

```bash
# Install Clang 16 with LLVM tools
./.github/workflows/install/install-clang-version.sh 16 --with-llvm-tools

# Install Clang 17 (without LLVM tools)
./.github/workflows/install/install-clang-version.sh 17

# Verify installation
clang-16 --version
clang-17 --version
```

### Script Features

- ✅ Adds LLVM repository automatically (for Clang ≥ 15)
- ✅ Installs compiler and development tools
- ✅ Creates symbolic links for easy access
- ✅ Validates installation
- ✅ Provides clear error messages

## Adding New Compiler Versions

### To Test a New Clang Version

1. Edit `.github/workflows/ci.yml`
2. Add matrix entry to `compiler-version-tests`:

```yaml
- name: "Ubuntu Clang 18 - C++23"
  os: ubuntu-latest
  compiler_name: "Clang"
  compiler_version: "18"
  compiler_c: "clang-18"
  compiler_cxx: "clang-18"
  cxx_std: 23
  build_type: Release
  generator: Ninja
  cache_path: ~/.cache
```

3. **That's it!** The CI automatically handles installation.

### To Test a New GCC Version

1. Edit `.github/workflows/ci.yml`
2. Add matrix entry to `compiler-version-tests`:

```yaml
- name: "Ubuntu GCC 14 - C++23"
  os: ubuntu-latest
  compiler_name: "GCC"
  compiler_version: "14"
  compiler_c: "gcc-14"
  compiler_cxx: "g++-14"
  cxx_std: 23
  build_type: Release
  generator: Ninja
  cache_path: ~/.cache
```

3. **Done!** GCC installation is handled automatically.

## Troubleshooting

### Problem: CI Still Failing

**Check:**
1. Verify matrix entry has correct `compiler_name` (must be "Clang" or "GCC")
2. Verify `compiler_cxx` format: "clang-16" or "g++-12"
3. Check CI logs for specific error messages

### Problem: Compiler Not Found After Installation

**Check:**
1. Verify installation script ran successfully
2. Check for network issues (LLVM repository access)
3. Verify Ubuntu version compatibility

### Problem: Multiple Clang Versions Still Conflicting

**Check:**
1. Verify `install-deps-ubuntu.sh` doesn't install version-specific packages
2. Verify per-matrix installation steps are running
3. Check CI logs for installation order

## Supported Versions

### Clang
- 15, 16, 17, 18, 19+ (via LLVM repository)

### GCC
- 11, 12, 13, 14+ (via Ubuntu repositories)

### macOS
- Xcode Clang (system compiler)

## CI Job Structure

```
compiler-version-tests
├─ Checkout
├─ Cache dependencies
├─ Install shared dependencies (install-deps-ubuntu.sh)
├─ Install specific compiler (per-matrix)
│  ├─ For Clang: install-clang-version.sh
│  └─ For GCC: apt-get install
├─ Setup Python
├─ Configure and Build
├─ Verify compiler version
└─ Upload results (on failure)
```

## Performance Impact

- ✅ **No negative impact** - Compiler installation is fast
- ✅ **Parallel execution** - Multiple matrix entries run simultaneously
- ✅ **Caching** - Dependencies cached between runs
- ✅ **Reduced conflicts** - Faster overall CI execution

## Related Documentation

- [Full Implementation Details](CI_COMPILER_INSTALLATION_FIX.md)
- [CI Pipeline Overview](.github/workflows/ci.yml)
- [LLVM Repository](https://apt.llvm.org/)

## Summary

The on-demand compiler installation system:
- ✅ Eliminates package conflicts
- ✅ Enables parallel compiler testing
- ✅ Simplifies adding new compiler versions
- ✅ Maintains clean, maintainable CI configuration
- ✅ Provides robust error handling

**Result:** XSigma CI pipeline now successfully tests multiple compiler versions without conflicts! 🎉

