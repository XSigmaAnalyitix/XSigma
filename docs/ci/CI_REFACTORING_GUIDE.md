# XSigma CI Pipeline Refactoring Guide

## Overview

The XSigma CI pipeline has been refactored to improve maintainability, reduce code duplication, and add comprehensive compiler version testing. This guide documents the changes and how to work with the new structure.

## Key Improvements

### 1. Script-Based Dependency Installation

**Before**: Inline installation commands in CI YAML (hundreds of lines)
**After**: Separate, reusable installation scripts

#### Installation Scripts

Located in `.github/workflows/install/`:

- **`install-deps-ubuntu.sh`**: Ubuntu/Linux dependencies
  - Installs: CMake, Ninja, Clang, GCC, Python, TBB, development libraries
  - Flags: `--with-cuda`, `--with-tbb`
  - Usage: `./.github/workflows/install/install-deps-ubuntu.sh --with-tbb`

- **`install-deps-macos.sh`**: macOS dependencies
  - Installs: Homebrew packages, Clang, GCC, Python, TBB
  - Flags: `--with-cuda`, `--with-tbb`
  - Usage: `./.github/workflows/install/install-deps-macos.sh --with-tbb`

- **`install-deps-windows.ps1`**: Windows dependencies (PowerShell)
  - Installs: Chocolatey packages, LLVM, CMake, Python
  - Flags: `-WithCuda`, `-WithTbb`
  - Usage: `.\.github\workflows\install\install-deps-windows.ps1 -WithTbb`

- **`install-sccache.sh`**: Sccache installation with platform detection
  - Automatic platform detection (Linux, macOS)
  - Downloads from official Mozilla releases
  - Usage: `./.github/workflows/install/install-sccache.sh 0.7.7`

#### Benefits

- **Maintainability**: Update dependencies in one place
- **Reusability**: Scripts can be used locally and in CI
- **Idempotency**: Scripts are safe to run multiple times
- **Error Handling**: Comprehensive error checking and logging
- **Cross-Platform**: Platform-specific logic encapsulated

### 2. setup.py Integration

**Before**: Direct CMake commands in CI
**After**: All builds use `setup.py` from Scripts/ directory

#### How It Works

```bash
cd Scripts
python setup.py ninja clang release test cxx17 loguru tbb
```

#### Benefits

- **Consistency**: Same build process locally and in CI
- **Flexibility**: Supports multiple compilers and configurations
- **Maintainability**: Build logic centralized in setup.py
- **Build Rule Compliance**: Automatically handles build_ninja_python directory

#### Supported Arguments

- **Generator**: `ninja`, `cninja`, `eninja`, `xcode`
- **Compiler**: `clang`, `clang++`, `gcc`, `g++`, `clang-15`, etc.
- **Build Type**: `debug`, `release`, `relwithdebinfo`
- **C++ Standard**: `cxx17`, `cxx20`, `cxx23`
- **Logging Backend**: `native`, `loguru`, `glog`
- **Features**: `tbb`, `cuda`, `test`, `benchmark`, etc.

### 3. Compiler Version Matrix Testing

**New Job**: `compiler-version-tests`

Tests XSigma with multiple compiler versions to ensure compatibility.

#### Tested Compilers

**Ubuntu**:
- GCC 11, 12, 13 (C++17, C++20)
- Clang 15, 16, 17 (C++17, C++20, C++23)

**macOS**:
- Xcode Clang (C++17, C++20)

#### Adding New Compiler Versions

1. Edit `.github/workflows/ci.yml`
2. Add entry to `compiler-version-tests` matrix:

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

3. Ensure compiler is available in the target platform

## CI Pipeline Structure

### Jobs (in execution order)

1. **build-matrix**: Primary testing (C++17 baseline + selective C++20/C++23)
2. **compiler-version-tests**: Multi-compiler compatibility testing
3. **tbb-specific-tests**: TBB functionality (Unix/macOS only)
4. **sanitizer-tests**: Memory/thread safety (Unix/macOS only)
5. **optimization-flags-test**: Compiler optimization testing
6. **lto-tests**: Link-time optimization testing
7. **benchmark-tests**: Performance regression testing (non-blocking)
8. **sccache-baseline-tests**: Build performance baseline
9. **sccache-enabled-tests**: Build performance with sccache
10. **ci-success**: Aggregates all results

### Job Dependencies

- `ci-success` depends on all other jobs
- Most jobs are required (fail-fast)
- `benchmark-tests` is non-blocking (warnings only)
- `compiler-version-tests` is non-blocking (warnings only)

## Maintenance Tasks

### Adding a New Test Configuration

1. Add entry to appropriate job's matrix
2. Update `ci-success.needs` if creating new job
3. Add result checking in `ci-success` job
4. Update maintenance guide

### Updating Dependencies

1. Edit appropriate script in `scripts/ci/`
2. Test locally: `./scripts/ci/install-deps-ubuntu.sh`
3. Commit and push
4. CI will use updated script

### Updating Build Configuration

1. Modify `setup.py` arguments in CI workflow
2. Or update `setup.py` itself for new features
3. Test locally: `cd Scripts && python setup.py ...`

### Updating Sccache Version

1. Edit `.github/workflows/ci.yml`
2. Change `SCCACHE_VERSION: "0.7.7"` to desired version
3. Verify version exists on GitHub releases

## Troubleshooting

### Compiler Not Found

If a specific compiler version isn't available:
- Check if it's installed in the CI environment
- Update installation script to include it
- Or use a different version

### Build Fails with setup.py

1. Check if `build_ninja_python` directory exists
2. Verify all required arguments are provided
3. Check setup.py logs in `Scripts/logs/`

### Script Execution Fails

1. Ensure scripts are executable: `chmod +x scripts/ci/*.sh`
2. Check script output for specific errors
3. Verify platform-specific requirements

## Performance Considerations

- **Compiler version tests**: Non-blocking to prevent delays
- **Sccache tests**: Separate jobs for fair comparison
- **Matrix size**: Optimized to balance coverage vs. time
- **Caching**: Separate cache namespaces per job type

## Future Enhancements

- [ ] Add MSVC version testing on Windows
- [ ] Add distributed sccache caching (S3 backend)
- [ ] Add performance trend tracking
- [ ] Add code coverage reporting
- [ ] Add static analysis (clang-tidy, cppcheck)

## References

- [setup.py Documentation](../Scripts/setup.py)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Sccache Documentation](https://github.com/mozilla/sccache)
