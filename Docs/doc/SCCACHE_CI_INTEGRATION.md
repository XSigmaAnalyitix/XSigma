# Sccache CI Integration Guide

## Overview

This document describes the sccache integration added to the XSigma CI/CD pipeline. Sccache is a distributed compiler cache that can significantly improve build times by caching compilation results.

## What is Sccache?

Sccache is a ccache-like tool for Rust and C/C++ that caches compilation results. It supports:
- Local caching (default)
- Distributed caching via S3, Azure Blob Storage, or other backends
- Multiple compiler toolchains (Clang, GCC, MSVC)
- Cross-platform support (Linux, macOS, Windows)

## CI Integration Structure

The XSigma CI pipeline now includes two separate test configurations for performance comparison:

### 1. Sccache Baseline Tests (`sccache-baseline-tests`)

**Purpose**: Establish baseline build performance without sccache

**Platforms**: Ubuntu, macOS, Windows

**Configuration**:
- C++17 Release builds
- TBB enabled (Ubuntu/macOS only)
- No sccache compiler launcher
- Identical build configuration to sccache-enabled tests

**Metrics Collected**:
- Build duration (in seconds)
- Test execution time
- Build configuration details

### 2. Sccache Enabled Tests (`sccache-enabled-tests`)

**Purpose**: Measure build performance with sccache enabled

**Platforms**: Ubuntu, macOS, Windows

**Configuration**:
- C++17 Release builds
- TBB enabled (Ubuntu/macOS only)
- Sccache enabled via `CMAKE_CXX_COMPILER_LAUNCHER=sccache`
- Identical build configuration to baseline tests

**Key Features**:
- Automatic sccache installation from official Mozilla releases
- Sccache cache directory caching between CI runs
- Build duration recording and reporting
- Sccache statistics display after build completion

## Implementation Details

### Environment Variables

```yaml
SCCACHE_VERSION: "0.7.7"  # Version of sccache to use
```

To update the sccache version, modify this variable in the CI workflow's `env` section.

### Sccache Installation

Sccache is installed from official Mozilla GitHub releases:

**Ubuntu**:
```bash
curl -L https://github.com/mozilla/sccache/releases/download/v0.7.7/sccache-v0.7.7-x86_64-unknown-linux-musl.tar.gz \
  | tar xz -C ~/.local/bin --strip-components=1
```

**macOS**:
```bash
curl -L https://github.com/mozilla/sccache/releases/download/v0.7.7/sccache-v0.7.7-x86_64-apple-darwin.tar.gz \
  | tar xz -C ~/.local/bin --strip-components=1
```

**Windows**:
```powershell
Invoke-WebRequest -Uri "https://github.com/mozilla/sccache/releases/download/v0.7.7/sccache-v0.7.7-x86_64-pc-windows-msvc.zip" `
  -OutFile "$env:TEMP\sccache.zip"
Expand-Archive -Path "$env:TEMP\sccache.zip" -DestinationPath "$env:USERPROFILE\.local\bin"
```

### CMake Integration

Sccache is enabled via the CMake compiler launcher flag:

```cmake
-DCMAKE_CXX_COMPILER_LAUNCHER=sccache
```

This tells CMake to use sccache as a wrapper around the C++ compiler.

### Cache Management

**Dependency Cache**:
- Separate cache namespaces for baseline and enabled tests
- Prevents cache conflicts between job types
- Uses `CACHE_VERSION` environment variable for invalidation

**Sccache Cache**:
- Cached between CI runs using GitHub Actions cache
- Platform-specific cache directories:
  - Linux: `~/.cache/sccache`
  - macOS: `~/Library/Caches/sccache`
  - Windows: `~/AppData/Local/sccache`
- Cache key includes commit SHA for per-commit isolation

## Performance Metrics

Build duration is recorded and reported for each job:

```
=== Sccache Baseline Build Metrics ===
Platform: Linux
Build Type: Release
C++ Standard: 17
Sccache Enabled: NO (Baseline)
Build Duration: 120 seconds

=== Sccache Enabled Build Metrics ===
Platform: Linux
Build Type: Release
C++ Standard: 17
Sccache Enabled: YES
Build Duration: 85 seconds
```

## Expected Performance Improvements

Typical improvements with sccache:

- **First build**: Similar or slightly slower (cache population overhead)
- **Subsequent builds**: 30-50% faster (cache hits)
- **Incremental builds**: 50-70% faster (most files cached)

Actual improvements depend on:
- Number of compilation units
- Code change patterns
- Cache hit rate
- System I/O performance

## Troubleshooting

### Sccache Installation Failures

If sccache installation fails:
1. Check the release URL is correct for your platform
2. Verify the version exists on GitHub releases
3. Check network connectivity in CI environment

### Cache Misses

If sccache shows low hit rates:
1. Verify cache is being persisted between runs
2. Check that compiler flags are consistent
3. Ensure source code changes are minimal

### Build Failures with Sccache

If builds fail only with sccache enabled:
1. Check sccache compatibility with your compiler version
2. Try disabling sccache for specific files if needed
3. Report issues to Mozilla sccache project

## Maintenance

### Updating Sccache Version

1. Update `SCCACHE_VERSION` in `.github/workflows/ci.yml`:
   ```yaml
   env:
     SCCACHE_VERSION: "0.8.0"  # New version
   ```

2. Verify new version is available on GitHub releases
3. Test in a PR to ensure compatibility

### Adding New Platforms

To add sccache testing for a new platform:

1. Add matrix entry to `sccache-baseline-tests`
2. Add matrix entry to `sccache-enabled-tests`
3. Add platform-specific sccache installation step if needed
4. Update cache paths for platform-specific directories

### Disabling Sccache Tests

To temporarily disable sccache tests:

1. Comment out or remove matrix entries from both jobs
2. Remove job names from `ci-success.needs` list
3. Remove result checking from `ci-success` job

## Future Enhancements

Potential improvements:

1. **Distributed Caching**: Configure S3 backend for shared cache
2. **Performance Dashboard**: Track build time trends over time
3. **Selective Caching**: Disable sccache for specific targets if needed
4. **Statistics Collection**: Aggregate cache hit rates across runs
5. **Cost Analysis**: Track storage and bandwidth costs

## References

- [Sccache GitHub Repository](https://github.com/mozilla/sccache)
- [Sccache Documentation](https://github.com/mozilla/sccache/blob/main/README.md)
- [CMake Compiler Launcher](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_LAUNCHER.html)
