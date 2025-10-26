# XSigma

A high-performance C++ library with a modern CMake build system providing cross-platform compatibility, advanced optimization, and flexible dependency management.
# XSigma

[![CI Status](https://github.com/XSigmaAnalyitix/XSigma/actions/workflows/ci.yml/badge.svg)](https://github.com/XSigmaAnalyitix/XSigma/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/XSigmaAnalyitix/XSigma/branch/main/graph/badge.svg)](https://codecov.io/gh/XSigmaAnalyitix/XSigma)
[![License: GPL-3.0 or Commercial](https://img.shields.io/badge/License-GPL--3.0%20or%20Commercial-blue.svg)](LICENSE)
## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Features](#features)
- [Documentation](#documentation)
- [Best Practices](#best-practices)
- [Build Speed Optimization](#build-speed-optimization)
- [Contributing](#contributing)
- [License](#license)

## Overview

XSigma is a production-ready C++ library featuring:

- **Cross-platform compatibility** - Windows, Linux, and macOS support
- **Modern CMake build system** - CMake 3.16+ with best practices
- **Advanced optimization** - LTO, vectorization, and compiler-specific flags
- **Flexible logging** - Three backend options (LOGURU, GLOG, NATIVE)
- **Comprehensive testing** - Sanitizers, coverage analysis, and static analysis tools

## Quick Start

### Basic Build

```bash
# Clone and build with default settings
git clone https://github.com/XSigmaAnalyitix/XSigma.git
cd XSigma

# Configure and build
cd Scripts
python setup.py ninja.clang.config.build
```

### Optimized Release Build

```bash
# High-performance build with AVX2 and LTO
cd Scripts
python setup.py ninja.clang.release.lto.avx2.config.build
```

### Using setup.py (Recommended)

```bash
cd Scripts
python setup.py ninja.clang.python.build.test
```

## Features

### Build Configuration

Configure your build with multiple options including build types (Debug, Release, RelWithDebInfo, MinSizeRel), C++ standard selection (C++17/20/23), and optimization settings.

**Key capabilities:**
- Multiple build types for different use cases
- Link-Time Optimization (LTO) for maximum performance
- Custom compiler flags and optimization levels

📖 **[Read more: Build Configuration Guide](docs/build-configuration.md)**

---

### Logging System

Flexible logging with three mutually-exclusive backends that can be selected at compile time, each offering different trade-offs between features, performance, and dependencies.

**Available backends:**
- **LOGURU** (default) - Full-featured logging with scopes and callbacks
- **GLOG** - Google's production-grade logging library
- **NATIVE** - Minimal native implementation with no dependencies

📖 **[Read more: Logging System Guide](docs/logging-system.md)**

---

### Third-Party Dependencies

Conditional compilation pattern where each library is controlled by `XSIGMA_ENABLE_XXX` options, allowing you to customize your build by including only the dependencies you need.

**Key features:**
- Git submodules or system-installed libraries
- Mandatory core libraries (fmt, cpuinfo)
- Optional libraries (magic_enum, loguru, mimalloc, Google Test, Benchmark)

📖 **[Read more: Third-Party Dependencies Guide](docs/third-party-dependencies.md)**

---

### Vectorization Support

CPU SIMD (Single Instruction, Multiple Data) instruction sets for high-performance computing, allowing the CPU to process multiple data elements simultaneously.

**Supported instruction sets:**
- SSE/SSE2 - Legacy compatibility
- AVX/AVX2 - Modern CPUs (default: AVX2)
- AVX-512 - Latest high-end systems

📖 **[Read more: Vectorization Guide](docs/vectorization.md)**

---

### Sanitizers

Comprehensive sanitizer support for memory debugging and analysis with all modern sanitizers across multiple compilers.

**Available sanitizers:**
- **AddressSanitizer** - Memory errors, buffer overflows
- **UndefinedBehaviorSanitizer** - Undefined behavior detection
- **ThreadSanitizer** - Data race detection
- **MemorySanitizer** - Uninitialized memory reads
- **LeakSanitizer** - Memory leak detection

📖 **[Read more: Sanitizers Guide](docs/sanitizer.md)**

---

### Code Coverage

Generate code coverage reports to measure test effectiveness and identify untested code paths using multiple compilers (Clang, GCC, MSVC).

**Key features:**
- Multi-compiler support (Clang, GCC, MSVC)
- HTML reports with line-by-line coverage highlighting
- JSON summaries for programmatic access
- Unified coverage runner for all compilers
- Automatic exclusion of third-party code
- Integrated with setup.py for easy use
- Cross-platform support (Windows, Linux, macOS)

#### Quick Start

**Generate coverage with Clang (using setup.py):**
```bash
cd Scripts
python setup.py ninja.clang.debug.coverage
# View HTML report: ../build_ninja_coverage/coverage_report/html/index.html
```

**Generate coverage with MSVC (using setup.py):**
```bash
cd Scripts
python setup.py vs22.debug.coverage
# View HTML report: ../build_vs22_coverage/coverage_report/html/index.html
```

**Generate coverage with GCC (using setup.py):**
```bash
cd Scripts
python setup.py ninja.gcc.debug.coverage
# View HTML report: ../build_ninja_coverage/coverage_report/html/index.html
```

#### Running Coverage Directly

For more control over coverage generation, use the `run_coverage.py` script directly:

**Basic usage:**
```bash
cd Tools/coverage
python run_coverage.py --build=../../build_ninja
```

**With custom source folder:**
```bash
cd Tools/coverage
python run_coverage.py --build=../../build_ninja --filter=Library
```

**With verbose output:**
```bash
cd Tools/coverage
python run_coverage.py --build=../../build_ninja --verbose
```

#### Report Formats

The coverage tools generate multiple output formats:

- **HTML** - Interactive visual reports with source code highlighting
  - Location: `<build_dir>/coverage_report/html/index.html`
  - Includes line-by-line coverage visualization
  - Supports navigation between files and coverage statistics

- **JSON** - Machine-readable format for CI/CD integration
  - Location: `<build_dir>/coverage_report/coverage_summary.json`
  - Contains overall coverage metrics and per-file statistics
  - Compatible with Codecov and other coverage services

- **LCOV** - Standard coverage format
  - Location: `<build_dir>/coverage_report/coverage.lcov`
  - Used by Codecov and other coverage tracking services
  - Can be processed by lcov tools for additional analysis

#### Expected Output

When coverage generation completes successfully, you should see:

```
=== Coverage Metrics ===
Line Coverage: 1234/5678 (21.73%)
Function Coverage: 89/120 (74.17%)
Files Analyzed: 45
```

The coverage report directory structure:
```
build_ninja_coverage/coverage_report/
├── html/                          # HTML reports
│   ├── index.html                # Main coverage report
│   └── [source files].html       # Per-file coverage
├── coverage_summary.json         # JSON summary
├── coverage.lcov                 # LCOV format
└── raw/                          # Raw coverage data (MSVC only)
```

#### CI/CD Integration

Coverage reports are automatically generated in CI/CD pipelines and can be:
- Uploaded to coverage tracking services (Codecov, Coveralls)
- Used to enforce minimum coverage thresholds
- Tracked over time to monitor code quality trends

**Codecov Integration:**
The CI/CD pipeline automatically uploads coverage reports to Codecov. To enable this:
1. Set up a Codecov account at https://codecov.io
2. Add your repository to Codecov
3. Set the `CODECOV_TOKEN` secret in your GitHub repository settings (optional for public repos)

📖 **[Read more: Code Coverage Guide](COVERAGE.md)** - Comprehensive guide with multi-compiler support, unified runner, and detailed usage examples

---

### Static Analysis Tools

Integrated static analysis tools to improve code quality: Include-What-You-Use (IWYU) for header dependency management and Cppcheck for comprehensive static code analysis.

**Available tools:**
- **IWYU** - Reduces unnecessary includes and enforces clean header dependencies
- **Cppcheck** - Detects errors, undefined behavior, style issues, and performance problems

📖 **[Read more: Static Analysis Guide](docs/static-analysis.md)**

---

### Spell Checking

Automated spell checking using codespell to maintain code quality and documentation consistency across the codebase.

**Configuration:**
- **`.codespellrc`** - Configuration file defining spell check behavior
- **Excluded directories** - `.git`, `.augment`, `.github`, `.vscode`, `build`, `Build`, `Cmake`, `ThirdParty`
- **Custom dictionary** - Project-specific words like "ThirdParty" are ignored
- **Hidden files** - Excluded from spell checking

#### Manual Usage

**Running codespell locally:**
```bash
# Install codespell (if not already installed)
pip install codespell

# Run spell check on the entire codebase
codespell
# Alternative if codespell is not in PATH:
python -m codespell

# Run with verbose output
codespell -v

# Check specific files or directories
codespell README.md docs/

# Show configuration being used
codespell --help
```

#### Adding Exceptions

To add words that should be ignored by codespell, edit the `.codespellrc` file:
```ini
ignore-words-list = ThirdParty,yourword,anotherword
```

#### Integration Options

- **setup.py integration** - Now available in the Scripts/setup.py workflow alongside other code quality tools (clangtidy, iwyu, cppcheck)
- **Pre-commit hooks** - Can be integrated for automatic checking before commits
- **CI/CD pipeline** - Suitable for continuous integration to ensure documentation quality
- **IDE integration** - Many editors support codespell plugins for real-time checking

#### setup.py Integration

**Enable spell checking with automatic corrections:**
```bash
# Enable spell checking during build (WARNING: modifies source files)
python setup.py ninja.clang.config.build.test.spell
python setup.py vs22.config.build.spell

# Combine with other tools
python setup.py ninja.clang.config.build.test.spell.cppcheck
```

**Important:** When `spell` is enabled, codespell will automatically apply spelling corrections to source files during the build process. Ensure you have committed your changes before building with this option.

**Benefits:**
- Maintains professional documentation standards
- Catches common spelling errors in code comments and documentation
- Configurable to ignore technical terms and project-specific vocabulary

---

### Cross-Platform Building

Full cross-platform compatibility across Windows, Linux, and macOS with platform-specific optimizations and build instructions.

**Supported platforms:**
- Windows (MSVC 2019+, Clang)
- Linux (GCC 9+, Clang 10+)
- macOS (Apple Clang, including Apple Silicon)

📖 **[Read more: Cross-Platform Building Guide](docs/cross-platform-building.md)**

---

## Documentation

### Core Documentation

- **[Build Configuration](docs/build-configuration.md)** - Build types, C++ standards, optimization options
- **[Logging System](docs/logging-system.md)** - Flexible logging with three backend options
- **[Third-Party Dependencies](docs/third-party-dependencies.md)** - Dependency management and integration
- **[Vectorization](docs/vectorization.md)** - CPU SIMD optimization (SSE, AVX, AVX2, AVX-512)
- **[Sanitizers](docs/sanitizers.md)** - Memory debugging and analysis tools
- **[Code Coverage](COVERAGE.md)** - Multi-compiler coverage analysis and reporting
- **[Static Analysis](docs/static-analysis.md)** - IWYU and Cppcheck tools
- **[Compiler Caching](docs/cache.md)** - Compiler cache types, installation, and configuration
- **[Spell Checking](#spell-checking)** - Codespell configuration and usage
- **[Cross-Platform Building](docs/cross-platform-building.md)** - Platform-specific build instructions
- **[Usage Examples](docs/usage-examples.md)** - Practical build configuration examples

### Additional Documentation

- **[CI/CD Pipeline](docs/CI_CD_PIPELINE.md)** - Continuous integration setup
- **[CI Quick Start](docs/CI_QUICK_START.md)** - Getting started with CI
- **[Valgrind Setup](docs/VALGRIND_SETUP.md)** - Memory debugging with Valgrind

### Legacy Documentation

The following documentation files are referenced in the codebase but may be located elsewhere:
- **LOGGING_BACKEND_USAGE_GUIDE.md** - Detailed logging usage guide
- **LOGGING_BACKEND_IMPLEMENTATION.md** - Logging implementation details
- **LOGGING_BACKEND_TEST_RESULTS.md** - Logging test results

## Getting Started

### Prerequisites

- CMake 3.16 or later
- C++17 compatible compiler:
  - Windows: MSVC 2019+ or Clang
  - Linux: GCC 9+ or Clang 10+
  - macOS: Apple Clang or Clang

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd XSigma

# Initialize submodules (if using Git submodules)
git submodule update --init --recursive

# Configure and build
cd Scripts
python setup.py ninja.clang.release.config.build
```

### Running Tests

```bash
# Enable testing during configuration
cd Scripts
python setup.py ninja.clang.debug.test.gtest.config.build.test
```

## Troubleshooting

### Common Issues

**Missing Third-Party Libraries**

If you see warnings about missing third-party libraries:

1. Add Git submodules:
   ```bash
   git submodule add https://github.com/fmtlib/fmt.git ThirdParty/fmt
   git submodule update --init --recursive
   ```

2. Use external libraries:
   ```bash
   cd Scripts
   python setup.py ninja.clang.external.config.build
   ```

3. Disable unused features:
   ```bash
   cd Scripts
   python setup.py ninja.clang.magic_enum.config.build
   ```

**Vectorization Issues**

If vectorization fails to compile:

1. Check compiler support
2. Use lower vectorization: add `avx` flag to setup.py command
3. Disable vectorization: omit vectorization flags from setup.py command

**LTO Issues**

If Link-Time Optimization fails:

1. Disable LTO: add `lto` flag to setup.py command (inverts default ON to OFF)
2. Check compiler version
3. Increase available memory

**Build Performance**

For faster builds:

1. Use external libraries: add `external` flag to setup.py command
2. Disable unused features
3. Use parallel compilation: (automatically handled by setup.py)
4. Enable build speed optimizations (see [Build Speed Optimization](#build-speed-optimization) section)

For more detailed troubleshooting, see the specific feature documentation.

## Migration Guide

### From Previous Build System

The new CMake system maintains **complete backward compatibility**:

- ✅ Same CMake options and values
- ✅ Same function names and signatures
- ✅ Same compiler flags and definitions
- ✅ Same target configuration behavior

### Vectorization Configuration

The vectorization system supports CPU SIMD instruction sets:

**CPU Vectorization:**
- SSE, AVX, AVX2, AVX512 instruction sets
- Cross-platform compiler support (MSVC, GCC, Clang)
- Automatic detection and configuration

**Configuration:**
- Set via `XSIGMA_VECTORIZATION_TYPE` option
- Options: `no`, `sse`, `avx`, `avx2`, `avx512`

**No migration required** - all existing configurations continue to work.

## Best Practices

### Development Workflow

**For daily development:**
```bash
cd Scripts
python setup.py ninja.clang.debug.test.config.build
```

**For production releases:**
```bash
cd Scripts
python setup.py ninja.clang.release.lto.config.build
```

**For CI/CD pipelines:**
```bash
cd Scripts
python setup.py ninja.clang.external.test.config.build
```

### Performance Optimization

1. **Enable high-performance libraries**: TBB, mimalloc
2. **Use appropriate vectorization**: AVX2 for modern CPUs
3. **Enable LTO**: For maximum optimization in release builds

### Resource-Constrained Environments

1. **Use minimal builds**: Disable optional features
2. **Disable vectorization**: For maximum compatibility
3. **Use external libraries**: Reduce build time and repository size

---

## Build Speed Optimization

XSigma includes configurable compiler caching and **faster linkers** to significantly reduce build times, especially for incremental builds and large projects.

### Overview

The build system supports multiple compiler cache types:
- **none** - No compiler caching (default)
- **ccache** - Caches compilation results to avoid recompiling unchanged code
- **sccache** - Distributed compiler cache with cloud storage support
- **buildcache** - Incremental build cache for faster rebuilds
- **Faster linkers** - Uses platform and compiler-specific optimized linkers

**Expected improvements:**
- **First build**: 5-15% faster with optimized linker
- **Incremental builds**: 50-80% faster with compiler cache (for unchanged files)
- **Clean rebuilds**: 10-30% faster with optimized linker

### Installation

#### Linux

**Install ccache:**
```bash
# Ubuntu/Debian
sudo apt-get install ccache

# Fedora/RHEL
sudo dnf install ccache

# Arch
sudo pacman -S ccache
```

**Install faster linkers:**
```bash
# For Clang: install lld and/or mold
# Ubuntu/Debian
sudo apt-get install lld mold

# Fedora/RHEL
sudo dnf install lld mold

# Arch
sudo pacman -S lld mold

# For GCC: install gold and/or mold
# Ubuntu/Debian (gold is usually included with binutils)
sudo apt-get install binutils mold

# Fedora/RHEL
sudo dnf install binutils mold

# Arch
sudo pacman -S binutils mold
```

#### macOS

**Install ccache:**
```bash
# Using Homebrew
brew install ccache

# Using MacPorts
sudo port install ccache
```

**Install faster linker (optional):**
```bash
# Install lld (LLVM linker)
brew install llvm
```

#### Windows

**Install ccache:**
```bash
# Using Chocolatey
choco install ccache

# Using vcpkg
vcpkg install ccache:x64-windows

# Or download from: https://github.com/ccache/ccache/releases
```

**Install faster linker (optional):**
```bash
# For Clang with lld-link (usually included with LLVM)
# Download LLVM from: https://github.com/llvm/llvm-project/releases
# Or use Chocolatey:
choco install llvm
```

### Configuration

Build speed optimization is **enabled by default**. You can select which cache type to use:

**Using setup.py (recommended):**
```bash
cd Scripts

# Use ccache (if available)
python setup.py ninja.clang.ccache.config.build

# Use sccache (if available)
python setup.py ninja.clang.sccache.config.build

# Use buildcache (if available)
python setup.py ninja.clang.buildcache.config.build

# Disable caching (use 'none')
python setup.py ninja.clang.none.config.build
```

**Using CMake directly:**
```bash
# Select cache type
cmake -DXSIGMA_CACHE_TYPE=ccache ..
cmake -DXSIGMA_CACHE_TYPE=sccache ..
cmake -DXSIGMA_CACHE_TYPE=buildcache ..
cmake -DXSIGMA_CACHE_TYPE=none ..

# Disable caching entirely
cmake -DXSIGMA_ENABLE_CACHE=OFF ..
```

### How It Works

#### Compiler Caching

The build system automatically uses the selected compiler cache as a compiler launcher if available:
- Caches compilation results based on source code and compiler flags
- Subsequent compilations of the same code are retrieved from cache
- Transparent to the build process - no configuration needed

**Supported cache types:**
- **ccache** - Local compilation cache, best for single-machine builds
- **sccache** - Distributed cache with cloud storage support (S3, Azure, GCS)
- **buildcache** - Incremental build cache optimized for CI/CD pipelines
- **none** - No caching

**Supported compilers (all cache types):**
- GCC
- Clang
- MSVC
- CUDA (if enabled)

#### Faster Linkers

The build system automatically detects and uses the fastest available linker:

| Platform | Compiler | Preferred Linker | Fallback |
|----------|----------|------------------|----------|
| Linux | Clang | mold | lld |
| Linux | GCC | mold | gold |
| macOS | Clang/Apple Clang | lld | system linker |
| Windows | Clang | lld-link | default |
| Windows | MSVC | default | - |

### Verification

**Check if ccache is being used:**
```bash
# View ccache statistics
ccache -s

# Clear ccache (if needed)
ccache -C

# Set ccache size limit (default is 5GB)
ccache -M 10G
```

**Check which linker is being used:**
```bash
# During build, look for messages like:
# "Linux/Clang: Using mold linker for faster linking"
# "Linux/GCC: Using gold linker for faster linking"

# Or check the build output for linker invocations
```

### Performance Tips

1. **Choose the right cache type for your use case:**
   - **ccache** - Best for local development and single-machine builds
   - **sccache** - Best for distributed CI/CD with cloud storage
   - **buildcache** - Best for incremental CI/CD pipelines

2. **Increase cache size** for large projects:
   ```bash
   ccache -M 20G  # Set ccache to 20GB
   ```

3. **Use consistent compiler flags** to maximize cache hits

4. **Enable on CI/CD** for faster pipelines:
   ```bash
   # In CI configuration, install your chosen cache tool
   # The build system will automatically detect and use it
   ```

5. **Monitor cache effectiveness**:
   ```bash
   ccache -s      # Shows ccache hit/miss statistics
   sccache --show-stats  # Shows sccache statistics
   ```

📖 **[Read more: Compiler Caching Guide](docs/cache.md)** - Comprehensive guide with installation instructions, performance comparisons, and advanced configuration

### Troubleshooting

**Compiler cache not being used:**
- Verify installation: `which ccache` / `which sccache` / `which buildcache` (Linux/macOS) or `where` (Windows)
- Check CMake output for "Found [cache_type]" message
- Ensure `XSIGMA_ENABLE_CACHE` is ON
- Verify `XSIGMA_CACHE_TYPE` is set to the correct cache type

**Wrong cache type selected:**
- Check current setting: `cmake -L | grep XSIGMA_CACHE_TYPE`
- Reconfigure with correct type: `python setup.py ninja.clang.[cache_type].config`
- Or use CMake directly: `cmake -DXSIGMA_CACHE_TYPE=[cache_type] ..`

**Linker not being used:**
- Verify installation: `which mold` / `which lld` / `which ld.gold`
- Check CMake output for linker detection messages
- Some systems may not have faster linkers available - fallback to default is automatic

**Cache misses or stale cache:**
- For ccache: `ccache -C` (clear) or `ccache -s` (stats)
- For sccache: `sccache --stop-server` then `sccache --start-server`
- For buildcache: Check `~/.buildcache` directory
- Verify compiler path consistency

---

## Contributing

When contributing to XSigma:

1. **Follow CMake best practices** - use modern target-based configuration
2. **Maintain backward compatibility** - existing configurations should continue working
3. **Test cross-platform** - verify Windows, Linux, and macOS compatibility
4. **Document changes** - update this README for new features
5. **Use conditional compilation** - follow the `XSIGMA_ENABLE_XXX` pattern

## License

See [LICENSE](LICENSE) file for details.

---

**XSigma** - High-performance C++ library with modern CMake build system
