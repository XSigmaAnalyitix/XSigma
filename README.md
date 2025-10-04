# XSigma

A high-performance C++ library with a modern CMake build system providing cross-platform compatibility, advanced optimization, and flexible dependency management.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Features](#features)
- [Documentation](#documentation)
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

📖 **[Read more: Sanitizers Guide](docs/sanitizers.md)**

---

### Code Coverage

Generate code coverage reports to measure test effectiveness and identify untested code paths using LLVM coverage tools (llvm-profdata and llvm-cov).

**Key features:**
- Line, function, and region coverage tracking
- HTML, text, and JSON report formats
- Automatic exclusion of third-party code
- Integrated with setup.py for easy use

📖 **[Read more: Code Coverage Guide](docs/code-coverage.md)**

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

- **Future setup.py integration** - Could be added to the Scripts/setup.py workflow alongside other code quality tools (clangtidy, iwyu, cppcheck)
- **Pre-commit hooks** - Can be integrated for automatic checking before commits
- **CI/CD pipeline** - Suitable for continuous integration to ensure documentation quality
- **IDE integration** - Many editors support codespell plugins for real-time checking

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
- **[Code Coverage](docs/code-coverage.md)** - Coverage analysis and reporting
- **[Static Analysis](docs/static-analysis.md)** - IWYU and Cppcheck tools
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
