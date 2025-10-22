# Code Coverage

Generate code coverage reports to measure test effectiveness and identify untested code paths. XSigma uses LLVM coverage tools (llvm-profdata and llvm-cov) for source-based coverage analysis.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [What Gets Measured](#what-gets-measured)
- [Automatic Coverage Analysis](#automatic-coverage-analysis)
- [Manual Coverage Commands](#manual-coverage-commands)
- [Coverage Requirements](#coverage-requirements)
- [Interpreting Coverage Reports](#interpreting-coverage-reports)
- [Best Practices](#best-practices)

## Overview

Code coverage analysis helps you:
- Measure test effectiveness
- Identify untested code paths
- Ensure critical code is tested
- Track coverage trends over time
- Meet coverage requirements (e.g., 98% minimum)

XSigma uses LLVM's source-based coverage tools for accurate, line-level coverage analysis.

## Quick Start

### Using setup.py (Recommended)

```bash
cd Scripts
python setup.py ninja.clang.config.tbb.build.coverage
```

The HTML coverage report will be generated at:
```
build_ninja_tbb_coverage/coverage_report/html/index.html
```

### Using CMake Directly

```bash
# Configure with coverage enabled
cmake -B build_coverage -S . \
    -DCMAKE_BUILD_TYPE=Debug \
    -DXSIGMA_ENABLE_COVERAGE=ON \
    -DXSIGMA_BUILD_TESTING=ON

# Build
cmake --build build_coverage

# Run tests to generate coverage data
ctest --test-dir build_coverage

# Generate coverage report
cmake --build build_coverage --target coverage-html
```

## What Gets Measured

Coverage analysis tracks:

- **Line coverage**: Which lines of code were executed during tests
- **Function coverage**: Which functions were called
- **Region coverage**: Which code branches were taken

### Automatic Exclusions

The following are automatically excluded from coverage reports:
- Third-party libraries (ThirdParty/ directory)
- Test files (Testing/ directories)
- Build artifacts
- External dependencies

## Automatic Coverage Analysis

Coverage analysis is integrated into the Python build helper in `Scripts/setup.py`.

### Integrated Workflow

Running `python setup.py ninja.clang.config.build.test.coverage` (from the `Scripts` directory) will automatically:
1. Configure the build with coverage enabled
2. Build the project
3. Run all tests
4. Collect coverage data
5. Analyze coverage and generate reports

**No need to add an extra `.analyze` step** - it's all done automatically!

### Standalone Analysis

If you want to re-analyze existing coverage data without rebuilding:

```bash
cd Scripts
python setup.py analyze
```

This is useful when you want to:
- Regenerate reports with different options
- Update reports after manual test runs
- Analyze coverage data from multiple test runs

## Manual Coverage Commands

After building with coverage enabled, you can generate reports manually:

```bash
# Navigate to build directory
cd build_ninja_tbb_coverage

# Merge raw coverage data
cmake --build . --target coverage-merge

# Generate text report (console output)
cmake --build . --target coverage-report

# Generate HTML report (interactive browser view)
cmake --build . --target coverage-html

# Generate JSON report (for CI/CD integration)
cmake --build . --target coverage-json
```

### Available Coverage Targets

| Target | Description | Output |
|--------|-------------|--------|
| `coverage-merge` | Merge raw coverage data | `.profdata` file |
| `coverage-report` | Generate text report | Console output |
| `coverage-html` | Generate HTML report | `coverage_report/html/` |
| `coverage-json` | Generate JSON report | `coverage_report/coverage.json` |

## Coverage Requirements

### Compiler Requirements

- **Compiler**: Clang with LLVM tools
- **Required tools**: `llvm-profdata`, `llvm-cov`
- **Build type**: Debug configuration with instrumentation flags

### Installing LLVM Tools

**Ubuntu/Debian:**
```bash
sudo apt-get install clang llvm
```

**macOS:**
```bash
brew install llvm
```

**Windows:**
Download and install LLVM from https://releases.llvm.org/

### Testing Requirements

- Google Test framework must be enabled (`XSIGMA_BUILD_TESTING=ON`)
- Tests must be run to generate coverage data
- Tests should cover all critical code paths

## Interpreting Coverage Reports

### HTML Report

The HTML report provides:
- **Summary page**: Overall coverage statistics
- **File list**: Coverage by file
- **Source view**: Line-by-line coverage with color coding
  - **Green**: Line executed
  - **Red**: Line not executed
  - **Gray**: Non-executable line (comments, declarations)

### Text Report

The text report shows:
```
Filename                      Regions    Missed Regions     Cover   Functions  Missed Functions  Executed
---------------------------------------------------------------------------------------------------------
Core/src/example.cpp              45                 3    93.33%          12                 1    91.67%
```

### Coverage Metrics

- **Line Coverage**: Percentage of executable lines that were run
- **Function Coverage**: Percentage of functions that were called
- **Region Coverage**: Percentage of code branches that were taken

### Target Coverage Levels

| Level | Description | Recommendation |
|-------|-------------|----------------|
| < 70% | Low coverage | Needs improvement |
| 70-90% | Good coverage | Acceptable for most projects |
| 90-98% | High coverage | Recommended for production code |
| 98%+ | Excellent coverage | XSigma target |

## Best Practices

### Writing Tests for Coverage

1. **Test all code paths**: Include positive and negative test cases
2. **Test edge cases**: Boundary conditions, error handling
3. **Test error paths**: Exception handling, error returns
4. **Avoid testing implementation details**: Focus on behavior

### Coverage-Driven Development

```bash
# 1. Write tests first
# 2. Run coverage analysis
cd Scripts
python setup.py ninja.clang.config.build.test.coverage

# 3. Review coverage report
open ../build_ninja_tbb_coverage/coverage_report/html/index.html

# 4. Add tests for uncovered code
# 5. Repeat until target coverage is reached
```

### CI/CD Integration

```yaml
# Example CI configuration
- name: "Code Coverage"
  run: |
    cd Scripts
    python setup.py ninja.clang.config.build.test.coverage
    
- name: "Upload Coverage Report"
  uses: codecov/codecov-action@v3
  with:
    files: ./build_ninja_tbb_coverage/coverage_report/coverage.json
```

### Maintaining High Coverage

1. **Set coverage requirements**: Enforce minimum coverage (e.g., 98%)
2. **Review coverage in PRs**: Check coverage changes in code reviews
3. **Track coverage trends**: Monitor coverage over time
4. **Focus on critical code**: Prioritize coverage for important modules

### Excluding Code from Coverage

For code that shouldn't be covered (e.g., debug code, platform-specific code):

```cpp
// LCOV_EXCL_START
void debug_only_function() {
    // This code won't be counted in coverage
}
// LCOV_EXCL_STOP

// Single line exclusion
void platform_specific() { /* LCOV_EXCL_LINE */
    // This line won't be counted
}
```

## Troubleshooting

### No Coverage Data Generated

**Problem**: Tests run but no coverage data is produced

**Solutions**:
1. Verify coverage is enabled: `XSIGMA_ENABLE_COVERAGE=ON`
2. Check that tests actually ran: `ctest --test-dir build --verbose`
3. Ensure LLVM tools are installed: `which llvm-profdata llvm-cov`

### Coverage Report Shows 0%

**Problem**: Coverage report shows 0% coverage for all files

**Solutions**:
1. Verify tests were run after building with coverage
2. Check that `.profraw` files were generated in the build directory
3. Ensure the correct build directory is being analyzed

### Missing LLVM Tools

**Problem**: `llvm-profdata` or `llvm-cov` not found

**Solutions**:
1. Install LLVM tools (see [Coverage Requirements](#coverage-requirements))
2. Add LLVM tools to PATH
3. Specify tool paths in CMake: `-DLLVM_PROFDATA=/path/to/llvm-profdata`

## Code Coverage Tools Architecture

The XSigma code coverage system is implemented in `Tools/code_coverage/` with the following components:

### Project Structure

```
Tools/code_coverage/
├── package/
│   ├── tool/              # Core coverage tools
│   │   ├── clang_coverage.py      # LLVM/Clang coverage implementation
│   │   ├── gcc_coverage.py        # GCC coverage implementation
│   │   ├── coverage_filters.py    # Shared file filtering logic
│   │   ├── summarize_jsons.py     # JSON parsing and summarization
│   │   ├── print_report.py        # HTML report generation
│   │   └── utils.py               # Utility functions
│   ├── oss/
│   │   └── utils.py               # OSS platform utilities
│   └── util/
│       └── setting.py             # Configuration and enums
├── tests/                 # Comprehensive test suite
│   ├── unit/             # Unit tests (110+ tests)
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test data and mocks
└── scripts/              # Standalone scripts
```

### Platform Support

- **Windows**: Uses LLVM/Clang with OpenCppCoverage or native LLVM tools
- **Linux**: Uses GCC or Clang with native LLVM tools
- **macOS**: Uses Clang with native LLVM tools

### Key Features

1. **Cross-Platform Compatibility**: Works on Windows, Linux, and macOS
2. **Error Handling**: Uses return values instead of exceptions (XSigma standard)
3. **File Filtering**: Automatically excludes test code, build artifacts, and third-party libraries
4. **Multiple Report Formats**: HTML, JSON, and terminal output
5. **Comprehensive Testing**: 110+ tests with >90% code coverage

### Customizing HTML Reports

The HTML report layout can be customized by modifying:
- `Tools/code_coverage/package/tool/print_report.py` - Report generation logic
- CSS styling in the generated HTML files

### Testing the Coverage Tools

Run the comprehensive test suite:

```bash
# Run all tests
pytest Tools/code_coverage/tests/ -v

# Run specific test category
pytest Tools/code_coverage/tests/unit/ -v
pytest Tools/code_coverage/tests/integration/ -v

# Generate coverage report for the tools themselves
pytest Tools/code_coverage/tests/ --cov=Tools/code_coverage/package --cov-report=html
```

### Architecture and Design Decisions

**Error Handling**: All functions return boolean or status codes instead of raising exceptions, following XSigma coding standards.

**File Filtering**: Common filtering logic is centralized in `coverage_filters.py` to avoid duplication and ensure consistency across different report generators.

**JSON Streaming**: JSON files are parsed line-by-line to handle large coverage files efficiently without loading entire files into memory.

**Platform Abstraction**: Platform-specific logic is isolated in separate modules (`clang_coverage.py`, `gcc_coverage.py`) while shared logic is in common modules.

## Troubleshooting

### Coverage Tools Issues

**Problem**: Coverage report shows 0% coverage

**Solutions**:
1. Verify tests were executed: Check for `.profraw` or `.gcda` files in build directory
2. Check file filtering: Ensure your source files aren't being excluded by filters
3. Verify LLVM tools: Run `llvm-profdata --version` and `llvm-cov --version`

**Problem**: "File not found" errors in coverage report

**Solutions**:
1. Verify source file paths are correct
2. Check that build directory structure matches source structure
3. Ensure coverage data was collected from the correct build

**Problem**: Tests fail during coverage collection

**Solutions**:
1. Run tests without coverage first: `pytest Tests/`
2. Check for permission errors on coverage data files
3. Verify disk space for coverage data files

## Additional Documentation

For detailed coverage configuration and advanced usage, see:
- `Cmake/tools/COVERAGE_USAGE.md` - Detailed CMake coverage configuration
- `Scripts/README_COVERAGE.md` - Coverage script documentation
- `Tools/code_coverage/README.md` - Coverage tools documentation

## Related Documentation

- [Build Configuration](build-configuration.md) - Build system configuration
- [Sanitizers](sanitizer.md) - Memory debugging and analysis
- [Static Analysis](static-analysis.md) - IWYU and Cppcheck tools

