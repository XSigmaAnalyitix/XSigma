# Static Analysis Tools

XSigma integrates two powerful static analysis tools to improve code quality: Include-What-You-Use (IWYU) for header dependency management and Cppcheck for comprehensive static code analysis.

## Table of Contents

- [Include-What-You-Use (IWYU)](#include-what-you-use-iwyu)
- [Cppcheck Static Analysis](#cppcheck-static-analysis)
- [Best Practices](#best-practices)
- [CI/CD Integration](#cicd-integration)

## Include-What-You-Use (IWYU)

IWYU helps reduce unnecessary includes and enforces clean header dependencies across the codebase. It analyzes your code and suggests which headers should be included or removed.

### Overview

- **CMake option**: `XSIGMA_ENABLE_IWYU` (default: OFF)
- **Applies to**: XSigma targets only (ThirdParty targets are skipped)
- **Logs**: `build/iwyu.log` with per-file analysis also recorded under `build/iwyu_logs/`
- **Mapping file** (optional): `Scripts/iwyu_exclusion.imp` (used if present)

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install iwyu
```

**Fedora/CentOS/RHEL:**
```bash
sudo dnf install iwyu
```

**macOS (Homebrew):**
```bash
brew install include-what-you-use
```

**Windows:**
Download from https://include-what-you-use.org/ or build from source

### Usage

#### Enable and Run

```bash
# Configure with IWYU enabled
cmake -B build -S . -DXSIGMA_ENABLE_IWYU=ON

# Build (IWYU runs during compilation and writes logs)
cmake --build build -j

# Inspect the log for include suggestions
less build/iwyu.log
```

#### Understanding IWYU Output

IWYU will suggest changes like:

```
example.cpp should add these lines:
#include <vector>
#include "core/utility.h"

example.cpp should remove these lines:
- #include <map>  // lines 5-5
- #include "unused_header.h"  // lines 8-8

The full include-list for example.cpp:
#include <vector>
#include "core/utility.h"
#include "example.h"
```

#### Customizing IWYU Behavior

Create or edit `Scripts/iwyu_exclusion.imp` to customize IWYU behavior:

```python
[
  # Map standard library headers
  { include: ["<bits/std_abs.h>", private, "<cmath>", public] },

  # Exclude third-party headers
  { include: ["@<ThirdParty/.*>", private, "<third_party.h>", public] },

  # Custom mappings
  { include: ["\"internal/impl.h\"", private, "\"public_api.h\"", public] }
]
```

### Notes

- IWYU is crash-resistant and uses conservative flags configured in `Cmake/tools/iwyu.cmake`
- If IWYU is not found and the option is ON, configuration fails with a helpful install hint
- IWYU analysis runs during compilation, so build times may increase
- Review suggestions carefully - not all suggestions may be appropriate for your codebase

## Cppcheck Static Analysis

Cppcheck provides comprehensive static analysis for C/C++ code quality, style, performance, and portability issues.

### Overview

- **CMake option**: `XSIGMA_ENABLE_CPPCHECK` (default: OFF)
- **Optional**: `XSIGMA_ENABLE_AUTOFIX` (WARNING: enables `--fix`, modifies source files!)
- **Suppressions file** (optional): `Scripts/cppcheck_suppressions.txt`
- **Output log**: `${CMAKE_BINARY_DIR}/cppcheckoutput.log`
- **Third-party code**: Automatically skipped

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install cppcheck
```

**Fedora/CentOS/RHEL:**
```bash
sudo dnf install cppcheck
```

**macOS (Homebrew):**
```bash
brew install cppcheck
```

**Windows:**
```bash
# Using Chocolatey
choco install cppcheck

# Using winget
winget install cppcheck
```

### Usage

#### Enable and Run

```bash
# Configure with Cppcheck enabled
cmake -B build -S . -DXSIGMA_ENABLE_CPPCHECK=ON

# Build (cppcheck runs as part of compilation and writes the log file)
cmake --build build -j

# Review analysis results
less build/cppcheckoutput.log
```

#### Enable Automatic Fixes (Use with Caution)

```bash
# Enable automatic fixes - WARNING: This modifies source files!
cmake -B build -S . \
    -DXSIGMA_ENABLE_CPPCHECK=ON \
    -DXSIGMA_ENABLE_AUTOFIX=ON

cmake --build build -j
```

**⚠️ Warning**: `XSIGMA_ENABLE_AUTOFIX` will modify your source files automatically. Always commit your changes before enabling this option, and review the changes carefully afterward.

#### Understanding Cppcheck Output

Cppcheck reports various issue types:

```
[example.cpp:42]: (error) Memory leak: ptr
[example.cpp:58]: (warning) Variable 'x' is assigned a value that is never used
[example.cpp:73]: (style) The scope of the variable 'temp' can be reduced
[example.cpp:91]: (performance) Function parameter 'str' should be passed by const reference
[example.cpp:105]: (portability) Casting between pointer types of different sizes
```

**Issue Severity Levels**:
- **error**: Code defects that will cause incorrect behavior
- **warning**: Potential issues that should be reviewed
- **style**: Code style suggestions
- **performance**: Performance optimization suggestions
- **portability**: Portability issues across platforms
- **information**: Informational messages

### Customizing Cppcheck Behavior

#### Suppressions File

Edit `Scripts/cppcheck_suppressions.txt` to silence known safe patterns:

```
# Suppress specific warnings
unusedFunction
missingInclude

# Suppress warnings for specific files
uninitvar:*/ThirdParty/*
*:*/external/*

# Suppress specific warning in specific file
nullPointer:src/legacy_code.cpp

# Suppress with inline comments in code
// cppcheck-suppress nullPointer
ptr->value = 42;
```

#### Inline Suppressions

Suppress warnings directly in code:

```cpp
// Suppress next line
// cppcheck-suppress nullPointer
ptr->value = 42;

// Suppress entire block
// cppcheck-suppress-begin nullPointer
void legacy_function() {
    // Complex legacy code
}
// cppcheck-suppress-end nullPointer
```

### Cppcheck Checks

Cppcheck performs various checks:

| Check Type | Description |
|------------|-------------|
| **Error detection** | Null pointer dereferences, memory leaks, buffer overflows |
| **Undefined behavior** | Uninitialized variables, out-of-bounds access |
| **Style** | Unused variables, redundant code, naming conventions |
| **Performance** | Inefficient operations, unnecessary copies |
| **Portability** | Platform-specific issues, type size assumptions |

### Notes

- The analysis is configured in `Cmake/tools/cppcheck.cmake` with platform-appropriate options
- Cppcheck runs during compilation, so build times may increase
- Review all suggestions - some may be false positives
- Use suppressions judiciously to avoid hiding real issues

## Best Practices

### Development Workflow

1. **Enable tools during development**:
   ```bash
   cmake -B build -S . \
       -DXSIGMA_ENABLE_IWYU=ON \
       -DXSIGMA_ENABLE_CPPCHECK=ON
   ```

2. **Review analysis results regularly**:
   ```bash
   # Review IWYU suggestions
   less build/iwyu.log

   # Review Cppcheck findings
   less build/cppcheckoutput.log
   ```

3. **Address issues incrementally**:
   - Fix critical errors first
   - Address warnings in high-priority code
   - Consider style suggestions for new code

### Code Review Integration

- Run static analysis before submitting code reviews
- Include analysis results in PR descriptions
- Address all errors and critical warnings
- Document reasons for suppressing warnings

### Maintaining Clean Code

1. **Regular analysis**: Run tools frequently during development
2. **Zero-warning policy**: Aim for zero warnings in new code
3. **Suppress carefully**: Only suppress false positives with documentation
4. **Update suppressions**: Review and update suppression files regularly

## CI/CD Integration

### Example CI Configuration

```yaml
# Static Analysis Job
- name: "Static Analysis"
  run: |
    # Configure with static analysis tools
    cmake -B build -S . \
        -DXSIGMA_ENABLE_IWYU=ON \
        -DXSIGMA_ENABLE_CPPCHECK=ON

    # Build (runs analysis)
    cmake --build build -j

    # Check for errors
    if grep -q "(error)" build/cppcheckoutput.log; then
      echo "Cppcheck found errors!"
      exit 1
    fi

- name: "Upload Analysis Results"
  uses: actions/upload-artifact@v3
  with:
    name: static-analysis-results
    path: |
      build/iwyu.log
      build/cppcheckoutput.log
```

### Automated Checks

```bash
# Fail build on Cppcheck errors
cmake -B build -S . -DXSIGMA_ENABLE_CPPCHECK=ON
cmake --build build -j

# Check for errors in output
if grep -q "(error)" build/cppcheckoutput.log; then
    echo "Static analysis found errors!"
    exit 1
fi
```

## Troubleshooting

### IWYU Not Found

**Problem**: CMake fails with "IWYU not found"

**Solutions**:
1. Install IWYU (see [Installation](#installation))
2. Disable IWYU: `cmake -B build -S . -DXSIGMA_ENABLE_IWYU=OFF`
3. Specify IWYU path: `cmake -B build -S . -DIWYU_PATH=/path/to/iwyu`

### Cppcheck Not Found

**Problem**: CMake fails with "Cppcheck not found"

**Solutions**:
1. Install Cppcheck (see [Installation](#installation))
2. Disable Cppcheck: `cmake -B build -S . -DXSIGMA_ENABLE_CPPCHECK=OFF`
3. Specify Cppcheck path: `cmake -B build -S . -DCPPCHECK_PATH=/path/to/cppcheck`

### Too Many False Positives

**Problem**: Analysis tools report many false positives

**Solutions**:
1. Add suppressions to `Scripts/cppcheck_suppressions.txt`
2. Use inline suppressions for specific cases
3. Update IWYU mappings in `Scripts/iwyu_exclusion.imp`
4. Report false positives to tool maintainers

## Related Documentation

- [Build Configuration](build-configuration.md) - Build system configuration
- [Code Coverage](code-coverage.md) - Test coverage analysis
- [Sanitizers](sanitizers.md) - Memory debugging and analysis
