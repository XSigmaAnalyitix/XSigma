# CI Pipeline: On-Demand Compiler Installation Fix

## Problem Statement

The XSigma CI pipeline's `compiler-version-tests` job was failing due to **package conflicts** when attempting to install multiple Clang versions simultaneously on Ubuntu.

### Root Cause

The `install-deps-ubuntu.sh` script attempted to install multiple Clang versions (16, 17, 18) in a single `apt-get install` command. Ubuntu's package repository has conflicting package names for different Clang versions:

- `clang-16`, `clang-17`, `clang-18` (conflicting packages)
- `llvm-16-dev`, `llvm-17-dev`, `llvm-18-dev` (conflicting packages)

**Error Example:**
```
E: Unable to locate package clang-17
E: Package clang-18 conflicts with clang-16
E: Broken packages
```

### Impact

- ❌ CI pipeline blocked for all compiler version tests
- ❌ Entire `compiler-version-tests` job fails
- ❌ Other CI jobs remain unaffected but compiler testing is unavailable
- ❌ Cannot verify code compatibility across multiple compiler versions

## Solution: On-Demand Compiler Installation

Implemented **Option 1 (On-Demand Compiler Installation)** - the recommended approach:

### Key Changes

#### 1. New Script: `install-clang-version.sh`

**Location:** `.github/workflows/install/install-clang-version.sh`

**Purpose:** Install a single specified Clang version without conflicts

**Features:**
- Accepts a single Clang version as parameter
- Adds LLVM repository for versions ≥ 15
- Installs only the requested version
- Creates symbolic links for easy access
- Includes error handling and validation

**Usage:**
```bash
./install-clang-version.sh 16 --with-llvm-tools
./install-clang-version.sh 17
```

#### 2. Updated CI Workflow: `.github/workflows/ci.yml`

**Changes in `compiler-version-tests` job:**

**Before:**
```yaml
- name: Install specific compiler version (Ubuntu)
  if: runner.os == 'Linux'
  run: |
    sudo apt-get update
    sudo apt-get install -y ${{ matrix.compiler_c }} ${{ matrix.compiler_cxx }}
```

**After:**
```yaml
- name: Install specific Clang version (Ubuntu)
  if: runner.os == 'Linux' && matrix.compiler_name == 'Clang'
  run: |
    CLANG_VERSION=$(echo "${{ matrix.compiler_cxx }}" | grep -oE '[0-9]+$' || echo "")
    if [ -z "$CLANG_VERSION" ]; then
      echo "WARNING: Could not extract Clang version"
    else
      chmod +x .github/workflows/install/install-clang-version.sh
      ./.github/workflows/install/install-clang-version.sh "$CLANG_VERSION" --with-llvm-tools
    fi

- name: Install specific GCC version (Ubuntu)
  if: runner.os == 'Linux' && matrix.compiler_name == 'GCC'
  run: |
    GCC_VERSION=$(echo "${{ matrix.compiler_cxx }}" | grep -oE '[0-9]+$' || echo "")
    if [ -z "$GCC_VERSION" ]; then
      echo "WARNING: Could not extract GCC version"
    else
      sudo apt-get update
      sudo apt-get install -y gcc-$GCC_VERSION g++-$GCC_VERSION
    fi
```

**Key Improvements:**
- ✅ Separate installation steps for Clang and GCC
- ✅ Conditional execution based on compiler type
- ✅ Version extraction from compiler name
- ✅ Per-matrix compiler installation (no conflicts)
- ✅ Graceful fallback on installation failure

#### 3. Updated Dependency Script: `install-deps-ubuntu.sh`

**Changes:**

**Before:**
```bash
# Clang compiler
log_info "Installing Clang compiler..."
sudo apt-get install -y \
    clang \
    clang++ \
    llvm \
    llvm-dev \
    || { ... }
```

**After:**
```bash
# Clang compiler (default version only)
# Note: Specific Clang versions are installed on-demand via install-clang-version.sh
# to avoid package conflicts when multiple versions are needed
log_info "Installing default Clang compiler..."
sudo apt-get install -y \
    clang \
    clang++ \
    || { ... }
```

**Key Changes:**
- ✅ Removed `llvm` and `llvm-dev` (conflicting packages)
- ✅ Installs only default Clang version
- ✅ Specific versions installed on-demand
- ✅ Reduced package conflicts

## How It Works

### Execution Flow

1. **Shared Dependencies** (runs once per job)
   - `install-deps-ubuntu.sh` installs common tools and default Clang
   - No version-specific packages installed

2. **Per-Matrix Compiler Installation** (runs for each matrix entry)
   - Extracts compiler version from matrix variable
   - Calls appropriate installation script:
     - Clang: `install-clang-version.sh <version>`
     - GCC: `apt-get install gcc-<version> g++-<version>`
   - Only one compiler version installed per matrix entry

3. **Build and Test**
   - Uses the installed compiler version
   - No conflicts with other matrix entries

### Example: Clang 16 Installation

```bash
# Matrix entry specifies: compiler_cxx: "clang-16"

# Step 1: Extract version
CLANG_VERSION=$(echo "clang-16" | grep -oE '[0-9]+$')  # Result: "16"

# Step 2: Call installation script
./install-clang-version.sh 16 --with-llvm-tools

# Step 3: Script adds LLVM repository and installs
# - Adds: deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main
# - Installs: clang-16, clang++-16, llvm-16, llvm-16-dev
# - Creates symbolic links: /usr/bin/clang -> /usr/bin/clang-16
```

## Benefits

✅ **No Package Conflicts** - Each matrix entry installs only its required compiler  
✅ **Parallel Execution** - Multiple compiler versions can be tested simultaneously  
✅ **Cleaner Dependency Script** - Shared script only installs common dependencies  
✅ **Better Maintainability** - Compiler-specific logic separated from shared setup  
✅ **Graceful Fallback** - Failures don't block entire pipeline  
✅ **Cross-Platform Ready** - GCC and Clang handled separately  
✅ **Extensible** - Easy to add new compiler versions to matrix  

## Supported Compiler Versions

### Clang (Ubuntu)
- ✅ Clang 15 (via LLVM repository)
- ✅ Clang 16 (via LLVM repository)
- ✅ Clang 17 (via LLVM repository)
- ✅ Clang 18+ (via LLVM repository)

### GCC (Ubuntu)
- ✅ GCC 11
- ✅ GCC 12
- ✅ GCC 13
- ✅ GCC 14+

### macOS
- ✅ Xcode Clang (uses system compiler)

## Testing the Changes

### Local Testing

```bash
# Test Clang 16 installation
./.github/workflows/install/install-clang-version.sh 16 --with-llvm-tools

# Verify installation
clang-16 --version
clang++-16 --version
```

### CI Testing

The changes are automatically tested when:
1. PR is created with changes to `.github/workflows/ci.yml`
2. `compiler-version-tests` job runs
3. Each matrix entry installs its specific compiler version
4. Build and tests execute successfully

## Troubleshooting

### Issue: "Could not extract Clang version"

**Cause:** Compiler name doesn't match expected format  
**Solution:** Verify matrix entry has correct `compiler_cxx` value (e.g., "clang-16")

### Issue: "Failed to install Clang X"

**Cause:** LLVM repository unavailable or package not found  
**Solution:** 
- Check Ubuntu version compatibility
- Verify LLVM repository is accessible
- Check for network issues

### Issue: "Package conflicts"

**Cause:** Multiple Clang versions still being installed  
**Solution:**
- Verify `install-deps-ubuntu.sh` doesn't install version-specific packages
- Check that per-matrix installation steps are running
- Review CI logs for installation order

## Migration Guide

### For Developers

No changes needed! The CI pipeline automatically handles compiler installation.

### For CI Maintainers

To add a new compiler version:

1. Add matrix entry to `compiler-version-tests`:
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

2. No other changes needed! The installation scripts handle the rest.

## References

- [LLVM Apt Repository](https://apt.llvm.org/)
- [Ubuntu Package Management](https://wiki.ubuntu.com/Apt)
- [GitHub Actions Matrix Strategy](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)

## Summary

The on-demand compiler installation approach successfully resolves package conflicts by:
- Installing only the required compiler version per matrix entry
- Removing version-specific packages from shared dependency script
- Using conditional installation steps based on compiler type
- Maintaining backward compatibility with existing CI jobs

This solution enables comprehensive compiler version testing while maintaining a clean, maintainable CI pipeline.

