# XSigma Installation Scripts Reference

## Overview

XSigma now includes platform-specific installation scripts for setting up build dependencies. These scripts are used in CI and can also be run locally for development.

## Scripts Location

All scripts are located in `.github/workflows/install/`:

```
.github/workflows/install/
├── install-deps-ubuntu.sh      # Ubuntu/Linux dependencies
├── install-deps-macos.sh       # macOS dependencies
├── install-deps-windows.ps1    # Windows dependencies (PowerShell)
└── install-sccache.sh          # Sccache installation
```

## Ubuntu/Linux: install-deps-ubuntu.sh

### Usage

```bash
chmod +x .github/workflows/install/install-deps-ubuntu.sh
./.github/workflows/install/install-deps-ubuntu.sh [OPTIONS]
```

### Options

- `--with-cuda`: Install CUDA Toolkit (optional)
- `--with-tbb`: Install Intel TBB (default: not installed)

### Examples

```bash
# Basic installation
./.github/workflows/install/install-deps-ubuntu.sh

# With TBB support
./.github/workflows/install/install-deps-ubuntu.sh --with-tbb

# With TBB and CUDA
./.github/workflows/install/install-deps-ubuntu.sh --with-tbb --with-cuda
```

### Installed Packages

- **Build Tools**: CMake, Ninja, Git, Curl, Wget, pkg-config
- **Compilers**: Clang, GCC, LLVM
- **Libraries**: OpenSSL, Zlib, NUMA, TBB (optional)
- **Python**: Python 3, pip, colorama, psutil

## macOS: install-deps-macos.sh

### Usage

```bash
chmod +x .github/workflows/install/install-deps-macos.sh
./.github/workflows/install/install-deps-macos.sh [OPTIONS]
```

### Prerequisites

- Homebrew must be installed
- Xcode Command Line Tools (auto-installed if needed)

### Options

- `--with-cuda`: Install CUDA Toolkit (limited support on Apple Silicon)
- `--with-tbb`: Install Intel TBB

### Examples

```bash
# Basic installation
./.github/workflows/install/install-deps-macos.sh

# With TBB support
./.github/workflows/install/install-deps-macos.sh --with-tbb
```

### Installed Packages

- **Build Tools**: CMake, Ninja, Git, Curl, Wget, pkg-config, Python 3
- **Compilers**: Clang (via Xcode), GCC
- **Libraries**: OpenSSL, Zlib, TBB (optional)

## Windows: install-deps-windows.ps1

### Usage

```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\.github\workflows\install\install-deps-windows.ps1 [-WithCuda] [-WithTbb]
```

### Prerequisites

- Chocolatey must be installed
- PowerShell (Administrator privileges required)

### Options

- `-WithCuda`: Install CUDA Toolkit
- `-WithTbb`: Install Intel TBB

### Examples

```powershell
# Basic installation
.\.github\workflows\install\install-deps-windows.ps1

# With TBB support
.\.github\workflows\install\install-deps-windows.ps1 -WithTbb

# With TBB and CUDA
.\.github\workflows\install\install-deps-windows.ps1 -WithTbb -WithCuda
```

### Installed Packages

- **Build Tools**: CMake, Ninja, Git, Curl, Wget, Python 3
- **Compilers**: LLVM/Clang, Visual Studio Build Tools
- **Libraries**: TBB (optional)

## Sccache: install-sccache.sh

### Usage

```bash
chmod +x .github/workflows/install/install-sccache.sh
./.github/workflows/install/install-sccache.sh [VERSION]
```

### Features

- Automatic platform detection (Linux, macOS)
- Downloads from official Mozilla releases
- Idempotent installation
- Adds to PATH automatically

### Examples

```bash
# Install default version (0.7.7)
./.github/workflows/install/install-sccache.sh

# Install specific version
./.github/workflows/install/install-sccache.sh 0.8.0

# Verify installation
sccache --version
```

### Installation Paths

- **Linux**: `~/.local/bin/sccache`
- **macOS**: `~/.local/bin/sccache`

## Common Tasks

### Install All Dependencies (Ubuntu)

```bash
./scripts/ci/install-deps-ubuntu.sh --with-tbb
```

### Install All Dependencies (macOS)

```bash
./scripts/ci/install-deps-macos.sh --with-tbb
```

### Install All Dependencies (Windows)

```powershell
.\scripts\ci\install-deps-windows.ps1 -WithTbb
```

### Install Sccache (All Platforms)

```bash
./scripts/ci/install-sccache.sh 0.7.7
```

### Verify Installations

```bash
# Check compilers
clang --version
gcc --version

# Check build tools
cmake --version
ninja --version

# Check Python
python3 --version

# Check sccache
sccache --version
```

## Troubleshooting

### Script Not Executable

```bash
chmod +x scripts/ci/install-deps-*.sh
chmod +x scripts/ci/install-sccache.sh
```

### Permission Denied (Ubuntu/macOS)

Scripts use `sudo` for system package installation. You may be prompted for your password.

### Chocolatey Not Found (Windows)

Install Chocolatey first:
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### Compiler Version Not Available

If a specific compiler version isn't available:
1. Check available versions: `apt-cache search gcc` (Ubuntu)
2. Update package lists: `sudo apt-get update`
3. Or use a different version

### Sccache Download Fails

1. Check internet connection
2. Verify version exists on GitHub releases
3. Try manual download and installation

## Script Features

### Error Handling

All scripts include:
- Exit on error (`set -e`)
- Comprehensive error messages
- Fallback options where applicable
- Non-fatal warnings for optional components

### Idempotency

Scripts are safe to run multiple times:
- Check if packages already installed
- Skip redundant installations
- Preserve existing configurations

### Logging

Scripts provide:
- Color-coded output (INFO, SUCCESS, WARNING, ERROR)
- Clear progress messages
- Verification steps
- Helpful suggestions on failure

## Integration with CI

Scripts are automatically called in `.github/workflows/ci.yml`:

```yaml
- name: Install dependencies (Ubuntu)
  if: runner.os == 'Linux'
  run: |
    chmod +x scripts/ci/install-deps-ubuntu.sh
    ./scripts/ci/install-deps-ubuntu.sh --with-tbb
```

## Local Development

Use these scripts to set up your development environment:

```bash
# Clone repository
git clone https://github.com/xsigma/xsigma.git
cd xsigma

# Install dependencies
./scripts/ci/install-deps-ubuntu.sh --with-tbb

# Build XSigma
cd Scripts
python setup.py ninja clang release test cxx17 loguru tbb
```

## References

- [XSigma CI Refactoring Guide](CI_REFACTORING_GUIDE.md)
- [setup.py Documentation](../Scripts/setup.py)
- [GitHub Actions CI Workflow](.github/workflows/ci.yml)
