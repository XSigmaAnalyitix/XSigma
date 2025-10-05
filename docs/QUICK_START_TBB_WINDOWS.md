# Quick Start: Building XSigma with TBB on Windows

## For Windows + Clang Users

If you're building XSigma on Windows with Clang and encounter a TBB error, follow these quick steps:

### Option 1: Install TBB via vcpkg (Recommended - 5 minutes)

```bash
# 1. Install vcpkg (if not already installed)
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat

# 2. Install TBB
.\vcpkg install tbb:x64-windows

# 3. Build XSigma
cd C:\dev\XSigma\Scripts
python setup.py config.ninja.clang.test.tbb -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
python setup.py build.ninja.clang.test.tbb
```

### Option 2: Use MSVC Instead of Clang (No installation needed)

```bash
cd C:\dev\XSigma\Scripts
python setup.py config.ninja.vs22.test.tbb
python setup.py build.ninja.vs22.test.tbb
```

## Why This is Needed

TBB cannot be built from source on Windows with Clang due to compiler compatibility issues. You must either:
- Install TBB via a package manager (vcpkg, Chocolatey), OR
- Use the MSVC compiler instead of Clang

## For Other Platforms

- **Linux**: TBB builds from source automatically ✅
- **macOS**: TBB builds from source automatically ✅
- **Windows + MSVC**: TBB builds from source automatically ✅

## Need Help?

See detailed documentation in `docs/TBB_WINDOWS_CLANG.md`

