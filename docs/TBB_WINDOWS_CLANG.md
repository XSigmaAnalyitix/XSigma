# Intel TBB on Windows with Clang

## Overview

When building XSigma on Windows with Clang as the compiler, Intel TBB (Threading Building Blocks) **must be installed via a package manager** rather than built from source. This is due to compatibility issues between TBB's build system and Clang when targeting the Windows MSVC ABI.

## Why System Installation is Required

Building TBB from source on Windows with Clang encounters several technical issues:

1. **Incompatible Compiler Flags**: TBB's CMake build system adds `-fPIC` (Position Independent Code) and `-fstack-protector-strong` flags, which are not supported by Clang when targeting the MSVC ABI (`x86_64-pc-windows-msvc`).

2. **DLL Export/Import Issues**: TBB's symbol visibility and DLL export/import mechanisms don't work correctly with Clang on Windows, leading to undefined symbol errors during linking.

3. **Linux-Specific Linker Flags**: TBB's build system adds linker flags like `--version-script`, `-z,relro`, and `-z,now` that are incompatible with Windows linkers (lld-link).

## Installation Methods

### Method 1: vcpkg (Recommended)

[vcpkg](https://vcpkg.io/) is Microsoft's C++ package manager and provides the most reliable TBB installation for Windows.

#### Installation Steps:

1. **Install vcpkg** (if not already installed):
   ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   ```

2. **Install TBB**:
   ```bash
   .\vcpkg install tbb:x64-windows
   ```

3. **Configure XSigma with vcpkg toolchain**:
   ```bash
   cd Scripts
   python setup.py config.ninja.clang.test.tbb -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
   ```

   Or set the environment variable:
   ```bash
   set CMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake
   python setup.py config.ninja.clang.test.tbb
   ```

### Method 2: Chocolatey

[Chocolatey](https://chocolatey.org/) is a Windows package manager.

```bash
choco install tbb
```

After installation, you may need to set the `TBB_ROOT` environment variable to help CMake find TBB.

### Method 3: Manual Installation

1. **Download TBB** from the official releases:
   - https://github.com/oneapi-src/oneTBB/releases

2. **Extract** the archive to a location (e.g., `C:\Program Files\TBB`)

3. **Set environment variable**:
   ```bash
   set TBB_ROOT=C:\Program Files\TBB
   ```

4. **Configure XSigma**:
   ```bash
   cd Scripts
   python setup.py config.ninja.clang.test.tbb
   ```

### Method 4: Use MSVC Compiler Instead

If you prefer not to install TBB separately, you can use the Visual Studio (MSVC) compiler instead of Clang. TBB builds successfully from source with MSVC:

```bash
cd Scripts
python setup.py config.ninja.vs22.test.tbb
```

This will build TBB from source using the MSVC compiler, which doesn't have the compatibility issues that Clang has.

## Verification

After installation, verify that CMake can find TBB:

```bash
cmake --find-package -DNAME=TBB -DCOMPILER_ID=Clang -DLANGUAGE=CXX -DMODE=EXIST
```

Or simply try configuring XSigma:

```bash
cd Scripts
python setup.py config.ninja.clang.test.tbb
```

If TBB is found, you should see:
```
-- ✅ Found system-installed Intel TBB
--    TBB Include Dir: [path]
--    TBB Library: [path]
```

## Troubleshooting

### CMake Cannot Find TBB

If CMake cannot find TBB after installation:

1. **Check TBB_ROOT environment variable**:
   ```bash
   echo %TBB_ROOT%
   ```

2. **Manually specify TBB location**:
   ```bash
   python setup.py config.ninja.clang.test.tbb -DTBB_ROOT="C:\path\to\tbb"
   ```

3. **Check PATH** - Ensure TBB's bin directory is in your PATH for runtime DLL loading.

### Linking Errors at Runtime

If you get DLL not found errors when running executables:

1. **Add TBB bin directory to PATH**:
   ```bash
   set PATH=%PATH%;C:\path\to\tbb\bin
   ```

2. **Copy TBB DLLs** to your executable directory (not recommended for development).

3. **Use vcpkg** which handles DLL deployment automatically.

## Technical Details

### Why vcpkg is Recommended

vcpkg provides several advantages:

1. **Pre-built binaries** - No compilation required
2. **Automatic dependency management** - Handles transitive dependencies
3. **CMake integration** - Seamless integration via toolchain file
4. **DLL deployment** - Automatically copies DLLs to output directory
5. **Version management** - Easy to update or switch versions

### Compiler Compatibility Matrix

| Compiler | Platform | Build from Source | System Install |
|----------|----------|-------------------|----------------|
| Clang    | Windows  | ❌ Not Supported  | ✅ Required    |
| Clang    | Linux    | ✅ Supported      | ✅ Supported   |
| Clang    | macOS    | ✅ Supported      | ✅ Supported   |
| MSVC     | Windows  | ✅ Supported      | ✅ Supported   |
| GCC      | Linux    | ✅ Supported      | ✅ Supported   |

## Alternative Solutions Considered

### 1. Building TBB as Static Libraries

**Attempted**: Build TBB as static libraries to avoid DLL export/import issues.

**Result**: This works but has drawbacks:
- Increases executable size significantly
- Each executable has its own copy of TBB code
- Not the intended use case for TBB
- Violates the principle of least surprise

### 2. Using MSVC to Build TBB, Clang for Everything Else

**Attempted**: Use ExternalProject to build TBB with MSVC compiler while keeping Clang for the main project.

**Result**: Complex and fragile:
- Requires Visual Studio installation
- Mixing compilers can cause ABI compatibility issues
- Difficult to maintain and debug
- Adds significant complexity to build system

### 3. Patching TBB's Build System

**Attempted**: Modify TBB's CMakeLists.txt to remove incompatible flags.

**Result**: Not sustainable:
- Requires maintaining patches across TBB versions
- Fragile and breaks with TBB updates
- Doesn't address fundamental ABI issues

## Conclusion

Requiring system-installed TBB on Windows with Clang is the **cleanest and most maintainable solution**. It:

- ✅ Provides clear, actionable error messages
- ✅ Leverages existing package management infrastructure
- ✅ Avoids complex build system workarounds
- ✅ Ensures ABI compatibility
- ✅ Follows best practices for dependency management
- ✅ Is easy to document and support

## References

- [Intel oneTBB GitHub](https://github.com/oneapi-src/oneTBB)
- [vcpkg Documentation](https://vcpkg.io/)
- [CMake FindTBB Module](https://cmake.org/cmake/help/latest/module/FindTBB.html)
- [Clang on Windows Documentation](https://clang.llvm.org/docs/MSVCCompatibility.html)

