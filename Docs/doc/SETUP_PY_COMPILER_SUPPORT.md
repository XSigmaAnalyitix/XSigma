# setup.py Compiler Support Documentation

## Overview

This document describes how `setup.py` handles compiler specifications and the newly added GCC support.

## Supported Compilers

### Clang

**Argument Format**: `clang`, `clang-15`, `clang-16`, `clang-17`, etc.

**How It Works**:
```python
def __is_clang_compiler(self, arg):
    return "clang" in arg and arg not in ["clang-cl", "clangtidy"]

def __set_clang_compiler(self, arg):
    self.__value["cmake_c_compiler"] = f"-DCMAKE_C_COMPILER={arg}"
    self.__value["cmake_cxx_compiler"] = (
        f"-DCMAKE_CXX_COMPILER={arg.replace('clang', 'clang++')}"
    )
```

**Examples**:
- Input: `clang` → C: `clang`, C++: `clang++`
- Input: `clang-15` → C: `clang-15`, C++: `clang++-15`

**Supported Platforms**: Linux, macOS, Windows

### GCC (NEW)

**Argument Format**: `gcc`, `g++`, `gcc-11`, `g++-11`, `gcc-12`, `g++-12`, etc.

**How It Works**:
```python
def __is_gcc_compiler(self, arg):
    return ("gcc" in arg or "g++" in arg) and arg not in ["cppcheck"]

def __set_gcc_compiler(self, arg):
    if "g++" in arg:
        # If it's g++ or g++-XX, use it as CXX and derive C compiler
        self.__value["cmake_cxx_compiler"] = f"-DCMAKE_CXX_COMPILER={arg}"
        c_compiler = arg.replace("g++", "gcc")
        self.__value["cmake_c_compiler"] = f"-DCMAKE_C_COMPILER={c_compiler}"
    else:
        # If it's gcc or gcc-XX, use it as C compiler and derive CXX compiler
        self.__value["cmake_c_compiler"] = f"-DCMAKE_C_COMPILER={arg}"
        cxx_compiler = arg.replace("gcc", "g++")
        self.__value["cmake_cxx_compiler"] = f"-DCMAKE_CXX_COMPILER={cxx_compiler}"
```

**Examples**:
- Input: `gcc` → C: `gcc`, C++: `g++`
- Input: `g++` → C: `gcc`, C++: `g++`
- Input: `gcc-11` → C: `gcc-11`, C++: `g++-11`
- Input: `g++-12` → C: `gcc-12`, C++: `g++-12`

**Supported Platforms**: Linux, macOS

### Visual Studio

**Argument Format**: `vs17`, `vs19`, `vs22`

**Supported Platforms**: Windows only

### Xcode

**Argument Format**: `xcode`

**Supported Platforms**: macOS only

## Compiler Argument Processing

### Argument Format Rules

1. **Base Compiler Name**: Pass the base compiler name, not the C++ variant
   - ✅ Correct: `clang`, `clang-15`, `gcc`, `gcc-11`
   - ❌ Incorrect: `clang++`, `clang++-15`, `g++`, `g++-11`

2. **Version Specification**: Use hyphen for version numbers
   - ✅ Correct: `clang-15`, `gcc-11`
   - ❌ Incorrect: `clang15`, `gcc11`

3. **Automatic Derivation**: setup.py automatically derives C++ compiler
   - Input: `clang` → Output: `clang++`
   - Input: `gcc` → Output: `g++`

### Processing Flow

```
User Input (e.g., "clang-15")
    ↓
parse_args() - splits by dots, underscores, spaces
    ↓
XsigmaConfiguration.__process_arg()
    ↓
__is_clang_compiler() - checks if "clang" in arg
    ↓
__set_clang_compiler() - sets CMAKE_C_COMPILER and CMAKE_CXX_COMPILER
    ↓
CMake Configuration
```

## Usage Examples

### Local Development

```bash
# Using default Clang
cd Scripts
python setup.py ninja clang release test cxx17 loguru tbb

# Using specific Clang version
python setup.py ninja clang-15 release test cxx20 loguru tbb

# Using GCC
python setup.py ninja gcc release test cxx17 loguru tbb

# Using specific GCC version
python setup.py ninja gcc-11 release test cxx17 loguru tbb
python setup.py ninja g++-12 release test cxx20 loguru tbb
```

### CI Workflow

```yaml
- name: Configure and Build
  working-directory: Scripts
  run: |
    python setup.py ninja ${{ matrix.compiler_cxx }} \
      $BUILD_TYPE_LOWER \
      test \
      cxx${{ matrix.cxx_std }} \
      loguru \
      tbb
```

Where `matrix.compiler_cxx` contains:
- `clang` (default Clang)
- `clang-15`, `clang-16`, `clang-17` (specific Clang versions)
- `gcc`, `gcc-11`, `gcc-12`, `gcc-13` (GCC versions)
- `g++`, `g++-11`, `g++-12`, `g++-13` (GCC C++ variants)

## CMake Integration

### Generated CMake Flags

When you pass `clang-15`:
```cmake
-DCMAKE_C_COMPILER=clang-15
-DCMAKE_CXX_COMPILER=clang++-15
```

When you pass `gcc-11`:
```cmake
-DCMAKE_C_COMPILER=gcc-11
-DCMAKE_CXX_COMPILER=g++-11
```

### Compiler Detection

CMake automatically detects:
- Compiler type (GCC, Clang, MSVC, etc.)
- Compiler version
- Supported C++ standards
- Available compiler flags

## Troubleshooting

### Compiler Not Found

**Error**: `cmake: error: clang-15: command not found`

**Solution**: Install the specific compiler version
```bash
# Ubuntu
sudo apt-get install clang-15

# macOS
brew install llvm@15

# Or use default compiler
python setup.py ninja clang release test cxx17
```

### Wrong Compiler Used

**Problem**: Build uses wrong compiler despite specifying one

**Solution**: Verify compiler argument format
```bash
# ✅ Correct
python setup.py ninja clang-15 ...

# ❌ Incorrect
python setup.py ninja clang++-15 ...
```

### CMake Compiler Mismatch

**Error**: `CMake Error: your C compiler: "clang-15" was not found`

**Solution**: Ensure compiler is in PATH
```bash
# Check if compiler is available
which clang-15
which g++-11

# Add to PATH if needed
export PATH=/usr/bin:$PATH
```

## Best Practices

1. **Use Base Compiler Name**: Always pass `clang` or `gcc`, not `clang++` or `g++`
2. **Specify Version Explicitly**: Use `clang-15` instead of relying on default
3. **Verify Compiler Availability**: Check `which clang-15` before building
4. **Use Consistent Format**: Don't mix `clang-15` and `clang15`
5. **Document Compiler Requirements**: Specify minimum compiler versions

## Future Enhancements

1. **Compiler Validation**: Check compiler availability before build
2. **Version Detection**: Automatically detect installed compiler versions
3. **Fallback Support**: Fall back to default compiler if version not found
4. **MSVC Versions**: Add support for `msvc-2019`, `msvc-2022`
5. **Intel Compiler**: Add support for Intel C++ Compiler

## References

- [setup.py Source Code](../Scripts/setup.py)
- [CMake Compiler Configuration](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)
- [XSigma CI Pipeline](.github/workflows/ci.yml)
