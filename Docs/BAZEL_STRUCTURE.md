# XSigma Bazel Build Structure

This document provides an overview of the Bazel build configuration files and their CMake equivalents.

## File Structure

```
XSigma/
├── WORKSPACE.bazel              # Workspace definition and external dependencies
├── BUILD.bazel                  # Root build file
├── .bazelrc                     # Build configuration flags
├── .bazelversion                # Specifies Bazel version (6.4.0)
├── BAZEL_BUILD.md              # User documentation
├── BAZEL_STRUCTURE.md          # This file
│
├── bazel/                       # Bazel helper files
│   ├── BUILD.bazel             # Config settings
│   └── xsigma.bzl              # Helper functions (copts, defines, linkopts)
│
├── third_party/                 # Third-party BUILD files
│   ├── fmt.BUILD               # fmt library
│   ├── cpuinfo.BUILD           # cpuinfo library
│   ├── magic_enum.BUILD        # magic_enum library
│   └── mimalloc.BUILD          # mimalloc library
│
├── Library/
│   ├── Core/
│   │   └── BUILD.bazel         # Core library build
│   └── Security/
│       └── BUILD.bazel         # Security library build
│
└── ThirdParty/                  # (Existing CMake third-party deps)
```

## CMake to Bazel Mapping

### Root Configuration Files

| CMake File | Bazel Equivalent | Purpose |
|------------|------------------|---------|
| `CMakeLists.txt` | `BUILD.bazel` + `WORKSPACE.bazel` | Project definition |
| `Cmake/flags/*.cmake` | `.bazelrc` | Compiler flags and options |
| `Cmake/tools/*.cmake` | `bazel/xsigma.bzl` | Helper functions |
| N/A | `bazel/BUILD.bazel` | Config settings |

### Library Build Files

| CMake File | Bazel Equivalent |
|------------|------------------|
| `Library/Core/CMakeLists.txt` | `Library/Core/BUILD.bazel` |
| `Library/Security/CMakeLists.txt` | `Library/Security/BUILD.bazel` |

### Third-Party Dependencies

| CMake Approach | Bazel Approach |
|----------------|----------------|
| `ThirdParty/CMakeLists.txt` | `WORKSPACE.bazel` + `third_party/*.BUILD` |
| `add_subdirectory(ThirdParty/xxx)` | `http_archive()` in WORKSPACE |
| `find_package()` | `http_archive()` or `local_repository()` |

## Key Concepts

### 1. WORKSPACE.bazel
- Defines the workspace name
- Declares external dependencies (third-party libraries)
- Equivalent to CMake's `find_package()` and `add_subdirectory(ThirdParty)`

### 2. BUILD.bazel
- Defines build targets (libraries, executables, tests)
- Equivalent to CMake's `add_library()`, `add_executable()`, `add_test()`

### 3. .bazelrc
- Sets default build flags
- Defines named configurations (`--config=release`, etc.)
- Equivalent to CMake cache variables and build types

### 4. bazel/xsigma.bzl
- Provides reusable functions for compiler flags and defines
- Equivalent to CMake functions in `Cmake/tools/*.cmake`

### 5. third_party/*.BUILD
- Build rules for external dependencies
- Created when the external library doesn't provide a BUILD file
- Equivalent to wrapper CMakeLists.txt for third-party libraries

## Configuration System

### CMake Options → Bazel Configs

CMake uses cache variables:
```cmake
option(XSIGMA_ENABLE_MIMALLOC "Enable mimalloc" ON)
```

Bazel uses config_setting + defines:
```python
config_setting(
    name = "enable_mimalloc",
    define_values = {"xsigma_enable_mimalloc": "true"},
)
```

Used in build with:
```bash
bazel build --config=mimalloc //...
# or
bazel build --define=xsigma_enable_mimalloc=true //...
```

### Conditional Compilation

**CMake:**
```cmake
if(XSIGMA_ENABLE_CUDA)
    target_sources(Core PRIVATE gpu/*.cpp)
endif()
```

**Bazel:**
```python
cc_library(
    name = "Core",
    srcs = ["core.cpp"] + select({
        "//bazel:enable_cuda": glob(["gpu/*.cpp"]),
        "//conditions:default": [],
    }),
)
```

## Build Targets

### Libraries

**CMake:**
```cmake
add_library(Core ${sources})
target_link_libraries(Core PUBLIC fmt::fmt)
```

**Bazel:**
```python
cc_library(
    name = "Core",
    srcs = [...],
    deps = ["@fmt//:fmt"],
)
```

### Tests

**CMake:**
```cmake
add_executable(core_test test.cpp)
add_test(NAME core_test COMMAND core_test)
```

**Bazel:**
```python
cc_test(
    name = "core_test",
    srcs = ["test.cpp"],
    deps = [":Core", "@com_google_googletest//:gtest_main"],
)
```

## Compiler Flags

### CMake Approach
```cmake
target_compile_options(Core PRIVATE -Wall -Wextra)
```

### Bazel Approach
```python
# In xsigma.bzl
def xsigma_copts():
    return ["-Wall", "-Wextra"]

# In BUILD.bazel
cc_library(
    name = "Core",
    copts = xsigma_copts(),
)
```

## Platform-Specific Configuration

### CMake
```cmake
if(UNIX AND NOT APPLE)
    target_link_libraries(Core PRIVATE pthread dl rt)
elseif(APPLE)
    target_link_libraries(Core PRIVATE "-framework Security")
elseif(WIN32)
    target_link_libraries(Core PRIVATE bcrypt)
endif()
```

### Bazel
```python
cc_library(
    name = "Core",
    linkopts = select({
        "@platforms//os:linux": ["-lpthread", "-ldl", "-lrt"],
        "@platforms//os:macos": ["-framework Security"],
        "@platforms//os:windows": ["-DEFAULTLIB:bcrypt.lib"],
        "//conditions:default": [],
    }),
)
```

## Dependency Management

### External Dependencies

**CMake:**
```cmake
find_package(fmt REQUIRED)
# or
add_subdirectory(ThirdParty/fmt)
```

**Bazel:**
```python
# In WORKSPACE.bazel
http_archive(
    name = "fmt",
    build_file = "//third_party:fmt.BUILD",
    urls = ["https://github.com/fmtlib/fmt/archive/10.1.1.tar.gz"],
)

# In BUILD.bazel
deps = ["@fmt//:fmt"]
```

### Local Dependencies

**CMake:**
```cmake
target_link_libraries(Security PUBLIC Core)
```

**Bazel:**
```python
cc_library(
    name = "Security",
    deps = ["//Library/Core:Core"],
)
```

## Feature Flags as Defines

### CMake
```cmake
if(XSIGMA_ENABLE_CUDA)
    target_compile_definitions(Core PUBLIC XSIGMA_ENABLE_CUDA)
endif()
```

### Bazel
```python
# Automatic via xsigma_defines() function
cc_library(
    name = "Core",
    defines = xsigma_defines(),  # Includes XSIGMA_ENABLE_CUDA when --config=cuda
)
```

## Generated Headers

### CMake
```cmake
configure_file(
    xsigma_version_macros.h.in
    xsigma_version_macros.h
    @ONLY
)
```

### Bazel
```python
genrule(
    name = "gen_version_macros",
    outs = ["xsigma_version_macros.h"],
    cmd = """
cat > $@ << 'EOF'
#define XSIGMA_VERSION_FULL "1.0.0"
EOF
    """,
)
```

## Common Build Commands

| Task | CMake | Bazel |
|------|-------|-------|
| Configure | `cmake -B build` | N/A (automatic) |
| Build all | `cmake --build build` | `bazel build //...` |
| Build library | `cmake --build build --target Core` | `bazel build //Library/Core:Core` |
| Run tests | `ctest --test-dir build` | `bazel test //...` |
| Clean | `rm -rf build` | `bazel clean` |
| Release build | `cmake -B build -DCMAKE_BUILD_TYPE=Release` | `bazel build --config=release //...` |

## Migration Checklist

If you're familiar with CMake and learning Bazel:

- [x] `CMakeLists.txt` → Create `BUILD.bazel` in each directory
- [x] `option()` → Create `config_setting()` in `bazel/BUILD.bazel`
- [x] `find_package()` → Add `http_archive()` in `WORKSPACE.bazel`
- [x] `target_compile_options()` → Add flags to `.bazelrc` or `xsigma_copts()`
- [x] `target_compile_definitions()` → Add to `xsigma_defines()`
- [x] `target_link_libraries()` → Add to `deps` attribute
- [x] `configure_file()` → Create `genrule()`
- [x] `if(CONDITION)` → Use `select({...})`

## Benefits of Bazel Over CMake

1. **Hermetic builds**: All dependencies explicitly declared
2. **Incremental builds**: Only rebuilds what's necessary
3. **Remote caching**: Share artifacts across machines
4. **Reproducible**: Same inputs → same outputs
5. **Scalable**: Handles large codebases efficiently
6. **Cross-platform**: Single build system for all platforms
7. **Parallel**: Better parallelization than Make

## When to Use Each

**Use Bazel when:**
- Building large, multi-language projects
- Need reproducible builds
- Want remote caching
- Building from multiple repositories

**Use CMake when:**
- Integrating with CMake ecosystem
- Using IDE with CMake support
- Existing CMake-based workflow
- Need package management (vcpkg, Conan)

## Further Reading

- [Bazel Documentation](https://bazel.build/)
- [CMake to Bazel Migration Guide](https://bazel.build/migrate/cmake)
- [Bazel C++ Tutorial](https://bazel.build/tutorials/cpp)
- See `BAZEL_BUILD.md` for usage examples
