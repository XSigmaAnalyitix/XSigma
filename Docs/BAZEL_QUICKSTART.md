# XSigma Bazel Quick Start

## Installation

```bash
# macOS
brew install bazelisk

# Linux
npm install -g @bazel/bazelisk

# Or download from https://github.com/bazelbuild/bazelisk/releases
```

## Build Examples

```bash
# Basic build (all libraries)
bazel build //...

# Release build with optimizations
bazel build --config=release --config=avx2 //...

# With optional features
bazel build --config=release \
            --config=avx2 \
            --config=mimalloc \
            --config=magic_enum \
            //...

# Build specific library
bazel build //Library/Core:Core
bazel build //Library/Security:Security

# Run tests
bazel test //...
```

## Most Common Configurations

```bash
# Production build
bazel build --config=release --config=lto --config=avx2 //...

# Development build  
bazel build --config=debug //...

# With sanitizers
bazel build --config=debug --config=asan //...
```

## Equivalent to Current CMake Build

Your current CMake command:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
               -DXSIGMA_VECTORIZATION_TYPE=avx2 \
               -DXSIGMA_ENABLE_MIMALLOC=ON \
               -DXSIGMA_ENABLE_MAGICENUM=ON
cmake --build build
```

Bazel equivalent:
```bash
bazel build --config=release \
            --config=avx2 \
            --config=mimalloc \
            --config=magic_enum \
            //...
```

## Create Personal Config

Create `.bazelrc.user` with your preferred settings:
```bash
cat > .bazelrc.user << 'RCEOF'
build --config=release
build --config=avx2
build --config=mimalloc
build --config=magic_enum
RCEOF
```

Then just run:
```bash
bazel build //...
```

## Files Created

- `WORKSPACE.bazel` - Dependency definitions
- `BUILD.bazel` - Root build file
- `.bazelrc` - Build configurations
- `.bazelversion` - Bazel version (6.4.0)
- `bazel/BUILD.bazel` - Config settings
- `bazel/xsigma.bzl` - Helper functions
- `Library/*/BUILD.bazel` - Library build files
- `third_party/*.BUILD` - Third-party dependencies
- `BAZEL_BUILD.md` - Full documentation
- `BAZEL_STRUCTURE.md` - Architecture overview

## Next Steps

1. Install Bazel/Bazelisk
2. Run `bazel build //...` to test
3. Create `.bazelrc.user` with your preferences
4. See `BAZEL_BUILD.md` for complete documentation
5. See `BAZEL_STRUCTURE.md` for architecture details

## Support

- CMake build still works as before
- Both build systems produce equivalent binaries
- Use whichever fits your workflow better
