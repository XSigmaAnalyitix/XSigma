# Valgrind Setup and Configuration Guide

## Overview

This guide explains how to set up and use Valgrind for memory checking in the XSigma project.

## Platform Support

### ✅ Supported Platforms
- **Linux (x86_64)**: Full support
- **Linux (ARM64)**: Limited support (Valgrind 3.19+)
- **macOS (x86_64/Intel)**: Supported with limitations

### ❌ Unsupported Platforms
- **macOS (ARM64/Apple Silicon)**: **NOT SUPPORTED**
  - Valgrind does not support Apple Silicon architecture
  - Use sanitizers as an alternative (see below)

## Installation

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install valgrind
```

### Linux (Fedora/RHEL/CentOS)
```bash
sudo dnf install valgrind
# or
sudo yum install valgrind
```

### Linux (Arch)
```bash
sudo pacman -S valgrind
```

### macOS (Intel only)
```bash
brew install valgrind
```

**Note**: On macOS, Valgrind support is limited and may not work with recent macOS versions.

## Building with Valgrind Support

### Standard Build
```bash
cd Scripts
python3 setup.py config.ninja.clang.valgrind.test
```

### Build with Specific Configuration
```bash
# Debug build with Valgrind
python3 setup.py config.ninja.clang.valgrind.test.debug

# Release build with Valgrind
python3 setup.py config.ninja.clang.valgrind.test.release
```

## Running Valgrind Tests

### Automatic Execution
When you build with the `valgrind` flag and include `test`, the tests will automatically run with Valgrind:

```bash
cd Scripts
python3 setup.py config.ninja.clang.valgrind.build.test
```

### Manual Execution
You can also run Valgrind tests manually:

```bash
cd Scripts
./valgrind_ctest.sh /path/to/build_directory
```

## Valgrind Configuration

### Suppression File
The project includes a suppression file to filter out known false positives:
- Location: `Cmake/xsigmaValgrindSuppression.txt`
- Automatically used by the build system
- Can be customized for project-specific suppressions

### Valgrind Options
The default configuration includes:
- `--tool=memcheck`: Memory error detector
- `--leak-check=full`: Detailed leak information
- `--show-leak-kinds=all`: Show all types of leaks
- `--track-origins=yes`: Track origins of uninitialized values
- `--gen-suppressions=all`: Generate suppression entries
- `--trace-children=yes`: Trace child processes
- `--show-reachable=yes`: Show reachable memory
- `--num-callers=50`: Stack trace depth

## Interpreting Results

### Memory Leak Types
1. **Definitely Lost**: Memory that is no longer accessible (real leak)
2. **Indirectly Lost**: Memory lost due to definitely lost blocks
3. **Possibly Lost**: Memory that might be leaked (pointer arithmetic)
4. **Still Reachable**: Memory still accessible at exit (not necessarily a leak)

### Exit Codes
- `0`: All tests passed, no memory errors
- `1`: Tests failed or memory errors detected
- Other: System or configuration errors

## Alternatives for Apple Silicon (ARM64)

Since Valgrind doesn't support Apple Silicon, use these alternatives:

### 1. AddressSanitizer (ASan)
Detects memory errors like buffer overflows, use-after-free, etc.

```bash
cd Scripts
python3 setup.py config.ninja.clang.test --sanitizer.address
```

### 2. LeakSanitizer (LSan)
Detects memory leaks (included with ASan on supported platforms).

```bash
cd Scripts
python3 setup.py config.ninja.clang.test --sanitizer.leak
```

### 3. UndefinedBehaviorSanitizer (UBSan)
Detects undefined behavior.

```bash
cd Scripts
python3 setup.py config.ninja.clang.test --sanitizer.undefined
```

### 4. ThreadSanitizer (TSan)
Detects data races and threading issues.

```bash
cd Scripts
python3 setup.py config.ninja.clang.test --sanitizer.thread
```

### 5. Docker with x86_64 Linux
Run Valgrind in a Docker container:

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    clang \
    valgrind \
    git
WORKDIR /workspace
EOF

# Build and run
docker build -t xsigma-valgrind .
docker run -v $(pwd):/workspace -it xsigma-valgrind bash
```

## Troubleshooting

### Issue: "Valgrind not found"
**Solution**: Install Valgrind using the platform-specific instructions above.

### Issue: "Valgrind does not support Apple Silicon"
**Solution**: Use sanitizers instead (see alternatives section).

### Issue: "Too many false positives"
**Solution**: Add suppressions to `Cmake/xsigmaValgrindSuppression.txt`.

### Issue: "Tests run very slowly"
**Solution**: This is normal. Valgrind significantly slows down execution (10-50x slower).

### Issue: "Valgrind crashes on macOS"
**Solution**: 
- Update to the latest Valgrind version
- Try using an older macOS version
- Use Docker with Linux
- Switch to sanitizers

## Performance Considerations

- **Slowdown**: Valgrind typically runs 10-50x slower than native execution
- **Memory Usage**: Valgrind requires significant additional memory
- **CI/CD**: Consider running Valgrind tests separately or on specific platforms

## Best Practices

1. **Regular Testing**: Run Valgrind tests regularly, not just before releases
2. **Fix Issues Promptly**: Address memory leaks and errors as they're discovered
3. **Use Suppressions Wisely**: Only suppress known false positives from third-party libraries
4. **Combine Tools**: Use both Valgrind (on Linux) and sanitizers for comprehensive coverage
5. **Document Suppressions**: Add comments explaining why each suppression is needed

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Valgrind Tests

on: [push, pull_request]

jobs:
  valgrind:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y valgrind cmake ninja-build clang
      - name: Build and test with Valgrind
        run: |
          cd Scripts
          python3 setup.py config.ninja.clang.valgrind.build.test
```

## Additional Resources

- [Valgrind Official Documentation](https://valgrind.org/docs/manual/manual.html)
- [Valgrind Quick Start Guide](https://valgrind.org/docs/manual/quick-start.html)
- [Suppression File Format](https://valgrind.org/docs/manual/manual-core.html#manual-core.suppress)
- [AddressSanitizer Documentation](https://clang.llvm.org/docs/AddressSanitizer.html)

## Support

For issues or questions:
1. Check this documentation
2. Review the Valgrind logs in the build directory
3. Consult the project's issue tracker
4. Contact the development team

