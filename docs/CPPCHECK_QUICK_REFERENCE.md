# Cppcheck Quick Reference Guide

## Quick Start

### Enable Cppcheck with setup.py
```bash
cd Scripts
python setup.py ninja.clang.config.build.cppcheck
```

### Enable Cppcheck with CMake
```bash
cmake -B build -S . -DXSIGMA_ENABLE_CPPCHECK=ON
cmake --build build
```

### Run Standalone Cppcheck Target
```bash
cmake -B build -S . -DXSIGMA_ENABLE_CPPCHECK=ON
cmake --build build --target run_cppcheck
```

## Installation

### Ubuntu/Debian
```bash
sudo apt-get install cppcheck
```

### Fedora/CentOS/RHEL
```bash
sudo dnf install cppcheck
```

### macOS
```bash
brew install cppcheck
```

### Windows
```bash
# Using Chocolatey
choco install cppcheck

# Using winget
winget install cppcheck
```

## Command Parameters

The following command is executed by both setup.py and CMake:

```bash
cppcheck . \
  --platform=unspecified \
  --enable=style \
  -q \
  --library=qt \
  --library=posix \
  --library=gnu \
  --library=bsd \
  --library=windows \
  --check-level=exhaustive \
  --template='{id},{file}:{line},{severity},{message}' \
  --suppressions-list=Scripts/cppcheck_suppressions.txt \
  -j8 \
  -I Library
```

## Suppressions

To suppress specific warnings, edit `Scripts/cppcheck_suppressions.txt`:

```
# Suppress all warnings in ThirdParty directory
*:ThirdParty/*

# Suppress all warnings in test files
*:Library/*/Testing/Cxx/*

# Suppress specific warning type
uninitvar:Library/Core/src/myfile.cxx

# Suppress specific warning on specific line
uninitvar:Library/Core/src/myfile.cxx:42
```

## Common Use Cases

### Development Build with Cppcheck
```bash
cd Scripts
python setup.py ninja.clang.config.build.test.cppcheck
```

### Release Build with Cppcheck
```bash
cd Scripts
python setup.py ninja.clang.release.config.build.cppcheck
```

### Cppcheck Only (No Build)
```bash
# After initial configuration
cmake --build build --target run_cppcheck
```

### Cppcheck with Automatic Fixes (⚠️ WARNING)
```bash
cd Scripts
python setup.py ninja.clang.config.build.cppcheck.cppcheck_autofix
```
**Note**: This will modify source files! Commit your changes first.

## Output Format

Cppcheck output follows this template:
```
{id},{file}:{line},{severity},{message}
```

Example:
```
uninitvar,Library/Core/src/example.cxx:42,error,Uninitialized variable: x
```

## Integration Points

### setup.py Integration
- Runs after the build step
- Only executes when `cppcheck` flag is present
- Runs from project root directory
- Displays command being executed
- Shows all output in terminal

### CMake Integration
- Creates `run_cppcheck` custom target
- Can be run independently of build
- Uses same parameters as setup.py
- Runs from project root directory

## Troubleshooting

### Cppcheck Not Found
**Error**: `cppcheck not found`

**Solution**: Install cppcheck using the installation commands above.

### Suppressions File Not Found
**Error**: Cannot find suppressions file

**Solution**: Ensure you're running from the project root and `Scripts/cppcheck_suppressions.txt` exists.

### Too Many Warnings
**Solution**: Add suppressions to `Scripts/cppcheck_suppressions.txt`

### Slow Analysis
**Solution**: The `-j8` flag uses 8 parallel jobs. Adjust based on your CPU:
- Edit the command in `Scripts/setup.py` (line 763)
- Edit the command in `Cmake/tools/cppcheck.cmake` (line 84)

## Best Practices

1. **Run regularly**: Enable cppcheck during development builds
2. **Review warnings**: Don't ignore all warnings - they often indicate real issues
3. **Use suppressions wisely**: Only suppress false positives
4. **Commit before autofix**: Always commit changes before using `cppcheck_autofix`
5. **CI/CD integration**: Consider running cppcheck in your CI pipeline

## Related Documentation

- [Static Analysis Guide](static-analysis.md) - Comprehensive static analysis documentation
- [Build Configuration](build-configuration.md) - Build system configuration
- [Cppcheck Official Documentation](https://cppcheck.sourceforge.io/)

## Examples

### Example 1: Full Development Workflow
```bash
cd Scripts

# Configure and build with cppcheck
python setup.py ninja.clang.config.build.test.cppcheck

# Review any warnings
# Add suppressions if needed to Scripts/cppcheck_suppressions.txt

# Rebuild
python setup.py ninja.clang.build.cppcheck
```

### Example 2: Quick Check Without Full Build
```bash
# Initial configuration (once)
cmake -B build -S . -DXSIGMA_ENABLE_CPPCHECK=ON

# Run cppcheck anytime
cmake --build build --target run_cppcheck
```

### Example 3: CI/CD Pipeline
```bash
# In your CI script
cd Scripts
python setup.py ninja.clang.release.config.build.test.cppcheck

# Check exit code
if [ $? -ne 0 ]; then
    echo "Cppcheck found issues"
    exit 1
fi
```

## Advanced Configuration

### Custom Cppcheck Parameters

To modify cppcheck parameters:

1. **For setup.py**: Edit `Scripts/setup.py`, lines 749-766
2. **For CMake**: Edit `Cmake/tools/cppcheck.cmake`, lines 71-85

### Platform-Specific Configuration

The existing CMake integration already handles platform-specific settings:
- Windows: `--platform=win64 --library=windows`
- Unix: `--platform=unix64 --library=posix`

The standalone command uses `--platform=unspecified` for cross-platform consistency.

## Performance Tips

- **Parallel jobs**: Adjust `-j8` based on CPU cores
- **Suppressions**: Use suppressions to skip third-party code
- **Incremental**: Use CMake target for quick checks without full rebuild
- **Selective**: Run on specific directories by modifying the command

## Exit Codes

- `0`: Success (no issues found)
- `1`: Warnings or errors found
- Other: Execution error (cppcheck not found, etc.)

## Support

For issues or questions:
1. Check this guide and [static-analysis.md](static-analysis.md)
2. Review cppcheck output for specific error messages
3. Check [Cppcheck documentation](https://cppcheck.sourceforge.io/)
4. Review suppressions file for examples

