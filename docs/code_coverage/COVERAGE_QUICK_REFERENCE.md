# Code Coverage Quick Reference

## Installation

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install llvm clang ninja-build cmake

# macOS
brew install llvm ninja cmake

# Windows
choco install llvm ninja cmake
```

## Quick Start

### Build with Coverage (Clang)
```bash
cd Scripts
python setup.py ninja.clang.config.build.test.coverage
```

### Build with Coverage (GCC)
```bash
cd Scripts
python setup.py ninja.gcc.config.build.test.coverage
```

### Re-analyze Coverage
```bash
cd Scripts
python setup.py analyze
```

## Report Locations

### Primary Reports (oss_coverage.py)
```
tools/code_coverage/profile/summary/    # Text summaries
tools/code_coverage/profile/json/       # JSON reports
tools/code_coverage/profile/html/       # HTML reports
tools/code_coverage/profile/log/        # Execution logs
```

### Fallback Reports (Legacy)
```
build_ninja_*/coverage/                 # Coverage data
build_ninja_*/coverage_report/          # Reports
build_ninja_*/coverage_report/html/     # HTML reports
```

## View Reports

### HTML Report
```bash
# Linux
xdg-open tools/code_coverage/profile/html/index.html

# macOS
open tools/code_coverage/profile/html/index.html

# Windows
start tools/code_coverage/profile/html/index.html
```

### Text Summary
```bash
cat tools/code_coverage/profile/summary/*.txt
```

### JSON Report
```bash
python -m json.tool tools/code_coverage/profile/json/*.json
```

## Common Commands

### Full Workflow
```bash
cd Scripts
python setup.py ninja.clang.config.build.test.coverage
```

### Just Configuration
```bash
cd Scripts
python setup.py ninja.clang.config.coverage
```

### Just Build
```bash
cd Scripts
python setup.py ninja.clang.build
```

### Just Tests
```bash
cd Scripts
python setup.py ninja.clang.test
```

### Just Coverage
```bash
cd Scripts
python setup.py ninja.clang.coverage
```

### Just Analysis
```bash
cd Scripts
python setup.py analyze
```

## Troubleshooting

### LLVM Tools Not Found
```bash
# Install LLVM
sudo apt-get install llvm  # Ubuntu/Debian
brew install llvm          # macOS

# Or set paths
export LLVM_COV_PATH=/usr/bin/llvm-cov
export LLVM_PROFDATA_PATH=/usr/bin/llvm-profdata
```

### No Coverage Data
```bash
# Check for profraw files
find build_ninja_* -name "*.profraw"

# Check for gcda files
find build_ninja_* -name "*.gcda"

# Run tests manually
cd build_ninja_* && ctest --verbose
```

### oss_coverage.py Not Found
```bash
# Verify file exists
ls -la tools/code_coverage/oss_coverage.py

# Check Python version
python --version  # Should be 3.7+
```

### Report Generation Failed
```bash
# Check logs
cat tools/code_coverage/profile/log/log.txt

# Try fallback
# System will automatically use legacy script
```

## Environment Variables

### LLVM Coverage (Clang)
```bash
export LLVM_PROFILE_FILE=coverage/default-%p.profraw
export LLVM_COV_PATH=/path/to/llvm-cov
export LLVM_PROFDATA_PATH=/path/to/llvm-profdata
```

### Build Directory
```bash
export XSIGMA_BUILD_DIR=/path/to/build
```

### Coverage Output
```bash
export XSIGMA_COVERAGE_OUTPUT=/path/to/output
```

## Build Flags

### Enable Coverage
```bash
-DXSIGMA_ENABLE_COVERAGE=ON
```

### Enable Testing
```bash
-DXSIGMA_BUILD_TESTING=ON
```

### Debug Build (Recommended)
```bash
-DCMAKE_BUILD_TYPE=Debug
```

## File Structure

```
tools/code_coverage/
├── oss_coverage.py              # Main entry point
├── README.md                    # Documentation
├── package/
│   ├── oss/                     # OSS implementation
│   │   ├── init.py
│   │   ├── cov_json.py
│   │   ├── run.py
│   │   └── utils.py
│   ├── tool/                    # Compiler tools
│   │   ├── clang_coverage.py
│   │   ├── gcc_coverage.py
│   │   ├── summarize_jsons.py
│   │   └── parser/
│   └── util/                    # Utilities
│       ├── setting.py
│       ├── utils.py
│       └── utils_init.py
└── profile/                     # Reports
    ├── json/                    # JSON reports
    ├── merged/                  # Merged profiles
    ├── summary/                 # Text summaries
    ├── html/                    # HTML reports
    ├── raw/                     # Raw coverage data
    └── log/                     # Execution logs
```

## Performance Tips

### Faster Builds
```bash
# Use Release build
-DCMAKE_BUILD_TYPE=Release

# Use more parallel jobs
ninja -j 16
```

### Reduce Disk Usage
```bash
# Clean old coverage data
rm -rf tools/code_coverage/profile/*

# Clean build directory
rm -rf build_ninja_*
```

### Faster Coverage Analysis
```bash
# Analyze specific files
python setup.py analyze --library-filter Library/Core/specific_module
```

## Documentation

- **Integration Summary**: `COVERAGE_INTEGRATION_SUMMARY.md`
- **Test Plan**: `COVERAGE_TEST_PLAN.md`
- **Verification Guide**: `COVERAGE_VERIFICATION_GUIDE.md`
- **Implementation Details**: `COVERAGE_IMPLEMENTATION_DETAILS.md`
- **Complete Status**: `COVERAGE_INTEGRATION_COMPLETE.md`

## Support Resources

1. Check logs: `tools/code_coverage/profile/log/log.txt`
2. Review documentation in `docs/`
3. Check PyTorch coverage tool documentation
4. Verify LLVM tools installation

## Key Metrics

- **Build Time Impact**: +10-20%
- **Test Time Impact**: +5-15%
- **Disk Space**: 50-200MB per run
- **Report Generation**: 5-30 seconds

## Success Indicators

✓ Coverage data files generated (.profraw or .gcda)
✓ Summary reports created in profile/summary/
✓ JSON reports valid and readable
✓ HTML report displays in browser
✓ Analysis completes without errors
✓ No critical errors in logs

## Next Steps

1. Run first coverage build
2. Verify report generation
3. Review HTML report
4. Run coverage analysis
5. Integrate into CI/CD
