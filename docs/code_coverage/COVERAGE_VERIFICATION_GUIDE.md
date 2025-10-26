# Code Coverage Integration Verification Guide

## Quick Start Verification

### Step 1: Verify Integration Files
```bash
# Check that oss_coverage.py exists
ls -la tools/code_coverage/oss_coverage.py

# Check profile directory structure
ls -la tools/code_coverage/profile/
```

**Expected Output**:
- oss_coverage.py file exists and is readable
- profile/ directory contains: json/, merged/, summary/, log/ subdirectories

### Step 2: Verify setup.py Integration
```bash
# Check that coverage methods are in setup.py
grep -n "__run_oss_coverage\|__run_legacy_coverage\|__check_coverage_reports" Scripts/setup.py
```

**Expected Output**:
- Three new methods found in setup.py
- Methods properly indented and part of XsigmaConfiguration class

### Step 3: Verify CMake Configuration
```bash
# Check coverage.cmake exists
ls -la Cmake/tools/coverage.cmake

# Verify XSIGMA_ENABLE_COVERAGE option
grep -n "XSIGMA_ENABLE_COVERAGE" Cmake/tools/coverage.cmake
```

**Expected Output**:
- coverage.cmake file exists
- XSIGMA_ENABLE_COVERAGE option defined
- Compiler-specific flags configured

## Build Verification

### Step 4: Configure Build with Coverage
```bash
cd Scripts
python setup.py ninja.clang.config.coverage
```

**Expected Output**:
```
[INFO] Starting build configuration for [OS]
[INFO] Build directory: [path]/build_ninja_coverage
[SUCCESS] Build configured successfully
```

**Verification**:
- Build directory created
- CMakeCache.txt contains `XSIGMA_ENABLE_COVERAGE:BOOL=ON`
- No configuration errors

### Step 5: Build with Coverage
```bash
cd Scripts
python setup.py ninja.clang.build
```

**Expected Output**:
- Compilation succeeds
- Coverage flags visible in compiler output
- No build errors

**Verification**:
```bash
# Check for coverage flags in build output
grep -i "fprofile-instr-generate\|fcoverage-mapping" build_ninja_coverage/CMakeFiles/*/flags.make
```

### Step 6: Run Tests with Coverage
```bash
cd Scripts
python setup.py ninja.clang.test
```

**Expected Output**:
- All tests pass
- Coverage data collected
- No test failures

**Verification**:
```bash
# Check for profraw files (Clang)
find build_ninja_coverage -name "*.profraw" | head -5

# Check for gcda files (GCC)
find build_ninja_coverage -name "*.gcda" | head -5
```

## Coverage Report Verification

### Step 7: Generate Coverage Reports
```bash
cd Scripts
python setup.py ninja.clang.coverage
```

**Expected Output**:
```
[INFO] Starting code coverage collection and report generation...
[INFO] Using PyTorch coverage tool: [path]/tools/code_coverage/oss_coverage.py
[SUCCESS] oss_coverage.py completed successfully
[SUCCESS] Coverage summary reports generated in [path]/tools/code_coverage/profile/summary/
```

### Step 8: Verify Report Files
```bash
# Check summary reports
ls -la tools/code_coverage/profile/summary/

# Check JSON reports
ls -la tools/code_coverage/profile/json/

# Check merged profiles
ls -la tools/code_coverage/profile/merged/

# Check logs
ls -la tools/code_coverage/profile/log/
```

**Expected Output**:
- Summary directory contains report files
- JSON directory contains coverage data
- Merged directory contains profdata files
- Log directory contains execution logs

### Step 9: Verify Report Content
```bash
# Check summary report content
cat tools/code_coverage/profile/summary/*.txt | head -50

# Check JSON report structure
python -m json.tool tools/code_coverage/profile/json/*.json | head -50
```

**Expected Output**:
- Summary reports contain coverage percentages
- JSON reports contain structured coverage data
- Files and line coverage information present

## HTML Report Verification

### Step 10: Check for HTML Reports
```bash
# Look for HTML reports
find tools/code_coverage/profile -name "*.html" -o -name "index.html"

# Alternative: Check legacy location
find build_ninja_coverage -name "index.html" -path "*/coverage_report/*"
```

**Expected Output**:
- HTML report file found
- Report path displayed

### Step 11: View HTML Report
```bash
# Open HTML report in browser
# Linux
xdg-open tools/code_coverage/profile/html/index.html

# macOS
open tools/code_coverage/profile/html/index.html

# Windows
start tools/code_coverage/profile/html/index.html
```

**Expected Output**:
- HTML report opens in default browser
- Coverage summary displayed
- File-level coverage visible
- Line-level coverage details available

## Analysis Verification

### Step 12: Run Coverage Analysis
```bash
cd Scripts
python setup.py analyze
```

**Expected Output**:
```
[INFO] Step 1: Validating LLVM tools...
[INFO] Step 2: Locating build directory...
[INFO] Step 3: Locating coverage data...
[INFO] Step 4: Locating test executables...
[INFO] Step 5: Running llvm-cov report...
[INFO] Step 6: Parsing coverage data...
[INFO] Step 7: Generating summary...
```

### Step 13: Verify Analysis Results
```bash
# Check for files below threshold
grep -i "below\|coverage" build_ninja_coverage/coverage_report/*.txt
```

**Expected Output**:
- Coverage percentages for each file
- Files below threshold identified
- Summary statistics provided

## Fallback Verification

### Step 14: Test Fallback Mechanism
```bash
# Temporarily rename oss_coverage.py
mv tools/code_coverage/oss_coverage.py tools/code_coverage/oss_coverage.py.bak

# Run coverage build
cd Scripts
python setup.py ninja.clang.coverage

# Restore oss_coverage.py
mv tools/code_coverage/oss_coverage.py.bak tools/code_coverage/oss_coverage.py
```

**Expected Output**:
- System detects missing oss_coverage.py
- Falls back to legacy script
- Coverage reports still generated
- No critical errors

## Troubleshooting

### Issue: LLVM tools not found
```bash
# Install LLVM tools
sudo apt-get install llvm  # Ubuntu/Debian
brew install llvm          # macOS

# Or set environment variables
export LLVM_COV_PATH=/path/to/llvm-cov
export LLVM_PROFDATA_PATH=/path/to/llvm-profdata
```

### Issue: No coverage data generated
```bash
# Verify coverage flags in build
grep -r "fprofile-instr-generate" build_ninja_coverage/

# Verify tests actually run
cd build_ninja_coverage && ctest --verbose
```

### Issue: oss_coverage.py fails
```bash
# Check Python version
python --version  # Should be 3.7+

# Check for missing dependencies
python -c "from tools.code_coverage.package.oss.init import initialization"

# Check logs
cat tools/code_coverage/profile/log/log.txt
```

## Success Checklist

- [ ] Integration files verified
- [ ] setup.py modifications confirmed
- [ ] CMake configuration correct
- [ ] Build succeeds with coverage flags
- [ ] Tests pass with coverage enabled
- [ ] Coverage data files generated
- [ ] Summary reports created
- [ ] JSON reports valid
- [ ] HTML report displays correctly
- [ ] Analysis runs successfully
- [ ] Fallback mechanism works
- [ ] No errors in logs

## Next Steps

1. Integrate coverage into CI/CD pipeline
2. Set up automated coverage reporting
3. Configure coverage thresholds
4. Add coverage badges to documentation
5. Monitor coverage trends over time
