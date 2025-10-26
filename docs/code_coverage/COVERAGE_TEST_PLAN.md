# Code Coverage Integration Test Plan

## Test Objectives

1. Verify oss_coverage.py integration works correctly
2. Validate coverage instrumentation in build configuration
3. Confirm HTML report generation
4. Test fallback to legacy coverage script
5. Verify cross-platform compatibility

## Test Environment Setup

### Prerequisites
- CMake 3.15+
- Ninja build system
- Clang or GCC compiler
- Python 3.7+
- LLVM tools (for Clang coverage):
  - `llvm-cov`
  - `llvm-profdata`

### Installation Commands

**Ubuntu/Debian:**
```bash
sudo apt-get install llvm clang ninja-build cmake
```

**macOS:**
```bash
brew install llvm ninja cmake
```

**Windows:**
```bash
choco install llvm ninja cmake
```

## Test Cases

### Test 1: Clang Coverage Build Configuration
**Objective**: Verify build configuration includes coverage flags

**Steps**:
1. Navigate to Scripts directory
2. Run: `python setup.py ninja.clang.config.coverage`
3. Check build directory for CMakeCache.txt
4. Verify `XSIGMA_ENABLE_COVERAGE=ON` in cache

**Expected Results**:
- Build directory created successfully
- CMakeCache.txt contains coverage configuration
- No configuration errors

### Test 2: Clang Coverage Build and Test
**Objective**: Build with coverage and run tests

**Steps**:
1. Run: `python setup.py ninja.clang.config.build.test.coverage`
2. Monitor build output for coverage flags
3. Verify tests execute with coverage instrumentation
4. Check for `.profraw` files in coverage directory

**Expected Results**:
- Build completes successfully
- Tests pass with coverage enabled
- Coverage data files generated
- No compilation errors

### Test 3: Coverage Report Generation
**Objective**: Verify oss_coverage.py generates reports

**Steps**:
1. After successful build, check report directories
2. Verify `tools/code_coverage/profile/summary/` contains reports
3. Check for JSON and text format reports
4. Validate report content

**Expected Results**:
- Summary reports generated
- JSON reports created
- Reports contain coverage data
- No report generation errors

### Test 4: HTML Report Generation
**Objective**: Verify HTML coverage reports are created

**Steps**:
1. Check for HTML reports in profile directory
2. Open HTML report in browser
3. Verify report displays coverage information
4. Check for file-level and line-level coverage

**Expected Results**:
- HTML report file exists
- Report displays correctly in browser
- Coverage metrics visible
- Navigation works properly

### Test 5: Fallback Mechanism
**Objective**: Verify fallback to legacy script works

**Steps**:
1. Temporarily rename oss_coverage.py
2. Run coverage build again
3. Verify fallback to compute_code_coverage_locally.sh
4. Check for coverage reports from legacy script

**Expected Results**:
- System detects missing oss_coverage.py
- Falls back to legacy script
- Coverage reports still generated
- No critical errors

### Test 6: GCC Coverage Build (if available)
**Objective**: Verify GCC coverage support

**Steps**:
1. Run: `python setup.py ninja.gcc.config.build.test.coverage`
2. Monitor for GCC-specific coverage flags
3. Verify `.gcda` files generated
4. Check for gcov reports

**Expected Results**:
- Build succeeds with GCC
- Coverage data collected
- Reports generated
- No compiler errors

### Test 7: Coverage Analysis
**Objective**: Verify coverage analysis tool works

**Steps**:
1. After coverage build, run: `python setup.py analyze`
2. Check for coverage threshold analysis
3. Verify files below threshold identified
4. Check analysis output

**Expected Results**:
- Analysis completes successfully
- Coverage percentages reported
- Files below threshold listed
- Analysis output readable

### Test 8: Cross-Platform Verification
**Objective**: Verify coverage works on different platforms

**Steps**:
1. Run coverage build on Windows
2. Run coverage build on Linux
3. Run coverage build on macOS
4. Compare report formats

**Expected Results**:
- Coverage works on all platforms
- Reports generated consistently
- No platform-specific errors
- Report formats compatible

## Success Criteria

- ✓ All tests pass without errors
- ✓ Coverage reports generated in expected locations
- ✓ HTML reports display correctly
- ✓ Fallback mechanism works
- ✓ Cross-platform compatibility verified
- ✓ No regression in existing functionality

## Troubleshooting

### Issue: LLVM tools not found
**Solution**: Install LLVM tools or set environment variables
```bash
export LLVM_COV_PATH=/path/to/llvm-cov
export LLVM_PROFDATA_PATH=/path/to/llvm-profdata
```

### Issue: No .profraw files generated
**Solution**: Verify tests actually execute and coverage flags are set

### Issue: oss_coverage.py not found
**Solution**: Verify tools/code_coverage directory exists and is accessible

## Regression Testing

After integration, verify:
1. Existing tests still pass
2. Build system still works without coverage flag
3. No performance degradation
4. No new compiler warnings

## Documentation

- See `COVERAGE_INTEGRATION_SUMMARY.md` for integration details
- See `code-coverage.md` for user documentation
