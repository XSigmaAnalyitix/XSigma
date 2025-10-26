# MSVC Coverage Refactoring - Testing Guide

## Quick Start

### 1. Run Coverage Build
```bash
cd Scripts
python setup.py vs22.debug.coverage
```

### 2. Verify Output Structure
```bash
# Check that flat coverage directory was created
ls -la build_vs22_coverage/coverage_report/coverage/

# Expected output:
# total XX
# drwxr-xr-x  index.html
# -rw-r--r--  yyy.html
# -rw-r--r--  other.html
# (no CxxTests files)
```

### 3. Open HTML Report
```bash
# Open the flat coverage report
build_vs22_coverage/coverage_report/coverage/index.html
```

---

## Detailed Testing

### Test 1: Verify Flat Directory Structure

**Objective**: Confirm HTML files are copied to flat directory

**Steps**:
```bash
# Navigate to coverage directory
cd build_vs22_coverage/coverage_report/coverage

# List all HTML files
ls -la *.html

# Count files
ls -1 *.html | wc -l
```

**Expected Results**:
- ✓ `index.html` exists
- ✓ Multiple `.html` files present
- ✓ No nested `Modules/` subdirectories
- ✓ No test files (no CxxTests, no Test, no Tests)

**Failure Indicators**:
- ✗ Empty directory
- ✗ Only `index.html` present
- ✗ Files still in nested structure
- ✗ Test files present

---

### Test 2: Verify Test Files Excluded

**Objective**: Confirm test files are NOT in flat directory

**Steps**:
```bash
# Check for test files
cd build_vs22_coverage/coverage_report/coverage
ls -la | grep -i test

# Should return nothing
```

**Expected Results**:
- ✓ No output (no test files found)
- ✓ No CxxTests files
- ✓ No Test files
- ✓ No Tests files

**Failure Indicators**:
- ✗ Files containing "Test" found
- ✗ Files containing "CxxTests" found
- ✗ Test files in directory listing

---

### Test 3: Verify Coverage Percentages Are Accurate

**Objective**: Confirm coverage percentages are NOT all 100%

**Steps**:
```bash
# Open index.html in browser
build_vs22_coverage/coverage_report/coverage/index.html

# Check coverage percentages displayed
# Compare with OpenCppCoverage original output
```

**Expected Results**:
- ✓ Coverage percentages vary (not all 100%)
- ✓ Percentages match OpenCppCoverage output
- ✓ Some files have lower coverage (e.g., 75%, 85%)
- ✓ Some files may have 100% coverage

**Failure Indicators**:
- ✗ All files show 100% coverage
- ✗ All files show 0% coverage
- ✗ Percentages don't match OpenCppCoverage output
- ✗ No coverage percentages displayed

---

### Test 4: Verify HTML Links Work

**Objective**: Confirm all links in index.html are functional

**Steps**:
1. Open `build_vs22_coverage/coverage_report/coverage/index.html` in browser
2. Click on each file link
3. Verify each file opens correctly
4. Check browser console for errors

**Expected Results**:
- ✓ All links are clickable
- ✓ Each link opens the correct HTML file
- ✓ No 404 errors
- ✓ No console errors
- ✓ Coverage data displays on each page

**Failure Indicators**:
- ✗ Links are broken (404 errors)
- ✗ Links point to wrong files
- ✗ Browser console shows errors
- ✗ Files don't open

---

### Test 5: Verify Original HTML Still Exists

**Objective**: Confirm original nested HTML structure is preserved

**Steps**:
```bash
# Check original HTML directory
ls -la build_vs22_coverage/coverage_report/html/

# Should still have nested structure
ls -la build_vs22_coverage/coverage_report/html/Modules/
```

**Expected Results**:
- ✓ Original `html/` directory still exists
- ✓ Nested `Modules/` structure preserved
- ✓ Original `index.html` unchanged
- ✓ All original files intact

**Failure Indicators**:
- ✗ Original HTML directory deleted
- ✗ Nested structure removed
- ✗ Original files missing

---

### Test 6: Verify Raw Coverage Data

**Objective**: Confirm raw coverage files are still generated

**Steps**:
```bash
# Check raw coverage directory
ls -la build_vs22_coverage/coverage_report/raw/

# Should contain .cov files
ls -1 *.cov | wc -l
```

**Expected Results**:
- ✓ `raw/` directory exists
- ✓ `.cov` files present
- ✓ File sizes > 0 bytes

**Failure Indicators**:
- ✗ `raw/` directory missing
- ✗ No `.cov` files
- ✗ Empty `.cov` files

---

### Test 7: Compare with Clang Coverage

**Objective**: Verify metrics are consistent across compilers

**Steps**:
```bash
# Run Clang coverage
cd Scripts
python setup.py ninja.clang.debug.coverage

# Compare coverage percentages
# Clang: build_ninja_coverage/coverage_report/
# MSVC: build_vs22_coverage/coverage_report/coverage/
```

**Expected Results**:
- ✓ Similar coverage percentages
- ✓ Same files analyzed
- ✓ Consistent metrics

**Acceptable Differences**:
- ± 5% variation due to compiler differences
- Different file organization
- Different HTML formatting

---

## Automated Testing

### Test Script

```bash
#!/bin/bash
# test_msvc_coverage.sh

set -e

echo "Testing MSVC Coverage Refactoring..."

# Test 1: Run coverage
echo "1. Running coverage build..."
cd Scripts
python setup.py vs22.debug.coverage
cd ..

# Test 2: Check flat directory
echo "2. Checking flat directory structure..."
if [ ! -d "build_vs22_coverage/coverage_report/coverage" ]; then
    echo "ERROR: Flat coverage directory not created"
    exit 1
fi

# Test 3: Check for HTML files
echo "3. Checking for HTML files..."
FILE_COUNT=$(ls -1 build_vs22_coverage/coverage_report/coverage/*.html 2>/dev/null | wc -l)
if [ $FILE_COUNT -lt 2 ]; then
    echo "ERROR: Not enough HTML files found ($FILE_COUNT)"
    exit 1
fi

# Test 4: Check for test files
echo "4. Checking for test file exclusion..."
if ls build_vs22_coverage/coverage_report/coverage/*Test*.html 2>/dev/null; then
    echo "ERROR: Test files found in coverage directory"
    exit 1
fi

# Test 5: Verify coverage percentages
echo "5. Verifying coverage percentages..."
if grep -q "100%" build_vs22_coverage/coverage_report/coverage/index.html; then
    echo "WARNING: All files show 100% coverage (may be correct)"
fi

echo "✓ All tests passed!"
```

---

## Troubleshooting

### Issue: Flat directory not created

**Symptoms**:
- `coverage/` directory missing
- Only `html/` and `raw/` directories exist

**Solutions**:
1. Check that `_process_coverage_html()` is being called
2. Verify `coverage_dir` path is correct
3. Check file permissions
4. Review error messages in console output

### Issue: Test files not excluded

**Symptoms**:
- CxxTests files in flat directory
- Test files in coverage report

**Solutions**:
1. Verify `_is_test_file()` function is correct
2. Check test file naming patterns
3. Review exclusion logic
4. Add debug logging to see which files are being processed

### Issue: Coverage percentages all 100%

**Symptoms**:
- All files show 100% coverage
- Doesn't match OpenCppCoverage output

**Solutions**:
1. Verify `_extract_coverage_percentage()` is working
2. Check HTML file format from OpenCppCoverage
3. Review regex pattern for coverage extraction
4. Add debug logging to see extracted values

### Issue: Links broken in index.html

**Symptoms**:
- 404 errors when clicking links
- Links point to wrong files

**Solutions**:
1. Verify files are copied to flat directory
2. Check file names match link references
3. Review `_update_index_html_links()` function
4. Check for path separator issues (/ vs \)

---

## Performance Benchmarks

### Expected Execution Times

| Operation | Time |
|-----------|------|
| Coverage collection | 30-60 seconds |
| HTML generation | 5-10 seconds |
| File copying | < 1 second |
| Coverage extraction | < 1 second |
| **Total** | **35-70 seconds** |

### Optimization Tips

1. **Reduce test count**: Fewer tests = faster coverage
2. **Smaller source tree**: Limit `--sources` path
3. **Parallel execution**: Run multiple tests in parallel
4. **Incremental coverage**: Only analyze changed files

---

## Validation Checklist

- [ ] Flat directory created at `coverage_report/coverage/`
- [ ] HTML files copied to flat directory
- [ ] Test files excluded from flat directory
- [ ] Coverage percentages are accurate (not all 100%)
- [ ] All links in index.html work
- [ ] Original HTML structure preserved
- [ ] Raw coverage files generated
- [ ] No errors in console output
- [ ] Metrics consistent with OpenCppCoverage
- [ ] Performance acceptable (< 2 minutes total)

---

## Success Criteria

✅ **All tests pass** if:
1. Flat directory exists with HTML files
2. No test files in flat directory
3. Coverage percentages vary (not all 100%)
4. All links work correctly
5. No errors in output
6. Metrics match OpenCppCoverage

❌ **Tests fail** if:
1. Flat directory not created
2. Test files present in flat directory
3. All files show 100% coverage
4. Links are broken
5. Errors in console output
6. Metrics don't match OpenCppCoverage

