# Valgrind Report Generation - Quick Reference

## Quick Start

### Run Valgrind Tests with Report Generation

```bash
cd Scripts
./valgrind_ctest.sh ../build_ninja_valgrind
```

### View the Generated Report

```bash
cat ../build_ninja_valgrind/valgrind_summary_report.txt
```

## Report Sections

### 1. Executive Summary
Quick statistics at a glance:
- Total Memory Errors
- Total Bytes Leaked
- Invalid Memory Accesses
- Uninitialized Value Uses
- File Descriptor Leaks

### 2. Memory Leaks Analysis
- **Status**: FOUND or NONE DETECTED
- **Leak Details**: Bytes and block counts
- **Leak Locations**: Stack traces showing where leaks originated

### 3. Invalid Memory Access
- **Status**: FOUND or NONE DETECTED
- **Error Count**: Number of invalid reads/writes
- **Locations**: Stack traces for each error

### 4. Uninitialized Value Usage
- **Status**: FOUND or NONE DETECTED
- **Occurrences**: Total count
- **Sample Locations**: Stack traces for debugging

### 5. File Descriptor Leaks
- **Status**: FOUND or NONE DETECTED
- **Open FDs**: List of unclosed file descriptors

### 6. Error Summary by Test
Per-test breakdown:
- Test name
- Error count for that test

### 7. Overall Status
Final result:
- **PASS ✓**: No memory issues detected
- **FAIL ✗**: Memory issues found

## Console Output

After tests complete, you'll see:

```
========================================
Valgrind Summary Report
========================================

=== EXECUTIVE SUMMARY ===
Total Memory Errors:        0
Total Bytes Leaked:         0 bytes
Invalid Memory Accesses:    0
Uninitialized Value Uses:   0
File Descriptor Leaks:      0

=== OVERALL STATUS ===
Result: PASS ✓
No memory issues detected during test execution.

[i] Full report saved to: /path/to/build_ninja_valgrind/valgrind_summary_report.txt
```

## File Locations

| Item | Location |
|------|----------|
| Summary Report | `<build_dir>/valgrind_summary_report.txt` |
| Raw Valgrind Logs | `<build_dir>/Testing/Temporary/MemoryChecker.*.log` |
| Test Results | `<build_dir>/Testing/Temporary/LastTest.log` |

## Interpreting Results

### PASS Result
```
Result: PASS ✓
No memory issues detected during test execution.
```
✅ All tests passed with no memory issues

### FAIL Result
```
Result: FAIL ✗
Memory issues detected. Please review the details above.
```
❌ Memory issues found - review the report sections above

## Common Issues and Solutions

### Issue: "No Valgrind log files found"
**Cause**: Tests didn't run or Valgrind wasn't invoked
**Solution**: 
- Verify build directory is correct
- Check that tests exist: `ctest -N`
- Ensure Valgrind is installed: `valgrind --version`

### Issue: Tests timed out
**Cause**: Valgrind slows tests 10-50x
**Solution**:
- Increase test timeouts in CMake
- Use `XSIGMA_VALGRIND_TIMEOUT_MULTIPLIER` (default: 20x)

### Issue: Memory leaks detected
**Solution**:
1. Review "Leak Locations" section for stack traces
2. Check the file and line numbers
3. Look for missing `delete` or improper resource cleanup
4. Consider using smart pointers

### Issue: Invalid memory access
**Solution**:
1. Review "Locations" section for stack traces
2. Check for buffer overflows
3. Verify array bounds
4. Look for use-after-free bugs

## Advanced Usage

### View Only Specific Section
```bash
# View only memory leaks
sed -n '/^MEMORY LEAKS ANALYSIS/,/^[A-Z]/p' valgrind_summary_report.txt

# View only invalid access
sed -n '/^INVALID MEMORY ACCESS/,/^[A-Z]/p' valgrind_summary_report.txt
```

### Compare Multiple Reports
```bash
# Generate reports for different builds
./valgrind_ctest.sh ../build_ninja_valgrind
cp ../build_ninja_valgrind/valgrind_summary_report.txt report_v1.txt

# Make changes and regenerate
./valgrind_ctest.sh ../build_ninja_valgrind
cp ../build_ninja_valgrind/valgrind_summary_report.txt report_v2.txt

# Compare
diff report_v1.txt report_v2.txt
```

### Extract Statistics
```bash
# Get just the executive summary
sed -n '/^EXECUTIVE SUMMARY/,/^$/p' valgrind_summary_report.txt
```

## Integration with CI/CD

The report is automatically:
- Generated during test execution
- Saved to a persistent file
- Displayed in console output
- Used to determine exit code (0 = pass, 1 = fail)

For CI/CD pipelines:
```bash
./valgrind_ctest.sh ../build_ninja_valgrind
exit_code=$?

# Archive the report
cp ../build_ninja_valgrind/valgrind_summary_report.txt artifacts/

exit $exit_code
```

## Tips and Best Practices

1. **Regular Testing**: Run Valgrind tests regularly to catch issues early
2. **Review Reports**: Always review the full report, not just the summary
3. **Fix Incrementally**: Fix one issue at a time and re-run tests
4. **Use Stack Traces**: Stack traces show the exact call path to the issue
5. **Check Raw Logs**: For complex issues, examine raw Valgrind logs
6. **Suppress Known Issues**: Use Valgrind suppression files for false positives

## Support

For more information:
- Full documentation: `docs/VALGRIND_REPORT_GENERATION.md`
- Valgrind manual: `man valgrind`
- XSigma Valgrind config: `Cmake/tools/valgrind.cmake`

