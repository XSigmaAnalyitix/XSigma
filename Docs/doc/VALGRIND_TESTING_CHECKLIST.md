# Valgrind Refactoring - Testing Checklist

## Pre-Testing Verification

### ✅ Code Quality Checks
- [x] Shell script syntax validation (`bash -n Scripts/valgrind_ctest.sh`)
- [x] CMake syntax validation (no diagnostics)
- [x] Documentation created and reviewed
- [ ] Code review completed

### ✅ File Integrity
- [x] `Cmake/tools/valgrind.cmake` - Enhanced with timeout configuration
- [x] `Scripts/valgrind_ctest.sh` - Refactored with modular functions
- [x] `CMakeLists.txt` - Updated with timeout function call
- [x] `docs/VALGRIND_SETUP.md` - Enhanced documentation
- [x] `docs/VALGRIND_REFACTORING.md` - Technical documentation
- [x] `docs/VALGRIND_QUICK_REFERENCE.md` - Quick reference guide
- [x] `VALGRIND_REFACTORING_SUMMARY.md` - Executive summary

## Build Testing

### Test 1: Clean Build with Valgrind
```bash
cd Scripts
python3 setup.py config.ninja.clang.valgrind.build
```

**Expected:**
- [ ] Build completes successfully
- [ ] No CMake errors
- [ ] Valgrind configuration messages appear
- [ ] Timeout multiplier message shown

**Verification:**
```bash
cd ../build_ninja_valgrind
cmake -L | grep VALGRIND
```

**Expected Output:**
```
XSIGMA_ENABLE_VALGRIND:BOOL=ON
XSIGMA_VALGRIND_TIMEOUT_MULTIPLIER:STRING=20
```

### Test 2: Build and Test with Valgrind
```bash
cd Scripts
python3 setup.py config.ninja.clang.valgrind.build.test
```

**Expected:**
- [ ] Build completes successfully
- [ ] Tests run under Valgrind
- [ ] Script shows progress messages
- [ ] Timeout handling works correctly
- [ ] Exit code is appropriate (0 or 1)

**Check:**
- [ ] No false timeout failures
- [ ] Memory issues detected correctly
- [ ] Status messages are clear

## Functional Testing

### Test 3: Timeout Scaling Verification
```bash
cd build_ninja_valgrind
ctest -N | grep -A 10 CoreCxxTests
```

**Expected:**
- [ ] CoreCxxTests timeout is scaled (should be 6000s if original was 300s)
- [ ] Other tests also have scaled timeouts

**Manual Verification:**
```bash
# Check the test properties
ctest --print-labels
ctest -R CoreCxxTests -V --timeout 10  # Should timeout
```

### Test 4: Script Execution
```bash
cd Scripts
./valgrind_ctest.sh ../build_ninja_valgrind
```

**Expected:**
- [ ] Script runs without errors
- [ ] Platform compatibility check works
- [ ] Valgrind installation check works
- [ ] Tests execute under Valgrind
- [ ] Results are analyzed correctly
- [ ] Status determination is intelligent

**Check Exit Codes:**
```bash
echo $?  # Should be 0 if no memory issues, 1 if issues detected
```

### Test 5: Timeout Handling
**Scenario:** Test times out but no memory issues

**Expected Behavior:**
- [ ] Script detects timeout
- [ ] Script checks for memory issues
- [ ] No memory issues found
- [ ] Status: WARNING (not ERROR)
- [ ] Exit code: 0 (success)
- [ ] Message: "Consider this a PASS for memory checking purposes"

**Verification:**
Check the script output for:
```
[!] Some tests timed out
[✓] No memory leaks detected
[✓] No memory errors detected
[!] WARNING: Tests timed out but no memory issues detected
[i] Consider this a PASS for memory checking purposes
```

### Test 6: Memory Issue Detection
**Scenario:** Actual memory leak exists

**Expected Behavior:**
- [ ] Script detects memory leak
- [ ] Detailed leak information shown
- [ ] Status: ERROR
- [ ] Exit code: 1 (failure)
- [ ] Clear error messages

### Test 7: Custom Timeout Multiplier
```bash
cd build_ninja_valgrind
cmake -DXSIGMA_VALGRIND_TIMEOUT_MULTIPLIER=30 ..
make
cd ../Scripts
./valgrind_ctest.sh ../build_ninja_valgrind
```

**Expected:**
- [ ] Configuration accepts custom multiplier
- [ ] Timeouts are scaled by 30x instead of 20x
- [ ] Tests have more time to complete

## Platform Testing

### Test 8: Linux (Primary Platform)
**Platform:** Ubuntu/Debian/Fedora

- [ ] Valgrind installs correctly
- [ ] Build succeeds
- [ ] Tests run successfully
- [ ] Timeout scaling works
- [ ] Script functions correctly

### Test 9: macOS Intel (If Available)
**Platform:** macOS x86_64

- [ ] Warning about limited support shown
- [ ] Valgrind installs (if available)
- [ ] Build succeeds
- [ ] Tests run (may have issues)
- [ ] Script handles platform correctly

### Test 10: macOS ARM64 (If Available)
**Platform:** macOS Apple Silicon

- [ ] Warning about no support shown
- [ ] Script suggests alternatives (sanitizers)
- [ ] Build fails gracefully or continues with warning
- [ ] Clear guidance provided

## Edge Cases

### Test 11: No Valgrind Installed
```bash
# Temporarily rename valgrind
sudo mv /usr/bin/valgrind /usr/bin/valgrind.bak

cd Scripts
./valgrind_ctest.sh ../build_ninja_valgrind

# Restore
sudo mv /usr/bin/valgrind.bak /usr/bin/valgrind
```

**Expected:**
- [ ] Script detects missing Valgrind
- [ ] Clear error message
- [ ] Installation instructions shown
- [ ] Exit code: 1

### Test 12: Invalid Build Directory
```bash
cd Scripts
./valgrind_ctest.sh /nonexistent/directory
```

**Expected:**
- [ ] Script detects invalid directory
- [ ] Clear error message
- [ ] Exit code: 1

### Test 13: No Tests Registered
**Scenario:** Build without tests

**Expected:**
- [ ] Script handles gracefully
- [ ] Warning about no tests
- [ ] Appropriate exit code

## Integration Testing

### Test 14: CI/CD Simulation
```bash
# Simulate CI environment
export CI=true

cd Scripts
python3 setup.py config.ninja.clang.valgrind.build.test

# Check exit code
if [ $? -eq 0 ]; then
    echo "CI would pass"
else
    echo "CI would fail"
fi
```

**Expected:**
- [ ] Build succeeds in CI-like environment
- [ ] Tests run correctly
- [ ] Exit codes are appropriate
- [ ] Logs are accessible

### Test 15: Suppression File
```bash
# Verify suppression file is used
cd Scripts
./valgrind_ctest.sh ../build_ninja_valgrind 2>&1 | grep -i suppression
```

**Expected:**
- [ ] Suppression file is detected
- [ ] Message confirms usage
- [ ] Suppressions are applied

## Documentation Testing

### Test 16: Documentation Accuracy
- [ ] All commands in documentation work
- [ ] Examples are correct
- [ ] File paths are accurate
- [ ] Configuration options are valid

### Test 17: Quick Reference Guide
- [ ] All quick commands work
- [ ] Troubleshooting steps are effective
- [ ] Configuration examples are correct

## Performance Testing

### Test 18: Timeout Adequacy
**Measure:** Time taken for CoreCxxTests under Valgrind

```bash
cd build_ninja_valgrind
time ctest -R CoreCxxTests -T memcheck
```

**Expected:**
- [ ] Test completes within timeout
- [ ] Timeout is not too generous (wastes time)
- [ ] Timeout is not too strict (causes false failures)

**Adjust if needed:**
- If tests consistently timeout: Increase multiplier
- If tests finish with lots of time left: Decrease multiplier

## Regression Testing

### Test 19: Existing Workflows
**Verify existing workflows still work:**

```bash
# Standard build
cd Scripts
python3 setup.py config.ninja.clang.build.test

# Coverage build
python3 setup.py config.ninja.clang.coverage.build.test

# Sanitizer build
python3 setup.py config.ninja.clang.test --sanitizer.address
```

**Expected:**
- [ ] All existing workflows unaffected
- [ ] No regressions introduced
- [ ] Backward compatibility maintained

## Final Verification

### Test 20: Complete End-to-End Test
```bash
# Clean slate
rm -rf build_ninja_valgrind

# Full workflow
cd Scripts
python3 setup.py config.ninja.clang.valgrind.build.test

# Verify results
echo "Exit code: $?"
ls -lh ../build_ninja_valgrind/Testing/Temporary/
```

**Expected:**
- [ ] Complete workflow succeeds
- [ ] All components work together
- [ ] Results are accurate
- [ ] Logs are generated
- [ ] Exit code is correct

## Sign-Off Checklist

### Code Quality
- [ ] No syntax errors
- [ ] No linting issues
- [ ] Code follows project standards
- [ ] Comments are clear and helpful

### Functionality
- [ ] Timeout issue fixed
- [ ] Configuration centralized
- [ ] Status reporting is intelligent
- [ ] All features work as designed

### Documentation
- [ ] User documentation complete
- [ ] Technical documentation complete
- [ ] Quick reference available
- [ ] Examples are accurate

### Testing
- [ ] All tests pass
- [ ] Edge cases handled
- [ ] Platform compatibility verified
- [ ] No regressions

### Deployment
- [ ] Changes are backward compatible
- [ ] Migration path is clear
- [ ] Team is notified
- [ ] CI/CD is updated (if needed)

## Post-Deployment Monitoring

### Week 1
- [ ] Monitor first Valgrind runs
- [ ] Check for unexpected timeouts
- [ ] Verify exit codes are correct
- [ ] Collect user feedback

### Week 2-4
- [ ] Review timeout multiplier effectiveness
- [ ] Adjust if needed
- [ ] Document any issues
- [ ] Update documentation based on feedback

## Notes

**Testing Environment:**
- OS: _______________
- Valgrind Version: _______________
- CMake Version: _______________
- Compiler: _______________

**Test Results:**
- Date: _______________
- Tester: _______________
- Overall Status: _______________
- Issues Found: _______________

**Recommendations:**
- _______________________________________________
- _______________________________________________
- _______________________________________________
