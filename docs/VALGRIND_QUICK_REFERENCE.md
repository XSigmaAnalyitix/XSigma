# Valgrind Quick Reference Guide

## Quick Start

### Run Valgrind Tests
```bash
cd Scripts
python3 setup.py config.ninja.clang.valgrind.build.test
```

### Manual Execution
```bash
cd Scripts
./valgrind_ctest.sh ../build_ninja_valgrind
```

## Configuration

### Location
All Valgrind configuration is in: `Cmake/tools/valgrind.cmake`

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `XSIGMA_VALGRIND_TIMEOUT_MULTIPLIER` | 20 | Timeout multiplier for tests under Valgrind |
| `CTEST_TEST_TIMEOUT` | 1800 | Global timeout in seconds (30 minutes) |
| Suppression File | `Scripts/valgrind_suppression.txt` | Known false positives |

### Customize Timeout
```bash
cmake -DXSIGMA_ENABLE_VALGRIND=ON \
      -DXSIGMA_VALGRIND_TIMEOUT_MULTIPLIER=30 \
      ..
```

## Understanding Results

### Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success or timeout without memory issues | ✅ Continue |
| 1 | Memory issues or test failures | ❌ Fix issues |

### Status Messages

#### ✅ Success
```
[✓] No memory leaks detected
[✓] No memory errors detected
[✓] SUCCESS: All tests passed with no memory issues!
```

#### ⚠️ Warning (Still Passes)
```
[!] Some tests timed out
[✓] No memory leaks detected
[✓] No memory errors detected
[i] Consider this a PASS for memory checking purposes
```

#### ❌ Failure
```
[✗] Memory leaks detected!
[✗] FAILED: Memory issues detected by Valgrind
```

## Common Issues

### Tests Timeout
**Quick Fix:**
```bash
# Increase timeout multiplier
cmake -DXSIGMA_VALGRIND_TIMEOUT_MULTIPLIER=30 ..
```

### False Positives
**Quick Fix:**
Add to `Scripts/valgrind_suppression.txt`:
```
{
   <insert_a_suppression_name_here>
   Memcheck:Leak
   fun:malloc
   fun:problematic_function
}
```

### Slow Execution
**Expected:** Valgrind runs 10-50x slower than normal
**Solution:** Be patient or increase timeout multiplier

## File Locations

| File | Purpose |
|------|---------|
| `Cmake/tools/valgrind.cmake` | Configuration |
| `Scripts/valgrind_ctest.sh` | Execution script |
| `Scripts/valgrind_suppression.txt` | Suppressions |
| `docs/VALGRIND_SETUP.md` | Full documentation |
| `docs/VALGRIND_REFACTORING.md` | Technical details |

## Valgrind Options

All options are in `valgrind.cmake`:

```cmake
--tool=memcheck              # Memory error detector
--leak-check=full            # Detailed leak info
--show-leak-kinds=all        # Show all leak types
--track-origins=yes          # Track uninitialized values
--track-fds=yes              # Track file descriptors
--error-exitcode=1           # Exit with error on issues
--verbose                    # Detailed output
--num-callers=50             # Stack trace depth
--trace-children=yes         # Trace child processes
```

## Timeout Calculation

```
Normal Test Timeout: 300 seconds
Multiplier: 20x
Valgrind Timeout: 300 × 20 = 6000 seconds (100 minutes)
```

## Platform Support

| Platform | Support | Notes |
|----------|---------|-------|
| Linux x86_64 | ✅ Full | Recommended |
| Linux ARM64 | ⚠️ Limited | Valgrind 3.19+ |
| macOS Intel | ⚠️ Limited | May have issues |
| macOS ARM64 | ❌ None | Use sanitizers |

## Alternatives to Valgrind

### AddressSanitizer (ASan)
```bash
python3 setup.py config.ninja.clang.test --sanitizer.address
```

### LeakSanitizer (LSan)
```bash
python3 setup.py config.ninja.clang.test --sanitizer.leak
```

### UndefinedBehaviorSanitizer (UBSan)
```bash
python3 setup.py config.ninja.clang.test --sanitizer.undefined
```

### ThreadSanitizer (TSan)
```bash
python3 setup.py config.ninja.clang.test --sanitizer.thread
```

## Viewing Logs

### Find Logs
```bash
ls -lh build_ninja_valgrind/Testing/Temporary/MemoryChecker.*.log
```

### Search for Errors
```bash
grep -i "error\|leak" build_ninja_valgrind/Testing/Temporary/MemoryChecker.*.log
```

### View Specific Log
```bash
cat build_ninja_valgrind/Testing/Temporary/MemoryChecker.1.log
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Install Valgrind
  run: sudo apt-get install -y valgrind

- name: Build and Test with Valgrind
  run: |
    cd Scripts
    python3 setup.py config.ninja.clang.valgrind.build.test
```

### Custom Timeout for CI
```yaml
- name: Configure with Valgrind
  run: |
    cmake -DXSIGMA_ENABLE_VALGRIND=ON \
          -DXSIGMA_VALGRIND_TIMEOUT_MULTIPLIER=25 \
          ..
```

## Memory Leak Types

| Type | Severity | Action |
|------|----------|--------|
| Definitely Lost | 🔴 Critical | Must fix |
| Indirectly Lost | 🔴 Critical | Must fix |
| Possibly Lost | 🟡 Warning | Investigate |
| Still Reachable | 🟢 Info | Optional |

## Best Practices

1. ✅ Run Valgrind regularly, not just before releases
2. ✅ Fix memory issues promptly
3. ✅ Use suppressions only for third-party false positives
4. ✅ Document all suppressions with comments
5. ✅ Combine Valgrind with sanitizers for full coverage
6. ✅ Monitor CI/CD Valgrind runs
7. ✅ Increase timeouts for complex tests

## Troubleshooting Commands

### Check Valgrind Installation
```bash
valgrind --version
```

### Test Script Syntax
```bash
bash -n Scripts/valgrind_ctest.sh
```

### Verify CMake Configuration
```bash
cd build_ninja_valgrind
cmake -L | grep VALGRIND
```

### List Tests with Timeouts
```bash
cd build_ninja_valgrind
ctest -N
```

### Run Single Test with Valgrind
```bash
cd build_ninja_valgrind
ctest -R CoreCxxTests -T memcheck -V
```

## Getting Help

1. **Quick Issues**: Check this guide
2. **Usage Questions**: See `docs/VALGRIND_SETUP.md`
3. **Technical Details**: See `docs/VALGRIND_REFACTORING.md`
4. **Bug Reports**: Contact development team

## Key Takeaways

- ⏱️ **Timeouts are automatic**: 20x multiplier by default
- 🎯 **Timeouts ≠ Failures**: Timeout without memory issues = PASS
- 📝 **Configuration centralized**: All settings in `valgrind.cmake`
- 🔍 **Memory issues = Failure**: Any leak or error causes failure
- 🛠️ **Customizable**: Adjust timeout multiplier as needed
- 📊 **Intelligent reporting**: Clear distinction between warnings and errors

## Quick Checklist

Before running Valgrind tests:
- [ ] Valgrind installed
- [ ] Build configured with `-DXSIGMA_ENABLE_VALGRIND=ON`
- [ ] Sufficient disk space for logs
- [ ] Adequate time (tests run 10-50x slower)

After running Valgrind tests:
- [ ] Check exit code
- [ ] Review status messages
- [ ] Examine logs if failures
- [ ] Fix memory issues if detected
- [ ] Update suppressions if needed

