# Code Coverage Implementation Details

## Architecture Overview

The coverage integration follows a layered architecture:

```
setup.py (User Interface)
    ↓
coverage() method (Main Entry Point)
    ↓
__run_oss_coverage() (Primary Implementation)
    ↓
oss_coverage.py (PyTorch Tool)
    ├── package/oss/init.py (Initialization)
    ├── package/oss/cov_json.py (Report Generation)
    ├── package/tool/clang_coverage.py (Clang Support)
    ├── package/tool/gcc_coverage.py (GCC Support)
    └── package/util/ (Utilities)

Fallback: __run_legacy_coverage() → compute_code_coverage_locally.sh
```

## Key Components

### 1. setup.py Integration

**Main Method: `coverage()`**
- Entry point for coverage workflow
- Tries oss_coverage.py first
- Falls back to legacy script if needed
- Handles error reporting

**Helper Methods**:
- `__run_oss_coverage()`: Executes PyTorch tool
- `__run_legacy_coverage()`: Fallback implementation
- `__check_coverage_reports()`: Verifies report generation

### 2. oss_coverage.py Tool

**Initialization Phase**:
- Detects compiler type (GCC or Clang)
- Creates necessary directories
- Parses command-line arguments
- Sets up environment variables

**Execution Phases**:
- **--run**: Execute tests with coverage instrumentation
- **--merge**: Merge LLVM profiles (Clang only)
- **--export**: Export coverage data to JSON
- **--summary**: Generate human-readable reports

### 3. Compiler-Specific Implementations

**Clang Coverage (package/tool/clang_coverage.py)**:
- Uses LLVM source-based coverage
- Generates `.profraw` files during test execution
- Merges profiles with `llvm-profdata`
- Exports reports with `llvm-cov`

**GCC Coverage (package/tool/gcc_coverage.py)**:
- Uses gcov-based coverage
- Generates `.gcda` files during test execution
- Processes with `gcov` tool
- Optional HTML generation with `lcov`/`genhtml`

### 4. Report Generation

**Report Types**:
1. **JSON Reports**: Structured coverage data
   - Location: `tools/code_coverage/profile/json/`
   - Format: JSON with file and line coverage

2. **Summary Reports**: Human-readable text
   - Location: `tools/code_coverage/profile/summary/`
   - Format: Text with coverage percentages

3. **HTML Reports**: Visual coverage display
   - Location: `tools/code_coverage/profile/html/`
   - Format: Interactive HTML with drill-down

## Data Flow

### Coverage Collection
```
1. CMake Configuration
   └─ XSIGMA_ENABLE_COVERAGE=ON
   └─ Compiler flags added

2. Compilation
   └─ Source files compiled with coverage instrumentation
   └─ Instrumentation code embedded in binaries

3. Test Execution
   └─ Tests run with LLVM_PROFILE_FILE set (Clang)
   └─ Coverage data written to .profraw/.gcda files

4. Profile Merging (Clang)
   └─ llvm-profdata merges .profraw files
   └─ Creates coverage.profdata

5. Report Generation
   └─ llvm-cov/gcov processes coverage data
   └─ Generates JSON, text, and HTML reports
```

## Environment Variables

**LLVM_PROFILE_FILE** (Clang):
- Controls where profraw files are written
- Format: `coverage/default-%p.profraw`
- `%p` replaced with process ID

**COMPILER_TYPE** (oss_coverage.py):
- Auto-detected from build directory name
- Can be overridden via environment
- Values: "clang" or "gcc"

## Configuration Options

### CMake Flags
```cmake
XSIGMA_ENABLE_COVERAGE=ON      # Enable coverage instrumentation
XSIGMA_BUILD_TESTING=ON        # Enable test building
CMAKE_BUILD_TYPE=Debug         # Recommended for coverage
```

### setup.py Arguments
```bash
python setup.py ninja.clang.config.build.test.coverage
                 ^^^^^ ^^^^^ ^^^^^^ ^^^^^ ^^^^ ^^^^^^^^
                 gen   comp  action action action feature
```

## Error Handling

### Graceful Degradation
1. If oss_coverage.py not found → Use legacy script
2. If LLVM tools not found → Provide installation instructions
3. If tests fail → Continue with coverage report generation
4. If report generation fails → Log error and continue

### Error Reporting
- Errors logged to `tools/code_coverage/profile/log/log.txt`
- Summary reporter tracks coverage status
- Exit codes indicate success/failure

## Performance Considerations

### Build Time Impact
- Coverage instrumentation adds ~10-20% to build time
- Compilation flags: `-fprofile-instr-generate -fcoverage-mapping`

### Runtime Impact
- Coverage collection adds ~5-15% to test execution time
- Profile merging: ~1-5 seconds for typical projects
- Report generation: ~5-30 seconds depending on project size

### Disk Space
- Coverage data: ~50-200MB per test run
- Merged profiles: ~10-50MB
- Reports: ~5-20MB

## Integration Points

### CMake Integration
- `Cmake/tools/coverage.cmake`: Compiler flag configuration
- `Library/Core/Testing/Cxx/CMakeLists.txt`: Test environment setup

### Build System Integration
- `Scripts/setup.py`: User-facing interface
- `Scripts/compute_code_coverage_locally.sh`: Legacy fallback

### CI/CD Integration
- Coverage flag: `XSIGMA_ENABLE_COVERAGE`
- Report location: `tools/code_coverage/profile/`
- Analysis tool: `Scripts/analyze_coverage.py`

## Extensibility

### Adding New Report Formats
1. Extend `package/tool/summarize_jsons.py`
2. Add format-specific generator
3. Update report output directory

### Supporting New Compilers
1. Create new compiler module in `package/tool/`
2. Implement coverage collection logic
3. Add compiler detection in `package/oss/utils.py`

### Custom Coverage Filters
1. Modify `package/oss/init.py` argument parsing
2. Add filter logic to report generation
3. Update summary report filtering

## Testing Coverage Integration

### Unit Tests
- Test compiler detection
- Test report parsing
- Test environment variable handling

### Integration Tests
- Full build with coverage
- Report generation verification
- Fallback mechanism testing

### Regression Tests
- Verify existing tests still pass
- Check build performance
- Validate report accuracy

## Maintenance

### Regular Updates
- Monitor PyTorch coverage tool updates
- Update oss_coverage.py when new versions available
- Test with new compiler versions

### Troubleshooting
- Check logs in `tools/code_coverage/profile/log/`
- Verify LLVM tools installation
- Validate CMake configuration

### Documentation
- Keep usage examples current
- Update troubleshooting guide
- Document known issues
