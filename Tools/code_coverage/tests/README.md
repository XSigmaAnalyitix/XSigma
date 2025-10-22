# Code Coverage Tools Test Suite

Comprehensive test suite for the XSigma code coverage tools, ensuring all recent fixes work correctly and preventing regressions.

## Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── README.md                   # This file
├── unit/                       # Unit tests for individual functions
│   ├── __init__.py
│   ├── test_path_handling.py   # Path normalization tests
│   ├── test_coverage_filters.py # Filter logic tests
│   ├── test_error_handling.py  # Error handling and return values
│   └── test_json_parsing.py    # JSON parsing and optimization
├── integration/                # Integration tests for workflows
│   ├── __init__.py
│   ├── test_cross_platform.py  # Cross-platform compatibility
│   └── test_workflow.py        # End-to-end workflow tests
└── fixtures/                   # Test data and mock files
    └── __init__.py
```

## Running Tests

### Prerequisites

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest Tools/code_coverage/tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest Tools/code_coverage/tests/unit/

# Integration tests only
pytest Tools/code_coverage/tests/integration/

# Specific test file
pytest Tools/code_coverage/tests/unit/test_path_handling.py

# Specific test class
pytest Tools/code_coverage/tests/unit/test_path_handling.py::TestPathNormalization

# Specific test function
pytest Tools/code_coverage/tests/unit/test_path_handling.py::TestPathNormalization::test_windows_path_with_backslashes
```

### Run with Coverage Report

```bash
pytest Tools/code_coverage/tests/ --cov=Tools/code_coverage/package --cov-report=html
```

### Run with Verbose Output

```bash
pytest Tools/code_coverage/tests/ -v
```

### Run with Markers

```bash
# Run only fast tests
pytest Tools/code_coverage/tests/ -m "not slow"

# Run only slow tests
pytest Tools/code_coverage/tests/ -m "slow"
```

## Test Coverage

### Unit Tests (4 files, ~150 tests)

#### test_path_handling.py
- Windows path normalization
- Unix path normalization
- Mixed path separators
- Relative vs absolute paths
- Path with dots and trailing separators
- Path conversion utilities

#### test_coverage_filters.py
- File inclusion/exclusion logic
- Testing folder exclusion
- CUDA file exclusion
- ThirdParty exclusion
- Interested folder filtering
- Windows path handling
- Backward compatibility with typo'd function name

#### test_error_handling.py
- export_target() return values
- run_oss_python_test() return values
- Input validation
- Error propagation through call chain
- No exception raising (XSigma standard)

#### test_json_parsing.py
- Valid JSON parsing
- Malformed JSON handling
- Empty file handling
- Large file memory efficiency
- Unicode and escaped characters
- Error status codes

### Integration Tests (2 files, ~40 tests)

#### test_cross_platform.py
- LLVM tool path discovery on Windows/Linux/macOS
- Shared library discovery (lib/ vs bin/)
- Path normalization across platforms
- Glob module usage
- Environment variable handling

#### test_workflow.py
- Workflow initialization
- GCDA file collection
- JSON export workflow
- Report generation
- Profile merging
- Coverage export
- File filtering
- Error recovery

## Fixtures

The `conftest.py` file provides reusable fixtures:

- `temp_dir`: Temporary directory for tests
- `mock_xsigma_folder`: Mock XSigma project structure
- `mock_build_folder`: Mock build folder structure
- `sample_json_file`: Sample JSON file for testing
- `large_json_file`: Large JSON file for memory testing
- `malformed_json_file`: Malformed JSON for error testing
- `empty_json_file`: Empty file for edge case testing
- `mock_subprocess`: Mocked subprocess module
- `mock_environment`: Mocked environment variables
- `test_file_paths`: Dictionary of test file paths
- `mock_platform_windows`: Mock Windows platform
- `mock_platform_linux`: Mock Linux platform
- `mock_platform_darwin`: Mock macOS platform

## Test Execution Time

- **Unit tests**: ~2-3 seconds
- **Integration tests**: ~1-2 seconds
- **Total**: ~5 seconds

## Code Coverage Goals

- **Target**: >90% coverage of modified files
- **Current**: Covers all critical fixes from code review
- **Files covered**:
  - `package/tool/clang_coverage.py`
  - `package/oss/utils.py`
  - `package/tool/summarize_jsons.py`
  - `package/tool/print_report.py`
  - `package/tool/utils.py`
  - `package/tool/gcc_coverage.py`
  - `package/tool/coverage_filters.py`

## Mocking Strategy

Tests use mocking to avoid dependencies on:
- LLVM tools (llvm-profdata, llvm-cov)
- Actual file system operations (where appropriate)
- Subprocess calls
- Platform-specific behavior

This allows tests to run quickly and reliably on any platform.

## Recent Fixes Tested

All tests verify the fixes from the code review:

1. ✅ Exception handling violations → Return values
2. ✅ Unix-only `find` command → Cross-platform `glob`
3. ✅ Missing error propagation → Boolean return values
4. ✅ Code duplication → Shared `coverage_filters` module
5. ✅ Missing input validation → Added validation
6. ✅ JSON parsing inefficiency → Line-by-line streaming
7. ✅ Hardcoded Windows paths → Expanded path search
8. ✅ Path normalization → `os.path.normpath()`
9. ✅ Typo in function name → Backward compatible wrapper
10. ✅ Missing docstrings → Added comprehensive docstrings

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```bash
# In CI/CD configuration
pytest Tools/code_coverage/tests/ \
  --cov=Tools/code_coverage/package \
  --cov-report=xml \
  --cov-report=term-missing \
  -v
```

## Troubleshooting

### Import Errors

If you get import errors, ensure the package root is in the Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Tools/code_coverage"
pytest Tools/code_coverage/tests/
```

### Platform-Specific Issues

Tests use mocking to handle platform differences. If you encounter platform-specific failures:

1. Check the mock fixtures in `conftest.py`
2. Verify the platform detection logic
3. Run with `-v` flag for detailed output

### Subprocess Mocking

Some tests mock subprocess calls. If subprocess behavior changes:

1. Update the mock in the test
2. Verify the actual command being called
3. Check error handling

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use descriptive test names
3. Add docstrings explaining what is tested
4. Use appropriate fixtures from `conftest.py`
5. Mock external dependencies
6. Aim for >90% code coverage

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Code Coverage Tools README](../README.md)
- [Code Review Fixes](../FIXES_IMPLEMENTED.md)

