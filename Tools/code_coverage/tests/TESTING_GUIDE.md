# Code Coverage Tools - Testing Guide

Complete guide for running, understanding, and maintaining the test suite.

## Quick Start

### 1. Install Dependencies

```bash
cd Tools/code_coverage
pip install pytest pytest-cov
```

### 2. Run All Tests

```bash
pytest tests/
```

### 3. View Coverage Report

```bash
pytest tests/ --cov=package --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Test Organization

### Unit Tests (Fast, Isolated)

Located in `tests/unit/`, these test individual functions in isolation:

```bash
pytest tests/unit/
```

**Files:**
- `test_path_handling.py` - Path normalization and conversion
- `test_coverage_filters.py` - File filtering logic
- `test_error_handling.py` - Error handling and return values
- `test_json_parsing.py` - JSON parsing and optimization

**Expected Time:** ~2-3 seconds

### Integration Tests (Moderate, Workflow-focused)

Located in `tests/integration/`, these test complete workflows:

```bash
pytest tests/integration/
```

**Files:**
- `test_cross_platform.py` - Cross-platform compatibility
- `test_workflow.py` - End-to-end workflows

**Expected Time:** ~1-2 seconds

## Common Test Commands

### Run Specific Test File

```bash
pytest tests/unit/test_path_handling.py
```

### Run Specific Test Class

```bash
pytest tests/unit/test_path_handling.py::TestPathNormalization
```

### Run Specific Test Function

```bash
pytest tests/unit/test_path_handling.py::TestPathNormalization::test_windows_path_with_backslashes
```

### Run Tests Matching Pattern

```bash
pytest tests/ -k "path"  # All tests with "path" in name
pytest tests/ -k "not slow"  # All tests except slow ones
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Extra Verbose Output

```bash
pytest tests/ -vv
```

### Run with Print Statements

```bash
pytest tests/ -s
```

### Run with Detailed Failure Info

```bash
pytest tests/ -vv --tb=long
```

### Run with Markers

```bash
pytest tests/ -m "unit"  # Only unit tests
pytest tests/ -m "integration"  # Only integration tests
pytest tests/ -m "not slow"  # Exclude slow tests
```

## Coverage Analysis

### Generate Coverage Report

```bash
pytest tests/ --cov=package --cov-report=html
```

### View Coverage by File

```bash
pytest tests/ --cov=package --cov-report=term-missing
```

### Coverage for Specific Module

```bash
pytest tests/ --cov=package.tool.clang_coverage --cov-report=term-missing
```

### Minimum Coverage Threshold

```bash
pytest tests/ --cov=package --cov-fail-under=90
```

## Debugging Tests

### Run Single Test with Debugging

```bash
pytest tests/unit/test_path_handling.py::TestPathNormalization::test_windows_path_with_backslashes -vv -s
```

### Use Python Debugger

```bash
pytest tests/ --pdb  # Drop into debugger on failure
pytest tests/ --pdbcls=IPython.terminal.debugger:TerminalPdb  # Use IPython debugger
```

### Show Local Variables on Failure

```bash
pytest tests/ -l
```

### Show Captured Output

```bash
pytest tests/ -s
```

## Test Fixtures

### Available Fixtures

All fixtures are defined in `conftest.py`:

```python
# Temporary directory
def test_something(temp_dir):
    # temp_dir is a Path object
    pass

# Mock XSigma folder structure
def test_something(mock_xsigma_folder):
    # mock_xsigma_folder has Library/, Testing/, etc.
    pass

# Mock build folder
def test_something(mock_build_folder):
    # mock_build_folder has bin/, lib/, profile/, etc.
    pass

# Sample JSON file
def test_something(sample_json_file):
    # sample_json_file is a Path to a valid JSON file
    pass

# Platform mocking
def test_something(mock_platform_windows):
    # platform.system() returns "Windows"
    pass
```

### Using Fixtures

```bash
# List all available fixtures
pytest --fixtures tests/

# Show fixture details
pytest --fixtures tests/ | grep -A 5 "temp_dir"
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install pytest pytest-cov
      - run: pytest Tools/code_coverage/tests/ --cov=Tools/code_coverage/package
```

### Local CI Simulation

```bash
# Run all checks locally
pytest Tools/code_coverage/tests/ \
  --cov=Tools/code_coverage/package \
  --cov-report=term-missing \
  --cov-fail-under=90 \
  -v
```

## Test Development

### Writing New Tests

1. **Choose location:**
   - Unit test → `tests/unit/test_*.py`
   - Integration test → `tests/integration/test_*.py`

2. **Use fixtures:**
   ```python
   def test_something(temp_dir, mock_platform_windows):
       # Your test code
       pass
   ```

3. **Follow naming:**
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test functions: `test_*`

4. **Add docstrings:**
   ```python
   def test_something(temp_dir):
       """Test that something works correctly."""
       # Your test code
   ```

5. **Use assertions:**
   ```python
   assert result is True
   assert len(items) == 3
   assert "error" in str(exception)
   ```

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

def test_with_mock():
    with patch("subprocess.check_call") as mock_call:
        mock_call.return_value = 0
        # Your test code
        mock_call.assert_called_once()
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'package'`

**Solution:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Tools/code_coverage"
pytest tests/
```

### Fixture Not Found

**Problem:** `fixture 'temp_dir' not found`

**Solution:**
- Ensure `conftest.py` is in the tests directory
- Check fixture name spelling
- Run `pytest --fixtures` to list available fixtures

### Tests Fail on Windows

**Problem:** Path-related tests fail on Windows

**Solution:**
- Tests use mocking for cross-platform compatibility
- Check `mock_platform_windows` fixture usage
- Verify path normalization logic

### Slow Tests

**Problem:** Tests take too long

**Solution:**
```bash
pytest tests/ --durations=10  # Show 10 slowest tests
pytest tests/ -m "not slow"  # Skip slow tests
```

## Performance Optimization

### Parallel Test Execution

```bash
pip install pytest-xdist
pytest tests/ -n auto  # Use all CPU cores
```

### Test Caching

```bash
pytest tests/ --cache-clear  # Clear cache
pytest tests/ -p no:cacheprovider  # Disable caching
```

## Test Maintenance

### Update Tests After Code Changes

1. Run tests to identify failures
2. Update test expectations
3. Add new tests for new functionality
4. Verify coverage remains >90%

### Regular Test Review

- Review test coverage monthly
- Update fixtures as needed
- Remove obsolete tests
- Add tests for bug fixes

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Plugins](https://docs.pytest.org/en/latest/reference.html#plugins)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Code Review Fixes](../FIXES_IMPLEMENTED.md)
- [Implementation Summary](../IMPLEMENTATION_COMPLETE.md)

