# Implementation Guide

## Module Dependencies

### Import Graph
```
run_coverage.py (Main Orchestrator)
├── gcc_coverage.py
│   └── html_report_generator.py
├── clang_coverage.py
├── msvc_coverage.py
├── html_report_generator.py
│   └── json (standard library)
└── coverage_summary.py
```

### Shared Utilities
- `get_platform_config()`: Duplicated in each compiler module for independence
- `CONFIG`: Centralized in `run_coverage.py`, copied to compiler modules

## File Structure

### Tools/coverage/
```
├── run_coverage.py              # Main orchestrator (578 lines)
├── gcc_coverage.py              # GCC/lcov implementation (~160 lines)
├── clang_coverage.py            # Clang/LLVM implementation (~180 lines)
├── msvc_coverage.py             # MSVC/OpenCppCoverage implementation (~180 lines)
├── coverage_summary.py          # JSON summary generator (~100 lines)
├── html_report_generator.py     # Enhanced HTML generator (~390 lines)
├── REFACTORING_SUMMARY.md       # This refactoring overview
├── FEATURES.md                  # Feature documentation
└── IMPLEMENTATION_GUIDE.md      # This file
```

## Key Design Decisions

### 1. Module Independence
Each compiler module is independent and can be used standalone:
```python
from gcc_coverage import generate_lcov_coverage
generate_lcov_coverage(build_dir, modules, exclude_patterns)
```

### 2. Configuration Duplication
Platform config is duplicated in each module to avoid circular imports:
- Keeps modules independent
- Simplifies testing
- Minimal code duplication (simple function)

### 3. Backward Compatibility
- Main `get_coverage()` API unchanged
- All existing scripts continue to work
- New features are additive, not breaking

### 4. Graceful Degradation
- HTML generation failures don't stop coverage
- JSON generation failures don't stop coverage
- Comprehensive error messages for debugging

## Testing Strategy

### Unit Tests
Test each module independently:

```python
# Test gcc_coverage.py
def test_parse_lcov_data():
    """Test lcov data parsing with execution counts."""
    # Create sample lcov file
    # Parse it
    # Verify covered/uncovered/execution counts

def test_generate_lcov_coverage():
    """Test GCC coverage generation."""
    # Mock subprocess calls
    # Verify lcov commands
    # Check output structure
```

### Integration Tests
Test full coverage pipeline:

```python
def test_gcc_coverage_pipeline():
    """Test complete GCC coverage generation."""
    # Build with coverage flags
    # Run tests
    # Generate coverage
    # Verify HTML output
    # Verify JSON summary

def test_compiler_detection():
    """Test automatic compiler detection."""
    # Create build directories with different compilers
    # Verify detection works correctly
```

### Cross-Platform Tests
Verify on all supported platforms:
- Linux (GCC/Clang)
- macOS (GCC/Clang)
- Windows (MSVC/Clang)

### Backward Compatibility Tests
```python
def test_get_coverage_api():
    """Verify get_coverage() API unchanged."""
    # Test with all parameter combinations
    # Verify return codes
    # Check output structure
```

## Performance Considerations

### Coverage Generation Time
- GCC: ~18-20 seconds (with LTO)
- Clang: ~15-25 seconds (depends on test count)
- MSVC: ~20-30 seconds (depends on DLL count)

### Memory Usage
- Typical: 100-200 MB
- Large projects: 500+ MB
- Profile data can be large (100+ MB for complex projects)

### Optimization Tips
1. Use LTO for faster builds (GCC/Clang)
2. Filter unnecessary files early
3. Run tests in parallel when possible
4. Use sparse profile data (LLVM)

## Troubleshooting

### Common Issues

#### 1. "lcov not found"
```bash
# Install lcov
sudo apt-get install lcov  # Linux
brew install lcov          # macOS
```

#### 2. "genhtml failed"
- Falls back to custom HTML generator
- Check lcov data validity
- Verify exclude patterns

#### 3. "No test executables found"
- Check test naming conventions
- Verify build directory structure
- Check executable permissions

#### 4. "Compiler detection failed"
- Ensure CMakeCache.txt exists
- Check CMakeCXXCompiler.cmake
- Verify build directory is valid

## Future Enhancements

### Planned Features
1. **Function-level coverage**: Track function coverage separately
2. **Branch coverage**: Include branch coverage metrics
3. **Coverage trends**: Track coverage over time
4. **Code review integration**: Comment on PRs with coverage changes
5. **Performance profiling**: Integrate with profiling tools
6. **Custom report templates**: Allow user-defined HTML templates

### Potential Optimizations
1. **Parallel test execution**: Run tests in parallel
2. **Incremental coverage**: Only analyze changed files
3. **Caching**: Cache coverage data between runs
4. **Streaming reports**: Generate reports while tests run

## Maintenance Guidelines

### Code Style
- Follow Google Python Style Guide
- Use type hints for all functions
- Document all public functions
- Keep functions focused and small

### Testing Requirements
- Minimum 80% code coverage
- All new features must have tests
- Cross-platform testing required
- Backward compatibility tests mandatory

### Documentation
- Update FEATURES.md for new features
- Update IMPLEMENTATION_GUIDE.md for architecture changes
- Add docstrings to all functions
- Include usage examples

## Version History

### v2.0 (Current - Refactored)
- Modular compiler-specific implementations
- Enhanced HTML reports with line-by-line coverage
- JSON summary generation
- Improved error handling
- Better cross-platform support

### v1.0 (Previous)
- Monolithic implementation
- Basic HTML reports
- Limited error handling
- GCC/Clang/MSVC support

