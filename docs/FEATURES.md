# Coverage System Features

## 1. Enhanced HTML Reports

### Line-by-Line Coverage Display
The refactored HTML report generator now displays:
- **Source code** with line numbers
- **Coverage status** for each line (covered/uncovered/neutral)
- **Execution counts** showing how many times each line was executed
- **Color coding**:
  - Green: Covered lines
  - Red: Uncovered lines
  - Gray: Non-executable lines

### Example HTML Report Structure
```
Coverage Report: path/to/file.cpp
├── Summary Section
│   ├── File path
│   ├── Coverage percentage with visual bar
│   ├── Covered/uncovered line counts
│   └── Total lines
└── Code Section
    ├── Line numbers (left column)
    ├── Coverage status (✓/✗/-)
    └── Source code with syntax highlighting
```

### Improved Styling
- Responsive design with better readability
- Hover effects for better interactivity
- Professional color scheme
- Mobile-friendly layout

## 2. JSON Coverage Summaries

### Purpose
Generate structured JSON files for CI/CD integration, automated reporting, and metrics tracking.

### File Location
```
build_dir/coverage_report/html/coverage_summary.json
```

### JSON Structure
```json
{
  "metadata": {
    "format_version": "1.0",
    "generator": "xsigma_coverage_tool"
  },
  "global_metrics": {
    "total_files": 42,
    "files_with_coverage": 40,
    "total_lines": 10000,
    "covered_lines": 8500,
    "uncovered_lines": 1500,
    "line_coverage_percent": 85.0
  },
  "files": {
    "Library/Core/memory/allocator.cpp": {
      "total_lines": 250,
      "covered_lines": 225,
      "uncovered_lines": 25,
      "line_coverage_percent": 90.0
    }
  }
}
```

### CI/CD Integration Examples

#### GitHub Actions
```yaml
- name: Generate Coverage
  run: python3 setup.py ninja.config.build.debug.coverage.test.lto

- name: Parse Coverage JSON
  run: |
    COVERAGE=$(jq '.global_metrics.line_coverage_percent' coverage_summary.json)
    echo "Coverage: $COVERAGE%"
    if (( $(echo "$COVERAGE < 80" | bc -l) )); then
      echo "Coverage below threshold!"
      exit 1
    fi
```

#### GitLab CI
```yaml
coverage_report:
  script:
    - python3 setup.py ninja.config.build.debug.coverage.test.lto
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage_report/html/coverage_summary.json
```

## 3. Modular Compiler Support

### Compiler-Specific Modules
Each compiler has its own dedicated module:

#### GCC (`gcc_coverage.py`)
- Uses lcov/genhtml for coverage generation
- Handles LTO compatibility issues
- Supports execution count tracking
- Fallback HTML generator if genhtml unavailable

#### Clang (`clang_coverage.py`)
- Uses llvm-profdata and llvm-cov
- Cross-platform test discovery
- Profile data merging
- Regex-based file filtering

#### MSVC (`msvc_coverage.py`)
- Uses OpenCppCoverage tool
- Windows-specific implementation
- Module-aware test discovery
- DLL/EXE filtering

### Adding New Compiler Support
To add support for a new compiler:

1. Create `new_compiler_coverage.py` with:
   ```python
   def generate_new_compiler_coverage(build_dir, modules, source_folder):
       """Generate coverage for new compiler."""
       # Implementation
   ```

2. Update `run_coverage.py`:
   ```python
   from new_compiler_coverage import generate_new_compiler_coverage

   # In get_coverage():
   elif compiler == "new_compiler":
       generate_new_compiler_coverage(build_path, modules, source_path)
   ```

## 4. Execution Count Tracking

### GCC/lcov Support
The `parse_lcov_data()` function extracts execution counts:
```python
covered_lines, uncovered_lines, execution_counts = parse_lcov_data(lcov_file)
# execution_counts: {file_path: {line_num: hit_count}}
```

### Usage in HTML Reports
Execution counts are displayed in the HTML reports:
- Shows how many times each line was executed
- Helps identify hot paths and optimization opportunities
- Useful for performance analysis

## 5. Cross-Platform Compatibility

### Supported Platforms
- Linux (GCC/Clang)
- macOS (GCC/Clang)
- Windows (MSVC/Clang)

### Platform Detection
Automatic detection via `get_platform_config()`:
```python
config = get_platform_config()
# Returns: {
#   "dll_extension": ".so" (Linux) / ".dylib" (macOS) / ".dll" (Windows),
#   "exe_extension": "" (Unix) / ".exe" (Windows),
#   "lib_folder": "lib" (Unix) / "bin" (Windows),
#   "os_name": "Linux" / "macOS" / "Windows"
# }
```

## 6. Backward Compatibility

### Unchanged API
The main `get_coverage()` function signature remains unchanged:
```python
get_coverage(
    compiler="auto",
    build_folder=".",
    source_folder="Library",
    output_folder=None,
    exclude=None,
    summary=True,
    xsigma_root=None
)
```

### Existing Scripts Continue to Work
All existing build scripts and CI/CD pipelines continue to work without modification.

## 7. Error Handling and Fallbacks

### Graceful Degradation
- If genhtml fails, falls back to custom HTML generator
- If JSON generation fails, coverage still completes
- Comprehensive error messages for debugging

### LTO Compatibility
- Automatically handles GCC LTO line number mismatches
- Uses `--ignore-errors` flags for lcov
- Prevents build failures due to coverage issues
