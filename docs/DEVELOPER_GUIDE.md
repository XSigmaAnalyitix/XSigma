# Developer Guide: Code Coverage Tools

## Quick Reference

### Project Structure

```
Tools/code_coverage/
├── oss_coverage.py              # Main entry point for OSS coverage
├── package/
│   ├── oss/                     # OSS-specific implementations
│   │   ├── utils.py             # Utility functions (binary/library discovery)
│   │   ├── cov_json.py          # JSON report generation orchestration
│   │   ├── init.py              # Initialization and argument parsing
│   │   └── run.py               # Test execution
│   ├── tool/                    # Coverage analysis tools
│   │   ├── clang_coverage.py    # LLVM/Clang coverage operations
│   │   ├── gcc_coverage.py      # GCC coverage operations
│   │   ├── summarize_jsons.py   # JSON parsing and summarization
│   │   ├── print_report.py      # Report generation
│   │   ├── utils.py             # Tool utilities
│   │   ├── html_report_generator.py  # HTML report generation
│   │   └── parser/              # Coverage data parsers
│   └── util/                    # Shared utilities
│       ├── setting.py           # Configuration and constants
│       └── utils.py             # Common utility functions
└── README.md                    # Documentation
```

---

## Key Concepts

### Coverage Workflow

1. **Run**: Execute tests with coverage instrumentation
   - Sets `LLVM_PROFILE_FILE` environment variable
   - Generates `.profraw` files

2. **Merge**: Combine profile data
   - Uses `llvm-profdata merge` to create `.merged` files
   - Suppresses harmless warnings

3. **Export**: Convert to JSON format
   - Uses `llvm-cov export` to create JSON files
   - **Important**: No filters applied at this stage (data loss prevention)

4. **Summarize**: Parse JSON and generate reports
   - Applies filters to exclude test files
   - Generates summary and HTML reports

### Filter Strategy

**Excluded Patterns**:
- `Library/Core/Testing/` - Test code
- `Test*.cxx`, `*Test.cxx` - Test files by naming
- `ThirdParty/` - Third-party code
- `cuda`, `aten/` - PyTorch-specific patterns

**Included Patterns**:
- Specified via `XSIGMA_INTERESTED_FOLDERS` (default: "Library")

---

## Important Implementation Details

### Path Handling

**Critical**: All paths must be normalized to forward slashes for matching:

```python
# Always normalize before pattern matching
normalized_path = file_path.replace("\\", "/")

# Check patterns
if "Library/" in normalized_path:
    # Include this file
```

### Shared Library Discovery

**Windows**: Searches both `lib/` and `bin/` directories
**Unix**: Searches `lib/` directory only

```python
# On Windows, DLLs are in bin/, not lib/
search_dirs = []
if os.path.isdir(lib_dir):
    search_dirs.append(lib_dir)
if system == "Windows" and os.path.isdir(bin_dir):
    search_dirs.append(bin_dir)
```

### Error Handling

**XSigma Standard**: No exception-based error handling

```python
# ❌ WRONG
try:
    do_something()
except Exception:
    pass

# ✅ CORRECT
if not do_something():
    print_error("Operation failed")
    return False
```

---

## Common Tasks

### Adding a New Filter

**File**: `Tools/code_coverage/package/tool/clang_coverage.py`

```python
def get_coverage_filters() -> list[str]:
    filters = [
        # Existing filters...
        # Add new filter here
        ".*[/\\\\]MyPattern[/\\\\].*",
    ]
    return filters
```

### Debugging Coverage Issues

1. **Check if files are in JSON export**:
   ```bash
   python -c "import json; data = json.load(open('coverage_report/json/CoreCxxTests.exe.json')); print(len(data['data'][0]['files']))"
   ```

2. **Check if files pass filters**:
   - Look at `coverage_report/summary/file_summary`
   - Files should be listed with coverage percentages

3. **Check environment variables**:
   ```bash
   echo $XSIGMA_INTERESTED_FOLDERS
   echo $XSIGMA_BUILD_FOLDER
   echo $XSIGMA_COVERAGE_DIR
   ```

### Running Coverage Manually

```bash
cd Scripts
python setup.py config.build.ninja.clang.TEST.debug.lto.cxx20.coverage
```

### Viewing Reports

- **Summary**: `build_ninja_lto_coverage/coverage_report/summary/file_summary`
- **HTML**: `build_ninja_lto_coverage/coverage_report/html_details/index.html`
- **Line Details**: `build_ninja_lto_coverage/coverage_report/summary/line_summary`

---

## Known Issues & Workarounds

### Issue 1: 0% Coverage
**Cause**: Path separator mismatch (Windows backslashes vs forward slashes)
**Fix**: Ensure path normalization in filter functions

### Issue 2: Missing Production Files
**Cause**: Shared libraries not found in `bin/` directory
**Fix**: Verify `get_oss_shared_library()` searches both `lib/` and `bin/`

### Issue 3: Testing Files Included
**Cause**: Filters not applied at report generation stage
**Fix**: Ensure filters are applied in `summarize_jsons.py`, not in `export_target()`

---

## Testing Checklist

Before committing changes:

- [ ] Path normalization works on Windows and Unix
- [ ] Shared libraries are discovered correctly
- [ ] Testing folder files are excluded
- [ ] Production files are included
- [ ] Coverage percentage is reasonable (>0%)
- [ ] HTML reports generate without errors
- [ ] No exception-based error handling
- [ ] All functions have docstrings
- [ ] Code follows PEP 8 style

---

## Performance Considerations

### Large JSON Files
- Current implementation loads entire file into memory
- For files >100MB, consider streaming JSON parser
- See `REVIEW_RECOMMENDATIONS.md` for optimization

### Shared Library Search
- Searches both `lib/` and `bin/` on Windows
- Could be optimized with caching if called frequently

### Path Normalization
- Simple string replacement is fast
- No performance concerns for typical use cases

---

## Code Quality Standards

### Required
- Type hints on all functions
- Docstrings with Args/Returns
- No exception-based error handling
- PEP 8 compliance
- Cross-platform path handling

### Recommended
- Unit tests for new functions
- Comments for non-obvious logic
- Error logging via `print_error()`
- Input validation

---

## Related Documentation

- `CODE_REVIEW.md` - Comprehensive code review
- `REVIEW_RECOMMENDATIONS.md` - Detailed action items
- `RECENT_FIXES_ANALYSIS.md` - Analysis of recent fixes
- `README.md` - User documentation

---

## Contact & Support

For issues or questions:
1. Check `CODE_REVIEW.md` for known issues
2. Review `RECENT_FIXES_ANALYSIS.md` for recent changes
3. Consult `REVIEW_RECOMMENDATIONS.md` for improvements
4. Check test results in `coverage_report/summary/`

---

## Version History

**2025-10-21**: 
- Fixed path separator normalization (Windows backslash issue)
- Fixed shared library discovery on Windows
- Removed filters from export stage (data loss prevention)
- Coverage report now includes 103 files (up from 29)
- Coverage percentage: 38.16%

