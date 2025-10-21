# XSigma Code Coverage - Source Exclusion Feature

**Date**: October 21, 2025
**Status**: ✓ IMPLEMENTED AND VERIFIED

## Overview

The XSigma code coverage workflow now supports excluding specific directories and patterns from coverage collection. This allows you to focus coverage reports on production code while excluding test code, mock implementations, or other non-production directories.

## Feature Implementation

### Changes Made

#### 1. `collect_coverage_data.py`
- Added `excluded_sources` parameter to `CoverageDataCollector` class
- Updated command building to include `--excluded_sources` flags for OpenCppCoverage
- Added `--excluded-sources` command-line argument (can be specified multiple times)

#### 2. `run_coverage_workflow.py`
- Added `excluded_sources` parameter to `CoverageWorkflow` class
- Updated `collect_coverage()` method to pass exclusion patterns to the collector
- Added `--excluded-sources` command-line argument to the workflow orchestrator

#### 3. Documentation
- Updated `WORKFLOW.md` with exclusion examples
- Updated `QUICK_START.md` with exclusion usage
- Added this feature documentation file

## Usage

### Basic Exclusion

Exclude the Testing folder:

```bash
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_report \
    --excluded-sources "*Testing*"
```

### Multiple Exclusions

Exclude multiple patterns:

```bash
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_report \
    --excluded-sources "*Testing*" \
    --excluded-sources "*Mock*" \
    --excluded-sources "*Stub*"
```

### Direct Script Usage

Exclude sources when using `collect_coverage_data.py` directly:

```bash
python collect_coverage_data.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_data \
    --excluded-sources "*Testing*"
```

## Verification Results

### Test Execution

✓ All 9 tests passing (5 unit + 4 integration)
✓ No regressions in existing functionality
✓ New exclusion feature working correctly

### Coverage Comparison

**Without Exclusion:**
- Overall Coverage: 63.36%
- Lines Covered: 7,116
- Lines Uncovered: 4,115
- Files Analyzed: 120

**With Exclusion (`*Testing*`):**
- Overall Coverage: 43.12%
- Lines Covered: 2,961
- Lines Uncovered: 3,906
- Files Analyzed: 97
- **Files Excluded**: 23 (19% reduction)

### Verification Steps

1. ✓ Ran workflow with `--excluded-sources "*Testing*"`
2. ✓ Verified no "Testing" files in coverage.xml
3. ✓ Verified no "Testing" files in HTML report
4. ✓ Confirmed coverage statistics reflect only Library code
5. ✓ Compared with and without exclusion
6. ✓ All tests still passing

## Technical Details

### OpenCppCoverage Integration

The exclusion feature uses OpenCppCoverage's `--excluded_sources` flag:

```bash
OpenCppCoverage.exe \
    --sources c:\dev\XSigma\Library \
    --excluded_sources "*Testing*" \
    --export_type cobertura:coverage.xml \
    -- CoreCxxTests.exe
```

### Pattern Matching

OpenCppCoverage supports wildcard patterns:
- `*Testing*` - Matches any path containing "Testing"
- `*Mock*` - Matches any path containing "Mock"
- `*/test/*` - Matches paths with "test" directory
- `C:\path\to\exclude` - Exact path exclusion

## Benefits

1. **Cleaner Reports**: Focus on production code coverage
2. **Accurate Metrics**: Coverage percentages reflect only production code
3. **Faster Analysis**: Fewer files to analyze and report
4. **Flexible**: Multiple exclusion patterns supported
5. **Backward Compatible**: Exclusion is optional

## Default Behavior

**Important**: By default, NO directories are excluded. The Testing folder is NOT automatically excluded.

To exclude Testing folder, you must explicitly use:
```bash
--excluded-sources "*Testing*"
```

## Examples

### Exclude Testing Folder Only
```bash
python run_coverage_workflow.py \
    --test-exe CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --excluded-sources "*Testing*"
```

### Exclude Multiple Test-Related Directories
```bash
python run_coverage_workflow.py \
    --test-exe CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --excluded-sources "*Testing*" \
    --excluded-sources "*test*" \
    --excluded-sources "*mock*"
```

### Exclude Specific Path
```bash
python run_coverage_workflow.py \
    --test-exe CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --excluded-sources "c:\dev\XSigma\Library\Testing"
```

## Integration with CI/CD

For CI/CD pipelines, add the exclusion flag to your coverage collection step:

```yaml
- name: Collect Coverage
  run: |
    python run_coverage_workflow.py \
      --test-exe build/bin/CoreCxxTests.exe \
      --sources Library \
      --output coverage_report \
      --excluded-sources "*Testing*"
```

## Troubleshooting

### Exclusion Not Working

1. Verify pattern syntax: `*Testing*` (with asterisks)
2. Check OpenCppCoverage version: 0.9.9.0+
3. Run with `--verbose` for debugging:
   ```bash
   python run_coverage_workflow.py ... --verbose
   ```

### Files Still Appearing

- Pattern may not match the file path
- Try more specific pattern: `*Testing*` vs `*/Testing/*`
- Check exact path in coverage.xml

## Future Enhancements

Potential improvements:
- Configuration file support for default exclusions
- Automatic Testing folder exclusion option
- Inclusion patterns (whitelist mode)
- Regex pattern support

## Conclusion

The source exclusion feature is fully implemented, tested, and ready for production use. It provides flexible control over which directories are included in coverage analysis, enabling more accurate and focused coverage reporting.

**Status**: ✓ COMPLETE AND VERIFIED

