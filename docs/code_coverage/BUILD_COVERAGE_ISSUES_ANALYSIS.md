# XSigma Build Coverage Issues - Analysis and Solutions

**Date**: October 21, 2025
**Status**: Analysis Complete - Solutions Provided

## Executive Summary

You have identified three critical issues with the code coverage workflow when running:
```bash
python setup.py config.build.ninja.clang.TEST.debug.lto.cxx20.buildcache.coverage
```

This document provides detailed analysis of each issue and the recommended solutions.

---

## Issue 1: Coverage Report Location

### Problem
The HTML coverage report is not being generated inside the build folder (e.g., `build_ninja_lto/coverage_report/`). The location is unclear.

### Root Cause Analysis

**Current Flow:**
1. `Scripts/setup.py` calls `coverage_helper.run_oss_coverage()` (line 1171)
2. `Scripts/helpers/coverage.py` sets environment variables:
   - `XSIGMA_COVERAGE_DIR = os.path.join(build_path, "coverage_report")` (line 86)
3. `Tools/code_coverage/oss_coverage.py` reads this environment variable (line 14)
4. `Tools/code_coverage/package/util/setting.py` uses it in `get_profile_dir()` (line 20)

**Issue:** The `oss_coverage.py` script uses the PyTorch OSS coverage tool which generates reports in:
- `Tools/code_coverage/profile/` (default)
- Or in `XSIGMA_COVERAGE_DIR` if set

However, the final HTML reports may be generated in the source directory, not the build directory.

### Solution

**Modify `Scripts/helpers/coverage.py` to ensure proper output directory:**

```python
# Line 86 - Already sets XSIGMA_COVERAGE_DIR correctly
coverage_dir = os.path.join(build_path, "coverage_report")
env["XSIGMA_COVERAGE_DIR"] = coverage_dir

# Add this to ensure the directory exists
os.makedirs(coverage_dir, exist_ok=True)
```

**Verify the output location:**
After running coverage, check:
```bash
ls -la build_ninja_lto/coverage_report/
```

---

## Issue 2: Incomplete Coverage - Missing Source Files

### Problem
Not all source files from the `Library` folder have corresponding HTML files in the coverage report. Some files are missing from the HTML output.

### Root Cause Analysis

**The Issue is in `Tools/code_coverage/package/tool/summarize_jsons.py`:**

The `is_intrested_file()` function (line 62) filters files based on:
1. Ignored patterns: `["cuda", "aten/gen_aten", "aten/aten_", "build/"]`
2. XSigma folder check: Files must be in the XSigma folder
3. **Interested folders filter**: If `interested_folders` is specified, ONLY those folders are included

**Current Problem:**
- `oss_coverage.py` calls `summarize_jsons(test_list, interested_folders, [""], TestPlatform.OSS)` (line 35)
- `interested_folders` is set to an empty list by default (line 66 in init.py)
- When empty, ALL files are included (line 84 in summarize_jsons.py)
- BUT: The coverage collection may not instrument all files

**Why Files Are Missing:**
1. Coverage instrumentation only includes files that are actually compiled and linked
2. Header-only files or unused source files may not be instrumented
3. Files in the Testing folder are being included (which you want to exclude)

### Solution

**Option A: Explicitly Include Library Folder (Recommended)**

Modify `Scripts/helpers/coverage.py` to pass interested folders:

```python
# Add after line 82
env["XSIGMA_INTERESTED_FOLDERS"] = "Library"
```

Then modify `Tools/code_coverage/oss_coverage.py` to use this:

```python
# Add after line 29
interested_folders = os.environ.get("XSIGMA_INTERESTED_FOLDERS", "").split(",")
interested_folders = [f.strip() for f in interested_folders if f.strip()]

# Pass to summarize_jsons
summarize_jsons(test_list, interested_folders, [""], TestPlatform.OSS)
```

**Option B: Use the New Exclusion Feature**

Use the `--excluded-sources` flag to exclude Testing folder:

```bash
python run_coverage_workflow.py \
    --test-exe build_ninja_lto/bin/CoreCxxTests.exe \
    --sources Library \
    --output build_ninja_lto/coverage_report \
    --excluded-sources "*Testing*"
```

---

## Issue 3: Testing Folder Inclusion

### Problem
Files from the Testing directory are appearing in the coverage report, but they should be excluded since they are test code, not production code.

### Root Cause Analysis

**Current Behavior:**
- `oss_coverage.py` doesn't exclude any directories by default
- The `is_intrested_file()` function doesn't have exclusion logic
- Testing folder files are included in the final report

### Solution

**Modify `Tools/code_coverage/package/tool/summarize_jsons.py`:**

Add exclusion patterns to the `is_intrested_file()` function:

```python
def is_intrested_file(
    file_path: str, interested_folders: list[str], platform: TestPlatform
) -> bool:
    # Add exclusion patterns
    excluded_patterns = ["Testing", "test", "mock", "stub"]
    if any(pattern in file_path for pattern in excluded_patterns):
        return False

    # ... rest of the function
```

**Or use environment variable approach:**

Modify `Scripts/helpers/coverage.py`:

```python
# Add after line 82
env["XSIGMA_EXCLUDED_SOURCES"] = "*Testing*"
```

Then modify `Tools/code_coverage/oss_coverage.py` to pass this to the collection step.

---

## Recommended Implementation Plan

### Step 1: Fix Report Location (Immediate)
- Verify `XSIGMA_COVERAGE_DIR` is being set correctly
- Ensure directory exists before running coverage
- **Status**: Already implemented in coverage.py (line 86)

### Step 2: Exclude Testing Folder (High Priority)
- Modify `summarize_jsons.py` to exclude Testing folder
- Add environment variable support for exclusions
- **Estimated effort**: 30 minutes

### Step 3: Include All Library Files (Medium Priority)
- Add interested_folders support to coverage.py
- Pass Library folder explicitly
- **Estimated effort**: 20 minutes

### Step 4: Documentation (Low Priority)
- Update WORKFLOW.md with build integration examples
- Document the three issues and solutions
- **Estimated effort**: 15 minutes

---

## Quick Reference: File Locations

| Issue | File | Line | Component |
|-------|------|------|-----------|
| Report Location | `Scripts/helpers/coverage.py` | 86 | XSIGMA_COVERAGE_DIR |
| Missing Files | `Tools/code_coverage/package/tool/summarize_jsons.py` | 62 | is_intrested_file() |
| Testing Inclusion | `Tools/code_coverage/package/tool/summarize_jsons.py` | 65 | ignored_patterns |
| OSS Coverage | `Tools/code_coverage/oss_coverage.py` | 35 | summarize_jsons() call |

---

## Next Steps

1. **Verify current behavior**: Run coverage and check output location
2. **Implement exclusions**: Add Testing folder exclusion to summarize_jsons.py
3. **Add interested folders**: Pass Library folder explicitly
4. **Test end-to-end**: Run full coverage workflow and verify results
5. **Document changes**: Update WORKFLOW.md with new behavior

---

## Related Documentation

- `WORKFLOW.md` - Coverage workflow documentation
- `EXCLUSION_FEATURE.md` - Source exclusion feature documentation
- `Scripts/helpers/coverage.py` - Coverage helper implementation
- `Tools/code_coverage/oss_coverage.py` - OSS coverage tool wrapper
