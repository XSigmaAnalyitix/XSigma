# CMake Format Bug Fix Summary

**Date**: 2025-10-27  
**Status**: ✅ **RESOLVED**  
**Severity**: 🔴 **CRITICAL** (Data Loss Bug)

---

## 🐛 Bug Description

The cmake-format linter adapter was producing **empty replacements** that would have **deleted all file content** when patches were applied through lintrunner.

### Impact
- **Severity**: Critical - would cause complete data loss
- **Scope**: All CMake files (*.cmake, CMakeLists.txt)
- **Trigger**: Running `lintrunner --take CMAKEFORMAT --apply-patches`
- **Result**: Files would be emptied (0 bytes) instead of being formatted

---

## 🔍 Root Cause Analysis

### The Problem

The `cmake-format` command-line tool behaves differently than other formatters like `clang-format`:

1. **clang-format behavior** (correct):
   ```bash
   clang-format file.cpp  # Outputs formatted content to stdout
   ```

2. **cmake-format behavior** (different):
   ```bash
   cmake-format file.cmake  # Modifies file in-place, NO stdout output
   ```

### The Bug

**File**: `Tools/linter/adapters/cmake_format_linter.py` (line 111)

**Broken Code**:
```python
proc = run_command(
    ["cmake-format", "--config-file", config, filename],
    retries=retries,
    timeout=timeout,
)
replacement = proc.stdout  # ❌ This was EMPTY (0 bytes)!
```

**What happened**:
1. cmake-format was called with just a filename argument
2. Without explicit output specification, cmake-format produces no stdout output
3. `proc.stdout` captured 0 bytes
4. The `replacement` field in JSON output was an empty string `""`
5. When lintrunner applied patches, it would replace file content with empty string
6. **Result**: All file content would be deleted

### Evidence

**Test showing the bug**:
```bash
$ python3 -c "
import subprocess
proc = subprocess.run(
    ['cmake-format', '--config-file', '.cmake-format.yaml', 'Library/Core/CMakeLists.txt'],
    capture_output=True
)
print('Stdout length:', len(proc.stdout))
"
Stdout length: 0  # ❌ EMPTY!
```

**JSON output from broken adapter**:
```json
{
  "path": "Library/Core/CMakeLists.txt",
  "code": "CMAKEFORMAT",
  "original": "cmake_minimum_required(VERSION 3.16)\n...",
  "replacement": "",  # ❌ EMPTY STRING - would delete all content!
  "description": "Run `lintrunner -a` to apply this patch."
}
```

---

## ✅ Solution Implemented

### The Fix

**File**: `Tools/linter/adapters/cmake_format_linter.py` (line 113)

**Fixed Code**:
```python
# Note: cmake-format requires '-o -' to output to stdout when given a filename
# Without this flag, it modifies the file in-place or produces no output
proc = run_command(
    ["cmake-format", "--config-file", config, "-o", "-", filename],
    retries=retries,
    timeout=timeout,
)
replacement = proc.stdout  # ✅ Now contains formatted content!
```

### What Changed

Added **`-o -`** flag to the cmake-format command:
- `-o` specifies the output file
- `-` means stdout (standard output)
- This forces cmake-format to output formatted content to stdout instead of modifying the file in-place

### Verification

**Test showing the fix**:
```bash
$ python3 -c "
import subprocess
proc = subprocess.run(
    ['cmake-format', '--config-file', '.cmake-format.yaml', '-o', '-', 'Library/Core/CMakeLists.txt'],
    capture_output=True
)
print('Stdout length:', len(proc.stdout))
"
Stdout length: 8968  # ✅ Contains formatted content!
```

**JSON output from fixed adapter**:
```json
{
  "path": "/tmp/test.cmake",
  "code": "CMAKEFORMAT",
  "original": "cmake_minimum_required(VERSION 3.16)\nif(WIN32)\n    set(VAR ON)\nendif()\n",
  "replacement": "cmake_minimum_required(VERSION 3.16)\nif(WIN32)\n  set(VAR ON)\nendif()\n",
  "description": "Run `lintrunner -a` to apply this patch."
}
```

✅ The `replacement` field now contains the properly formatted content!

---

## 🧪 Testing & Verification

### Automated Test Suite

Created comprehensive test script: `Scripts/test_cmake_format_no_data_loss.sh`

**Test Coverage**:
1. ✅ Direct cmake-format execution produces non-empty output
2. ✅ Linter adapter captures formatted content correctly
3. ✅ In-place formatting preserves file content
4. ✅ Real project files format correctly without data loss
5. ✅ Content verification (specific CMake commands are preserved)

**Test Results**:
```bash
$ bash Scripts/test_cmake_format_no_data_loss.sh
==========================================
CMake Format Data Loss Prevention Test
==========================================

1. Test file created:
   Path: /tmp/test_cmake_XXXXXX.txt
   Size:      780 bytes
   Lines:       42

2. Testing cmake-format direct output...
✓ cmake-format produced output: 717 bytes
✓ Content preserved: cmake_minimum_required found
✓ Content preserved: add_library found

3. Testing lintrunner adapter...
✓ Linter adapter produced replacement: 718 bytes
✓ Replacement preserves content: cmake_minimum_required found

4. Testing in-place formatting...
   Original:      780 bytes,       42 lines
   Formatted:      718 bytes,       30 lines
✓ In-place formatting preserved content
✓ Content check: cmake_minimum_required found
✓ Content check: add_library found

5. Testing with real project file...
   Original file:     9204 bytes
   Formatted output: 8967 bytes
✓ Real file content preserved

==========================================
✓ All data loss prevention tests passed!
==========================================
```

### Manual Testing

**Test 1: Linter adapter produces valid output**
```bash
$ python3 Tools/linter/adapters/cmake_format_linter.py --config=.cmake-format.yaml /tmp/test.cmake
{"path": "/tmp/test.cmake", "replacement": "cmake_minimum_required(VERSION 3.16)\nif(WIN32)\n  set(VAR ON)\nendif()\n", ...}
```
✅ Replacement field contains formatted content

**Test 2: Lintrunner detects formatting issues**
```bash
$ lintrunner --take CMAKEFORMAT /tmp/test.cmake
>>> Lint for ../../../private/tmp/test.cmake:
  Warning (CMAKEFORMAT) format
    3    |-    set(VAR ON)
       4 |+  set(VAR ON)
```
✅ Formatting differences detected correctly

**Test 3: Lintrunner applies patches correctly**
```bash
$ lintrunner --take CMAKEFORMAT --apply-patches /tmp/test.cmake
ok No lint issues.
Successfully applied all patches.
```
✅ Patches applied successfully, file content preserved

---

## 📊 Impact Assessment

### Before Fix
- ❌ Linter adapter produced empty replacements
- ❌ Applying patches would delete all file content
- ❌ Critical data loss risk
- ❌ cmake-format integration unusable

### After Fix
- ✅ Linter adapter produces correct formatted content
- ✅ Applying patches preserves file content
- ✅ No data loss risk
- ✅ cmake-format integration fully functional

---

## 📝 Files Modified

1. **`Tools/linter/adapters/cmake_format_linter.py`** (line 113)
   - Added `-o -` flag to cmake-format command
   - Added explanatory comment

2. **`Docs/README_CMAKE_FORMAT.md`**
   - Added bug fix documentation section
   - Documented root cause and solution

3. **`Docs/CMAKE_FORMAT_CONFLICT_RESOLUTION.md`**
   - Added empty replacement bug to problem statement
   - Documented fix in solution section
   - Updated verification results

4. **`Scripts/test_cmake_format_no_data_loss.sh`** (new file)
   - Comprehensive automated test suite
   - Verifies data loss prevention
   - Tests all aspects of the fix

---

## 🚀 Usage

### Format CMake files safely
```bash
# Check for formatting issues
lintrunner --take CMAKEFORMAT <file>

# Apply formatting patches
lintrunner --take CMAKEFORMAT --apply-patches <file>

# Format all CMake files in project
lintrunner --take CMAKEFORMAT --all-files --apply-patches
```

### Verify the fix
```bash
# Run automated test suite
bash Scripts/test_cmake_format_no_data_loss.sh

# Run compatibility verification
bash Scripts/verify_cmake_format_compatibility.sh
```

---

## ✅ Status: RESOLVED

- ✅ Bug identified and root cause analyzed
- ✅ Fix implemented and tested
- ✅ Automated test suite created
- ✅ Documentation updated
- ✅ All tests passing
- ✅ Production-ready

**The cmake-format linter integration is now fully functional and safe to use.**

---

## 📚 Related Documentation

- **Configuration**: `.cmake-format.yaml`
- **Linter Adapter**: `Tools/linter/adapters/cmake_format_linter.py`
- **Integration Config**: `.lintrunner.toml` (lines 779-806)
- **Full Documentation**: `Docs/README_CMAKE_FORMAT.md`
- **Conflict Resolution**: `Docs/CMAKE_FORMAT_CONFLICT_RESOLUTION.md`
- **Test Script**: `Scripts/test_cmake_format_no_data_loss.sh`
- **Verification Script**: `Scripts/verify_cmake_format_compatibility.sh`

---

**Author**: Augment Agent  
**Date**: 2025-10-27  
**Version**: 1.0.0

