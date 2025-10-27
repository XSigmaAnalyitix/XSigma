# CMake Format and CMakeLint Conflict Resolution

**Date**: 2025-10-27  
**Status**: ✅ Resolved  
**Tools**: cmake-format (cmakelang 0.6.13), cmakelint (1.4.1)

---

## Executive Summary

Two critical issues were identified and resolved in the cmake-format integration:

1. **Configuration Conflict**: Incompatible whitespace formatting rules between cmake-format and cmakelint
2. **Empty Replacement Bug**: cmake-format linter adapter was producing empty replacements that would delete file content

**Resolution**:
- Updated `.cmake-format.yaml` configuration to align with cmakelint requirements
- Fixed linter adapter to correctly capture cmake-format output using `-o -` flag

Both issues are now fully resolved and verified.

---

## Problem Statement

### Symptoms
1. **Configuration Conflict**: cmake-format produced warnings about unrecognized configuration options
2. **Configuration Conflict**: Files formatted by cmake-format failed cmakelint validation with `[whitespace/extra]` errors
3. **Configuration Conflict**: Configuration file contained deprecated options not supported by cmake-format 0.6.13
4. **Empty Replacement Bug**: Linter adapter produced empty `replacement` field in JSON output, which would delete all file content when patches were applied

### Root Cause Analysis

#### Issue 1: Control Flow Spacing Conflict
**Configuration**: `.cmake-format.yaml` had `separate_ctrl_name_with_space: true`

**cmake-format behavior**:
```cmake
if (condition)
  # code
endif ()
```

**cmakelint requirement**:
```cmake
if(condition)
  # code
endif()
```

**Error**: cmakelint flagged 54 instances of `Extra spaces between 'if' and its ()` in a single test file.

#### Issue 2: Deprecated Configuration Options
The following options were present but not recognized by cmake-format 0.6.13:
- `indent_width` (should use `tab_size` instead)
- `comment_prefix` (not a valid format option)
- `enable_markup` (belongs in `markup` section, not `format`)
- `max_paren_depth` (not a recognized option)
- `explicit_start`, `explicit_end`, `implicit_block_start`, `implicit_block_end`, `implicit_paragraph_start` (not valid markup options)

**Warnings produced**:
```
WARNING config_util.py:307: The following configuration options were ignored:
  indent_width
  comment_prefix
  enable_markup
  max_paren_depth
```

#### Issue 3: Empty Replacement Bug (Critical)

**Location**: `Tools/linter/adapters/cmake_format_linter.py`

**Problem**: The linter adapter was calling cmake-format without the `-o -` flag:
```python
# BROKEN CODE:
proc = run_command(
    ["cmake-format", "--config-file", config, filename],
    retries=retries,
    timeout=timeout,
)
replacement = proc.stdout  # This was EMPTY (0 bytes)!
```

**Behavior**:
- When `cmake-format` is given a filename without `-o -`, it modifies the file in-place or produces no stdout output
- The subprocess captured empty stdout (0 bytes)
- The `replacement` field in JSON output was an empty string `""`
- If patches were applied via `lintrunner -a`, this would **delete all file content**

**Impact**: Critical data loss bug that would have corrupted all CMake files if patches were applied.

---

## Solution Implemented

### 1. Fixed Control Flow Spacing (Configuration Conflict)

**Change**: Set `separate_ctrl_name_with_space: false`

**Before**:
```yaml
format:
  separate_ctrl_name_with_space: true  # ❌ Conflicts with cmakelint
```

**After**:
```yaml
format:
  # Do NOT separate flow control names (if, while, etc.) from their parentheses with a space
  # Example: if(condition) instead of if (condition)
  # This aligns with cmakelint's whitespace/extra check
  separate_ctrl_name_with_space: false  # ✅ Compatible with cmakelint
```

**Impact**: Eliminates all `[whitespace/extra]` errors from cmakelint.

### 2. Removed Deprecated Options

**Removed from `format` section**:
- `indent_width` → Use `tab_size` instead
- `comment_prefix` → Not a valid option
- `enable_markup` → Moved to `markup` section
- `max_paren_depth` → Not recognized

**Removed from `markup` section**:
- `explicit_start`, `explicit_end`
- `implicit_block_start`, `implicit_block_end`, `implicit_paragraph_start`

### 3. Added Missing Sections

**Added `parse` section** for listfile parsing options:
```yaml
parse:
  additional_commands: {}
  override_spec: {}
  vartags: []
  proptags: []
```

### 4. Enhanced Documentation

- Added comprehensive comments for each configuration option
- Explained the rationale for each setting
- Documented compatibility requirements with cmakelint
- Validated all options against `cmake-format --dump-config` output

### 5. Fixed Empty Replacement Bug (Critical Fix)

**File**: `Tools/linter/adapters/cmake_format_linter.py` (line 113)

**Change**: Added `-o -` flag to cmake-format command

**Before (BROKEN)**:
```python
proc = run_command(
    ["cmake-format", "--config-file", config, filename],
    retries=retries,
    timeout=timeout,
)
replacement = proc.stdout  # Empty! (0 bytes)
```

**After (FIXED)**:
```python
# Note: cmake-format requires '-o -' to output to stdout when given a filename
# Without this flag, it modifies the file in-place or produces no output
proc = run_command(
    ["cmake-format", "--config-file", config, "-o", "-", filename],
    retries=retries,
    timeout=timeout,
)
replacement = proc.stdout  # Now contains formatted content!
```

**Impact**:
- ✅ cmake-format now outputs formatted content to stdout
- ✅ Linter adapter correctly captures the formatted content
- ✅ No data loss when applying patches
- ✅ All file content is preserved during formatting

---

## Verification Results

### Test Methodology
1. **Configuration Conflict**: Formatted `Library/Core/CMakeLists.txt` with cmake-format and verified cmakelint compatibility
2. **Empty Replacement Bug**: Created test files and verified linter adapter produces non-empty replacements
3. **Integration Test**: Ran complete workflow through lintrunner with patch application
4. **Data Loss Prevention**: Verified file content is preserved during formatting operations

### Before Fix
```bash
$ cmake-format --check --config-file=.cmake-format.yaml Library/Core/CMakeLists.txt
WARNING config_util.py:307: The following configuration options were ignored:
  indent_width
  comment_prefix
  enable_markup
  max_paren_depth
ERROR __main__.py:618: Check failed: Library/Core/CMakeLists.txt

$ cmakelint --config=.cmakelintrc Library/Core/CMakeLists.txt
Library/Core/CMakeLists.txt:11: Extra spaces between 'if' and its () [whitespace/extra]
Library/Core/CMakeLists.txt:14: Extra spaces between 'else' and its () [whitespace/extra]
... (54 total errors)
Total Errors: 54
```

### After Fix (Configuration Conflict)
```bash
$ cmake-format --check --config-file=.cmake-format.yaml Library/Core/CMakeLists.txt
ERROR __main__.py:618: Check failed: Library/Core/CMakeLists.txt
# (This is expected - file needs formatting)

$ cmake-format --config-file=.cmake-format.yaml -i Library/Core/CMakeLists.txt
# (No warnings - clean execution)

$ cmakelint --config=.cmakelintrc Library/Core/CMakeLists.txt
Total Errors: 0
# ✅ Success!
```

### After Fix (Empty Replacement Bug)
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

---

## Configuration Summary

### Key Settings in `.cmake-format.yaml`

| Setting | Value | Rationale |
|---------|-------|-----------|
| `line_width` | 100 | Matches `.clang-format` ColumnLimit |
| `tab_size` | 2 | Matches project indentation standard |
| `use_tabchars` | false | Use spaces, not tabs |
| `separate_ctrl_name_with_space` | **false** | **Aligns with cmakelint** |
| `separate_fn_name_with_space` | false | Consistent with control flow |
| `dangle_parens` | true | Improves readability |
| `enable_sort` | true | Enables argument sorting |
| `autosort` | false | Prevents unexpected reordering |
| `command_case` | canonical | Preserves canonical case |

### cmakelint Configuration (`.cmakelintrc`)

```
filter=-convention/filename,-linelength,-package/consistency,-readability/logic,-readability/mixedcase,-readability/wonkycase,-syntax,-whitespace/eol,+whitespace/extra,-whitespace/indent,-whitespace/mismatch,-whitespace/newline,-whitespace/tabs
```

**Key point**: `+whitespace/extra` is enabled, which checks for extra spaces after control flow keywords.

---

## Best Practices

### For Developers

1. **Always run cmake-format before committing CMake files**:
   ```bash
   cmake-format -i --config-file=.cmake-format.yaml CMakeLists.txt
   ```

2. **Verify with cmakelint**:
   ```bash
   cmakelint --config=.cmakelintrc CMakeLists.txt
   ```

3. **Use lintrunner for integrated workflow**:
   ```bash
   lintrunner --take CMAKEFORMAT --apply-patches
   lintrunner --only=CMAKE
   ```

### For CI/CD

1. **Check formatting in CI**:
   ```bash
   cmake-format --check --config-file=.cmake-format.yaml $(find . -name "CMakeLists.txt")
   ```

2. **Run linting in CI**:
   ```bash
   lintrunner --only=CMAKE
   lintrunner --only=CMAKEFORMAT
   ```

---

## Lessons Learned

1. **Configuration validation is critical**: Always validate configuration files against the tool's specification using `--dump-config`.

2. **Tool compatibility must be verified**: When using multiple tools (formatter + linter), ensure their configurations don't conflict.

3. **Documentation prevents regressions**: Comprehensive comments in configuration files help prevent future misconfigurations.

4. **Test with real files**: Always test configuration changes with actual project files, not just examples.

---

## References

- [cmake-format Documentation](https://cmake-format.readthedocs.io/)
- [cmakelang GitHub](https://github.com/cheshirekow/cmake_format)
- [cmakelint GitHub](https://github.com/cmake-lint/cmake-lint)
- [XSigma Coding Standards](.augment/rules/coding.md)
- [XSigma CMake Format README](README_CMAKE_FORMAT.md)

---

## Appendix: Configuration Diff

### Before (Problematic)
```yaml
format:
  line_width: 100
  indent_width: 2  # ❌ Not recognized
  tab_size: 2
  use_tabchars: false
  dangle_parens: true
  separate_ctrl_name_with_space: true  # ❌ Conflicts with cmakelint
  separate_fn_name_with_space: false
  comment_prefix: '  #'  # ❌ Not recognized
  enable_markup: true  # ❌ Wrong section
  max_subgroups_hwrap: 3
  max_paren_depth: 6  # ❌ Not recognized
  enable_sort: true
  autosort: false
```

### After (Fixed)
```yaml
format:
  disable: false
  line_width: 100
  tab_size: 2
  use_tabchars: false
  fractional_tab_policy: use-space
  max_subgroups_hwrap: 3
  max_pargs_hwrap: 6
  max_rows_cmdline: 2
  separate_ctrl_name_with_space: false  # ✅ Fixed
  separate_fn_name_with_space: false
  dangle_parens: true
  dangle_align: prefix
  min_prefix_chars: 4
  max_prefix_chars: 10
  max_lines_hwrap: 2
  line_ending: unix
  command_case: canonical
  keyword_case: unchanged
  always_wrap: []
  enable_sort: true
  autosort: false
  require_valid_layout: false
  layout_passes: {}
```

---

**Status**: ✅ Conflict Resolved  
**Verification**: ✅ Passed  
**Documentation**: ✅ Updated  
**Ready for Production**: ✅ Yes

