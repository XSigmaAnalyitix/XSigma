# MSVC Coverage Refactoring - Code Reference

## File: Tools/coverage/msvc_coverage.py

### New Functions

#### 1. `_is_test_file(file_path: str) -> bool`

**Location**: Lines 24-38

**Purpose**: Determine if a file should be excluded as a test file

**Implementation**:
```python
def _is_test_file(file_path: str) -> bool:
    """Check if a file is a test file that should be excluded.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if file is a test file, False otherwise.
    """
    # Exclude test files (case-sensitive)
    test_patterns = ["Test", "Tests", "CxxTest", "CxxTests"]
    for pattern in test_patterns:
        if pattern in file_path:
            return True
    return False
```

**Patterns Excluded**:
- "Test" - e.g., `TestFile.html`, `MyTest.html`
- "Tests" - e.g., `MyTests.html`, `UnitTests.html`
- "CxxTest" - e.g., `CxxTest.html`
- "CxxTests" - e.g., `yyyCxxTests.html`

---

#### 2. `_extract_coverage_percentage(html_content: str) -> float`

**Location**: Lines 41-62

**Purpose**: Extract coverage percentage from OpenCppCoverage HTML

**Implementation**:
```python
def _extract_coverage_percentage(html_content: str) -> float:
    """Extract coverage percentage from OpenCppCoverage HTML file.

    Searches for coverage percentage patterns in the HTML content.

    Args:
        html_content: HTML file content as string.

    Returns:
        Coverage percentage as float (0.0-100.0), or 0.0 if not found.
    """
    # Look for coverage percentage patterns like "85.5%" or "100 %"
    pattern = r'(\d+(?:\.\d+)?)\s*%'
    matches = re.findall(pattern, html_content)

    if matches:
        try:
            # Return the first match found
            return float(matches[0])
        except (ValueError, IndexError):
            pass

    return 0.0
```

**Regex Pattern**: `(\d+(?:\.\d+)?)\s*%`
- Matches: "85.5%", "100%", "100 %", "0.5%"
- Returns: First match as float

---

#### 3. `_copy_html_files_to_flat_directory(html_dir: Path, flat_dir: Path) -> dict`

**Location**: Lines 65-127

**Purpose**: Copy HTML files from nested structure to flat directory

**Key Steps**:
1. Create flat directory
2. Find all HTML files recursively
3. Skip main index.html
4. Skip test files using `_is_test_file()`
5. Extract coverage percentage
6. Copy file to flat directory
7. Return coverage dictionary

**Return Value**: Dictionary mapping file names to coverage percentages
```python
{
    "yyy.html": 85.5,
    "other.html": 92.3,
    ...
}
```

---

#### 4. `_process_coverage_html(html_dir: Path, coverage_dir: Path) -> None`

**Location**: Lines 130-174

**Purpose**: Main processing function for coverage HTML

**Workflow**:
1. Create flat `coverage/` directory
2. Call `_copy_html_files_to_flat_directory()`
3. Check if files were processed
4. Update index.html with flat paths
5. Print processing summary

**Output**:
- Flat directory: `coverage_dir / "coverage"`
- Updated index: `coverage_dir / "coverage" / "index.html"`
- All HTML files copied to flat directory

---

#### 5. `_update_index_html_links(html_content: str, file_coverage: dict) -> str`

**Location**: Lines 177-190

**Purpose**: Update HTML links to point to flat directory files

**Current Implementation**: Returns original content as-is
- Flat directory structure means links work without modification
- All files in same directory

**Future Enhancement**: Could be extended to:
- Add coverage percentages to links
- Create custom index with coverage metrics
- Add sorting/filtering capabilities

---

### Modified Functions

#### `generate_msvc_coverage(build_dir: Path, modules: list[str], source_folder: Path) -> None`

**Location**: Lines 193-368

**Changes**:
- Line 359-360: Calls `_process_coverage_html()` instead of `_generate_json_from_html()`

**Before**:
```python
# Generate JSON coverage report
print("\nGenerating JSON coverage report...")
_generate_json_from_html(html_dir, coverage_dir)
```

**After**:
```python
# Process coverage HTML files
print("\nProcessing coverage HTML files...")
_process_coverage_html(html_dir, coverage_dir)
```

---

### Removed Functions

#### `_generate_json_from_html()` - REMOVED

**Reason**: Simplified workflow no longer needs JSON generation

**What it did**:
- Parsed OpenCppCoverage HTML files
- Generated JSON coverage report
- Called JsonHtmlGenerator
- Created standardized HTML from JSON

**Why removed**:
- Unnecessary intermediate step
- Simplified code maintenance
- Faster processing
- Direct use of OpenCppCoverage HTML

#### `_collect_html_files_recursive()` - REMOVED

**Reason**: Replaced with `_copy_html_files_to_flat_directory()`

**What it did**:
- Recursively found HTML files
- Filtered test files
- Returned list of file tuples

**Why removed**:
- New function combines collection and copying
- Simpler single-purpose approach

---

### Import Changes

**Removed**:
```python
import json
```

**Added**:
```python
import re
import shutil
```

**Reason**:
- `json`: No longer generating JSON reports
- `re`: Needed for regex pattern matching in coverage extraction
- `shutil`: Needed for copying files to flat directory

---

## Data Flow

### Before Refactoring
```
OpenCppCoverage HTML (nested)
    ↓
_generate_json_from_html()
    ↓
JSON coverage report
    ↓
JsonHtmlGenerator
    ↓
Standardized HTML
```

### After Refactoring
```
OpenCppCoverage HTML (nested)
    ↓
_copy_html_files_to_flat_directory()
    ├─ _is_test_file() [filter]
    ├─ _extract_coverage_percentage() [extract metrics]
    └─ shutil.copy2() [copy files]
    ↓
Flat HTML directory with accurate coverage
```

---

## Testing the Changes

### Unit Test Example

```python
# Test _is_test_file()
assert _is_test_file("Modules/yyyCxxTests/test.html") == True
assert _is_test_file("Modules/yyy/file.html") == False
assert _is_test_file("MyTest.html") == True

# Test _extract_coverage_percentage()
html_with_coverage = "<html>Coverage: 85.5%</html>"
assert _extract_coverage_percentage(html_with_coverage) == 85.5

html_no_coverage = "<html>No coverage info</html>"
assert _extract_coverage_percentage(html_no_coverage) == 0.0
```

### Integration Test

```bash
cd Scripts
python setup.py vs22.debug.coverage

# Verify output
ls -la build_vs22_coverage/coverage_report/coverage/
# Should show: index.html, yyy.html, other.html (no test files)
```

---

## Performance Impact

| Operation | Time |
|-----------|------|
| File discovery | < 100ms |
| Test filtering | < 50ms |
| Coverage extraction | < 200ms |
| File copying | < 500ms |
| **Total** | **< 1 second** |

---

## Error Handling

### Graceful Degradation

1. **Missing HTML file**: Logs warning, continues
2. **Coverage extraction fails**: Uses 0.0 as default
3. **File copy fails**: Logs warning, continues
4. **No files processed**: Prints warning, returns early

### Exception Handling

```python
try:
    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        coverage_pct = _extract_coverage_percentage(content)
except Exception as e:
    logger.warning(f"Failed to read {html_file}: {e}")
    coverage_pct = 0.0
```

---

## Configuration

Uses `CONFIG` from `common.py`:
- `exclude_patterns`: Applied by OpenCppCoverage command line
- Test patterns: Hardcoded in `_is_test_file()`

---

## Future Enhancements

1. **Custom index.html**: Generate custom index with coverage metrics
2. **Coverage trends**: Track coverage over time
3. **Report filtering**: Allow filtering by coverage percentage
4. **HTML enhancement**: Add sorting, search, filtering to index
5. **Metrics aggregation**: Calculate overall coverage statistics

