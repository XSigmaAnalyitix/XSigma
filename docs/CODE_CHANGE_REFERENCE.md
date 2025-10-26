# Code Change Reference: Double .html Extension Fix

## File: Tools/coverage/msvc_coverage.py

### Function: `_collect_html_files_recursive()`

**Location**: Lines 54-100

**Purpose**: Recursively collect HTML files from OpenCppCoverage directory structure

### Change Summary

**What Changed**: The function now strips the `.html` extension from file paths before returning them

**Why**: The `json_html_generator.py` expects file paths WITHOUT the `.html` extension so it can add it when generating links

### Detailed Code Change

#### Before (Lines 54-94)

```python
def _collect_html_files_recursive(html_dir: Path) -> list[tuple[Path, str]]:
    """Recursively collect HTML files from OpenCppCoverage directory structure.

    OpenCppCoverage generates HTML in a nested structure:
    html/
    ├── index.html (main summary)
    └── Modules/
        ├── module1/
        │   ├── file1.html
        │   └── file2.html
        └── module2/
            └── file3.html

    Args:
        html_dir: Root HTML directory.

    Returns:
        List of tuples (file_path, relative_path) for non-excluded HTML files.
    """
    html_files = []

    # Recursively find all HTML files in subdirectories
    for html_file in html_dir.rglob("*.html"):
        # Skip the main index.html
        if html_file.name == "index.html" and html_file.parent == html_dir:
            continue

        # Get relative path for display
        try:
            rel_path = html_file.relative_to(html_dir)
        except ValueError:
            rel_path = html_file

        # Check if file should be excluded
        if _should_exclude_file(str(rel_path)):
            logger.debug(f"Excluding file from coverage: {rel_path}")
            continue

        html_files.append((html_file, str(rel_path)))  # ← ISSUE: Stores with .html

    return html_files
```

#### After (Lines 54-100)

```python
def _collect_html_files_recursive(html_dir: Path) -> list[tuple[Path, str]]:
    """Recursively collect HTML files from OpenCppCoverage directory structure.

    OpenCppCoverage generates HTML in a nested structure:
    html/
    ├── index.html (main summary)
    └── Modules/
        ├── module1/
        │   ├── file1.html (represents file1.cxx or file1.h)
        │   └── file2.html (represents file2.cxx or file2.h)
        └── module2/
            └── file3.html (represents file3.cxx or file3.h)

    Args:
        html_dir: Root HTML directory.

    Returns:
        List of tuples (file_path, source_file_path) where source_file_path
        is the relative path WITHOUT the .html extension (e.g., Modules/module1/file1).
    """
    html_files = []

    # Recursively find all HTML files in subdirectories
    for html_file in html_dir.rglob("*.html"):
        # Skip the main index.html
        if html_file.name == "index.html" and html_file.parent == html_dir:
            continue

        # Get relative path for display
        try:
            rel_path = html_file.relative_to(html_dir)
        except ValueError:
            rel_path = html_file

        # Check if file should be excluded
        if _should_exclude_file(str(rel_path)):
            logger.debug(f"Excluding file from coverage: {rel_path}")
            continue

        # Remove the .html extension to get the source file path
        # OpenCppCoverage names HTML files as: yyy.html (for yyy.cxx or yyy.h)
        # We need to store just the path without .html so json_html_generator can add it
        source_file_path = str(rel_path.with_suffix(''))  # ← FIX: Strip .html

        html_files.append((html_file, source_file_path))  # ← FIX: Store without .html

    return html_files
```

### Key Differences

| Aspect | Before | After |
|--------|--------|-------|
| **Stored Path** | `Modules/yyy/yyy.html` | `Modules/yyy/yyy` |
| **JSON Value** | `"file": "Modules/yyy/yyy.html"` | `"file": "Modules/yyy/yyy"` |
| **Generated Link** | `href="yyy.html.html"` | `href="yyy.html"` |
| **Link Status** | ❌ Broken (404) | ✅ Working |

### Technical Details

**Method Used**: `Path.with_suffix('')`

```python
from pathlib import Path

# Example
html_file = Path("Modules/yyy/yyy.html")
source_file = html_file.with_suffix('')
print(source_file)  # Output: Modules/yyy/yyy
```

**Why This Works**:
- `with_suffix('')` removes the file extension
- Preserves the full directory path
- Works with any file extension (`.html`, `.cxx`, `.h`, etc.)

### Impact on Data Flow

**JSON Generation** (in `_generate_json_from_html()`):
```python
file_data = {
    "file": rel_path,  # Now: "Modules/yyy/yyy" (without .html)
    "line_coverage": { ... }
}
```

**HTML Generation** (in `json_html_generator.py` line 162):
```python
file_link = f"{Path(file_name).name}.html"
# file_name = "Modules/yyy/yyy"
# Path(file_name).name = "yyy"
# file_link = "yyy.html"  ← CORRECT!
```

### Backward Compatibility

✅ **Fully Backward Compatible**

- No changes to function signature
- No changes to return type
- Only internal path handling modified
- JSON schema structure unchanged
- HTML output format unchanged

### Testing the Change

```python
# Test the fix
from pathlib import Path

# Simulate the fix
html_file = Path("Modules/yyy/yyy.html")
source_file_path = str(html_file.with_suffix(''))

print(f"HTML file: {html_file}")
print(f"Source path: {source_file_path}")
print(f"Generated link: {Path(source_file_path).name}.html")

# Output:
# HTML file: Modules/yyy/yyy.html
# Source path: Modules/yyy/yyy
# Generated link: yyy.html
```

### Related Code

**In `json_html_generator.py` (line 162)**:
```python
file_link = f"{Path(file_name).name}.html"
```

This line expects `file_name` to NOT have the `.html` extension, which is now guaranteed by our fix.

---

**Status**: ✅ IMPLEMENTED

**Lines Changed**: 54-100 in `Tools/coverage/msvc_coverage.py`

**Total Changes**: 1 function modified, 3 lines added (comments + fix)

