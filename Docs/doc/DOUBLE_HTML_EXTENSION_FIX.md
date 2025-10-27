# Fix for Double .html Extension Issue

## Problem

The generated `index.html` file contained links with double `.html` extensions:
- **Incorrect**: `Modules/yyy/yyy.html.html`
- **Expected**: `Modules/yyy/yyy.html`

## Root Cause

The issue was in how file paths were being stored in the JSON coverage report:

1. **In `_collect_html_files_recursive()`**:
   - Was storing the full relative path including `.html` extension
   - Example: `Modules/yyy/yyy.html`

2. **In `json_html_generator.py` (line 162)**:
   - Takes the file name from JSON and appends `.html`
   - Code: `file_link = f"{Path(file_name).name}.html"`
   - Result: `yyy.html` + `.html` = `yyy.html.html`

## Solution

Modified `_collect_html_files_recursive()` in `Tools/coverage/msvc_coverage.py` to:

1. **Remove the `.html` extension** from the relative path before storing in JSON
2. **Use `Path.with_suffix('')`** to strip the extension
3. **Store source file path** without extension (e.g., `Modules/yyy/yyy`)

### Code Change

**Before:**
```python
# Get relative path for display
try:
    rel_path = html_file.relative_to(html_dir)
except ValueError:
    rel_path = html_file

# Check if file should be excluded
if _should_exclude_file(str(rel_path)):
    logger.debug(f"Excluding file from coverage: {rel_path}")
    continue

html_files.append((html_file, str(rel_path)))  # ← Stored with .html
```

**After:**
```python
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
source_file_path = str(rel_path.with_suffix(''))  # ← Stored without .html

html_files.append((html_file, source_file_path))
```

## Data Flow

### Before Fix
```
OpenCppCoverage HTML:
  Modules/yyy/yyy.html

↓ _collect_html_files_recursive()

JSON (coverage_summary.json):
  "file": "Modules/yyy/yyy.html"

↓ json_html_generator.py (line 162)

HTML Link:
  file_link = f"{Path('Modules/yyy/yyy.html').name}.html"
  file_link = f"yyy.html.html"  ← WRONG!
```

### After Fix
```
OpenCppCoverage HTML:
  Modules/yyy/yyy.html

↓ _collect_html_files_recursive()

JSON (coverage_summary.json):
  "file": "Modules/yyy/yyy"  ← No .html extension

↓ json_html_generator.py (line 162)

HTML Link:
  file_link = f"{Path('Modules/yyy/yyy').name}.html"
  file_link = f"yyy.html"  ← CORRECT!
```

## File Paths in JSON

### Before Fix
```json
{
  "files": [
    {
      "file": "Modules/yyy/yyy.html",
      "line_coverage": { ... }
    },
    {
      "file": "Modules/yyy/other.html",
      "line_coverage": { ... }
    }
  ]
}
```

### After Fix
```json
{
  "files": [
    {
      "file": "Modules/yyy/yyy",
      "line_coverage": { ... }
    },
    {
      "file": "Modules/yyy/other",
      "line_coverage": { ... }
    }
  ]
}
```

## Generated HTML Links

### Before Fix
```html
<a href="yyy.html.html">Modules/yyy/yyy</a>  ← BROKEN LINK
```

### After Fix
```html
<a href="yyy.html">Modules/yyy/yyy</a>  ← CORRECT LINK
```

## Testing the Fix

### Step 1: Run Coverage
```bash
cd Scripts
python setup.py vs22.debug.coverage
```

### Step 2: Check JSON File
```bash
# Verify file paths don't have double extensions
python -m json.tool build_vs22_coverage/coverage_report/coverage_summary.json | grep '"file"'

# Expected output:
# "file": "Modules/yyy/yyy",
# "file": "Modules/yyy/other",
# NOT:
# "file": "Modules/yyy/yyy.html",
```

### Step 3: Check HTML Links
```bash
# Open the generated HTML report
build_vs22_coverage/coverage_report/html/index.html

# Verify:
# - Links point to files like "yyy.html" (not "yyy.html.html")
# - Clicking links opens the correct coverage pages
# - No 404 errors in browser console
```

### Step 4: Verify Generated Files
```bash
# Check that HTML files exist with correct names
ls -la build_vs22_coverage/coverage_report/html/

# Expected:
# index.html
# yyy.html (not yyy.html.html)
# other.html (not other.html.html)
```

## Impact

✅ **Fixes**: Double `.html` extension issue
✅ **Maintains**: Backward compatibility with JSON schema
✅ **Improves**: HTML report usability
✅ **No Breaking Changes**: Only internal path handling changed

## Related Files

- `Tools/coverage/msvc_coverage.py` - Fixed `_collect_html_files_recursive()`
- `Tools/coverage/html_report/json_html_generator.py` - Uses corrected paths
- `build_vs22_coverage/coverage_report/coverage_summary.json` - Now has correct file paths

## Verification Checklist

- [ ] Run: `python setup.py vs22.debug.coverage`
- [ ] Check: JSON file paths don't have `.html` extension
- [ ] Check: HTML links are correct (single `.html` extension)
- [ ] Check: All links in index.html are clickable
- [ ] Check: No 404 errors when clicking links
- [ ] Check: Coverage data displays correctly on each page
