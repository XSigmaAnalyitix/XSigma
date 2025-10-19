# Multi-File HTML Coverage Report - Quick Start Guide

## What's New?

The XSigma project now generates **rich, multi-file HTML coverage reports** instead of a single monolithic HTML file. The new reports include:

✅ **Index/Summary Page** - Overall coverage statistics and file listing  
✅ **Individual File Reports** - Line-by-line coverage visualization  
✅ **Color-Coded Coverage** - Visual indicators for coverage quality  
✅ **Professional Design** - Responsive, easy-to-navigate interface  

## Quick Start (5 Minutes)

### Step 1: Build with Coverage

```bash
cd Scripts
python setup.py ninja.clang.config.build.test.coverage
```

Or with GCC:
```bash
python setup.py ninja.gcc.config.build.test.coverage
```

### Step 2: Wait for Report Generation

The build will automatically generate the multi-file HTML report. Look for output like:
```
[INFO] start multi-file html report generation
[INFO] multi-file html generation take time: X.XX seconds
```

### Step 3: Open the Report

**Windows:**
```bash
start tools/code_coverage/profile/html_details/index.html
```

**macOS:**
```bash
open tools/code_coverage/profile/html_details/index.html
```

**Linux:**
```bash
xdg-open tools/code_coverage/profile/html_details/index.html
```

## Report Structure

### Index Page (`index.html`)

Shows:
- **Overall Coverage %** - Total coverage across all files
- **Lines Covered** - Total covered lines
- **Lines Uncovered** - Total uncovered lines
- **Total Lines** - Total lines analyzed
- **Files Analyzed** - Number of source files

Plus a **File Summary Table** with:
- File name (clickable link)
- Covered lines
- Uncovered lines
- Total lines
- Coverage percentage (color-coded)

### Individual File Reports

Each file gets its own report showing:
- **File path** and coverage stats
- **Line-by-line visualization**:
  - 🟢 Green = Covered line
  - 🔴 Red = Uncovered line
  - ⚪ White = Non-executable line
- **Source code** with line numbers
- **Back link** to summary

## Color Coding

| Coverage | Color | Meaning |
|----------|-------|---------|
| ≥ 80% | 🟢 Green | Excellent |
| 60-79% | 🟠 Orange | Good |
| < 60% | 🔴 Red | Needs Improvement |

## Report Location

```
tools/code_coverage/profile/html_details/
├── index.html                    ← Start here
├── Core_Math_Vector.h.html
├── Core_Math_Matrix.h.html
├── Core_Util_Helper.cpp.html
└── ... (one file per source file)
```

## Common Tasks

### View Coverage Summary
1. Open `index.html`
2. Review overall statistics
3. Identify files with low coverage

### Examine Specific File
1. Click on file name in summary table
2. Review line-by-line coverage
3. Identify uncovered lines (red)

### Share Report
1. Zip the entire `html_details/` directory
2. Send to team members
3. They can open `index.html` in any browser

### Archive Report
```bash
# Create timestamped archive
zip -r coverage_report_$(date +%Y%m%d_%H%M%S).zip tools/code_coverage/profile/html_details/
```

### View with HTTP Server
```bash
cd tools/code_coverage/profile/html_details
python -m http.server 8000
# Open http://localhost:8000 in browser
```

## Troubleshooting

### Reports Not Generated?

1. **Check coverage is enabled:**
   ```bash
   grep XSIGMA_ENABLE_COVERAGE build_ninja_*/CMakeCache.txt
   ```

2. **Verify tests ran:**
   - Look for test output in build log
   - Check for `.profraw` files in build directory

3. **Check for errors:**
   - Look for error messages in build output
   - Check `tools/code_coverage/profile/log/` for logs

### Source Code Not Showing?

- Ensure source files exist at their original paths
- Check file permissions
- Verify source root is set correctly

### Styling Issues?

- Clear browser cache (Ctrl+Shift+Delete)
- Try a different browser
- Check for browser extensions blocking CSS

## Tips & Tricks

### Find Worst Coverage
1. Open `index.html`
2. Files are sorted by coverage (worst first)
3. Focus on red/orange files

### Track Coverage Over Time
1. Archive each report with timestamp
2. Compare coverage percentages
3. Identify improving/declining areas

### Share with Team
1. Generate report
2. Zip `html_details/` directory
3. Upload to shared drive or email
4. Team can view in any browser

### Integrate with CI/CD
```bash
# In CI script
python setup.py ninja.clang.config.build.test.coverage

# Archive report
zip -r coverage_report.zip tools/code_coverage/profile/html_details/

# Upload to artifact storage
# (e.g., S3, Azure Blob, etc.)
```

## Report Statistics Example

```
Overall Coverage: 85.42%
Lines Covered: 12,450
Lines Uncovered: 2,130
Total Lines: 14,580
Files Analyzed: 156

Top 3 Best Coverage:
  1. Core/Math/Vector.h - 98.5%
  2. Core/Math/Matrix.h - 96.2%
  3. Core/Util/Helper.cpp - 94.8%

Top 3 Worst Coverage:
  1. Core/Advanced/Algorithm.h - 45.3%
  2. Core/Experimental/Feature.cpp - 52.1%
  3. Core/Legacy/OldCode.h - 58.7%
```

## Next Steps

1. ✅ Generate your first coverage report
2. ✅ Review the index page
3. ✅ Click on a file to see details
4. ✅ Identify areas for improvement
5. ✅ Add tests for uncovered code
6. ✅ Re-run coverage to verify improvement

## Need Help?

- See [MULTIFILE_HTML_COVERAGE_REPORT.md](MULTIFILE_HTML_COVERAGE_REPORT.md) for detailed documentation
- Check [COVERAGE_QUICK_REFERENCE.md](COVERAGE_QUICK_REFERENCE.md) for coverage commands
- Review [COVERAGE_VERIFICATION_GUIDE.md](COVERAGE_VERIFICATION_GUIDE.md) for troubleshooting

## Summary

The new multi-file HTML coverage reports provide:
- 📊 Clear overview of coverage statistics
- 📁 Individual file analysis
- 🎨 Professional, easy-to-read design
- 🔗 Easy navigation between files
- 📦 Self-contained, shareable reports

**Happy coverage analysis! 🎉**

