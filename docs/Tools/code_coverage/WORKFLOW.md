# XSigma Code Coverage Workflow

## Overview

This document describes the modular code coverage workflow for XSigma. The workflow is split into two independent, composable steps:

1. **Coverage Data Collection** (`collect_coverage_data.py`)
2. **HTML Report Generation** (`generate_html_report.py`)

These can be run independently or orchestrated together using the unified workflow script.

## Architecture

### Separation of Concerns

The workflow follows a clean architectural pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                  Coverage Workflow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐      ┌──────────────────────┐   │
│  │  Data Collection     │      │  Report Generation   │   │
│  │  (OpenCppCoverage)   │─────▶│  (HTML Generator)    │   │
│  │                      │      │                      │   │
│  │ • Runs test exe      │      │ • Parses coverage    │   │
│  │ • Collects raw data  │      │ • Generates HTML     │   │
│  │ • Exports to XML     │      │ • Creates index      │   │
│  └──────────────────────┘      └──────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
Tools/code_coverage/
├── README.md                          # Original PyTorch coverage tool docs
├── WORKFLOW.md                        # This file
├── oss_coverage.py                    # PyTorch OSS coverage tool
├── collect_coverage_data.py           # NEW: Data collection script
├── generate_html_report.py            # NEW: Report generation script
├── run_coverage_workflow.py           # NEW: Unified workflow orchestrator
├── package/
│   ├── tool/
│   │   ├── html_report_generator.py   # HTML report generation
│   │   ├── clang_coverage.py          # Clang coverage support
│   │   ├── gcc_coverage.py            # GCC coverage support
│   │   └── ...
│   ├── oss/
│   │   └── ...
│   └── util/
│       └── ...
```

## Quick Start

### Option 1: Complete Workflow (Recommended)

Run the entire workflow in one command:

```bash
cd Tools/code_coverage
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_report
```

This will:
1. Collect coverage data from CoreCxxTests.exe
2. Generate HTML reports in `coverage_report/html/`
3. Display the path to `index.html`

### Option 2: Step-by-Step (For Advanced Users)

#### Step 1: Collect Coverage Data

```bash
python collect_coverage_data.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_data
```

This generates:
- `coverage_data/coverage.cov` - Raw coverage file
- `coverage_data/coverage.xml` - Cobertura XML format

#### Step 2: Generate HTML Report

```bash
python generate_html_report.py \
    --coverage-data coverage_data/coverage.xml \
    --output html_report \
    --source-root c:\dev\XSigma
```

This generates:
- `html_report/index.html` - Summary page
- `html_report/*.html` - Individual file reports

## Usage Examples

### Basic Usage

```bash
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library
```

### With Custom Output Directory

```bash
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output my_coverage_report
```

### Excluding Directories from Coverage

To exclude specific directories (e.g., Testing folder) from coverage collection:

```bash
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_report \
    --excluded-sources "*Testing*"
```

You can specify multiple exclusion patterns:

```bash
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_report \
    --excluded-sources "*Testing*" \
    --excluded-sources "*Mock*" \
    --excluded-sources "*Stub*"
```

**Note**: By default, the Testing folder is NOT automatically excluded. Use the `--excluded-sources` flag to exclude it.

### With Verbose Output

```bash
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --verbose
```

### Collecting Data Only

```bash
python collect_coverage_data.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_data \
    --verbose
```

### Generating Report from Existing Data

```bash
python generate_html_report.py \
    --coverage-data coverage_data/coverage.xml \
    --output html_report \
    --source-root c:\dev\XSigma \
    --verbose
```

## Requirements

### Windows (OpenCppCoverage)

- **OpenCppCoverage.exe** installed and in PATH
  - Download from: https://github.com/OpenCppCoverage/OpenCppCoverage
  - Or install via: `choco install opencppcoverage`

### All Platforms

- Python 3.7+
- XSigma source code
- Built test executables

## Output

### Directory Structure

```
coverage_report/
├── data/
│   ├── coverage.cov          # Raw coverage file
│   └── coverage.xml          # Cobertura XML format
└── html/
    ├── index.html            # Summary page with statistics
    ├── Library_Core_*.html    # Individual file reports
    └── ...
```

### HTML Report Features

- **Summary Page** (`index.html`)
  - Overall coverage percentage
  - Lines covered/uncovered
  - Files analyzed
  - File-by-file statistics table

- **Individual File Reports**
  - Line-by-line coverage visualization
  - Color-coded lines (green=covered, red=uncovered)
  - Coverage statistics per file
  - Navigation back to summary

## Troubleshooting

### OpenCppCoverage Not Found

**Error**: `OpenCppCoverage.exe not found`

**Solution**:
1. Install OpenCppCoverage from https://github.com/OpenCppCoverage/OpenCppCoverage
2. Add to PATH or specify full path
3. Verify installation: `where OpenCppCoverage.exe`

### No Coverage Data Generated

**Error**: Coverage files are empty or missing

**Solution**:
1. Verify test executable exists and runs: `CoreCxxTests.exe --help`
2. Check source directory path is correct
3. Run with `--verbose` flag for detailed output
4. Ensure test executable was built with coverage instrumentation

### HTML Report Not Generated

**Error**: `No coverage data file found`

**Solution**:
1. Verify coverage data collection completed successfully
2. Check that `coverage.xml` exists in data directory
3. Run collection step again with `--verbose`

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Collect Coverage
  run: |
    cd Tools/code_coverage
    python run_coverage_workflow.py \
      --test-exe ${{ github.workspace }}/build/bin/CoreCxxTests.exe \
      --sources ${{ github.workspace }}/Library \
      --output coverage_report

- name: Upload Coverage Report
  uses: actions/upload-artifact@v2
  with:
    name: coverage-report
    path: Tools/code_coverage/coverage_report/html/
```

## Advanced Configuration

### Custom Source Root

For reading source files in HTML reports:

```bash
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --source-root c:\dev\XSigma
```

### Separate Data and Report Generation

For CI/CD pipelines that need to separate concerns:

```bash
# On build machine
python collect_coverage_data.py --test-exe ... --sources ... --output data

# On reporting machine
python generate_html_report.py --coverage-data data/coverage.xml --output report
```

## See Also

- `README.md` - Original PyTorch coverage tool documentation
- `package/tool/html_report_generator.py` - HTML generation implementation
- `package/tool/clang_coverage.py` - Clang coverage support
- `package/tool/gcc_coverage.py` - GCC coverage support

