# XSigma Code Coverage - Quick Start Guide

## 30-Second Setup

### Prerequisites
- Windows with OpenCppCoverage installed
- XSigma built with test executables
- Python 3.7+

### One-Command Coverage Report

```bash
cd Tools/code_coverage
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output coverage_report
```

**Output**: Open `coverage_report/html/index.html` in your browser

## Common Tasks

### Generate Coverage Report
```bash
python run_coverage_workflow.py \
    --test-exe <path-to-test-exe> \
    --sources <path-to-library> \
    --output <output-dir>
```

### Collect Coverage Data Only
```bash
python collect_coverage_data.py \
    --test-exe <path-to-test-exe> \
    --sources <path-to-library> \
    --output <output-dir>
```

### Generate Report from Existing Data
```bash
python generate_html_report.py \
    --coverage-data <path-to-coverage.xml> \
    --output <output-dir> \
    --source-root <path-to-xsigma>
```

### Enable Verbose Output
Add `--verbose` flag to any command:
```bash
python run_coverage_workflow.py \
    --test-exe <path> \
    --sources <path> \
    --verbose
```

### Exclude Directories from Coverage
To exclude test code or other directories:
```bash
python run_coverage_workflow.py \
    --test-exe <path> \
    --sources <path> \
    --excluded-sources "*Testing*"
```

Multiple exclusions:
```bash
python run_coverage_workflow.py \
    --test-exe <path> \
    --sources <path> \
    --excluded-sources "*Testing*" \
    --excluded-sources "*Mock*"
```

## File Locations

| File | Purpose |
|------|---------|
| `collect_coverage_data.py` | Collect coverage data from tests |
| `generate_html_report.py` | Generate HTML reports from coverage data |
| `run_coverage_workflow.py` | Run complete workflow (recommended) |
| `WORKFLOW.md` | Detailed documentation |
| `IMPLEMENTATION_SUMMARY.md` | Implementation details |

## Troubleshooting

### OpenCppCoverage Not Found
```bash
# Install via Chocolatey
choco install opencppcoverage

# Or download from
# https://github.com/OpenCppCoverage/OpenCppCoverage
```

### No Coverage Data Generated
1. Verify test executable runs: `CoreCxxTests.exe --help`
2. Check source directory path is correct
3. Run with `--verbose` for details

### HTML Report Not Generated
1. Verify coverage data collection succeeded
2. Check `coverage.xml` exists in data directory
3. Run collection step again

## Example Workflow

```bash
# 1. Navigate to coverage tools
cd Tools/code_coverage

# 2. Run complete workflow
python run_coverage_workflow.py \
    --test-exe C:\dev\build_ninja_lto\bin\CoreCxxTests.exe \
    --sources c:\dev\XSigma\Library \
    --output my_coverage

# 3. Open report in browser
start my_coverage/html/index.html
```

## Output Structure

```
coverage_report/
├── data/
│   ├── coverage.cov          # Raw coverage file
│   └── coverage.xml          # Cobertura XML
└── html/
    ├── index.html            # Summary page
    ├── Library_Core_*.html    # File reports
    └── ...
```

## Key Features

✓ **Modular**: Use components independently or together
✓ **Fast**: Efficient coverage collection and report generation
✓ **Beautiful**: Rich HTML reports with line-by-line visualization
✓ **Flexible**: Customize output locations and source roots
✓ **Documented**: Comprehensive guides and examples

## Next Steps

1. **Run your first report** using the one-command setup above
2. **Review the HTML report** to understand coverage
3. **Read WORKFLOW.md** for advanced usage
4. **Integrate with CI/CD** using examples in WORKFLOW.md

## Support

For detailed information, see:
- `WORKFLOW.md` - Complete workflow documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `README.md` - Original PyTorch coverage tool docs

## Quick Reference

| Task | Command |
|------|---------|
| Full workflow | `python run_coverage_workflow.py --test-exe <exe> --sources <src> --output <out>` |
| Collect only | `python collect_coverage_data.py --test-exe <exe> --sources <src> --output <out>` |
| Report only | `python generate_html_report.py --coverage-data <xml> --output <out>` |
| Help | `python <script> --help` |
| Verbose | Add `--verbose` to any command |
