# XSigma Code Coverage User Guide

## Overview

The XSigma coverage system provides comprehensive code coverage analysis for C++ projects across multiple compilers (Clang, GCC, MSVC). It generates both JSON and HTML reports for easy integration into CI/CD pipelines and local development workflows.

### Key Features

- **Multi-compiler support**: Clang, GCC, and MSVC
- **Multiple coverage metrics**: Line, function, and region coverage
- **JSON reports**: Machine-readable format for CI/CD integration
- **HTML reports**: Professional, interactive reports for human review
- **Cobertura-compatible**: Standard format for industry tools
- **Cross-platform**: Works on Windows, macOS, and Linux

---

## Running Coverage for Each Compiler

### Clang Coverage

Run coverage build with Clang:

```bash
cd Scripts
python setup.py config.build.ninja.clang.TEST.coverage
```

This will:
1. Configure the build with coverage flags (`-fprofile-instr-generate -fcoverage-mapping`)
2. Build the project
3. Run tests
4. Collect coverage data
5. Generate JSON and HTML reports

**Output files:**
- `build_ninja_coverage/coverage_report/coverage_summary.json` - JSON report
- `build_ninja_coverage/coverage_report/coverage.lcov` - LCOV format
- `build_ninja_coverage/coverage_report/html/` - HTML reports

### GCC Coverage

Run coverage build with GCC:

```bash
cd Scripts
python setup.py config.build.ninja.gcc.TEST.coverage
```

**Output files:**
- `build_ninja_coverage/coverage_report/coverage_summary.json` - JSON report
- `build_ninja_coverage/coverage_report/coverage.gcov` - GCC coverage format
- `build_ninja_coverage/coverage_report/html/` - HTML reports

### MSVC Coverage

Run coverage build with MSVC:

```bash
cd Scripts
python setup.py config.build.ninja.msvc.TEST.coverage
```

**Output files:**
- `build_ninja_coverage/coverage_report/coverage_summary.json` - JSON report
- `build_ninja_coverage/coverage_report/html/` - HTML reports

---

## Generating JSON Reports

JSON reports are automatically generated during the coverage build process. The JSON format is Cobertura-compatible and includes:

### JSON Structure

```json
{
  "metadata": {
    "format_version": "2.0",
    "generator": "xsigma_coverage_tool",
    "schema": "cobertura-compatible"
  },
  "summary": {
    "line_coverage": {
      "total": 12360,
      "covered": 4307,
      "uncovered": 8053,
      "percent": 34.85
    },
    "function_coverage": {
      "total": 1695,
      "covered": 553,
      "uncovered": 1142,
      "percent": 32.63
    },
    "region_coverage": {
      "total": 0,
      "covered": 0,
      "uncovered": 0,
      "percent": 0.0
    }
  },
  "files": [
    {
      "file": "path/to/file.cpp",
      "line_coverage": {...},
      "function_coverage": {...},
      "region_coverage": {...}
    }
  ]
}
```

### Accessing JSON Reports

The JSON report is located at:
```
build_ninja_coverage/coverage_report/coverage_summary.json
```

---

## Generating HTML Reports from JSON

### Using the Command-Line Tool

Generate HTML reports from an existing JSON file:

```bash
cd Tools/coverage
python -m html_report --json=path/to/coverage_summary.json --output=path/to/html_output
```

### Using Python API

```python
from html_report.json_html_generator import JsonHtmlGenerator
from pathlib import Path

# Generate HTML from JSON file
generator = JsonHtmlGenerator(Path("output_dir"))
index_file = generator.generate_from_json(Path("coverage_summary.json"))

# Or generate from dictionary
data = {...}  # Your coverage data
index_file = generator.generate_from_dict(data)
```

### HTML Report Features

- **Summary page**: Overall coverage metrics with visual progress bars
- **Per-file pages**: Detailed coverage for each source file
- **Interactive navigation**: Links between summary and file pages
- **Professional styling**: Clean, responsive design
- **Mobile-friendly**: Works on all devices

---

## Coverage Metrics Explanation

### Line Coverage

Percentage of executable lines that were executed during testing.

- **Formula**: `(covered_lines / total_lines) * 100`
- **Interpretation**: Higher is better; 100% means all lines were executed
- **Limitations**: Doesn't account for all code paths

### Function Coverage

Percentage of functions that were called during testing.

- **Formula**: `(covered_functions / total_functions) * 100`
- **Interpretation**: Indicates which functions are tested
- **Use case**: Identify untested functions

### Region Coverage

Percentage of code regions (branches, conditions) that were executed.

- **Formula**: `(covered_regions / total_regions) * 100`
- **Interpretation**: More granular than line coverage
- **Compiler-specific**: Clang-specific metric

---

## CI/CD Integration Guide

### GitHub Actions Example

Add to `.github/workflows/ci.yml`:

```yaml
- name: Generate Coverage Report
  run: |
    cd Scripts
    python setup.py config.build.ninja.clang.TEST.coverage

- name: Upload Coverage Artifacts
  uses: actions/upload-artifact@v3
  with:
    name: coverage-reports
    path: build_ninja_coverage/coverage_report/

- name: Comment Coverage on PR
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v6
  with:
    script: |
      const fs = require('fs');
      const coverage = JSON.parse(fs.readFileSync('build_ninja_coverage/coverage_report/coverage_summary.json'));
      const line_cov = coverage.summary.line_coverage.percent;
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: `📊 Coverage Report\n\nLine Coverage: ${line_cov}%`
      });
```

### Setting Coverage Thresholds

Check coverage against thresholds:

```bash
#!/bin/bash
COVERAGE=$(python -c "
import json
with open('build_ninja_coverage/coverage_report/coverage_summary.json') as f:
    data = json.load(f)
    print(data['summary']['line_coverage']['percent'])
")

THRESHOLD=80
if (( $(echo "$COVERAGE < $THRESHOLD" | bc -l) )); then
    echo "Coverage $COVERAGE% is below threshold $THRESHOLD%"
    exit 1
fi
```

---

## Troubleshooting

### Issue: Coverage data not generated

**Solution**: Ensure coverage flags are enabled in CMake configuration:
```bash
cmake -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" ...
```

### Issue: JSON file is empty

**Solution**: Verify tests ran successfully and generated profile data:
```bash
ls -la build_ninja_coverage/coverage_report/*.profraw
```

### Issue: HTML reports not generated

**Solution**: Check that JSON file exists and is valid:
```bash
python -m json.tool build_ninja_coverage/coverage_report/coverage_summary.json
```

### Issue: Low coverage percentage

**Solution**:
1. Ensure all test suites are running
2. Check that coverage flags are applied to all source files
3. Verify test execution completed successfully

### Issue: MSVC/OpenCppCoverage generates empty report

**Symptoms**:
- Coverage report generated but contains no data
- HTML report shows 0% coverage
- No coverage data collected from test executables

**Root Causes**:
1. Test executables not built with debug information
2. OpenCppCoverage not finding instrumented binaries
3. Test executables not running successfully
4. Source paths not correctly specified

**Solution**:

1. **Verify debug information is included**:
   ```bash
   # Check that binaries have debug symbols
   dumpbin /headers build_ninja_coverage/bin/CoreCxxTests.exe | findstr /i "debug"
   ```

2. **Ensure CMake coverage flags are applied**:
   - The CMake coverage configuration now automatically adds `/Zi /Od` flags for MSVC
   - Verify in CMakeCache.txt:
   ```bash
   grep "CMAKE_CXX_FLAGS" build_ninja_coverage/CMakeCache.txt
   ```

3. **Test executable verification**:
   ```bash
   # Run test executable directly to verify it works
   cd build_ninja_coverage/bin
   CoreCxxTests.exe
   ```

4. **Check OpenCppCoverage command**:
   - The coverage script now provides detailed debugging output
   - Run with verbose output to see the exact command:
   ```bash
   cd Tools/coverage
   python run_coverage.py --build=../../build_ninja_coverage --verbose
   ```

5. **Verify source paths**:
   - Ensure source folder path is correct and uses Windows path separators
   - Check that source files exist at the specified path

6. **Check for test failures**:
   - The coverage script now verifies test executables run successfully
   - Look for warnings about non-zero exit codes in the output

---

## Best Practices

1. **Run coverage regularly**: Include in CI/CD pipeline
2. **Track trends**: Archive coverage reports over time
3. **Set thresholds**: Enforce minimum coverage requirements
4. **Review reports**: Identify untested code paths
5. **Update tests**: Increase coverage for critical code
6. **Document exclusions**: Mark code that shouldn't be covered

---

## Advanced Usage

### Custom Coverage Thresholds

Set different thresholds for different modules:

```python
from html_report.json_html_generator import JsonHtmlGenerator

# Generate report
generator = JsonHtmlGenerator(Path("output"))
generator.generate_from_json(Path("coverage.json"))

# Check thresholds
with open("coverage.json") as f:
    data = json.load(f)
    line_cov = data["summary"]["line_coverage"]["percent"]
    if line_cov < 80:
        print("Warning: Line coverage below 80%")
```

### Merging Coverage Reports

Combine coverage from multiple test runs:

```bash
# Merge profile data
llvm-cov export -format=lcov *.profdata > merged.lcov

# Generate JSON from merged data
python -c "
from coverage_summary import CoverageSummaryGenerator
gen = CoverageSummaryGenerator()
gen.generate_from_lcov('merged.lcov')
"
```

---

## Support and Resources

- **Documentation**: See `Tools/coverage/` directory
- **Tests**: Run `pytest Tools/coverage/test_html_report.py`
- **Issues**: Report bugs in the project issue tracker
- **Contributing**: Follow XSigma coding standards

---

## Version History

- **2.0**: JSON report generation, HTML from JSON conversion
- **1.0**: Initial coverage system with HTML reports

