#!/usr/bin/env python3
"""MSVC-specific code coverage generation.

Handles coverage generation for MSVC compiler using OpenCppCoverage tool.
Generates both HTML and JSON coverage reports.
"""

import subprocess
import json
from pathlib import Path
import logging

from common import (
    CONFIG,
    get_platform_config,
    find_opencppcoverage,
    discover_test_executables,
)

logger = logging.getLogger(__name__)


def _generate_json_from_html(html_dir: Path, output_dir: Path) -> None:
    """Generate JSON coverage report from OpenCppCoverage HTML output.

    Args:
        html_dir: Directory containing OpenCppCoverage HTML reports.
        output_dir: Directory where JSON report will be saved.
    """
    try:
        summary = {
            "metadata": {
                "format_version": "2.0",
                "generator": "xsigma_coverage_tool",
                "schema": "cobertura-compatible"
            },
            "summary": {
                "line_coverage": {
                    "total": 0,
                    "covered": 0,
                    "uncovered": 0,
                    "percent": 0.0
                }
            },
            "files": []
        }

        # Parse HTML files to extract coverage data
        # OpenCppCoverage generates index.html with coverage summary
        index_file = html_dir / "index.html"
        if not index_file.exists():
            print(f"Warning: OpenCppCoverage index.html not found at {index_file}")
            return

        # Extract coverage data from HTML
        with open(index_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # Look for coverage percentage in HTML
            # OpenCppCoverage format: look for coverage metrics
            import re
            # Pattern to find coverage percentages
            coverage_pattern = r'(\d+(?:\.\d+)?)\s*%'
            matches = re.findall(coverage_pattern, content)

            if matches:
                # Try to extract overall coverage
                try:
                    overall_coverage = float(matches[0])
                    summary["summary"]["line_coverage"]["percent"] = round(overall_coverage, 2)
                except (ValueError, IndexError):
                    pass

        # Look for individual file reports
        html_files = list(html_dir.glob("*.html"))
        for html_file in html_files:
            if html_file.name == "index.html":
                continue

            try:
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()

                    # Extract file path and coverage info
                    file_data = {
                        "file": html_file.stem,
                        "line_coverage": {
                            "total": 0,
                            "covered": 0,
                            "uncovered": 0,
                            "percent": 0.0
                        }
                    }

                    # Try to extract coverage percentage from file
                    import re
                    coverage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', file_content)
                    if coverage_match:
                        try:
                            file_coverage = float(coverage_match.group(1))
                            file_data["line_coverage"]["percent"] = round(file_coverage, 2)
                        except ValueError:
                            pass

                    summary["files"].append(file_data)

            except Exception as e:
                logger.warning(f"Failed to parse {html_file}: {e}")

        # Save JSON
        json_file = output_dir / "coverage_summary.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"JSON coverage report saved to: {json_file}")

        # Generate HTML from JSON using the new html_report package
        try:
            from html_report import JsonHtmlGenerator
            json_html_dir = output_dir / "html_from_json"
            generator = JsonHtmlGenerator(json_html_dir)
            index_file = generator.generate_from_dict(summary)
            print(f"HTML report generated from JSON at: {index_file}")
        except Exception as e:
            print(f"Warning: Failed to generate HTML from JSON: {e}")

    except Exception as e:
        print(f"Warning: Failed to generate JSON from HTML: {e}")


def generate_msvc_coverage(build_dir: Path, modules: list[str],
                           source_folder: Path) -> None:
    """Generate code coverage using opencppcoverage (for MSVC on Windows).

    Generates HTML reports in html/ folder and raw coverage data in raw/ folder.
    Uses CONFIG dictionary for all configurable parameters including exclude patterns.

    Args:
        build_dir: Path to build directory.
        modules: List of module names to analyze.
        source_folder: Path to source folder containing modules.

    Raises:
        RuntimeError: If not on Windows or opencppcoverage not found.
    """
    build_dir = Path(build_dir)
    coverage_dir = build_dir / "coverage_report"
    html_dir = coverage_dir / "html"
    raw_dir = coverage_dir / "raw"

    # Create output directories
    html_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = get_platform_config()
    excludes = CONFIG.get("exclude_patterns", [])

    if config["os_name"] != "Windows":
        raise RuntimeError("MSVC coverage only supported on Windows")

    # Find OpenCppCoverage
    opencpp_path = find_opencppcoverage()
    if not opencpp_path:
        raise RuntimeError("OpenCppCoverage not found. Please install it.")

    # Discover test executables
    print("Discovering test executables...")
    test_executables = discover_test_executables(build_dir)

    if not test_executables:
        print("Warning: No test executables found")
        return

    print(f"Found {len(test_executables)} test executable(s)")

    # Run coverage for each test executable
    print("Running coverage analysis...")
    print(f"Analyzing coverage for: {source_folder}")

    failed_tests = []
    successful_tests = 0

    for test_exe in test_executables:
        test_name = test_exe.stem
        print(f"Running coverage for: {test_name}")

        # Build OpenCppCoverage command
        cov_cmd = [
            str(opencpp_path),
            "--quiet",
        ]

        # Add HTML export
        cov_cmd.append(f"--export_type=html:{html_dir}")

        # Add binary export for raw coverage data
        raw_file = raw_dir / f"{test_name}.cov"
        cov_cmd.append(f"--export_type=binary:{raw_file}")

        # Add source filter for specific folder (Windows path separators)
        windows_source_path = str(source_folder).replace("/", "\\")
        cov_cmd.append(f"--sources={windows_source_path}")

        # Add exclusion patterns with Windows path separators
        for exclude_pattern in excludes:
            windows_pattern = exclude_pattern.replace("/", "\\")
            cov_cmd.append(f"--excluded_sources={windows_pattern}")

        # Add the test executable to run
        cov_cmd.append("--")
        cov_cmd.append(str(test_exe))

        try:
            result = subprocess.run(
                cov_cmd,
                cwd=build_dir,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                print(f"✓ Coverage generated for: {test_name}")
                successful_tests += 1
            else:
                print(f"✗ Coverage failed for: {test_name}")
                if result.stderr:
                    print(f"  Error: {result.stderr}")
                failed_tests.append(test_name)
        except Exception as e:
            print(f"✗ Exception running coverage for {test_name}: {e}")
            failed_tests.append(test_name)

    # Verify output
    html_files = list(html_dir.glob("**/*.html"))
    raw_files = list(raw_dir.glob("*.cov"))

    print(f"\n{'='*60}")
    print(f"Coverage Report Summary")
    print(f"{'='*60}")
    print(f"Tests processed: {successful_tests}/{len(test_executables)}")
    print(f"HTML files generated: {len(html_files)}")
    print(f"Raw coverage files: {len(raw_files)}")
    print(f"HTML report location: {html_dir}")
    print(f"Raw data location: {raw_dir}")

    if not html_files and not raw_files:
        print("\n⚠ Warning: No coverage output generated!")
        raise RuntimeError("Coverage generation produced no output")

    # Generate JSON coverage report
    print("\nGenerating JSON coverage report...")
    _generate_json_from_html(html_dir, coverage_dir)

    if failed_tests:
        print(f"\n{len(failed_tests)} test(s) had issues:")
        for test_name in failed_tests:
            print(f"  - {test_name}")
    else:
        print(f"\n✓ All {len(test_executables)} test(s) processed successfully!")