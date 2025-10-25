#!/usr/bin/env python3
"""GCC/gcov-specific code coverage generation.

Handles coverage generation for GCC compiler using lcov/genhtml tools.
Generates both HTML and JSON coverage reports.
"""

import os
import subprocess
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from common import CONFIG
logger = logging.getLogger(__name__)



def _generate_json_from_lcov(lcov_file: Path, output_dir: Path) -> None:
    """Generate JSON coverage report from LCOV data.

    Args:
        lcov_file: Path to the LCOV coverage file.
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
                },
                "function_coverage": {
                    "total": 0,
                    "covered": 0,
                    "uncovered": 0,
                    "percent": 0.0
                }
            },
            "files": []
        }

        # Parse LCOV file
        current_file = None
        file_data = {}

        with open(lcov_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()

                if line.startswith("SF:"):
                    current_file = line[3:]
                    file_data = {
                        "file": current_file,
                        "line_coverage": {
                            "total": 0,
                            "covered": 0,
                            "uncovered": 0,
                            "percent": 0.0
                        }
                    }

                elif line.startswith("FN:") and current_file:
                    # Function definition
                    if "functions" not in file_data:
                        file_data["functions"] = []

                elif line.startswith("FNDA:") and current_file:
                    # Function data (execution count)
                    pass

                elif line.startswith("FNF:") and current_file:
                    # Total functions
                    total_funcs = int(line[4:])
                    file_data["function_coverage"] = {
                        "total": total_funcs,
                        "covered": 0,
                        "uncovered": total_funcs,
                        "percent": 0.0
                    }

                elif line.startswith("FNH:") and current_file:
                    # Functions hit
                    covered_funcs = int(line[4:])
                    if "function_coverage" in file_data:
                        file_data["function_coverage"]["covered"] = covered_funcs
                        file_data["function_coverage"]["uncovered"] = (
                            file_data["function_coverage"]["total"] - covered_funcs
                        )
                        if file_data["function_coverage"]["total"] > 0:
                            func_percent = (covered_funcs / file_data["function_coverage"]["total"]) * 100
                            file_data["function_coverage"]["percent"] = round(func_percent, 2)

                elif line.startswith("LF:") and current_file:
                    # Total lines
                    total_lines = int(line[3:])
                    file_data["line_coverage"]["total"] = total_lines

                elif line.startswith("LH:") and current_file:
                    # Lines hit
                    covered_lines = int(line[3:])
                    file_data["line_coverage"]["covered"] = covered_lines
                    file_data["line_coverage"]["uncovered"] = (
                        file_data["line_coverage"]["total"] - covered_lines
                    )
                    if file_data["line_coverage"]["total"] > 0:
                        line_percent = (covered_lines / file_data["line_coverage"]["total"]) * 100
                        file_data["line_coverage"]["percent"] = round(line_percent, 2)

                elif line == "end_of_record":
                    if current_file and file_data:
                        summary["files"].append(file_data)
                        summary["summary"]["line_coverage"]["total"] += file_data["line_coverage"]["total"]
                        summary["summary"]["line_coverage"]["covered"] += file_data["line_coverage"]["covered"]
                        summary["summary"]["line_coverage"]["uncovered"] += file_data["line_coverage"]["uncovered"]

                        if "function_coverage" in file_data:
                            summary["summary"]["function_coverage"]["total"] += file_data["function_coverage"]["total"]
                            summary["summary"]["function_coverage"]["covered"] += file_data["function_coverage"]["covered"]
                            summary["summary"]["function_coverage"]["uncovered"] += file_data["function_coverage"]["uncovered"]

                    current_file = None
                    file_data = {}

        # Calculate summary percentages
        if summary["summary"]["line_coverage"]["total"] > 0:
            line_percent = (summary["summary"]["line_coverage"]["covered"] /
                           summary["summary"]["line_coverage"]["total"]) * 100
            summary["summary"]["line_coverage"]["percent"] = round(line_percent, 2)

        if summary["summary"]["function_coverage"]["total"] > 0:
            func_percent = (summary["summary"]["function_coverage"]["covered"] /
                           summary["summary"]["function_coverage"]["total"]) * 100
            summary["summary"]["function_coverage"]["percent"] = round(func_percent, 2)

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
        print(f"Warning: Failed to generate JSON from LCOV: {e}")


def generate_lcov_coverage(build_dir: Path, modules: List[str],
                          exclude_patterns: List[str] = None) -> None:
    """
    Generate LCOV coverage report from build directory.
    
    Args:
        build_dir: Path to the build directory containing .gcda files
        modules: List of modules to include in coverage
        exclude_patterns: List of patterns to exclude (e.g., ['/usr/*', '*/ThirdParty/*'])
    """
    import subprocess
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    # Default exclusions
    default_excludes = CONFIG["exclude_patterns"]
    exclude_patterns = list(set(default_excludes + exclude_patterns))
    
    coverage_info = build_dir / "coverage.info"
    coverage_filtered = build_dir / "coverage_filtered.info"
    coverage_report_dir = build_dir / "coverage_report"
    
    print(f"Capturing coverage data from {build_dir}...")
    
    # Capture coverage data
    capture_cmd = [
        "lcov",
        "--directory", str(build_dir),
        "--capture",
        "--ignore-errors", "mismatch,negative,gcov",
        "--output-file", str(coverage_info)
    ]

    result = subprocess.run(capture_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error capturing coverage: {result.stderr}")
        return
    
    print("Coverage data captured successfully.")
    print(f"Filtering exclusions: {exclude_patterns}")
    
    # Remove excluded patterns
    remove_cmd = ["lcov", "--remove", str(coverage_info)]
    remove_cmd.extend(exclude_patterns)
    remove_cmd.extend([
        "--output-file", str(coverage_filtered)
    ])
    
    result = subprocess.run(remove_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error filtering coverage: {result.stderr}")
        return
    
    print("Coverage data filtered.")
    print(f"Generating HTML report to {coverage_report_dir}...")
    
    # Generate HTML report
    genhtml_cmd = [
        "genhtml",
        str(coverage_filtered),
        "--output-directory", str(coverage_report_dir),
        "--title", "Code Coverage Report",
        "--num-spaces", "4"
    ]
    
    result = subprocess.run(genhtml_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating HTML: {result.stderr}")
        return

    print(f"✓ Coverage report generated at {coverage_report_dir}/index.html")

    # Generate JSON coverage report
    print("Generating JSON coverage report...")
    _generate_json_from_lcov(coverage_filtered, coverage_report_dir)