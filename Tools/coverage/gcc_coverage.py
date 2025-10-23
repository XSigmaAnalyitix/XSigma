#!/usr/bin/env python3
"""GCC/gcov-specific code coverage generation.

Handles coverage generation for GCC compiler using lcov/genhtml tools.
"""

import os
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from common import CONFIG
logger = logging.getLogger(__name__)



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