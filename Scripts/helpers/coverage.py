"""
Code Coverage Analysis Helper Module

This module handles code coverage collection and analysis.
Extracted from setup.py for better modularity and maintainability.
"""

import os
import subprocess
import sys
from typing import Optional


def run_oss_coverage(source_path: str, build_path: str, cmake_cxx_compiler: str) -> int:
    """
    Run coverage using the oss_coverage.py tool from tools/code_coverage.

    This tool provides:
    - Support for both GCC and Clang compilers
    - Automatic compiler detection
    - JSON-based coverage reports
    - HTML report generation
    - Cross-platform compatibility

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    oss_coverage_script = os.path.join(source_path, "tools", "code_coverage", "oss_coverage.py")

    if not os.path.exists(oss_coverage_script):
        return 1

    # Build command to run oss_coverage.py
    oss_cov_cmd = [
        sys.executable,
        oss_coverage_script,
        "--run",
        "--export",
        "--summary",
    ]

    # Add merge step for Clang (required for LLVM coverage)
    if "clang" in cmake_cxx_compiler.lower():
        oss_cov_cmd.insert(3, "--merge")

    try:
        # Prepare environment for oss_coverage.py
        env = os.environ.copy()

        # Set HOME environment variable for Windows compatibility
        if "HOME" not in env:
            if "USERPROFILE" in env:
                env["HOME"] = env["USERPROFILE"]
            else:
                env["HOME"] = os.path.expanduser("~")

        # Set CXX environment variable to help compiler detection
        if "clang" in cmake_cxx_compiler.lower():
            env["CXX"] = "clang++"
        elif "gcc" in cmake_cxx_compiler.lower():
            env["CXX"] = "g++"

        # Set XSIGMA_FOLDER to help oss_coverage.py find the correct paths
        env["XSIGMA_FOLDER"] = source_path

        # Change to source directory for oss_coverage.py to work correctly
        original_dir = os.getcwd()
        os.chdir(source_path)

        result = subprocess.run(
            oss_cov_cmd,
            cwd=source_path,
            env=env,
            check=False,
            shell=(os.name == 'nt')
        )

        os.chdir(original_dir)
        return result.returncode

    except Exception:
        return 1


def check_coverage_reports(source_path: str) -> None:
    """Check for generated coverage reports."""
    report_paths = [
        os.path.join(source_path, "coverage_report"),
        os.path.join(source_path, "htmlcov"),
    ]

    for report_path in report_paths:
        if os.path.exists(report_path):
            pass  # Report found


def run_analyze_coverage(source_path: str, build_path: Optional[str] = None, verbose: bool = False) -> int:
    """
    Run coverage analysis to identify files below coverage threshold.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    analyze_script = os.path.join(source_path, "Scripts", "analyze_coverage.py")

    if not os.path.exists(analyze_script):
        return 1

    # Build command to run analyze_coverage.py
    analyze_cmd = [sys.executable, analyze_script]

    # Add build directory if provided
    if build_path and os.path.isdir(build_path):
        analyze_cmd.extend(["--build-dir", build_path])

    # Add verbose flag if requested
    if verbose:
        analyze_cmd.append("--verbose")

    try:
        result = subprocess.run(
            analyze_cmd,
            cwd=source_path,
            check=False,
            shell=(os.name == 'nt')
        )
        return result.returncode

    except Exception:
        return 1

