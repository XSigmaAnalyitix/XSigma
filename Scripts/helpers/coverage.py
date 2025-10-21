"""
Code Coverage Analysis Helper Module

This module handles code coverage collection and analysis.
Extracted from setup.py for better modularity and maintainability.
"""

import os
import subprocess
import sys
from typing import Optional

def _detect_compiler_from_cmake_cache(build_path: str) -> Optional[str]:
    """
    Detect the compiler used in the build by reading CMakeCache.txt.

    Args:
        build_path: Path to the build directory

    Returns:
        Compiler type: "clang", "gcc", or None if not detected
    """
    cmake_cache_path = os.path.join(build_path, "CMakeCache.txt")
    if not os.path.exists(cmake_cache_path):
        return None

    try:
        with open(cmake_cache_path, 'r') as f:
            for line in f:
                if line.startswith("CMAKE_CXX_COMPILER:"):
                    # Extract the compiler path/name
                    compiler_path = line.split("=", 1)[1].strip()
                    compiler_name = os.path.basename(compiler_path).lower()

                    if "clang" in compiler_name:
                        return "clang"
                    elif "gcc" in compiler_name or "g++" in compiler_name:
                        return "gcc"
    except Exception:
        pass

    return None


def run_oss_coverage(source_path: str, build_path: str, cmake_cxx_compiler: str) -> int:
    """
    Run coverage using the oss_coverage.py tool from tools/code_coverage.

    This tool provides:
    - Support for both GCC and Clang compilers
    - Automatic compiler detection
    - JSON-based coverage reports
    - HTML report generation
    - Cross-platform compatibility

    Args:
        source_path: Path to the source directory
        build_path: Path to the build directory
        cmake_cxx_compiler: C++ compiler being used (can be empty string)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    oss_coverage_script = os.path.join(
        source_path, "Tools", "code_coverage", "oss_coverage.py"
    )

    if not os.path.exists(oss_coverage_script):
        return 1

    # Extract build folder name from build_path
    build_folder = os.path.basename(build_path)

    # Build command to run oss_coverage.py
    oss_cov_cmd = [
        sys.executable,
        oss_coverage_script,
        "--build-folder",
        build_folder,
        "--run",
        "--export",
        "--summary",
    ]

    # Determine the actual compiler used in the build
    # First, check if cmake_cxx_compiler was explicitly specified
    detected_compiler = None
    if cmake_cxx_compiler and cmake_cxx_compiler.strip():
        if "clang" in cmake_cxx_compiler.lower():
            detected_compiler = "clang"
        elif "gcc" in cmake_cxx_compiler.lower():
            detected_compiler = "gcc"

    # If not detected from cmake_cxx_compiler, try to read from CMakeCache.txt
    if not detected_compiler:
        detected_compiler = _detect_compiler_from_cmake_cache(build_path)

    # Add merge step for Clang (required for LLVM coverage)
    if detected_compiler == "clang":
        oss_cov_cmd.insert(5, "--merge")

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
        # This is critical for oss_coverage.py to detect the compiler type
        if detected_compiler == "clang":
            env["CXX"] = "clang++"
            env["CC"] = "clang"
        elif detected_compiler == "gcc":
            env["CXX"] = "g++"
            env["CC"] = "gcc"
        else:
            # Default to gcc for unknown compilers (safer default)
            env["CXX"] = "g++"
            env["CC"] = "gcc"

        # Set XSIGMA_FOLDER to help oss_coverage.py find the correct paths
        env["XSIGMA_FOLDER"] = source_path

        # Set build folder environment variables for oss_coverage.py
        # Pass the full build path so the coverage tool can resolve it correctly
        env["XSIGMA_BUILD_FOLDER"] = build_folder
        env["XSIGMA_BUILD_PATH"] = build_path
        env["XSIGMA_TEST_SUBFOLDER"] = "bin"

        # Set coverage output directory to build folder
        # This ensures coverage reports are generated in the build folder
        coverage_dir = os.path.join(build_path, "coverage_report")
        env["XSIGMA_COVERAGE_DIR"] = coverage_dir

        # Ensure coverage directory exists
        os.makedirs(coverage_dir, exist_ok=True)

        # Set interested folders to focus on Library code only
        # This ensures only production code is analyzed, not test code
        env["XSIGMA_INTERESTED_FOLDERS"] = "Library"

        # Set excluded patterns to exclude Testing folder and other test code
        # This prevents test code from appearing in coverage reports
        env["XSIGMA_EXCLUDED_PATTERNS"] = "Testing,test,tests,mock,stub"

        # Change to source directory for oss_coverage.py to work correctly
        original_dir = os.getcwd()
        os.chdir(source_path)

        result = subprocess.run(
            oss_cov_cmd,
            cwd=source_path,
            env=env,
            check=False,
            shell=(os.name == "nt"),
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

