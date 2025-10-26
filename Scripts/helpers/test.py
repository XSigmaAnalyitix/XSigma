"""
Test Execution Helper Module

This module handles test execution using ctest and valgrind.
Extracted from setup.py for better modularity and maintainability.
"""

import os
import platform
import subprocess


def run_ctest(
    builder: str, build_enum: str, system: str, verbosity: str, shell_flag: bool
) -> int:
    """
    Run tests using ctest.

    Args:
        builder: Build system (ninja, make, xcodebuild, cmake)
        build_enum: Build type (Release, Debug, RelWithDebInfo)
        system: Operating system (Linux, Darwin, Windows)
        verbosity: Verbosity flag
        shell_flag: Whether to use shell execution

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        ctest_cmd = ["ctest"]

        if system == "Windows":
            if builder != "ninja":
                ctest_cmd.extend(["-C", build_enum])

        # Handle Xcode configuration
        if builder == "xcodebuild":
            ctest_cmd.extend(["-C", build_enum])

        if verbosity:
            ctest_cmd.append(verbosity)

        return subprocess.check_call(
            ctest_cmd, stderr=subprocess.STDOUT, shell=shell_flag
        )

    except subprocess.CalledProcessError:
        return 1
    except Exception:
        return 1


def run_valgrind_test(source_path: str, build_path: str, shell_flag: bool) -> int:
    """
    Run tests with Valgrind memory checking.

    Args:
        source_path: Path to source directory
        build_path: Path to build directory
        shell_flag: Whether to use shell execution

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    script_full_path = os.path.join(source_path, "Scripts", "valgrind_ctest.sh")

    # Check if script exists
    if not os.path.exists(script_full_path):
        return 1

    # Make script executable
    try:
        os.chmod(script_full_path, 0o755)
    except Exception:
        pass

    # Check platform compatibility
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        pass  # Valgrind doesn't support Apple Silicon

    try:
        return subprocess.call(
            [script_full_path, build_path],
            stdin=subprocess.PIPE,
            shell=shell_flag,
        )
    except Exception:
        return 1
