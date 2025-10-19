"""
Build Operations Helper Module

This module handles build operations using CMake and various build systems.
Extracted from setup.py for better modularity and maintainability.
"""

import os
import platform
import subprocess
import sys
from typing import List


def get_logical_processor_count():
    """Get the number of logical processors available."""
    try:
        import psutil
        return psutil.cpu_count(logical=True)
    except ImportError:
        try:
            return os.cpu_count()
        except AttributeError:
            import multiprocessing
            return multiprocessing.cpu_count()


def build_project(builder: str, build_enum: str, system: str, shell_flag: bool) -> int:
    """
    Build the project using the specified builder.

    Args:
        builder: Build system (ninja, make, xcodebuild, cmake)
        build_enum: Build type (Release, Debug, RelWithDebInfo)
        system: Operating system (Linux, Darwin, Windows)
        shell_flag: Whether to use shell execution

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        if system == "Linux" or builder == "ninja":
            n = get_logical_processor_count()
            cmake_cmd_build = [builder, "-j", str(n)]
        elif builder == "xcodebuild":
            n = get_logical_processor_count()
            cmake_cmd_build = [
                "xcodebuild",
                "-configuration", build_enum,
                "-parallelizeTargets",
                "-jobs", str(n)
            ]
        elif system == "Windows":
            cmake_cmd_build = [
                builder,
                "--build",
                ".",
                "--config",
                build_enum,
            ]
        else:
            return 1

        subprocess.check_call(
            cmake_cmd_build,
            stderr=subprocess.STDOUT,
            shell=shell_flag
        )
        return 0

    except subprocess.CalledProcessError:
        return 1
    except Exception:
        return 1


def setup_windows_path(builder: str, build_enum: str) -> None:
    """Setup Windows PATH for testing."""
    try:
        if builder != "ninja":
            path_bat_file = f"windows_path.{build_enum}.bat"
            if os.path.exists(path_bat_file):
                subprocess.check_call(path_bat_file)
        else:
            if os.path.exists("windows_path.bat"):
                subprocess.check_call("windows_path.bat")
    except (subprocess.CalledProcessError, Exception):
        pass

