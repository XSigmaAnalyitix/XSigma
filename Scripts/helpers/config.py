"""
Configuration Generation Helper Module

This module handles CMake configuration generation.
Extracted from setup.py for better modularity and maintainability.
"""

import os
import subprocess


def configure_build(
    source_path: str,
    build_path: str,
    cmake_generator: str,
    cmake_cxx_compiler: str,
    cmake_c_compiler: str,
    cmake_flags: list[str],
    arg_cmake_verbose: str,
    shell_flag: bool,
) -> int:
    """
    Configure the build system using CMake.

    Args:
        source_path: Path to source directory
        build_path: Path to build directory
        cmake_generator: CMake generator to use
        cmake_cxx_compiler: C++ compiler specification
        cmake_c_compiler: C compiler specification
        cmake_flags: Additional CMake flags
        arg_cmake_verbose: Verbosity level
        shell_flag: Whether to use shell execution

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        build_folder = f"-B {build_path}"
        source_folder = f"-S {source_path}"

        cmake_cmd = [
            "cmake",
            source_folder,
            build_folder,
            "-G",
            cmake_generator,
            arg_cmake_verbose,
        ]

        if cmake_cxx_compiler:
            cmake_cmd.append(cmake_cxx_compiler)
        if cmake_c_compiler:
            cmake_cmd.append(cmake_c_compiler)

        # Add additional CMake flags
        cmake_cmd.extend(cmake_flags)

        subprocess.check_call(cmake_cmd, stderr=subprocess.STDOUT, shell=shell_flag)
        return 0

    except subprocess.CalledProcessError:
        return 1
    except Exception:
        return 1


def handle_xcode_project_opening() -> None:
    """Handle opening the Xcode project after generation."""
    try:
        xcodeproj_files = [f for f in os.listdir(".") if f.endswith(".xcodeproj")]
        if xcodeproj_files:
            project_file = xcodeproj_files[0]
            try:
                response = (
                    input("Would you like to open the Xcode project? (y/N): ")
                    .strip()
                    .lower()
                )
                if response in ["y", "yes"]:
                    subprocess.run(["open", project_file], check=True)
            except (KeyboardInterrupt, EOFError):
                pass
    except Exception:
        pass
