#!/usr/bin/env python3
"""Common utilities for code coverage generation.

Provides shared functions used across multiple coverage modules to eliminate
code duplication and improve maintainability.
"""

import subprocess
import platform
import re
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Centralized hardcoded values
# ============================================================================

CONFIG = {
    # Default filter folder (subfolder within source containing modules)
    "filter": "Library",

    # File patterns to exclude from coverage reports
    "exclude_patterns": [
        "*ThirdParty*",
        "*Testing*",
        "/usr/*",
    ],

    # Regex patterns for LLVM to ignore
    "llvm_ignore_regex": [
        ".*Testing[/\\\\].*",
        ".*Serialization[/\\\\].*",
        ".*ThirdParty[/\\\\].*",
        ".*xsigmasys[/\\\\].*",
    ],

    # Markers to detect project root
    "project_markers": [".git", ".gitignore", "pyproject.toml"],
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


def get_platform_config() -> dict:
    """Infer platform-specific extensions and paths.

    Returns:
        Dictionary with platform-specific configuration including
        dll_extension, exe_extension, lib_folder, and os_name.

    Raises:
        RuntimeError: If platform is not supported.
    """
    system = platform.system()

    if system == "Windows":
        return {
            "dll_extension": ".dll",
            "exe_extension": ".exe",
            "lib_folder": "bin",
            "os_name": "Windows"
        }
    elif system == "Darwin":
        return {
            "dll_extension": ".dylib",
            "exe_extension": "",
            "lib_folder": "lib",
            "os_name": "macOS"
        }
    elif system == "Linux":
        return {
            "dll_extension": ".so",
            "exe_extension": "",
            "lib_folder": "lib",
            "os_name": "Linux"
        }
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def find_opencppcoverage() -> Optional[str]:
    """Find OpenCppCoverage executable in system PATH.

    Searches system PATH and common installation directories for the
    OpenCppCoverage executable.

    Returns:
        Path to OpenCppCoverage.exe or None if not found.
    """
    try:
        result = subprocess.run(
            ["OpenCppCoverage.exe", "--help"],
            capture_output=True,
            check=True,
            text=True
        )
        return "OpenCppCoverage.exe"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try common installation paths
    common_paths = [
        Path("C:\\Program Files\\OpenCppCoverage\\OpenCppCoverage.exe"),
        Path("C:\\Program Files (x86)\\OpenCppCoverage\\OpenCppCoverage.exe"),
    ]

    for path in common_paths:
        if path.exists():
            return str(path)

    return None


def discover_test_executables(build_dir: Path) -> List[Path]:
    """Discover all test executables in the build directory.

    Searches for executables matching common test patterns in bin and lib folders.

    Args:
        build_dir: Path to the build directory.

    Returns:
        List of Path objects pointing to test executables.
    """
    build_dir = Path(build_dir)
    test_executables = []
    config = get_platform_config()
    exe_ext = config["exe_extension"]

    search_dirs = [
        build_dir / "bin",
        build_dir / "lib",
        build_dir / "tests",
    ]

    test_patterns = ["*Test*", "*test*", "*CxxTests*"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in test_patterns:
            for exe_file in search_dir.glob(f"{pattern}{exe_ext}"):
                if exe_file.is_file():
                    test_executables.append(exe_file)

    # Remove duplicates while preserving order
    seen = set()
    unique_executables = []
    for exe in test_executables:
        exe_str = str(exe.resolve())
        if exe_str not in seen:
            seen.add(exe_str)
            unique_executables.append(exe)

    return unique_executables


def find_library(build_dir: Path, lib_folder: str, module_name: str,
                 dll_extension: str) -> Optional[str]:
    """Find library file for a module.

    Searches for library files matching the module name pattern.

    Args:
        build_dir: Path to build directory.
        lib_folder: Folder name containing libraries (e.g., "lib", "bin").
        module_name: Name of the module to find.
        dll_extension: File extension for libraries (e.g., ".so", ".dll").

    Returns:
        Path to the library file or None if not found.
    """
    lib_path = build_dir / lib_folder
    if not lib_path.exists():
        return None

    patterns = [
        f"{module_name}{dll_extension}",
        f"lib{module_name}{dll_extension}",
        f"{module_name}*{dll_extension}",
    ]

    for pattern in patterns:
        matches = list(lib_path.glob(pattern))
        if matches:
            selected = matches[0]
            print(f"Found library for module '{module_name}': {selected.name} "
                  f"using {selected.name}")
            return str(selected)

    return None


def get_project_root() -> Path:
    """Find the project root directory by searching for common markers.

    Searches upward from the script directory for markers like .git,
    .gitignore, or pyproject.toml.

    Returns:
        Path to project root directory.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Search up to 10 levels
        for marker in CONFIG["project_markers"]:
            if (current / marker).exists():
                return current
        current = current.parent

    return Path(__file__).resolve().parent


def resolve_build_dir(build_dir_arg: str,
                      project_root: Optional[Path] = None) -> Path:
    """Resolve build directory from argument (absolute or relative).

    Tries multiple resolution strategies:
    1. Absolute path
    2. Relative to current working directory
    3. Relative to script directory
    4. Relative to project root

    Args:
        build_dir_arg: Build directory argument (can be absolute or relative).
        project_root: Optional project root for relative resolution.

    Returns:
        Resolved Path to build directory.

    Raises:
        ValueError: If build directory cannot be resolved.
    """
    build_dir_arg = str(build_dir_arg)

    # Try as absolute path
    abs_path = Path(build_dir_arg).resolve()
    if abs_path.exists():
        return abs_path

    # Try relative to CWD
    cwd_relative = Path.cwd() / build_dir_arg
    if cwd_relative.exists():
        return cwd_relative

    # Try relative to script directory
    script_dir = Path(__file__).resolve().parent
    script_relative = script_dir / build_dir_arg
    if script_relative.exists():
        return script_relative

    # Try relative to project root
    if project_root is None:
        project_root = get_project_root()
    project_relative = project_root / build_dir_arg
    if project_relative.exists():
        return project_relative

    raise ValueError(
        f"Build directory '{build_dir_arg}' not found. Tried:\n"
        f"    - Absolute: {abs_path}\n"
        f"    - CWD relative: {cwd_relative}\n"
        f"    - Script relative: {script_relative}\n"
        f"    - Project relative: {project_relative}"
    )


def validate_build_structure(build_dir: Path, config: dict,
                             library_folder: str) -> dict:
    """Validate that the build directory has expected structure.

    Args:
        build_dir: Path to build directory.
        config: Configuration dictionary.
        library_folder: Name of library folder to check.

    Returns:
        Dictionary with 'valid' boolean and 'issues' list.
    """
    issues = []

    if not (build_dir / "CMakeCache.txt").exists():
        issues.append(f"CMakeCache.txt not found in {build_dir}")

    if not (build_dir / library_folder).exists():
        issues.append(f"Expected source folder not found: "
                      f"{build_dir / library_folder}")

    return {"valid": len(issues) == 0, "issues": issues}


def detect_compiler(build_dir: Path | str) -> str:
    """Detect the compiler used in the build directory.

    Looks for XSIGMA_COMPILER_ID in CMakeCache.txt to determine the compiler.

    Args:
        build_dir: Path to build directory.

    Returns:
        Compiler name: "gcc", "clang", "msvc", or "intel".

    Raises:
        RuntimeError: If compiler cannot be detected.
    """
    build_dir = Path(build_dir)
    cmake_cache = build_dir / "CMakeCache.txt"

    if not cmake_cache.exists():
        logger.error(f"CMakeCache.txt not found in {build_dir}")
        raise RuntimeError(f"CMakeCache.txt not found in {build_dir}")

    try:
        with open(cmake_cache, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # Look for XSIGMA_COMPILER_ID
            for line in content.split('\n'):
                if 'XSIGMA_COMPILER_ID' in line:
                    # Extract the value after the '='
                    if '=' in line:
                        compiler_id = line.split('=', 1)[-1].strip()
                        
                        # Normalize to lowercase for comparison
                        compiler_id_lower = compiler_id.lower()
                        
                        # Map XSIGMA compiler IDs to supported compilers
                        print(f"XSIGMA_COMPILER_ID found: {compiler_id}")
                        if compiler_id_lower == "gcc":
                            logger.info("GCC detected")
                            return "gcc"
                        elif compiler_id_lower == "clang":
                            logger.info("Clang detected")
                            return "clang"
                        elif compiler_id_lower == "msvc":
                            logger.info("MSVC detected")
                            return "msvc"
                        elif compiler_id_lower == "intel":
                            logger.info("Intel compiler detected")
                            return "intel"
                        else:
                            logger.warning(
                                f"Unknown compiler ID found: {compiler_id}"
                            )

    except Exception as e:
        logger.error(f"Error reading CMakeCache.txt: {e}")
        raise RuntimeError(f"Error reading CMakeCache.txt: {e}")

    logger.error(
        f"Could not determine compiler from CMakeCache.txt in {build_dir}. "
        "XSIGMA_COMPILER_ID not found."
    )
    raise RuntimeError(
        "Could not determine compiler from CMakeCache.txt. "
        "XSIGMA_COMPILER_ID not found. "
        "Please ensure the build was configured with GCC, Clang, MSVC, or Intel."
    )