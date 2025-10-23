#!/usr/bin/env python3
"""MSVC-specific code coverage generation.

Handles coverage generation for MSVC compiler using OpenCppCoverage tool.
"""

import subprocess
import platform
from pathlib import Path
from typing import List, Optional

from common import get_platform_config, find_opencppcoverage, discover_test_executables


def generate_msvc_coverage(build_dir: Path, modules: List[str],
                          source_folder: Path) -> None:
    """Generate code coverage using opencppcoverage (for MSVC on Windows).

    Args:
        build_dir: Path to build directory.
        modules: List of module names to analyze.
        source_folder: Path to source folder containing modules.

    Raises:
        RuntimeError: If not on Windows or opencppcoverage not found.
    """
    build_dir = Path(build_dir)
    coverage_dir = build_dir / "coverage_report"
    raw_dir = coverage_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = get_platform_config()

    if config["os_name"] != "Windows":
        raise RuntimeError("MSVC coverage only supported on Windows")

    # Find OpenCppCoverage
    opencpp_path = find_opencppcoverage()
    if not opencpp_path:
        raise RuntimeError("OpenCppCoverage not found. Please install it.")

    # Discover test executables
    print("Discovering test executables...")
    test_exes = discover_test_executables(build_dir)
    print(f"Found {len(test_exes)} test executables")

    if not test_exes:
        print("Warning: No test executables found")
        return

    # Run coverage for each test
    print("Running coverage analysis...")
    html_dir = coverage_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    for test_exe in test_exes:
        print(f"Running coverage for: {test_exe}")
        cov_cmd = [
            opencpp_path,
            "--quiet",
            "--export_type=html:" + str(html_dir),
            str(test_exe),
        ]

        try:
            subprocess.run(cov_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Coverage failed for {test_exe}: {e}")

    print(f"Coverage report generated at: {html_dir}")

