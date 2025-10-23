#!/usr/bin/env python3
"""Clang/LLVM-specific code coverage generation.

Handles coverage generation for Clang compiler using LLVM coverage tools
(llvm-profdata, llvm-cov).
"""

import os
import subprocess
import platform
from pathlib import Path
from typing import List, Optional
import logging

from common import CONFIG, get_platform_config, find_library

logger = logging.getLogger(__name__)

def prepare_llvm_coverage(build_dir: Path, module_name: str, binaries_list: str, profraw_list: str) -> List[Path]:
    """Discover and run all test executables for a specific module.

    Searches for test executables containing the module name in their filename
    using flexible pattern matching.

    Args:
        build_dir: Path to build directory.
        module_name: Name of the module to find tests for.

    Returns:
        List of Path objects pointing to test executables for the module.
    """
    config = get_platform_config()
    exe_extension = config["exe_extension"]
    bin_folder = config["lib_folder"]
    dll_extension = config["dll_extension"]
    coverage_dir = build_dir / "coverage_report"
    coverage_dir.mkdir(exist_ok=True)

    dll_path = find_library(build_dir, bin_folder, module_name, dll_extension)
    test_dir = build_dir / "Library" / module_name / "Testing" / "Cxx"
    test_executable = build_dir / "bin" / f"{module_name}CxxTests{exe_extension}"
    profraw_file = build_dir / "coverage_report" / f"{module_name}CxxTests.profraw"
    
    if not test_executable.exists():
        print(f"Warning: Test executable not found for {module_name}, skipping")
        return False
    
    with open(binaries_list, 'a') as f:
        print(f"Adding {dll_path} to binaries list")    
        f.write(f"-object={dll_path}\n")
    
    env = os.environ.copy()
    env['LLVM_PROFILE_FILE'] = str(profraw_file)
    try:
        subprocess.run([str(test_executable)], env=env, check=False, cwd=str(test_dir), capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Test execution failed for {module_name}: {e}")
        return False
    except FileNotFoundError as e:
        print(f"Warning: Could not execute {module_name}CxxTests: {e}")
        return False
    
    with open(binaries_list, 'a') as f:
        print(f"Adding {test_executable} to binaries list")
        f.write(f"-object={test_executable}\n")
    with open(profraw_list, 'a') as f:
        print(f"Adding {profraw_file} to profraw list")
        f.write(f"{profraw_file}\n")
    
    return True

def generate_llvm_coverage(build_dir: Path, modules: List[str],
                          source_folder: Path,
                          llvm_ignore_regex: List[str] = None) -> None:
    """Generate code coverage using LLVM (for Clang).

    Args:
        build_dir: Path to build directory.
        modules: List of module names to analyze.
        source_folder: Path to source folder containing modules.
        llvm_ignore_regex: List of regex patterns to ignore. If None, uses CONFIG.
    """
    if llvm_ignore_regex is None:
        llvm_ignore_regex = CONFIG["llvm_ignore_regex"]

    build_dir = Path(build_dir)
    coverage_dir = build_dir / "coverage_report"
    coverage_dir.mkdir(exist_ok=True)
    
    binaries_list = coverage_dir / "binaries.list"
    profraw_list = coverage_dir / "profraw.list"
    
    binaries_list.write_text("")
    profraw_list.write_text("")

    # Discover and run test executables
    print("Discovering test executables...")
    all_profraw_files = []
    successful_modules = 0
    for module in modules:
        if prepare_llvm_coverage(build_dir, module, str(binaries_list), str(profraw_list)):
             successful_modules += 1

    if successful_modules == 0:
        print("Error: No modules processed successfully")
        return

    # Find all .profraw files
    print(f"Successfully processed {successful_modules}/{len(modules)} modules")
    
    print("Merging profile data...")
    profraw_files = profraw_list.read_text().strip().split('\n')
    profraw_files = [f for f in profraw_files if f]
    
    if not profraw_files:
        print("Error: No profraw files generated")
        return
    
    merge_cmd = [
        "llvm-profdata", "merge", "-o",
        str(coverage_dir / "all-merged.profdata"), "-sparse"
    ] + profraw_files
    subprocess.run(merge_cmd, check=True)
    
    print("Creating coverage report...")
    binaries = binaries_list.read_text().strip().split('\n')
    binaries = [b for b in binaries if b]
    
    show_cmd = [
        "llvm-cov", "show"
    ] + binaries + [
        f"-instr-profile={coverage_dir / 'all-merged.profdata'}",
        "-use-color",
        f"-output-dir={coverage_dir / 'html'}",
        "-format=html"
    ]
    
    # Add ignore patterns from config
    for pattern in CONFIG["llvm_ignore_regex"]:
        show_cmd.insert(-3, f"-ignore-filename-regex={pattern}")
    
    subprocess.run(show_cmd, check=True)
    print(f"Coverage report generated in {coverage_dir / 'html'}")

