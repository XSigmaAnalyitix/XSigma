#!/usr/bin/env python3
"""Code coverage generation tool for XSigma project.

Supports multiple compilers (Clang, GCC, MSVC) with automatic detection.
Generates HTML reports and JSON summaries for code coverage analysis.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
from typing import List, Optional

# ============================================================================
# CONFIGURATION - Centralized hardcoded values
# ============================================================================

CONFIG = {
    # Default filter folder (subfolder within source containing modules)
    "filter": "Library",

    # File patterns to exclude from coverage reports
    "exclude_patterns": [
        "*ThirdParty*",
        "xsigmaType*Array.*",
        "*Python.cxx",
        "*Wrapping*",
        "*Examples*",
        "*Testing*",
        "*Utilities*",
        "*_s.cxx",
        "*_vs*.cxx",
        "*_fs*.cxx",
        "*GS.cxx",
        "*VS.cxx",
        "*FS.cxx",
        "*FP*.cxx",
        "*VP*.cxx",
        "xsigmagl.cxx",
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
    
def get_project_root() -> Path:
    """Find the project root directory by searching for common markers.

    Searches upward from the script directory for markers like .git,
    .gitignore, or pyproject.toml.

    Returns:
        Path to project root directory.
    """
    current = Path(__file__).resolve().parent

    while current != current.parent:
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
        build_dir_arg: Build directory path (absolute or relative).
        project_root: Project root directory. If None, auto-detected.

    Returns:
        Resolved absolute Path to build directory.

    Raises:
        ValueError: If build directory cannot be found.
    """
    if not build_dir_arg:
        raise ValueError("Build directory argument is required")

    build_path = Path(build_dir_arg)

    if build_path.is_absolute():
        if not build_path.exists():
            raise ValueError(f"Build directory does not exist: {build_path}")
        return build_path.resolve()

    cwd_relative = (Path.cwd() / build_path).resolve()
    if cwd_relative.exists():
        print(f"Using build directory relative to CWD: {cwd_relative}")
        return cwd_relative

    script_relative = (Path(__file__).resolve().parent / build_path).resolve()
    if script_relative.exists():
        print(f"Using build directory relative to script: {script_relative}")
        return script_relative

    if project_root is None:
        project_root = get_project_root()

    project_relative = (project_root / build_path).resolve()
    if project_relative.exists():
        print(f"Using build directory relative to project root: "
              f"{project_relative}")
        return project_relative

    raise ValueError(
        f"Build directory not found: {build_dir_arg}\n"
        f"  Tried:\n"
        f"    - CWD relative: {cwd_relative}\n"
        f"    - Script relative: {script_relative}\n"
        f"    - Project relative: {project_relative}"
    )

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


def detect_compiler(build_dir: Path | str) -> str:
    """Detect the compiler used in the build directory.

    Looks for CMakeCache.txt to determine if MSVC, Clang, or GCC was used.

    Args:
        build_dir: Path to build directory.

    Returns:
        Compiler type: "msvc", "clang", or "gcc".

    Raises:
        RuntimeError: If compiler cannot be determined.
    """
    cmake_cache = Path(build_dir) / "CMakeCache.txt"

    if not cmake_cache.exists():
        raise RuntimeError(
            f"CMakeCache.txt not found in {build_dir}. "
            "Ensure the build directory is a valid CMake build directory."
        )

    cache_content = cmake_cache.read_text().lower()

    # Check for MSVC
    if "msvc" in cache_content or "visual studio" in cache_content:
        return "msvc"

    # Check for Clang
    if "clang" in cache_content or "clang++" in cache_content:
        return "clang"

    # Check for GCC
    if "gcc" in cache_content or "g++" in cache_content:
        return "gcc"

    # If neither explicit marker found, check CXX_COMPILER path
    if "CMAKE_CXX_COMPILER" in cache_content:
        if "clang" in cache_content:
            return "clang"
        elif "gcc" in cache_content or "g++" in cache_content:
            return "gcc"
        elif "cl.exe" in cache_content or "msvc" in cache_content:
            return "msvc"

    raise RuntimeError(
        "Could not determine compiler from CMakeCache.txt. "
        "Please ensure the build was configured with either MSVC, Clang, "
        "or GCC."
    )

def find_library(build_dir: Path, lib_folder: str, module_name: str,
                 dll_extension: str) -> Optional[str]:
    """Find library file for a module.

    Searches for library files matching the module name pattern.

    Args:
        build_dir: Path to build directory.
        lib_folder: Library folder name within build directory.
        module_name: Module name to search for.
        dll_extension: Dynamic library extension (e.g., ".dll", ".so").

    Returns:
        Path to library file or None if not found.
    """
    lib_path = Path(build_dir) / lib_folder

    if not lib_path.exists():
        return None

    pattern = f"{module_name}*{dll_extension}"
    matches = sorted(lib_path.glob(pattern))

    if matches:
        selected = matches[0]
        if len(matches) > 1:
            print(f"  Found {len(matches)} libraries matching {module_name}; "
                  f"using {selected.name}")
        return str(selected)

    return None


def validate_build_structure(build_dir: Path, config: dict,
                             library_folder: str) -> dict:
    """Validate that the build directory has expected structure.

    Args:
        build_dir: Path to build directory.
        config: Platform configuration dictionary.
        library_folder: Expected library folder name.

    Returns:
        Dictionary with 'valid' (bool) and 'issues' (list) keys.
    """
    issues = []

    lib_folder = build_dir / config["lib_folder"]
    if not lib_folder.exists():
        issues.append(f"Library folder not found: {lib_folder}")

    if not (build_dir / library_folder).exists():
        issues.append(f"Expected source folder not found: "
                      f"{build_dir / library_folder}")

    return {"valid": len(issues) == 0, "issues": issues}

def prepare_llvm_coverage(build_dir: Path, module_name: str) -> List[Path]:
    """Discover and run all test executables for a specific module.

    Searches for test executables containing the module name in their filename
    using flexible pattern matching. Patterns include:
    - {module_name}CxxTests
    - Test{module_name}
    - {module_name}Test
    - Any other executable with {module_name} in the name

    Args:
        build_dir: Path to build directory.
        module_name: Name of the module to find tests for.

    Returns:
        List of Path objects pointing to test executables for the module.
    """
    config = get_platform_config()
    exe_ext = config["exe_extension"]
    test_executables = []

    # Search in common binary locations
    search_dirs = [
        build_dir / "bin",
        build_dir / "lib",
        build_dir / "bin" / "Debug",
        build_dir / "bin" / "Release",
    ]

    # Flexible patterns to match module-specific test executables
    patterns = [
        f"{module_name}CxxTests{exe_ext}",
        f"Test{module_name}{exe_ext}",
        f"{module_name}Test{exe_ext}",
        f"*{module_name}*Test*{exe_ext}",
        f"*Test*{module_name}*{exe_ext}",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for pattern in patterns:
            for exe_file in search_dir.glob(pattern):
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


def generate_llvm_coverage(build_dir: Path, modules: List[str],
                          source_folder: Path) -> None:
    """Generate code coverage using LLVM (for Clang).

    Args:
        build_dir: Path to build directory.
        modules: List of module names to analyze.
        source_folder: Path to source folder containing modules.
    """
    build_dir = Path(build_dir)
    coverage_dir = build_dir / "coverage_report"
    raw_dir = coverage_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    binaries_list = raw_dir / "binaries.list"
    profraw_list = raw_dir / "profraw.list"

    binaries_list.write_text("")
    profraw_list.write_text("")

    config = get_platform_config()

    print(f"Generating LLVM coverage data for modules: {', '.join(modules)}")

    # Discover test executables for each module
    all_test_executables = []
    for module in modules:
        module_tests = prepare_llvm_coverage(build_dir, module)
        all_test_executables.extend(module_tests)

    # Remove duplicates
    seen = set()
    unique_tests = []
    for exe in all_test_executables:
        exe_str = str(exe.resolve())
        if exe_str not in seen:
            seen.add(exe_str)
            unique_tests.append(exe)

    if not unique_tests:
        print("Warning: No test executables found")
        return

    successful_tests = 0
    for test_exe in unique_tests:
        profraw_file = raw_dir / f"{test_exe.stem}.profraw"

        # Add test executable to binaries list
        with open(binaries_list, 'a') as f:
            f.write(f"-object={test_exe}\n")

        env = os.environ.copy()
        env['LLVM_PROFILE_FILE'] = str(profraw_file)

        try:
            print(f"Running: {test_exe.name}")
            subprocess.run([str(test_exe)], env=env, check=True)
            successful_tests += 1

            # Add profraw file to list
            with open(profraw_list, 'a') as f:
                f.write(f"{profraw_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Test execution failed for {test_exe.name}: {e}")
        except FileNotFoundError as e:
            print(f"Warning: Could not execute {test_exe.name}: {e}")

    if successful_tests == 0:
        print("Error: No test executables processed successfully")
        return

    print(f"Successfully processed {successful_tests}/"
          f"{len(unique_tests)} test executables")

    print("Merging profile data...")
    profraw_files = profraw_list.read_text().strip().split('\n')
    profraw_files = [f for f in profraw_files if f]

    if not profraw_files:
        print("Error: No profraw files generated")
        return

    merge_cmd = [
        "llvm-profdata", "merge", "-o",
        str(raw_dir / "all-merged.profdata"), "-sparse"
    ] + profraw_files
    subprocess.run(merge_cmd, check=True)

    print("Creating coverage report...")
    binaries = binaries_list.read_text().strip().split('\n')
    binaries = [b for b in binaries if b]

    show_cmd = [
        "llvm-cov", "show"
    ] + binaries + [
        f"-instr-profile={raw_dir / 'all-merged.profdata'}",
        "-use-color",
        f"-output-dir={coverage_dir / 'html'}",
        "-format=html"
    ]

    # Add ignore patterns from config
    for pattern in CONFIG["llvm_ignore_regex"]:
        show_cmd.insert(-3, f"-ignore-filename-regex={pattern}")

    subprocess.run(show_cmd, check=True)
    print(f"Coverage report generated in {coverage_dir / 'html'}")

def generate_lcov_coverage(build_dir: Path, modules: List[str]) -> None:
    """Generate code coverage using lcov (for GCC).

    Args:
        build_dir: Path to build directory.
        modules: List of module names to analyze.
    """
    build_dir = Path(build_dir)
    coverage_dir = build_dir / "coverage_report"
    raw_dir = coverage_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    original_dir = os.getcwd()
    try:
        os.chdir(build_dir)

        print(f"Generating lcov coverage data for modules: {', '.join(modules)}")
        print("Clearing previous coverage data...")
        subprocess.run(["lcov", "--quiet", "--directory", ".",
                        "--zerocounters"], check=True)

        print("Running tests...")
        subprocess.run(["ctest"], check=True)

        print("Capturing coverage data...")
        app_info = raw_dir / "app.info"
        result = subprocess.run(
            ["lcov", "--quiet", "--directory", ".", "--capture",
             "--output-file", str(app_info)],
            capture_output=True, text=True
        )
        for line in result.stderr.split('\n'):
            if "WARNING" not in line and line.strip():
                print(line)

        print("Removing coverage for excluded files...")
        app_info2 = raw_dir / "app.info2"
        remove_cmd = (["lcov", "--quiet", "--remove", str(app_info)] +
                      CONFIG["exclude_patterns"] +
                      ["--output-file", str(app_info2)])
        subprocess.run(remove_cmd, check=True)

        print("Generating HTML report...")
        subprocess.run(["genhtml", "-o", str(coverage_dir / "html"),
                        "--quiet", str(app_info2)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)

        config = get_platform_config()
        if config["os_name"] == "macOS":
            print("To view results, type: open ./coverage_report/html/index.html")
        else:
            print("To view results, type: firefox ./coverage_report/html/index.html")
    finally:
        os.chdir(original_dir)

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
        raise RuntimeError("MSVC coverage is only supported on Windows")

    print(f"Generating MSVC coverage data for modules: {', '.join(modules)}")

    # Find opencppcoverage executable with full path
    opencppcoverage_path = find_opencppcoverage()
    if not opencppcoverage_path:
        raise RuntimeError(
            "opencppcoverage not found. Please install it from:\n"
            "https://github.com/OpenCppCoverage/OpenCppCoverage\n"
            "Or add it to your system PATH."
        )

    print(f"Using opencppcoverage: {opencppcoverage_path}")

    # Discover all test executables
    test_executables = discover_test_executables(build_dir)
    if not test_executables:
        print("Warning: No test executables found")
        return

    successful_tests = 0
    for test_exe in test_executables:
        try:
            # Build opencppcoverage command with correct parameters
            cmd = [
                opencppcoverage_path,
                "--modules", "*.dll",
                "--sources", str(source_folder),
                "--export_type=html:" + str(coverage_dir / "html"),
                "--excluded_modules", "*.exe",
                "--no_aggregate_by_file",
                "--working_dir", str(raw_dir),
                "--",
                str(test_exe)
            ]

            print(f"Running: {test_exe.name}")
            subprocess.run(cmd, check=True, cwd=str(raw_dir))
            successful_tests += 1
            print(f"Coverage report generated for {test_exe.name}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Test execution failed for {test_exe.name}: {e}")
        except FileNotFoundError as e:
            print(f"Warning: Could not execute {test_exe.name}: {e}")

    if successful_tests == 0:
        print("Error: No test executables processed successfully")
        return

    print(f"Successfully processed {successful_tests}/{len(test_executables)} "
          "test executables")
    print(f"Coverage reports generated in {coverage_dir}")


def discover_test_executables(build_dir: Path) -> List[Path]:
    """Discover all test executables in the build directory.

    Searches for executables matching common test patterns in bin and lib folders.

    Args:
        build_dir: Path to the build directory.

    Returns:
        List of Path objects pointing to test executables.
    """
    test_executables = []
    config = get_platform_config()
    exe_ext = config["exe_extension"]

    # Search in common binary locations
    search_dirs = [
        build_dir / "bin",
        build_dir / "lib",
        build_dir / "bin" / "Debug",
        build_dir / "bin" / "Release",
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


def get_coverage(
    compiler: str = "auto",
    build_folder: str | Path = ".",
    source_folder: str | Path = "Library",
    output_folder: Optional[str | Path] = None,
    exclude: Optional[List[str]] = None,
    summary: bool = True,
    xsigma_root: Optional[str | Path] = None,
) -> int:
    """Generate code coverage report for the XSigma project.

    Provides a programmatic interface for coverage generation with automatic
    compiler detection. Supports Clang, GCC, and MSVC compilers.

    Args:
        compiler: Compiler to use ('clang', 'gcc', 'msvc', or 'auto' for
            automatic detection). Default: 'auto'.
        build_folder: Path to build directory (absolute or relative to
            xsigma_root). Default: '.'.
        source_folder: Path to source directory (absolute or relative to
            xsigma_root). Default: 'Library'.
        output_folder: Path to output directory. If None, uses
            build_folder/coverage_report. Default: None.
        exclude: List of folders to exclude from coverage analysis.
            Default: None.
        summary: Whether to generate and display summary report.
            Default: True.
        xsigma_root: XSigma root directory for resolving relative paths.
            If None, uses current directory. Default: None.

    Returns:
        Exit code (0 for success, non-zero for failure).

    Raises:
        ValueError: If paths are invalid or compiler cannot be detected.
        RuntimeError: If required tools are not found.
    """
    try:
        # Resolve paths
        if xsigma_root is None:
            xsigma_root = Path.cwd()
        else:
            xsigma_root = Path(xsigma_root)

        build_path = Path(build_folder)
        if not build_path.is_absolute():
            build_path = xsigma_root / build_path
        build_path = build_path.resolve()

        source_path = Path(source_folder)
        if not source_path.is_absolute():
            source_path = xsigma_root / source_path
        source_path = source_path.resolve()

        if output_folder is None:
            output_path = build_path / "coverage_report"
        else:
            output_path = Path(output_folder)
            if not output_path.is_absolute():
                output_path = xsigma_root / output_path
            output_path = output_path.resolve()

        # Validate paths
        if not build_path.exists():
            raise ValueError(f"Build directory does not exist: {build_path}")
        if not source_path.exists():
            raise ValueError(f"Source directory does not exist: {source_path}")

        # Detect compiler if needed
        if compiler == "auto":
            compiler = detect_compiler(build_path)

        compiler = compiler.lower()
        if compiler not in ["clang", "gcc", "msvc"]:
            raise ValueError(f"Unsupported compiler: {compiler}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Discover modules from source directory
        modules = [d.name for d in source_path.iterdir()
                   if d.is_dir() and not d.name.startswith("_")]

        if not modules:
            raise ValueError(f"No modules found in {source_path}")

        # Run coverage based on compiler
        if compiler == "msvc":
            generate_msvc_coverage(build_path, modules, source_path)
        elif compiler == "clang":
            generate_llvm_coverage(build_path, modules, source_path)
        elif compiler == "gcc":
            generate_lcov_coverage(build_path, modules)

        return 0

    except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Coverage generation failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error during coverage generation: {e}",
              file=sys.stderr)
        return 1


def print_help():
    """Print usage instructions."""
    print("""
    Code Coverage Generator

    Usage:
        python run_coverage.py --build=<build_directory> [--filter=<folder>] [--verbose]

    Arguments:
        --build=PATH        Build directory (required). Can be absolute or relative to:
                            1. Current working directory
                            2. Script directory
                            3. Project root (auto-detected from .git, etc.)

        --filter=FOLDER     Filter folder within source (default: Library).
                            All subfolders are treated as modules.

        --verbose           Print additional debug information

    Compiler Detection:
        The script automatically detects the compiler from CMakeCache.txt:
        - MSVC: Uses opencppcoverage coverage tool (Windows only)
        - Clang: Uses LLVM coverage tools (llvm-profdata, llvm-cov)
        - GCC: Uses lcov coverage tool

    Requirements for MSVC (opencppcoverage):
        - Windows platform
        - MSVC compiler
        - opencppcoverage installed (https://github.com/OpenCppCoverage/OpenCppCoverage)
        - Added to system PATH

    Requirements for Clang (LLVM coverage):
        - Clang compiler with instrumentation
        - LLVM tools: llvm-profdata, llvm-cov

    Requirements for GCC (lcov coverage):
        - Linux or macOS
        - lcov installed
        - GCC compiler with -fprofile-arcs -ftest-coverage flags

    Examples:
        # Auto-detect compiler
        python run_coverage.py --build=build

        # With custom filter folder
        python run_coverage.py --build=build --filter=Src

        # Relative to project root with verbose output
        python run_coverage.py --build=build --verbose
    """)

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Code Coverage Generator",
        add_help=False
    )

    parser.add_argument("--build", required=True, help="Build directory path")
    parser.add_argument("--filter", default=CONFIG["filter"],
                        help="Filter folder name (default: Library)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("-h", "--help", action="store_true",
                        help="Show help message")

    args = parser.parse_args()

    if args.help:
        print_help()
        return

    try:
        project_root = get_project_root()
        if args.verbose:
            print(f"Project root detected at: {project_root}")

        build_dir = resolve_build_dir(args.build, project_root)
        if args.verbose:
            print(f"Resolved build directory: {build_dir}")

        source_dir = project_root / args.filter
        if args.verbose:
            print(f"Filter folder: {args.filter}")
            print(f"Source directory: {source_dir}")

        # Detect compiler
        compiler = detect_compiler(build_dir)
        print(f"Detected compiler: {compiler.upper()}")

        # Discover modules from filter directory
        modules = [d.name for d in source_dir.iterdir()
                   if d.is_dir() and not d.name.startswith("_")]

        if not modules:
            raise ValueError(f"No modules found in {source_dir}")

        if args.verbose:
            print(f"Modules to analyze: {', '.join(modules)}")

        # Choose coverage tool based on compiler
        if compiler == "msvc":
            print("Using opencppcoverage tool...")
            generate_msvc_coverage(build_dir, modules, source_dir)
        elif compiler == "clang":
            print("Using LLVM coverage tools...")
            generate_llvm_coverage(build_dir, modules, source_dir)
        elif compiler == "gcc":
            print("Using lcov coverage tool...")
            generate_lcov_coverage(build_dir, modules)
        else:
            raise RuntimeError(f"Unsupported compiler: {compiler}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: Required tool not found: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
