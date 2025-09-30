#!/usr/bin/env python3
"""
Cross-platform coverage analysis tool for XSigma project.

This script analyzes LLVM coverage reports to identify Core library files
below a specified coverage threshold (default: 95%).

Features:
- Fully cross-platform (Windows, Linux, macOS)
- Auto-detects LLVM tools (llvm-cov, llvm-profdata)
- Configurable via command-line arguments, environment variables, or config file
- Machine-independent (no hardcoded paths)
- Supports dry-run and verbose modes for debugging
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Configuration and Constants
# =============================================================================

class Config:
    """Configuration container for coverage analysis."""

    def __init__(self):
        # Get script and repository root directories
        self.script_dir = Path(__file__).resolve().parent
        self.repo_root = self.script_dir.parent

        # Default paths (relative to repo root)
        self.build_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.source_dir: Path = self.repo_root

        # Tool paths
        self.llvm_cov: Optional[str] = None
        self.llvm_profdata: Optional[str] = None

        # Coverage settings
        self.coverage_threshold: float = 95.0
        self.library_filter: str = "Library/Core"  # Use forward slash for cross-platform

        # Runtime options
        self.dry_run: bool = False
        self.verbose: bool = False

    def __repr__(self):
        return (
            f"Config(\n"
            f"  script_dir={self.script_dir}\n"
            f"  repo_root={self.repo_root}\n"
            f"  build_dir={self.build_dir}\n"
            f"  output_dir={self.output_dir}\n"
            f"  source_dir={self.source_dir}\n"
            f"  llvm_cov={self.llvm_cov}\n"
            f"  llvm_profdata={self.llvm_profdata}\n"
            f"  coverage_threshold={self.coverage_threshold}\n"
            f"  library_filter={self.library_filter}\n"
            f"  dry_run={self.dry_run}\n"
            f"  verbose={self.verbose}\n"
            f")"
        )


# =============================================================================
# Logging and Output
# =============================================================================

def log_info(message: str, config: Optional[Config] = None):
    """Print informational message."""
    print(f"[INFO] {message}")


def log_verbose(message: str, config: Config):
    """Print verbose message if verbose mode is enabled."""
    if config.verbose:
        print(f"[VERBOSE] {message}")


def log_warning(message: str):
    """Print warning message."""
    print(f"[WARNING] {message}", file=sys.stderr)


def log_error(message: str):
    """Print error message."""
    print(f"[ERROR] {message}", file=sys.stderr)


# =============================================================================
# Platform Detection
# =============================================================================

def get_platform_info() -> Dict[str, str]:
    """Get platform information."""
    system = platform.system()
    return {
        "system": system,
        "machine": platform.machine(),
        "platform": sys.platform,
        "is_windows": system == "Windows",
        "is_linux": system == "Linux",
        "is_macos": system == "Darwin",
    }


def get_executable_extension() -> str:
    """Get the executable extension for the current platform."""
    return ".exe" if platform.system() == "Windows" else ""


# =============================================================================
# Tool Detection and Validation
# =============================================================================

def find_llvm_tool(tool_name: str, config: Config) -> Optional[str]:
    """
    Find LLVM tool executable on the system.

    Search order:
    1. Environment variable (e.g., LLVM_COV_PATH)
    2. System PATH (using shutil.which)
    3. Common installation directories for each platform

    Args:
        tool_name: Name of the tool (e.g., 'llvm-cov', 'llvm-profdata')
        config: Configuration object

    Returns:
        Path to the tool executable, or None if not found
    """
    platform_info = get_platform_info()
    exe_ext = get_executable_extension()
    tool_exe = f"{tool_name}{exe_ext}"

    # 1. Check environment variable
    env_var = f"{tool_name.upper().replace('-', '_')}_PATH"
    env_path = os.environ.get(env_var)
    if env_path:
        env_path_obj = Path(env_path)
        if env_path_obj.is_file() and os.access(env_path_obj, os.X_OK):
            log_verbose(f"Found {tool_name} via environment variable {env_var}: {env_path}", config)
            return str(env_path_obj)
        else:
            log_warning(f"Environment variable {env_var} is set but points to invalid path: {env_path}")

    # 2. Check system PATH
    which_result = shutil.which(tool_name)
    if which_result:
        log_verbose(f"Found {tool_name} in system PATH: {which_result}", config)
        return which_result

    # 3. Check common installation directories
    common_paths: List[Path] = []

    if platform_info["is_windows"]:
        # Windows common paths
        common_paths.extend([
            Path("C:/Program Files/LLVM/bin") / tool_exe,
            Path("C:/Program Files (x86)/LLVM/bin") / tool_exe,
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "LLVM" / "bin" / tool_exe,
        ])
    elif platform_info["is_macos"]:
        # macOS common paths (Homebrew, MacPorts)
        common_paths.extend([
            Path("/usr/local/opt/llvm/bin") / tool_name,
            Path("/opt/homebrew/opt/llvm/bin") / tool_name,
            Path("/opt/local/bin") / tool_name,
            Path("/usr/local/bin") / tool_name,
        ])
    elif platform_info["is_linux"]:
        # Linux common paths
        common_paths.extend([
            Path("/usr/bin") / tool_name,
            Path("/usr/local/bin") / tool_name,
            Path("/opt/llvm/bin") / tool_name,
        ])

    for path in common_paths:
        if path.is_file() and os.access(path, os.X_OK):
            log_verbose(f"Found {tool_name} at common location: {path}", config)
            return str(path)

    return None


def get_installation_instructions(tool_name: str) -> str:
    """Get installation instructions for LLVM tools based on platform."""
    platform_info = get_platform_info()

    instructions = [
        f"\n{tool_name} not found. Please install LLVM tools:\n"
    ]

    if platform_info["is_windows"]:
        instructions.extend([
            "Windows:",
            "  1. Download from: https://releases.llvm.org/",
            "  2. Or install via Chocolatey: choco install llvm",
            "  3. Or install via Scoop: scoop install llvm",
            f"  4. Or set environment variable: {tool_name.upper().replace('-', '_')}_PATH",
        ])
    elif platform_info["is_macos"]:
        instructions.extend([
            "macOS:",
            "  1. Install via Homebrew: brew install llvm",
            "  2. Or install via MacPorts: sudo port install llvm",
            f"  3. Or set environment variable: {tool_name.upper().replace('-', '_')}_PATH",
            "  Note: You may need to add LLVM to PATH:",
            "    export PATH=\"/usr/local/opt/llvm/bin:$PATH\"",
            "    or",
            "    export PATH=\"/opt/homebrew/opt/llvm/bin:$PATH\"",
        ])
    elif platform_info["is_linux"]:
        instructions.extend([
            "Linux:",
            "  1. Debian/Ubuntu: sudo apt install llvm",
            "  2. Fedora/RHEL: sudo dnf install llvm",
            "  3. Arch: sudo pacman -S llvm",
            f"  4. Or set environment variable: {tool_name.upper().replace('-', '_')}_PATH",
        ])

    return "\n".join(instructions)


def validate_tools(config: Config) -> bool:
    """
    Validate that required LLVM tools are available.

    Returns:
        True if all tools are found, False otherwise
    """
    tools_found = True

    # Find llvm-cov
    if not config.llvm_cov:
        config.llvm_cov = find_llvm_tool("llvm-cov", config)

    if not config.llvm_cov:
        log_error("llvm-cov not found")
        print(get_installation_instructions("llvm-cov"))
        tools_found = False
    else:
        log_verbose(f"Using llvm-cov: {config.llvm_cov}", config)

    # Find llvm-profdata (optional, but recommended)
    if not config.llvm_profdata:
        config.llvm_profdata = find_llvm_tool("llvm-profdata", config)

    if config.llvm_profdata:
        log_verbose(f"Using llvm-profdata: {config.llvm_profdata}", config)
    else:
        log_verbose("llvm-profdata not found (optional)", config)

    return tools_found


# =============================================================================
# Path Resolution and Validation
# =============================================================================

def find_build_directory(config: Config) -> Optional[Path]:
    """
    Find the build directory containing coverage data.

    Search order:
    1. Explicitly configured build_dir
    2. Environment variable XSIGMA_BUILD_DIR
    3. Default patterns: build_ninja_*_coverage, build_ninja_python, build

    Returns:
        Path to build directory, or None if not found
    """
    if config.build_dir and config.build_dir.is_dir():
        return config.build_dir

    # Check environment variable
    env_build_dir = os.environ.get("XSIGMA_BUILD_DIR")
    if env_build_dir:
        env_path = Path(env_build_dir)
        if env_path.is_dir():
            log_verbose(f"Found build directory via XSIGMA_BUILD_DIR: {env_path}", config)
            return env_path
        else:
            log_warning(f"XSIGMA_BUILD_DIR is set but directory doesn't exist: {env_build_dir}")

    # Search for common build directory patterns
    search_patterns = [
        "build_ninja_*_coverage",
        "build_ninja_python",
        "build_ninja_clang",
        "build",
    ]

    for pattern in search_patterns:
        matches = list(config.repo_root.glob(pattern))
        if matches:
            # Prefer coverage builds
            for match in matches:
                if "coverage" in match.name.lower():
                    log_verbose(f"Found coverage build directory: {match}", config)
                    return match
            # Otherwise use first match
            log_verbose(f"Found build directory: {matches[0]}", config)
            return matches[0]

    return None


def find_coverage_profdata(build_dir: Path, config: Config) -> Optional[Path]:
    """
    Find the coverage.profdata file in the build directory.

    Args:
        build_dir: Build directory to search
        config: Configuration object

    Returns:
        Path to coverage.profdata file, or None if not found
    """
    # Common locations for coverage.profdata
    search_paths = [
        build_dir / "coverage_report" / "coverage.profdata",
        build_dir / "coverage" / "coverage.profdata",
        build_dir / "coverage.profdata",
    ]

    for path in search_paths:
        if path.is_file():
            log_verbose(f"Found coverage.profdata: {path}", config)
            return path

    # Search recursively as fallback
    profdata_files = list(build_dir.rglob("coverage.profdata"))
    if profdata_files:
        log_verbose(f"Found coverage.profdata (recursive search): {profdata_files[0]}", config)
        return profdata_files[0]

    return None


def find_test_executables(build_dir: Path, config: Config) -> List[Path]:
    """
    Find test executables in the build directory.

    Args:
        build_dir: Build directory to search
        config: Configuration object

    Returns:
        List of paths to test executables
    """
    exe_ext = get_executable_extension()

    # Common locations for test executables
    bin_dir = build_dir / "bin"

    test_executables = []

    if bin_dir.is_dir():
        # Look for files matching test patterns
        patterns = [f"*Test{exe_ext}", f"*Tests{exe_ext}"]
        for pattern in patterns:
            test_executables.extend(bin_dir.glob(pattern))

    # Remove duplicates and sort
    test_executables = sorted(set(test_executables))

    if test_executables:
        log_verbose(f"Found {len(test_executables)} test executable(s):", config)
        for exe in test_executables:
            log_verbose(f"  - {exe}", config)

    return test_executables



# =============================================================================
# Coverage Report Execution
# =============================================================================

def run_llvm_cov_report(
    profdata_path: Path,
    test_executables: List[Path],
    config: Config
) -> Optional[str]:
    """
    Run llvm-cov report and return the output.

    Args:
        profdata_path: Path to coverage.profdata file
        test_executables: List of test executable paths
        config: Configuration object

    Returns:
        Coverage report output as string, or None on error
    """
    if not test_executables:
        log_error("No test executables found")
        return None

    # Build command
    cmd = [
        config.llvm_cov,
        "report",
        str(test_executables[0]),  # Primary executable
    ]

    # Add additional executables with -object flag
    for exe in test_executables[1:]:
        cmd.extend(["-object", str(exe)])

    # Add profdata and options
    cmd.extend([
        f"-instr-profile={profdata_path}",
        "-show-region-summary=false",
    ])

    log_verbose(f"Running command: {' '.join(cmd)}", config)

    if config.dry_run:
        log_info("DRY RUN: Would execute the above command")
        return None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            check=False
        )

        if result.returncode != 0:
            log_error(f"llvm-cov failed with exit code {result.returncode}")
            if result.stderr:
                log_error(f"stderr: {result.stderr}")
            return None

        return result.stdout

    except Exception as e:
        log_error(f"Error running llvm-cov: {e}")
        return None


# =============================================================================
# Coverage Report Parsing
# =============================================================================

def normalize_path_separators(path_str: str) -> str:
    """
    Normalize path separators to forward slashes for cross-platform comparison.

    Args:
        path_str: Path string with platform-specific separators

    Returns:
        Path string with forward slashes
    """
    return path_str.replace('\\', '/')


def parse_coverage_report(report_text: str, config: Config) -> List[Tuple[str, str, float]]:
    """
    Parse the llvm-cov report output.

    Args:
        report_text: Output from llvm-cov report command
        config: Configuration object

    Returns:
        List of tuples: (full_path, basename, coverage_percentage)
    """
    files_data = []

    # Normalize library filter for cross-platform comparison
    library_filter_normalized = normalize_path_separators(config.library_filter)

    lines = report_text.split('\n')
    for line in lines:
        # Skip header lines, separator lines, and empty lines
        if not line.strip() or line.startswith('---') or line.startswith('Filename'):
            continue
        if line.startswith('TOTAL') or line.startswith('Files which'):
            continue

        # Skip warning lines
        if 'warning:' in line.lower():
            continue

        # Normalize path separators in the line for cross-platform matching
        line_normalized = normalize_path_separators(line)

        # Look for lines with .cxx files from the specified library (but not Testing)
        if library_filter_normalized in line_normalized and '.cxx' in line_normalized:
            if 'Testing' not in line_normalized and 'Test' not in line_normalized:
                # Parse the line to extract filename and coverage
                # Format: Filename   Functions  Missed Functions  Executed  Lines  Missed Lines  Cover  Branches  Missed Branches  Cover
                parts = line.split()
                if len(parts) >= 7:
                    filename = parts[0]
                    # Find the line coverage percentage (it's after "Lines" and "Missed Lines")
                    try:
                        # Find the coverage percentage - it should be the value after missed lines
                        for i, part in enumerate(parts):
                            if '%' in part and i > 0:
                                # This is likely a coverage percentage
                                coverage_str = part.replace('%', '')
                                try:
                                    coverage = float(coverage_str)
                                    # Extract just the filename without path
                                    file_basename = Path(filename).name
                                    files_data.append((filename, file_basename, coverage))
                                    log_verbose(f"Parsed: {file_basename} -> {coverage}%", config)
                                    break
                                except ValueError:
                                    continue
                    except Exception as e:
                        log_warning(f"Error parsing line: {line}")
                        log_warning(f"Error: {e}")

    return files_data


# =============================================================================
# Report Generation
# =============================================================================

def print_coverage_summary(
    files_data: List[Tuple[str, str, float]],
    config: Config
):
    """
    Print coverage analysis summary.

    Args:
        files_data: List of tuples (full_path, basename, coverage_percentage)
        config: Configuration object
    """
    if not files_data:
        print("\nNo library .cxx files found in coverage report.")
        print("This might mean:")
        print("  1. The files are header-only or fully inlined")
        print("  2. The coverage report format has changed")
        print("  3. The files are not being compiled into the test executable")
        print(f"  4. No files match the filter: {config.library_filter}")
        return

    # Categorize files by coverage threshold
    below_threshold = [
        (full, base, cov) for full, base, cov in files_data
        if cov < config.coverage_threshold
    ]
    above_threshold = [
        (full, base, cov) for full, base, cov in files_data
        if cov >= config.coverage_threshold
    ]

    # Print files below threshold
    if below_threshold:
        print(f"\n{'='*100}")
        print(f"Files BELOW {config.coverage_threshold}% coverage ({len(below_threshold)} files):")
        print(f"{'='*100}")
        for full_path, basename, coverage in sorted(below_threshold, key=lambda x: x[2]):
            # Normalize path for display
            display_path = normalize_path_separators(full_path)
            print(f"  {basename:50s} {coverage:6.2f}%  ({display_path})")

    # Print files above threshold
    if above_threshold:
        print(f"\n{'='*100}")
        print(f"Files AT OR ABOVE {config.coverage_threshold}% coverage ({len(above_threshold)} files):")
        print(f"{'='*100}")
        for full_path, basename, coverage in sorted(above_threshold, key=lambda x: x[2], reverse=True):
            print(f"  {basename:50s} {coverage:6.2f}%")

    # Summary
    total = len(below_threshold) + len(above_threshold)
    print(f"\n{'='*100}")
    print(f"Summary:")
    print(f"  Total .cxx files analyzed: {total}")
    print(f"  Files below {config.coverage_threshold}%: {len(below_threshold)}")
    print(f"  Files at or above {config.coverage_threshold}%: {len(above_threshold)}")
    if total > 0:
        percentage = len(above_threshold) / total * 100
        print(f"  Percentage at or above {config.coverage_threshold}%: {percentage:.1f}%")
    print(f"{'='*100}")

    return below_threshold


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config_file(config: Config) -> Dict:
    """
    Load configuration from file if it exists.

    Searches for:
    - coverage_config.json in repo root
    - .coveragerc in repo root (simple key=value format)

    Args:
        config: Configuration object

    Returns:
        Dictionary of configuration values
    """
    config_data = {}

    # Try JSON config file
    json_config_path = config.repo_root / "coverage_config.json"
    if json_config_path.is_file():
        try:
            with open(json_config_path, 'r') as f:
                config_data = json.load(f)
            log_verbose(f"Loaded configuration from {json_config_path}", config)
        except Exception as e:
            log_warning(f"Failed to load {json_config_path}: {e}")

    return config_data


def apply_configuration(config: Config, args: argparse.Namespace):
    """
    Apply configuration from multiple sources in priority order:
    1. Hardcoded defaults (already set in Config.__init__)
    2. Configuration file
    3. Environment variables
    4. Command-line arguments (highest priority)

    Args:
        config: Configuration object to update
        args: Parsed command-line arguments
    """
    # 2. Load from configuration file
    config_file_data = load_config_file(config)

    # Apply build_dir
    if 'build_dir' in config_file_data:
        config.build_dir = Path(config_file_data['build_dir'])

    if os.environ.get('XSIGMA_BUILD_DIR'):
        config.build_dir = Path(os.environ['XSIGMA_BUILD_DIR'])

    if args.build_dir:
        config.build_dir = Path(args.build_dir)

    # Apply output_dir
    if 'output_dir' in config_file_data:
        config.output_dir = Path(config_file_data['output_dir'])

    if os.environ.get('XSIGMA_COVERAGE_OUTPUT'):
        config.output_dir = Path(os.environ['XSIGMA_COVERAGE_OUTPUT'])

    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    # Apply source_dir
    if args.source_dir:
        config.source_dir = Path(args.source_dir)

    # Apply tool paths
    if args.llvm_cov:
        config.llvm_cov = args.llvm_cov

    if args.llvm_profdata:
        config.llvm_profdata = args.llvm_profdata

    # Apply coverage threshold
    if args.threshold:
        config.coverage_threshold = args.threshold

    # Apply library filter
    if hasattr(args, 'library_filter') and args.library_filter:
        config.library_filter = args.library_filter

    # Apply runtime options
    config.dry_run = args.dry_run
    config.verbose = args.verbose


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-platform coverage analysis tool for XSigma project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect everything)
  python analyze_coverage.py

  # Specify build directory
  python analyze_coverage.py --build-dir ../build_ninja_clang_coverage

  # Use custom threshold
  python analyze_coverage.py --threshold 90

  # Dry run to see what would be executed
  python analyze_coverage.py --dry-run --verbose

  # Specify LLVM tools explicitly
  python analyze_coverage.py --llvm-cov /usr/local/bin/llvm-cov

Environment Variables:
  XSIGMA_BUILD_DIR          - Path to build directory
  XSIGMA_COVERAGE_OUTPUT    - Path for coverage report output
  LLVM_COV_PATH             - Path to llvm-cov executable
  LLVM_PROFDATA_PATH        - Path to llvm-profdata executable
        """
    )

    # Path arguments
    parser.add_argument(
        '--build-dir',
        type=str,
        help='Path to the build directory (default: auto-detect)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Path for coverage report output (default: build_dir/coverage_report)'
    )

    parser.add_argument(
        '--source-dir',
        type=str,
        help='Path to source code root (default: repository root)'
    )

    # Tool paths
    parser.add_argument(
        '--llvm-cov',
        type=str,
        help='Path to llvm-cov executable (default: auto-detect)'
    )

    parser.add_argument(
        '--llvm-profdata',
        type=str,
        help='Path to llvm-profdata executable (default: auto-detect)'
    )

    # Analysis options
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Coverage threshold percentage (default: 95.0)'
    )

    parser.add_argument(
        '--library-filter',
        type=str,
        default='Library/Core',
        help='Library path filter (default: Library/Core)'
    )

    # Runtime options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


# =============================================================================
# Main Function
# =============================================================================

def main() -> int:
    """
    Main function to analyze coverage.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    args = parse_arguments()

    # Initialize configuration
    config = Config()
    apply_configuration(config, args)

    # Print header
    print("=" * 100)
    print("XSigma Cross-Platform Coverage Analysis")
    print("=" * 100)
    print()

    # Print platform information
    platform_info = get_platform_info()
    log_info(f"Platform: {platform_info['system']} ({platform_info['machine']})")
    log_info(f"Python: {sys.version.split()[0]}")
    print()

    if config.verbose:
        log_verbose("Configuration:", config)
        log_verbose(str(config), config)
        print()

    # Step 1: Validate LLVM tools
    log_info("Step 1: Validating LLVM tools...")
    if not validate_tools(config):
        log_error("Required tools not found. Please install LLVM tools.")
        return 1
    print()

    # Step 2: Find build directory
    log_info("Step 2: Locating build directory...")
    build_dir = find_build_directory(config)
    if not build_dir:
        log_error("Build directory not found.")
        log_error("Please specify with --build-dir or set XSIGMA_BUILD_DIR environment variable.")
        log_error(f"Searched in: {config.repo_root}")
        return 1

    config.build_dir = build_dir
    log_info(f"Using build directory: {build_dir}")
    print()

    # Step 3: Find coverage data
    log_info("Step 3: Locating coverage data...")
    profdata_path = find_coverage_profdata(build_dir, config)
    if not profdata_path:
        log_error("coverage.profdata not found in build directory.")
        log_error("Please run tests with coverage enabled first.")
        log_error(f"Expected location: {build_dir}/coverage_report/coverage.profdata")
        return 1

    log_info(f"Found coverage data: {profdata_path}")
    print()

    # Step 4: Find test executables
    log_info("Step 4: Locating test executables...")
    test_executables = find_test_executables(build_dir, config)
    if not test_executables:
        log_error("No test executables found in build directory.")
        log_error(f"Expected location: {build_dir}/bin/*Test{get_executable_extension()}")
        return 1

    log_info(f"Found {len(test_executables)} test executable(s)")
    for exe in test_executables:
        log_info(f"  - {exe.name}")
    print()

    # Step 5: Run coverage report
    log_info("Step 5: Running llvm-cov report...")
    report_text = run_llvm_cov_report(profdata_path, test_executables, config)

    if config.dry_run:
        log_info("Dry run complete. No actual coverage analysis performed.")
        return 0

    if not report_text:
        log_error("Failed to generate coverage report")
        return 1

    log_info("Coverage report generated successfully")
    print()

    # Step 6: Parse coverage data
    log_info("Step 6: Parsing coverage data...")
    files_data = parse_coverage_report(report_text, config)
    log_info(f"Parsed {len(files_data)} file(s)")
    print()

    # Step 7: Print summary
    log_info("Step 7: Generating summary...")
    print()
    below_threshold = print_coverage_summary(files_data, config)

    # Return exit code based on results
    if below_threshold:
        log_warning(f"\n{len(below_threshold)} file(s) below {config.coverage_threshold}% coverage threshold")
        return 1
    else:
        log_info(f"\nAll files meet or exceed {config.coverage_threshold}% coverage threshold!")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log_error("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        log_error(f"\nUnexpected error: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)

