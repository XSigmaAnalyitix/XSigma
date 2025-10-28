"""
Cppcheck Static Analysis Helper Module
This module handles static code analysis using cppcheck.
Extracted from setup.py for better modularity and maintainability.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class CppcheckIssue:
    """Represents a single cppcheck issue."""

    id: str
    file: str
    line: int
    severity: str
    message: str

    @classmethod
    def from_line(cls, line: str) -> Optional["CppcheckIssue"]:
        """Parse a cppcheck output line into an issue object."""
        if not line or "," not in line:
            return None

        parts = line.split(",", 4)
        if len(parts) < 4:
            return None

        try:
            issue_id = parts[0].strip()
            file_line = parts[1].strip()
            severity = parts[2].strip()
            message = parts[3].strip() if len(parts) > 3 else ""

            # Parse file:line
            if ":" in file_line:
                file_path, line_num = file_line.rsplit(":", 1)
                line_int = int(line_num) if line_num.isdigit() else 0
            else:
                file_path = file_line
                line_int = 0

            return cls(
                id=issue_id,
                file=file_path,
                line=line_int,
                severity=severity,
                message=message,
            )
        except (ValueError, IndexError):
            return None


def get_logical_processor_count() -> int:
    """Get the number of logical processors available."""
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil.cpu_count(logical=True)  # type: ignore[no-untyped-call,no-any-return]
    except ImportError:
        try:
            count = os.cpu_count()
            return count if count is not None else 1
        except AttributeError:
            import multiprocessing

            return multiprocessing.cpu_count()


def build_cppcheck_command(
    source_path: str,
    output_file: str,
    max_parallel_jobs: int = 8,
    additional_includes: Optional[list[str]] = None,
    check_level: str = "exhaustive",
    exclude_patterns: Optional[list[str]] = None,
) -> list[str]:
    """
    Build the cppcheck command with appropriate settings.

    Args:
        source_path: Root path of the source code
        output_file: Path to write cppcheck output
        max_parallel_jobs: Maximum number of parallel jobs (default: 8)
        additional_includes: Additional include directories
        check_level: Cppcheck check level (normal/exhaustive)
        exclude_patterns: Additional patterns to exclude (beyond default build*)

    Returns:
        List of command arguments for subprocess
    """
    cpu_count = get_logical_processor_count()
    parallel_jobs = min(cpu_count, max_parallel_jobs)

    cmd = [
        "cppcheck",
        ".",
        "--platform=unspecified",
        "--enable=all",
        "--inline-suppr",
        "-q",
        "--library=qt",
        "--library=posix",
        "--library=gnu",
        "--library=bsd",
        "--library=windows",
        f"--check-level={check_level}",
        "--template={id},{file}:{line},{severity},{message}",
        "--suppress=missingInclude",
        "--suppress=missingIncludeSystem",
        "--suppress=toomanyconfigs",  # Suppress informational noise
        "--suppress=unmatchedSuppression",  # Suppress unmatched suppression warnings
        "--suppress=checkersReport",  # Suppress checkers report information
        f"-j{parallel_jobs}",
        "-I",
        "Library",
        f"--output-file={output_file}",
    ]

    # Exclude build directories (all folders starting with "build")
    default_excludes = [
        "-ibuild",
        "-ibuild_*",
        "-i./build",
        "-i./build_*",
    ]
    cmd.extend(default_excludes)

    # Add custom exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            cmd.append(f"-i{pattern}")

    # Add additional include directories
    if additional_includes:
        for inc_dir in additional_includes:
            cmd.extend(["-I", inc_dir])

    # Add suppressions file if it exists
    suppressions_file = os.path.join(
        source_path, "Scripts", "suppressions", "cppcheck_suppressions.txt"
    )
    if os.path.exists(suppressions_file):
        cmd.append(f"--suppressions-list={suppressions_file}")

    return cmd


def parse_cppcheck_output(output_file: str) -> list[CppcheckIssue]:
    """
    Parse cppcheck output file into structured issues.

    Args:
        output_file: Path to cppcheck output file

    Returns:
        List of CppcheckIssue objects
    """
    issues: list[CppcheckIssue] = []

    if not os.path.exists(output_file):
        return issues

    try:
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    issue = CppcheckIssue.from_line(line)
                    if issue:
                        issues.append(issue)
    except Exception as e:
        print(f"Warning: Error parsing cppcheck output: {e}")

    return issues


def summarize_issues(issues: list[CppcheckIssue]) -> dict[str, int]:
    """
    Summarize issues by severity.

    Args:
        issues: List of CppcheckIssue objects

    Returns:
        Dictionary mapping severity to count
    """
    summary: dict[str, int] = {}
    for issue in issues:
        summary[issue.severity] = summary.get(issue.severity, 0) + 1
    return summary


def print_issue_summary(issues: list[CppcheckIssue]) -> None:
    """Print a human-readable summary of cppcheck issues."""
    if not issues:
        print("✓ No cppcheck issues found!")
        return

    summary = summarize_issues(issues)

    print(f"\n{'=' * 60}")
    print("Cppcheck Analysis Summary")
    print(f"{'=' * 60}")
    print(f"Total issues found: {len(issues)}")
    print("\nBy severity:")

    # Sort by severity priority
    severity_order = [
        "error",
        "warning",
        "style",
        "performance",
        "portability",
        "information",
    ]
    for severity in severity_order:
        if severity in summary:
            print(f"  {severity:15s}: {summary[severity]:4d}")

    # Print any other severities not in the list
    for severity, count in sorted(summary.items()):
        if severity not in severity_order:
            print(f"  {severity:15s}: {count:4d}")

    print(f"{'=' * 60}\n")

    # Print critical errors
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        print("Critical Errors:")
        for err in errors[:10]:  # Limit to first 10
            print(f"  {err.file}:{err.line} - {err.message}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print()


def process_cppcheck_results(
    result: subprocess.CompletedProcess[str], output_file: str, verbose: bool = False
) -> int:
    """
    Process cppcheck results and return exit code.

    Args:
        result: Completed subprocess result
        output_file: Path to cppcheck output file
        verbose: Whether to print detailed summary

    Returns:
        Exit code (0 for success, 1 for errors found)
    """
    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} not found")
        return 1

    try:
        issues = parse_cppcheck_output(output_file)

        if verbose or any(i.severity == "error" for i in issues):
            print_issue_summary(issues)

        # Return error code if critical issues found
        has_errors = any(i.severity == "error" for i in issues)
        return 1 if has_errors else 0

    except Exception as e:
        print(f"Error processing cppcheck results: {e}")
        return 1


def run_cppcheck_analysis(
    source_path: str, output_file: str = "cppcheck_output.txt", verbose: bool = True
) -> int:
    """
    Run complete cppcheck analysis workflow.

    Args:
        source_path: Root path of source code
        output_file: Path to write cppcheck output
        verbose: Whether to print detailed output

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    cmd = build_cppcheck_command(source_path, output_file)

    if verbose:
        print(f"Running cppcheck with {cmd[cmd.index('-j') + 1][2:]} parallel jobs...")

    try:
        result = subprocess.run(
            cmd, cwd=source_path, capture_output=True, text=True, check=False
        )

        return process_cppcheck_results(result, output_file, verbose)

    except FileNotFoundError:
        print("Error: cppcheck not found. Please install cppcheck.")
        print("  macOS: brew install cppcheck")
        print("  Ubuntu: sudo apt-get install cppcheck")
        return 1
    except Exception as e:
        print(f"Error running cppcheck: {e}")
        return 1


if __name__ == "__main__":
    # Example usage
    import sys

    source_path = sys.argv[1] if len(sys.argv) > 1 else "."
    exit_code = run_cppcheck_analysis(source_path)
    sys.exit(exit_code)
