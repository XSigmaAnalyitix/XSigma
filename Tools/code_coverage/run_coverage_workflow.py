#!/usr/bin/env python3
"""
Complete Code Coverage Workflow for XSigma

This script orchestrates the complete code coverage workflow:
1. Collects coverage data using OpenCppCoverage.exe
2. Generates HTML reports from the collected data

Usage:
    python run_coverage_workflow.py --test-exe <path> --sources <path> [--output <path>]

Example:
    python run_coverage_workflow.py \
        --test-exe C:\\dev\\build_ninja_lto\\bin\\CoreCxxTests.exe \
        --sources c:\\dev\\XSigma\\Library \
        --output coverage_report
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


class CoverageWorkflow:
    """Orchestrates the complete coverage workflow."""

    def __init__(
        self,
        test_exe: str,
        sources: str,
        output_dir: str = "coverage_report",
        source_root: Optional[str] = None,
        verbose: bool = False,
        excluded_sources: Optional[list] = None,
    ):
        """Initialize the coverage workflow.

        Args:
            test_exe: Path to the test executable
            sources: Path to source files to collect coverage for
            output_dir: Output directory for all coverage artifacts
            source_root: Root directory of source files (for HTML report)
            verbose: Enable verbose output
            excluded_sources: List of patterns to exclude from coverage (e.g., ['*Testing*'])
        """
        self.test_exe = Path(test_exe)
        self.sources = Path(sources)
        self.output_dir = Path(output_dir)
        self.source_root = source_root or str(self.sources.parent)
        self.verbose = verbose
        self.excluded_sources = excluded_sources or []

        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.html_dir = self.output_dir / "html"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        """Print log message if verbose is enabled."""
        if self.verbose:
            print(f"[CoverageWorkflow] {message}")

    def _run_command(self, cmd: list, cwd: Optional[Path] = None) -> bool:
        """Run a command and return success status.

        Args:
            cmd: Command to run as list
            cwd: Working directory

        Returns:
            True if command succeeded, False otherwise
        """
        # Convert all Path objects to strings with proper escaping
        cmd_str = [str(c) for c in cmd]
        self._log(f"Running: {' '.join(cmd_str)}")
        try:
            result = subprocess.run(
                cmd_str,
                check=False,
                cwd=str(cwd) if cwd else None,
            )
            return result.returncode == 0
        except Exception as e:
            print(f"ERROR: Failed to run command: {e}", file=sys.stderr)
            return False

    def collect_coverage(self) -> bool:
        """Collect coverage data.

        Returns:
            True if collection succeeded, False otherwise
        """
        print("=" * 70)
        print("STEP 1: Collecting Coverage Data")
        print("=" * 70)

        script_dir = Path(__file__).parent
        collect_script = script_dir / "collect_coverage_data.py"

        cmd = [
            sys.executable,
            str(collect_script),
            "--test-exe",
            str(self.test_exe),
            "--sources",
            str(self.sources),
            "--output",
            str(self.data_dir),
        ]

        # Add excluded sources
        for excluded in self.excluded_sources:
            cmd.extend(["--excluded-sources", excluded])

        if self.verbose:
            cmd.append("--verbose")

        if not self._run_command(cmd):
            print("ERROR: Coverage data collection failed", file=sys.stderr)
            return False

        print("[OK] Coverage data collection completed")
        return True

    def generate_html_report(self) -> bool:
        """Generate HTML report from collected coverage data.

        Returns:
            True if generation succeeded, False otherwise
        """
        print("\n" + "=" * 70)
        print("STEP 2: Generating HTML Report")
        print("=" * 70)

        # Find coverage data file
        coverage_file = None
        for ext in [".xml", ".cov", ".json"]:
            candidate = self.data_dir / f"coverage{ext}"
            if candidate.exists():
                coverage_file = candidate
                break

        if not coverage_file:
            print("ERROR: No coverage data file found", file=sys.stderr)
            return False

        script_dir = Path(__file__).parent
        generate_script = script_dir / "generate_html_report.py"

        cmd = [
            sys.executable,
            str(generate_script),
            "--coverage-data",
            str(coverage_file),
            "--output",
            str(self.html_dir),
            "--source-root",
            self.source_root,
        ]

        if self.verbose:
            cmd.append("--verbose")

        if not self._run_command(cmd):
            print("ERROR: HTML report generation failed", file=sys.stderr)
            return False

        print("[OK] HTML report generation completed")
        return True

    def run(self) -> bool:
        """Run the complete coverage workflow.

        Returns:
            True if workflow succeeded, False otherwise
        """
        print("\n" + "=" * 70)
        print("XSigma Code Coverage Workflow")
        print("=" * 70)
        print(f"Test executable: {self.test_exe}")
        print(f"Source directory: {self.sources}")
        print(f"Output directory: {self.output_dir}")
        print()

        # Step 1: Collect coverage data
        if not self.collect_coverage():
            return False

        # Step 2: Generate HTML report
        if not self.generate_html_report():
            return False

        # Print summary
        print("\n" + "=" * 70)
        print("Coverage Workflow Completed Successfully!")
        print("=" * 70)
        print(f"Coverage data: {self.data_dir}")
        print(f"HTML report: {self.html_dir}")
        print(f"Open {self.html_dir / 'index.html'} in a browser to view the report")
        print()

        return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete code coverage workflow for XSigma"
    )
    parser.add_argument(
        "--test-exe",
        required=True,
        help="Path to the test executable",
    )
    parser.add_argument(
        "--sources",
        required=True,
        help="Path to source files to collect coverage for",
    )
    parser.add_argument(
        "--output",
        default="coverage_report",
        help="Output directory for coverage artifacts (default: coverage_report)",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Root directory of source files (for HTML report)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--excluded-sources",
        action="append",
        default=[],
        help="Patterns to exclude from coverage (e.g., '*Testing*'). Can be specified multiple times.",
    )

    args = parser.parse_args()

    try:
        workflow = CoverageWorkflow(
            test_exe=args.test_exe,
            sources=args.sources,
            output_dir=args.output,
            source_root=args.source_root,
            verbose=args.verbose,
            excluded_sources=args.excluded_sources,
        )

        if not workflow.run():
            return 1

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
