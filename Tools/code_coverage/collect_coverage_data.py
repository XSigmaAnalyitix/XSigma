#!/usr/bin/env python3
"""
Coverage Data Collection Script for XSigma

This script collects code coverage data using OpenCppCoverage.exe on Windows.
It runs a specified test executable and captures coverage information for
source files in the Library folder.

Usage:
    python collect_coverage_data.py --test-exe <path> --sources <path> [--output <path>]

Example:
    python collect_coverage_data.py \
        --test-exe C:\\dev\\build_ninja_lto\\bin\\CoreCxxTests.exe \
        --sources c:\\dev\\XSigma\\Library \
        --output coverage_data
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


class CoverageDataCollector:
    """Collects coverage data using OpenCppCoverage.exe."""

    def __init__(
        self,
        test_exe: str,
        sources: str,
        output_dir: str = "coverage_data",
        verbose: bool = False,
        excluded_sources: Optional[list] = None,
    ):
        """Initialize the coverage data collector.

        Args:
            test_exe: Path to the test executable
            sources: Path to source files to collect coverage for
            output_dir: Directory to store coverage data
            verbose: Enable verbose output
            excluded_sources: List of patterns to exclude from coverage (e.g., ['*Testing*'])
        """
        self.test_exe = Path(test_exe)
        self.sources = Path(sources)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.excluded_sources = excluded_sources or []

        # Validate inputs
        if not self.test_exe.exists():
            raise FileNotFoundError(f"Test executable not found: {test_exe}")
        if not self.sources.exists():
            raise FileNotFoundError(f"Sources directory not found: {sources}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        """Print log message if verbose is enabled."""
        if self.verbose:
            print(f"[CoverageCollector] {message}")

    def _find_opencppcoverage(self) -> Optional[str]:
        """Find OpenCppCoverage.exe in system PATH or common locations."""
        # Check if it's in PATH
        try:
            result = subprocess.run(
                ["where", "OpenCppCoverage.exe"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass

        # Check common installation locations
        common_paths = [
            "C:\\Program Files\\OpenCppCoverage\\OpenCppCoverage.exe",
            "C:\\Program Files (x86)\\OpenCppCoverage\\OpenCppCoverage.exe",
        ]

        for path in common_paths:
            if Path(path).exists():
                return path

        return None

    def collect(self) -> bool:
        """Collect coverage data by running the test executable.

        Returns:
            True if collection succeeded, False otherwise
        """
        self._log(f"Starting coverage collection for {self.test_exe}")

        # Find OpenCppCoverage
        opencppcoverage = self._find_opencppcoverage()
        if not opencppcoverage:
            print(
                "ERROR: OpenCppCoverage.exe not found. "
                "Please install it or add it to PATH.",
                file=sys.stderr,
            )
            return False

        self._log(f"Found OpenCppCoverage at: {opencppcoverage}")

        # Build command
        # Note: OpenCppCoverage 0.9.9.0 uses --export_type with colon-separated format
        # Format: --export_type cobertura:output_file
        # Use absolute path to avoid nesting issues
        cobertura_file = (self.output_dir / "coverage.xml").resolve()
        cmd = [
            opencppcoverage,
            "--sources",
            str(self.sources),
        ]

        # Add excluded sources if specified
        for excluded in self.excluded_sources:
            cmd.extend(["--excluded_sources", excluded])

        cmd.extend([
            "--export_type",
            f"cobertura:{cobertura_file}",
            "--",
            str(self.test_exe),
        ])

        self._log(f"Running command: {' '.join(cmd)}")

        try:
            # Run from current directory, not output_dir, to avoid path nesting
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(
                    f"ERROR: OpenCppCoverage failed with return code {result.returncode}",
                    file=sys.stderr,
                )
                return False

            self._log("Coverage collection completed successfully")
            return True

        except Exception as e:
            print(f"ERROR: Failed to run OpenCppCoverage: {e}", file=sys.stderr)
            return False

    def get_coverage_files(self) -> dict:
        """Get information about generated coverage files.

        Returns:
            Dictionary with coverage file information
        """
        files = {
            "coverage_file": str(self.output_dir / "coverage.cov"),
            "cobertura_file": str(self.output_dir / "coverage.xml"),
            "output_dir": str(self.output_dir),
        }

        # Check which files exist
        existing_files = {}
        for key, path in files.items():
            if Path(path).exists():
                existing_files[key] = path

        return existing_files


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect code coverage data using OpenCppCoverage.exe"
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
        default="coverage_data",
        help="Output directory for coverage data (default: coverage_data)",
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
        collector = CoverageDataCollector(
            test_exe=args.test_exe,
            sources=args.sources,
            output_dir=args.output,
            verbose=args.verbose,
            excluded_sources=args.excluded_sources,
        )

        if not collector.collect():
            return 1

        # Print coverage files information
        files = collector.get_coverage_files()
        print("\nCoverage data collection completed!")
        print("Generated files:")
        for key, path in files.items():
            print(f"  {key}: {path}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
