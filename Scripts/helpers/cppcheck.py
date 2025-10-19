"""
Cppcheck Static Analysis Helper Module

This module handles static code analysis using cppcheck.
Extracted from setup.py for better modularity and maintainability.
"""

import os
import subprocess
import sys
from typing import List
from pathlib import Path


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


def build_cppcheck_command(source_path: str, output_file: str) -> List[str]:
    """Build the cppcheck command with appropriate settings."""
    cpu_count = get_logical_processor_count()
    parallel_jobs = min(cpu_count, 8)

    cmd = [
        "cppcheck",
        ".",
        "--platform=unspecified",
        "--enable=style,performance,portability,information",
        "--inline-suppr",
        "-q",
        "--library=qt",
        "--library=posix",
        "--library=gnu",
        "--library=bsd",
        "--library=windows",
        "--check-level=exhaustive",
        "--template={id},{file}:{line},{severity},{message}",
        "--suppress=missingInclude",
        f"-j{parallel_jobs}",
        "-I", "Library",
        f"--output-file={output_file}"
    ]

    # Add suppressions file if it exists
    suppressions_file = os.path.join(source_path, "Scripts", "suppressions", "cppcheck_suppressions.txt")
    if os.path.exists(suppressions_file):
        cmd.append(f"--suppressions-list={suppressions_file}")

    return cmd


def process_cppcheck_results(result: subprocess.CompletedProcess, output_file: str) -> int:
    """Process cppcheck results and return exit code."""
    if not os.path.exists(output_file):
        return 1

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            return 0

        # Count issues by type
        lines = content.split('\n')
        issue_counts = {}
        for line in lines:
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    severity = parts[2]
                    issue_counts[severity] = issue_counts.get(severity, 0) + 1

        # Return appropriate exit code
        if any(severity in ['error'] for severity in issue_counts.keys()):
            return 1
        else:
            return 0

    except Exception:
        return 1

