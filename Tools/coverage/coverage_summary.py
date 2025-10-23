#!/usr/bin/env python3
"""JSON coverage summary generation for CI/CD integration.

Generates structured JSON files with per-file and global coverage metrics.
"""

import json
from pathlib import Path
from typing import Dict, Set


class CoverageSummaryGenerator:
    """Generates JSON coverage summaries from coverage data."""

    def __init__(self):
        """Initialize the coverage summary generator."""
        self.summary = {
            "metadata": {
                "format_version": "1.0",
                "generator": "xsigma_coverage_tool"
            },
            "global_metrics": {
                "total_files": 0,
                "files_with_coverage": 0,
                "total_lines": 0,
                "covered_lines": 0,
                "uncovered_lines": 0,
                "line_coverage_percent": 0.0,
                "total_functions": 0,
                "covered_functions": 0,
                "function_coverage_percent": 0.0,
            },
            "files": {}
        }

    def generate_summary(self, covered_lines: Dict[str, Set[int]],
                        uncovered_lines: Dict[str, Set[int]],
                        execution_counts: Dict = None) -> dict:
        """Generate coverage summary from coverage data.

        Args:
            covered_lines: Dictionary mapping file paths to sets of covered line numbers.
            uncovered_lines: Dictionary mapping file paths to sets of uncovered line numbers.
            execution_counts: Optional dictionary with execution counts per line.

        Returns:
            Dictionary containing the coverage summary.
        """
        if execution_counts is None:
            execution_counts = {}

        total_lines = 0
        total_covered = 0
        total_uncovered = 0

        # Process each file
        for file_path in set(list(covered_lines.keys()) + list(uncovered_lines.keys())):
            covered = covered_lines.get(file_path, set())
            uncovered = uncovered_lines.get(file_path, set())

            file_total = len(covered) + len(uncovered)
            file_covered = len(covered)
            file_uncovered = len(uncovered)

            if file_total > 0:
                coverage_percent = (file_covered / file_total) * 100
            else:
                coverage_percent = 0.0

            self.summary["files"][file_path] = {
                "total_lines": file_total,
                "covered_lines": file_covered,
                "uncovered_lines": file_uncovered,
                "line_coverage_percent": round(coverage_percent, 2)
            }

            total_lines += file_total
            total_covered += file_covered
            total_uncovered += file_uncovered

        # Calculate global metrics
        self.summary["global_metrics"]["total_files"] = len(self.summary["files"])
        self.summary["global_metrics"]["files_with_coverage"] = len(
            [f for f in self.summary["files"].values() if f["total_lines"] > 0]
        )
        self.summary["global_metrics"]["total_lines"] = total_lines
        self.summary["global_metrics"]["covered_lines"] = total_covered
        self.summary["global_metrics"]["uncovered_lines"] = total_uncovered

        if total_lines > 0:
            global_coverage = (total_covered / total_lines) * 100
        else:
            global_coverage = 0.0

        self.summary["global_metrics"]["line_coverage_percent"] = round(global_coverage, 2)

        return self.summary

    def save_to_file(self, output_path: Path) -> None:
        """Save the coverage summary to a JSON file.

        Args:
            output_path: Path where the JSON file should be saved.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2)

        print(f"Coverage summary saved to: {output_path}")

