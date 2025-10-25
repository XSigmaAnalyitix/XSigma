"""HTML coverage report generator for direct coverage data.

This module provides functionality to generate professional HTML coverage reports
from direct coverage data (covered/uncovered lines and execution counts).
"""

from pathlib import Path
from typing import Dict, Set, Optional

from .templates import get_index_html_template, get_file_html_template, get_line_html


class HtmlGenerator:
    """Generates HTML coverage reports from direct coverage data.

    This class creates professional HTML reports with line-by-line coverage
    display, execution counts, and summary statistics.
    """

    def __init__(self, coverage_dir: Path, source_root: str = ""):
        """Initialize the HTML report generator.

        Args:
            coverage_dir: Directory where HTML reports will be generated.
            source_root: Root directory for source files (for relative paths).
        """
        self.coverage_dir = Path(coverage_dir)
        self.source_root = source_root
        self.coverage_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, covered_lines: Dict[str, Set[int]],
                       uncovered_lines: Dict[str, Set[int]],
                       execution_counts: Optional[Dict] = None) -> None:
        """Generate HTML coverage report.

        Args:
            covered_lines: Dictionary mapping file paths to sets of covered line numbers.
            uncovered_lines: Dictionary mapping file paths to sets of uncovered line numbers.
            execution_counts: Optional dictionary with execution counts per line.
        """
        if execution_counts is None:
            execution_counts = {}

        # Generate index page
        self._generate_index(covered_lines, uncovered_lines)

        # Generate per-file pages
        for file_path in set(list(covered_lines.keys()) + list(uncovered_lines.keys())):
            covered = covered_lines.get(file_path, set())
            uncovered = uncovered_lines.get(file_path, set())
            file_counts = execution_counts.get(file_path, {})

            self._generate_file_report(file_path, covered, uncovered, file_counts)

    def _generate_index(self, covered_lines: Dict[str, Set[int]],
                       uncovered_lines: Dict[str, Set[int]]) -> None:
        """Generate the index/summary page.

        Args:
            covered_lines: Dictionary mapping file paths to sets of covered line numbers.
            uncovered_lines: Dictionary mapping file paths to sets of uncovered line numbers.
        """
        total_covered = sum(len(lines) for lines in covered_lines.values())
        total_uncovered = sum(len(lines) for lines in uncovered_lines.values())
        total_lines = total_covered + total_uncovered

        if total_lines > 0:
            coverage_percent = (total_covered / total_lines) * 100
        else:
            coverage_percent = 0.0

        # Build file list with coverage percentages for sorting
        file_list = []
        for file_path in set(list(covered_lines.keys()) + list(uncovered_lines.keys())):
            covered = len(covered_lines.get(file_path, set()))
            uncovered = len(uncovered_lines.get(file_path, set()))
            total = covered + uncovered

            if total > 0:
                file_coverage = (covered / total) * 100
            else:
                file_coverage = 0.0

            file_list.append((file_path, covered, uncovered, total, file_coverage))

        # Sort by coverage percentage in descending order (highest first)
        file_list.sort(key=lambda x: x[4], reverse=True)

        # Build file table rows
        files_table_rows = ""
        for file_path, covered, uncovered, total, file_coverage in file_list:
            file_name = Path(file_path).name
            html_file = f"{file_name}.html"

            files_table_rows += f"""            <tr>
                <td><a href="{html_file}">{file_path}</a></td>
                <td>{file_coverage:.1f}%</td>
                <td>{covered}</td>
                <td>{uncovered}</td>
                <td>{total}</td>
            </tr>
"""

        html_content = get_index_html_template(
            coverage_percent, total_covered, total_uncovered, total_lines,
            files_table_rows
        )

        index_path = self.coverage_dir / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_file_report(self, file_path: str, covered: Set[int],
                             uncovered: Set[int], execution_counts: Dict) -> None:
        """Generate a per-file coverage report.

        Args:
            file_path: Path to the source file.
            covered: Set of covered line numbers.
            uncovered: Set of uncovered line numbers.
            execution_counts: Dictionary with execution counts per line.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Warning: Source file not found: {file_path}")
            return

        total_lines = len(lines)
        covered_count = len(covered)
        uncovered_count = len(uncovered)

        if total_lines > 0:
            if (covered_count + uncovered_count) > 0:
                coverage_percent = (covered_count /
                                   (covered_count + uncovered_count)) * 100
            else:
                coverage_percent = 0
        else:
            coverage_percent = 0.0

        # Build lines HTML
        lines_html = ""
        for line_num, line_content in enumerate(lines, 1):
            if line_num in covered:
                status_class = "covered"
                status_text = "✓"
                line_class = "covered-line"
            elif line_num in uncovered:
                status_class = "uncovered"
                status_text = "✗"
                line_class = "uncovered-line"
            else:
                status_class = "neutral"
                status_text = "-"
                line_class = ""

            hit_count = execution_counts.get(line_num, 0)
            line_content = line_content.rstrip('\n')
            line_content = (line_content.replace('&', '&amp;')
                           .replace('<', '&lt;').replace('>', '&gt;'))

            lines_html += get_line_html(line_num, status_class, status_text,
                                       line_content, hit_count, line_class)

        html_content = get_file_html_template(
            file_path, coverage_percent, covered_count, uncovered_count,
            total_lines, lines_html
        )

        file_name = Path(file_path).name
        output_file = self.coverage_dir / f"{file_name}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)



