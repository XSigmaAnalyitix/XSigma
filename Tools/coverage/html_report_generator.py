#!/usr/bin/env python3
"""Custom HTML coverage report generator.

Generates professional HTML coverage reports with line-by-line coverage display,
execution counts, and JSON summaries for CI/CD integration.
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, Optional


class HtmlReportGenerator:
    """Generates custom HTML coverage reports."""

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
                       execution_counts: Dict = None) -> None:
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

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Code Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .coverage-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 3px; overflow: hidden; margin-top: 5px; }}
        .coverage-fill {{ height: 100%; background-color: #28a745; width: {coverage_percent}%; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background-color: #007bff; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        a {{ color: #007bff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Code Coverage Report</h1>
        
        <div class="summary">
            <div class="metric">
                <div class="metric-label">Overall Coverage</div>
                <div class="metric-value">{coverage_percent:.1f}%</div>
                <div class="coverage-bar">
                    <div class="coverage-fill"></div>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Lines Covered</div>
                <div class="metric-value">{total_covered}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Lines Uncovered</div>
                <div class="metric-value">{total_uncovered}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Lines</div>
                <div class="metric-value">{total_lines}</div>
            </div>
        </div>

        <h2>Files</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Coverage</th>
                <th>Covered</th>
                <th>Uncovered</th>
                <th>Total</th>
            </tr>
"""

        for file_path in sorted(set(list(covered_lines.keys()) + list(uncovered_lines.keys()))):
            covered = len(covered_lines.get(file_path, set()))
            uncovered = len(uncovered_lines.get(file_path, set()))
            total = covered + uncovered

            if total > 0:
                file_coverage = (covered / total) * 100
            else:
                file_coverage = 0.0

            file_name = Path(file_path).name
            html_file = f"{file_name}.html"

            html_content += f"""            <tr>
                <td><a href="{html_file}">{file_path}</a></td>
                <td>{file_coverage:.1f}%</td>
                <td>{covered}</td>
                <td>{uncovered}</td>
                <td>{total}</td>
            </tr>
"""

        html_content += """        </table>
    </div>
</body>
</html>
"""

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
            coverage_percent = (covered_count / (covered_count + uncovered_count)) * 100 if (covered_count + uncovered_count) > 0 else 0
        else:
            coverage_percent = 0.0

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Coverage: {file_path}</title>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 0; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; }}
        .header {{ background-color: #007bff; color: white; padding: 20px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 5px 0 0 0; font-size: 14px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 20px; background-color: #f8f9fa; }}
        .stat {{ padding: 10px; background-color: white; border-radius: 5px; border-left: 4px solid #007bff; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .stat-value {{ font-size: 18px; font-weight: bold; }}
        .code {{ padding: 20px; }}
        .line {{ display: flex; border-bottom: 1px solid #eee; }}
        .line:hover {{ background-color: #f9f9f9; }}
        .line-number {{ width: 50px; text-align: right; padding-right: 10px; color: #999; background-color: #f5f5f5; user-select: none; }}
        .coverage-status {{ width: 30px; text-align: center; padding: 0 10px; font-weight: bold; user-select: none; }}
        .covered {{ background-color: #d4edda; color: #155724; }}
        .uncovered {{ background-color: #f8d7da; color: #721c24; }}
        .neutral {{ background-color: #e2e3e5; color: #383d41; }}
        .line-content {{ flex: 1; padding: 0 10px; white-space: pre-wrap; word-wrap: break-word; }}
        .back-link {{ padding: 20px; }}
        .back-link a {{ color: #007bff; text-decoration: none; }}
        .back-link a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Coverage Report: {file_path}</h1>
            <p>Line-by-line coverage analysis</p>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Coverage</div>
                <div class="stat-value">{coverage_percent:.1f}%</div>
            </div>
            <div class="stat">
                <div class="stat-label">Covered Lines</div>
                <div class="stat-value">{covered_count}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Uncovered Lines</div>
                <div class="stat-value">{uncovered_count}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Total Lines</div>
                <div class="stat-value">{total_lines}</div>
            </div>
        </div>

        <div class="code">
"""

        for line_num, line_content in enumerate(lines, 1):
            if line_num in covered:
                status_class = "covered"
                status_text = "✓"
            elif line_num in uncovered:
                status_class = "uncovered"
                status_text = "✗"
            else:
                status_class = "neutral"
                status_text = "-"

            hit_count = execution_counts.get(line_num, 0)
            hit_text = f" ({hit_count}x)" if hit_count > 0 else ""

            line_content = line_content.rstrip('\n')
            line_content = line_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            html_content += f"""            <div class="line">
                <div class="line-number">{line_num}</div>
                <div class="coverage-status {status_class}">{status_text}{hit_text}</div>
                <div class="line-content">{line_content}</div>
            </div>
"""

        html_content += """        </div>
        
        <div class="back-link">
            <a href="index.html">← Back to Summary</a>
        </div>
    </div>
</body>
</html>
"""

        file_name = Path(file_path).name
        output_file = self.coverage_dir / f"{file_name}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def generate_json_summary(self, covered_lines: Dict[str, Set[int]],
                             uncovered_lines: Dict[str, Set[int]]) -> None:
        """Generate JSON coverage summary.

        Args:
            covered_lines: Dictionary mapping file paths to sets of covered line numbers.
            uncovered_lines: Dictionary mapping file paths to sets of uncovered line numbers.
        """
        from coverage_summary import CoverageSummaryGenerator

        generator = CoverageSummaryGenerator()
        summary = generator.generate_summary(covered_lines, uncovered_lines)

        output_file = self.coverage_dir / "coverage_summary.json"
        generator.save_to_file(output_file)

