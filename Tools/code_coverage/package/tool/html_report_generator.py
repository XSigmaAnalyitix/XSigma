"""
Multi-file HTML Report Generator for Code Coverage

Generates a rich, multi-file HTML report structure with:
- Index/Summary page with overall coverage statistics
- Individual file reports with line-by-line coverage visualization
- Syntax highlighting for readability
- Navigation between files
"""

import os
import html
from typing import Dict, Set, Tuple
from datetime import datetime


class HtmlReportGenerator:
    """Generate multi-file HTML coverage reports."""

    def __init__(self, output_dir: str):
        """Initialize the HTML report generator.
        
        Args:
            output_dir: Directory where HTML reports will be generated
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        covered_lines: Dict[str, Set[int]],
        uncovered_lines: Dict[str, Set[int]],
        source_root: str = "",
    ) -> None:
        """Generate complete multi-file HTML report.
        
        Args:
            covered_lines: Dict mapping file paths to sets of covered line numbers
            uncovered_lines: Dict mapping file paths to sets of uncovered line numbers
            source_root: Root directory of source files for reading content
        """
        # Calculate statistics
        stats = self._calculate_statistics(covered_lines, uncovered_lines)
        
        # Generate index page
        self._generate_index_page(stats, covered_lines, uncovered_lines)
        
        # Generate individual file reports
        self._generate_file_reports(
            covered_lines, uncovered_lines, source_root
        )

    def _calculate_statistics(
        self,
        covered_lines: Dict[str, Set[int]],
        uncovered_lines: Dict[str, Set[int]],
    ) -> Dict:
        """Calculate overall coverage statistics."""
        total_covered = 0
        total_uncovered = 0
        file_stats = []

        for file_path in covered_lines:
            covered = len(covered_lines[file_path])
            uncovered = len(uncovered_lines.get(file_path, set()))
            total = covered + uncovered

            if total > 0:
                percentage = (covered / total) * 100
            else:
                percentage = 0

            file_stats.append({
                'path': file_path,
                'covered': covered,
                'uncovered': uncovered,
                'total': total,
                'percentage': percentage,
            })

            total_covered += covered
            total_uncovered += uncovered

        total_lines = total_covered + total_uncovered
        if total_lines > 0:
            overall_percentage = (total_covered / total_lines) * 100
        else:
            overall_percentage = 0

        # Sort by percentage (ascending) to show worst coverage first
        file_stats.sort(key=lambda x: x['percentage'])

        return {
            'total_covered': total_covered,
            'total_uncovered': total_uncovered,
            'total_lines': total_lines,
            'overall_percentage': overall_percentage,
            'file_stats': file_stats,
            'num_files': len(file_stats),
        }

    def _generate_index_page(
        self,
        stats: Dict,
        covered_lines: Dict[str, Set[int]],
        uncovered_lines: Dict[str, Set[int]],
    ) -> None:
        """Generate the index/summary page."""
        html_content = self._get_index_html(stats)
        
        index_path = os.path.join(self.output_dir, 'index.html')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_file_reports(
        self,
        covered_lines: Dict[str, Set[int]],
        uncovered_lines: Dict[str, Set[int]],
        source_root: str,
    ) -> None:
        """Generate individual file reports."""
        for file_path in covered_lines:
            covered = covered_lines[file_path]
            uncovered = uncovered_lines.get(file_path, set())
            
            # Generate HTML for this file
            html_content = self._get_file_html(
                file_path, covered, uncovered, source_root
            )
            
            # Create safe filename
            safe_filename = self._get_safe_filename(file_path)
            file_report_path = os.path.join(self.output_dir, safe_filename)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(file_report_path), exist_ok=True)
            
            with open(file_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

    def _get_safe_filename(self, file_path: str) -> str:
        """Convert file path to safe HTML filename."""
        # Replace path separators with underscores and add .html
        safe_name = file_path.replace('\\', '_').replace('/', '_')
        return safe_name + '.html'

    def _get_index_html(self, stats: Dict) -> str:
        """Generate HTML for index page."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build file table rows
        file_rows = []
        for file_stat in stats['file_stats']:
            safe_filename = self._get_safe_filename(file_stat['path'])
            coverage_class = self._get_coverage_class(file_stat['percentage'])
            
            file_rows.append(f"""
            <tr>
                <td><a href="{safe_filename}">{html.escape(file_stat['path'])}</a></td>
                <td class="number">{file_stat['covered']}</td>
                <td class="number">{file_stat['uncovered']}</td>
                <td class="number">{file_stat['total']}</td>
                <td class="{coverage_class}">{file_stat['percentage']:.2f}%</td>
            </tr>
            """)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Code Coverage Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f9f9f9;
        }}
        .number {{
            text-align: right;
            font-family: monospace;
        }}
        .coverage-excellent {{
            color: #27ae60;
            font-weight: bold;
        }}
        .coverage-good {{
            color: #f39c12;
            font-weight: bold;
        }}
        .coverage-poor {{
            color: #e74c3c;
            font-weight: bold;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Coverage Report</h1>
        <p>Generated on {timestamp}</p>
    </div>

    <div class="summary">
        <div class="stat-box">
            <div class="stat-label">Overall Coverage</div>
            <div class="stat-value">{stats['overall_percentage']:.2f}%</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Lines Covered</div>
            <div class="stat-value">{stats['total_covered']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Lines Uncovered</div>
            <div class="stat-value">{stats['total_uncovered']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Total Lines</div>
            <div class="stat-value">{stats['total_lines']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Files Analyzed</div>
            <div class="stat-value">{stats['num_files']}</div>
        </div>
    </div>

    <h2>File Coverage Summary</h2>
    <table>
        <thead>
            <tr>
                <th>File</th>
                <th>Covered</th>
                <th>Uncovered</th>
                <th>Total</th>
                <th>Coverage</th>
            </tr>
        </thead>
        <tbody>
            {''.join(file_rows)}
        </tbody>
    </table>

    <div class="timestamp">
        <p>Report generated at {timestamp}</p>
    </div>
</body>
</html>
"""

    def _resolve_source_file(self, file_path: str, source_root: str) -> str:
        """Resolve the actual source file path.

        Handles paths like 'dev\\XSigma\\Library\\...' by finding the actual file.

        Args:
            file_path: Path from coverage data
            source_root: Root directory to search from

        Returns:
            Absolute path to source file, or empty string if not found
        """
        from pathlib import Path

        # Try direct path first
        if os.path.exists(file_path):
            return file_path

        # Try joining with source_root
        if source_root:
            full_path = os.path.join(source_root, file_path)
            if os.path.exists(full_path):
                return full_path

            # Try extracting just the filename and searching from source_root
            # For paths like 'dev\\XSigma\\Library\\Core\\...', extract 'Library\\Core\\...'
            parts = Path(file_path).parts
            if 'Library' in parts:
                lib_index = parts.index('Library')
                relative_path = os.path.join(*parts[lib_index:])
                full_path = os.path.join(source_root, relative_path)
                if os.path.exists(full_path):
                    return full_path

        return ""

    def _get_file_html(
        self,
        file_path: str,
        covered: Set[int],
        uncovered: Set[int],
        source_root: str,
    ) -> str:
        """Generate HTML for individual file report."""
        # Try to read source file
        source_lines = []
        if source_root:
            full_path = self._resolve_source_file(file_path, source_root)
            if full_path:
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        source_lines = f.readlines()
                except Exception:
                    pass

        # Generate line-by-line coverage
        line_rows = []
        for line_num in range(1, len(source_lines) + 1):
            line_content = source_lines[line_num - 1].rstrip('\n')
            
            if line_num in covered:
                coverage_class = 'covered'
                coverage_text = '✓'
            elif line_num in uncovered:
                coverage_class = 'uncovered'
                coverage_text = '✗'
            else:
                coverage_class = 'neutral'
                coverage_text = ' '

            line_rows.append(f"""
            <tr class="{coverage_class}">
                <td class="line-num">{line_num}</td>
                <td class="coverage-marker">{coverage_text}</td>
                <td class="line-content"><code>{html.escape(line_content)}</code></td>
            </tr>
            """)

        covered_count = len(covered)
        uncovered_count = len(uncovered)
        total_count = covered_count + uncovered_count
        percentage = (covered_count / total_count * 100) if total_count > 0 else 0

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Coverage: {html.escape(file_path)}</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            margin: 0;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 15px 20px;
            margin-bottom: 20px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin: 10px 0;
            font-size: 14px;
        }}
        .nav {{
            margin: 20px;
        }}
        .nav a {{
            color: #3498db;
            text-decoration: none;
            margin-right: 15px;
        }}
        .nav a:hover {{
            text-decoration: underline;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            margin: 20px 0;
        }}
        td {{
            padding: 2px 8px;
            border-right: 1px solid #ecf0f1;
        }}
        .line-num {{
            width: 50px;
            text-align: right;
            color: #7f8c8d;
            background-color: #f9f9f9;
            font-weight: bold;
        }}
        .coverage-marker {{
            width: 30px;
            text-align: center;
            font-weight: bold;
        }}
        .line-content {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        tr.covered {{
            background-color: #d4edda;
        }}
        tr.covered .coverage-marker {{
            color: #28a745;
        }}
        tr.uncovered {{
            background-color: #f8d7da;
        }}
        tr.uncovered .coverage-marker {{
            color: #dc3545;
        }}
        tr.neutral {{
            background-color: #fff;
        }}
        code {{
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{html.escape(file_path)}</h2>
        <div class="stats">
            <span>Covered: {covered_count}</span>
            <span>Uncovered: {uncovered_count}</span>
            <span>Total: {total_count}</span>
            <span>Coverage: {percentage:.2f}%</span>
        </div>
    </div>

    <div class="nav">
        <a href="index.html">← Back to Summary</a>
    </div>

    <table>
        <tbody>
            {''.join(line_rows) if line_rows else '<tr><td colspan="3">Source file not available</td></tr>'}
        </tbody>
    </table>
</body>
</html>
"""

    def _get_coverage_class(self, percentage: float) -> str:
        """Get CSS class based on coverage percentage."""
        if percentage >= 80:
            return 'coverage-excellent'
        elif percentage >= 60:
            return 'coverage-good'
        else:
            return 'coverage-poor'

