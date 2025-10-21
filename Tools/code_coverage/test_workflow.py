#!/usr/bin/env python3
"""
Test script for the XSigma code coverage workflow.

This script validates that the coverage workflow components work correctly
by testing individual modules and the complete workflow.

Usage:
    python test_workflow.py [--verbose]
"""

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set


def test_html_report_generator() -> bool:
    """Test the HTML report generator module.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing HtmlReportGenerator...")

    try:
        from package.tool.html_report_generator import HtmlReportGenerator

        # Create test data
        covered_lines: Dict[str, Set[int]] = {
            "Library/Core/test.h": {1, 2, 3, 4, 5},
            "Library/Core/test.cpp": {10, 11, 12, 13, 14, 15},
        }
        uncovered_lines: Dict[str, Set[int]] = {
            "Library/Core/test.h": {6, 7, 8},
            "Library/Core/test.cpp": {16, 17, 18},
        }

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HtmlReportGenerator(tmpdir)
            generator.generate_report(covered_lines, uncovered_lines)

            # Verify index.html was created
            index_file = Path(tmpdir) / "index.html"
            if not index_file.exists():
                print("  ✗ index.html not created")
                return False

            # Verify content
            content = index_file.read_text()
            if "Code Coverage Report" not in content:
                print("  ✗ index.html missing expected content")
                return False

            print("  ✓ HtmlReportGenerator works correctly")
            return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_coverage_data_parser() -> bool:
    """Test the coverage data parser.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing CoverageDataParser...")

    try:
        # Import the parser from generate_html_report
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_html_report import CoverageDataParser

        # Create test Cobertura XML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            xml_content = """<?xml version="1.0" ?>
<coverage version="1.0">
    <package name="Library">
        <class name="test" filename="Library/Core/test.cpp">
            <line number="1" hits="1"/>
            <line number="2" hits="1"/>
            <line number="3" hits="0"/>
        </class>
    </package>
</coverage>"""
            f.write(xml_content)
            xml_file = f.name

        try:
            covered, uncovered = CoverageDataParser.parse_cobertura_xml(xml_file)

            if "Library/Core/test.cpp" not in covered:
                print("  ✗ Parser didn't extract covered lines")
                return False

            if len(covered["Library/Core/test.cpp"]) != 2:
                print("  ✗ Parser didn't correctly count covered lines")
                return False

            if len(uncovered["Library/Core/test.cpp"]) != 1:
                print("  ✗ Parser didn't correctly count uncovered lines")
                return False

            print("  ✓ CoverageDataParser works correctly")
            return True

        finally:
            Path(xml_file).unlink()

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_collect_coverage_data_module() -> bool:
    """Test the collect_coverage_data module imports.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing collect_coverage_data module...")

    try:
        from collect_coverage_data import CoverageDataCollector

        # Test that class can be instantiated (without running actual coverage)
        # We'll just verify the class exists and has expected methods
        if not hasattr(CoverageDataCollector, "collect"):
            print("  ✗ CoverageDataCollector missing collect method")
            return False

        if not hasattr(CoverageDataCollector, "get_coverage_files"):
            print("  ✗ CoverageDataCollector missing get_coverage_files method")
            return False

        print("  ✓ collect_coverage_data module is valid")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_workflow_module() -> bool:
    """Test the workflow module imports.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing run_coverage_workflow module...")

    try:
        from run_coverage_workflow import CoverageWorkflow

        # Test that class can be instantiated
        if not hasattr(CoverageWorkflow, "run"):
            print("  ✗ CoverageWorkflow missing run method")
            return False

        if not hasattr(CoverageWorkflow, "collect_coverage"):
            print("  ✗ CoverageWorkflow missing collect_coverage method")
            return False

        if not hasattr(CoverageWorkflow, "generate_html_report"):
            print("  ✗ CoverageWorkflow missing generate_html_report method")
            return False

        print("  ✓ run_coverage_workflow module is valid")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_file_structure() -> bool:
    """Test that all required files exist.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing file structure...")

    required_files = [
        "collect_coverage_data.py",
        "generate_html_report.py",
        "run_coverage_workflow.py",
        "WORKFLOW.md",
        "package/tool/html_report_generator.py",
    ]

    script_dir = Path(__file__).parent
    all_exist = True

    for file_path in required_files:
        full_path = script_dir / file_path
        if not full_path.exists():
            print(f"  ✗ Missing: {file_path}")
            all_exist = False

    if all_exist:
        print("  ✓ All required files exist")
        return True
    else:
        return False


def main() -> int:
    """Run all tests.

    Returns:
        0 if all tests passed, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Test the coverage workflow")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    print("=" * 70)
    print("XSigma Code Coverage Workflow - Test Suite")
    print("=" * 70)
    print()

    tests = [
        ("File Structure", test_file_structure),
        ("HtmlReportGenerator", test_html_report_generator),
        ("CoverageDataParser", test_coverage_data_parser),
        ("collect_coverage_data Module", test_collect_coverage_data_module),
        ("run_coverage_workflow Module", test_workflow_module),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append((test_name, False))
        print()

    # Print summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print()

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

