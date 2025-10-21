#!/usr/bin/env python3
"""
Integration test for the XSigma code coverage workflow.

This test simulates the complete workflow with mock data to verify
that all components work together correctly.

Usage:
    python test_integration.py [--verbose]
"""

import argparse
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def create_mock_coverage_xml(output_file: Path) -> None:
    """Create a mock Cobertura XML coverage file.

    Args:
        output_file: Path to write the XML file
    """
    root = ET.Element("coverage", version="1.0")

    # Create a package
    package = ET.SubElement(root, "package", name="Library")

    # Create a class with coverage data
    cls = ET.SubElement(
        package,
        "class",
        name="TestClass",
        filename="Library/Core/test.cpp",
    )

    # Add covered lines
    for line_num in [1, 2, 3, 4, 5, 10, 11, 12]:
        ET.SubElement(cls, "line", number=str(line_num), hits="1")

    # Add uncovered lines
    for line_num in [6, 7, 8, 9]:
        ET.SubElement(cls, "line", number=str(line_num), hits="0")

    # Write XML
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def test_html_generation_from_xml() -> bool:
    """Test HTML report generation from mock XML data.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing HTML generation from mock XML data...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create mock XML file
            xml_file = tmpdir_path / "coverage.xml"
            create_mock_coverage_xml(xml_file)

            # Create output directory
            html_dir = tmpdir_path / "html_output"
            html_dir.mkdir()

            # Run generate_html_report.py
            script_dir = Path(__file__).parent
            cmd = [
                sys.executable,
                str(script_dir / "generate_html_report.py"),
                "--coverage-data",
                str(xml_file),
                "--output",
                str(html_dir),
                "--source-root",
                str(tmpdir_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  ✗ Command failed: {result.stderr}")
                return False

            # Verify index.html was created
            index_file = html_dir / "index.html"
            if not index_file.exists():
                print("  ✗ index.html not created")
                return False

            # Verify content
            content = index_file.read_text()
            if "Code Coverage Report" not in content:
                print("  ✗ index.html missing expected content")
                return False

            if "Overall Coverage" not in content:
                print("  ✗ index.html missing coverage statistics")
                return False

            print("  ✓ HTML generation from XML works correctly")
            return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_workflow_help() -> bool:
    """Test that workflow scripts have proper help.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing workflow script help...")

    try:
        script_dir = Path(__file__).parent
        scripts = [
            "run_coverage_workflow.py",
            "collect_coverage_data.py",
            "generate_html_report.py",
        ]

        for script in scripts:
            cmd = [sys.executable, str(script_dir / script), "--help"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  ✗ {script} help failed")
                return False

            if "usage:" not in result.stdout.lower():
                print(f"  ✗ {script} help output invalid")
                return False

        print("  ✓ All workflow scripts have proper help")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_workflow_documentation() -> bool:
    """Test that workflow documentation exists and is valid.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing workflow documentation...")

    try:
        script_dir = Path(__file__).parent
        workflow_doc = script_dir / "WORKFLOW.md"

        if not workflow_doc.exists():
            print("  ✗ WORKFLOW.md not found")
            return False

        # Read with UTF-8 encoding and error handling
        content = workflow_doc.read_text(encoding="utf-8", errors="ignore")

        required_sections = [
            "Overview",
            "Architecture",
            "Quick Start",
            "Usage Examples",
            "Requirements",
            "Output",
            "Troubleshooting",
        ]

        for section in required_sections:
            if section not in content:
                print(f"  ✗ WORKFLOW.md missing section: {section}")
                return False

        print("  ✓ Workflow documentation is complete")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_modular_architecture() -> bool:
    """Test that the architecture is properly modular.

    Returns:
        True if test passed, False otherwise
    """
    print("Testing modular architecture...")

    try:
        script_dir = Path(__file__).parent

        # Verify collect_coverage_data can be imported independently
        sys.path.insert(0, str(script_dir))
        from collect_coverage_data import CoverageDataCollector

        # Verify generate_html_report can be imported independently
        from generate_html_report import HtmlReportGenerator_Wrapper

        # Verify run_coverage_workflow can be imported independently
        from run_coverage_workflow import CoverageWorkflow

        # Verify each has expected methods
        if not hasattr(CoverageDataCollector, "collect"):
            print("  ✗ CoverageDataCollector missing collect method")
            return False

        if not hasattr(HtmlReportGenerator_Wrapper, "generate"):
            print("  ✗ HtmlReportGenerator_Wrapper missing generate method")
            return False

        if not hasattr(CoverageWorkflow, "run"):
            print("  ✗ CoverageWorkflow missing run method")
            return False

        print("  ✓ Modular architecture is properly implemented")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main() -> int:
    """Run all integration tests.

    Returns:
        0 if all tests passed, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Integration test for coverage workflow")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    print("=" * 70)
    print("XSigma Code Coverage Workflow - Integration Tests")
    print("=" * 70)
    print()

    tests = [
        ("Workflow Documentation", test_workflow_documentation),
        ("Workflow Script Help", test_workflow_help),
        ("Modular Architecture", test_modular_architecture),
        ("HTML Generation from XML", test_html_generation_from_xml),
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
    print("Integration Test Summary")
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
        print("✓ All integration tests passed!")
        return 0
    else:
        print("✗ Some integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
