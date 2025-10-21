from __future__ import annotations

import os
import subprocess
from typing import IO, Optional

from ..util.setting import SUMMARY_FOLDER_DIR, TestList, TestStatusType
from .html_report_generator import HtmlReportGenerator
from ..oss.utils import get_xsigma_folder


CoverageItem = tuple[str, float, int, int]


def key_by_percentage(x: CoverageItem) -> float:
    return x[1]


def key_by_name(x: CoverageItem) -> str:
    return x[0]


def is_intrested_file(file_path: str, interested_folders: list[str]) -> bool:
    # Normalize path separators to forward slashes for consistent matching
    normalized_path = file_path.replace("\\", "/")

    if "cuda" in normalized_path:
        return False
    if "aten/gen_aten" in normalized_path or "aten/aten_" in normalized_path:
        return False
    for folder in interested_folders:
        # Normalize folder path to forward slashes
        normalized_folder = folder.replace("\\", "/")
        if normalized_folder in normalized_path:
            return True
    return False


def is_this_type_of_tests(target_name: str, test_set_by_type: set[str]) -> bool:
    # tests are divided into three types: success / partial success / fail to collect coverage
    for test in test_set_by_type:
        if target_name in test:
            return True
    return False


def print_test_by_type(
    tests: TestList, test_set_by_type: set[str], type_name: str, summary_file: IO[str]
) -> None:
    print("Tests " + type_name + " to collect coverage:", file=summary_file)
    for test in tests:
        if is_this_type_of_tests(test.name, test_set_by_type):
            print(test.target_pattern, file=summary_file)
    print(file=summary_file)


def print_test_condition(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: list[str],
    coverage_only: list[str],
    summary_file: IO[str],
    summary_type: str,
) -> None:
    print_test_by_type(tests, tests_type["success"], "fully success", summary_file)
    print_test_by_type(tests, tests_type["partial"], "partially success", summary_file)
    print_test_by_type(tests, tests_type["fail"], "failed", summary_file)
    print(
        "\n\nCoverage Collected Over Interested Folders:\n",
        interested_folders,
        file=summary_file,
    )
    print(
        "\n\nCoverage Compilation Flags Only Apply To: \n",
        coverage_only,
        file=summary_file,
    )
    print(
        "\n\n---------------------------------- "
        + summary_type
        + " ----------------------------------",
        file=summary_file,
    )


def line_oriented_report(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: list[str],
    coverage_only: list[str],
    covered_lines: dict[str, set[int]],
    uncovered_lines: dict[str, set[int]],
) -> None:
    with open(os.path.join(SUMMARY_FOLDER_DIR, "line_summary"), "w+") as report_file:
        print_test_condition(
            tests,
            tests_type,
            interested_folders,
            coverage_only,
            report_file,
            "LINE SUMMARY",
        )
        for file_name in covered_lines:
            covered = covered_lines[file_name]
            uncovered = uncovered_lines[file_name]
            print(
                f"{file_name}\n  covered lines: {sorted(covered)}\n  unconvered lines:{sorted(uncovered)}",
                file=report_file,
            )


def print_file_summary(
    covered_summary: int, total_summary: int, summary_file: IO[str]
) -> float:
    # print summary first
    try:
        coverage_percentage = 100.0 * covered_summary / total_summary
    except ZeroDivisionError:
        coverage_percentage = 0
    print(
        f"SUMMARY\ncovered: {covered_summary}\nuncovered: {total_summary}\npercentage: {coverage_percentage:.2f}%\n\n",
        file=summary_file,
    )
    if coverage_percentage == 0:
        print("Coverage is 0, Please check if json profiles are valid")
    return coverage_percentage


def print_file_oriented_report(
    tests_type: TestStatusType,
    coverage: list[CoverageItem],
    covered_summary: int,
    total_summary: int,
    summary_file: IO[str],
    tests: TestList,
    interested_folders: list[str],
    coverage_only: list[str],
) -> None:
    coverage_percentage = print_file_summary(
        covered_summary, total_summary, summary_file
    )
    # print test condition (interested folder / tests that are successful or failed)
    print_test_condition(
        tests,
        tests_type,
        interested_folders,
        coverage_only,
        summary_file,
        "FILE SUMMARY",
    )
    # print each file's information
    for item in coverage:
        print(
            item[0].ljust(75),
            (str(item[1]) + "%").rjust(10),
            str(item[2]).rjust(10),
            str(item[3]).rjust(10),
            file=summary_file,
        )

    print(f"summary percentage:{coverage_percentage:.2f}%")


def file_oriented_report(
    tests: TestList,
    tests_type: TestStatusType,
    interested_folders: list[str],
    coverage_only: list[str],
    covered_lines: dict[str, set[int]],
    uncovered_lines: dict[str, set[int]],
) -> None:
    with open(os.path.join(SUMMARY_FOLDER_DIR, "file_summary"), "w+") as summary_file:
        covered_summary = 0
        total_summary = 0
        coverage = []
        for file_name in covered_lines:
            # get coverage number for this file
            covered_count = len(covered_lines[file_name])
            total_count = covered_count + len(uncovered_lines[file_name])
            try:
                percentage = round(covered_count / total_count * 100, 2)
            except ZeroDivisionError:
                percentage = 0
            # store information in a list to be sorted
            coverage.append((file_name, percentage, covered_count, total_count))
            # update summary
            covered_summary = covered_summary + covered_count
            total_summary = total_summary + total_count
        # sort
        coverage.sort(key=key_by_name)
        coverage.sort(key=key_by_percentage)
        # print
        print_file_oriented_report(
            tests_type,
            coverage,
            covered_summary,
            total_summary,
            summary_file,
            tests,
            interested_folders,
            coverage_only,
        )


def get_html_ignored_pattern() -> list[str]:
    # Patterns to exclude from coverage reports
    # Note: Only include patterns that actually match files in the coverage data
    # Using patterns that don't match any files causes lcov to fail with exit code 25
    return ["/usr/*", "*/third_party/*"]


def html_oriented_report() -> None:
    # use lcov to generate the coverage report
    # Get the build folder from PROFILE_DIR (which is set to the actual build folder)
    # PROFILE_DIR is typically something like /path/to/build_ninja_coverage_lto/coverage_report
    # So we need to go up two levels to get the build folder
    from ..util.setting import PROFILE_DIR

    # Extract the build folder path from PROFILE_DIR
    # PROFILE_DIR = /path/to/build_ninja_coverage_lto/coverage_report
    # We need /path/to/build_ninja_coverage_lto
    build_folder = os.path.dirname(PROFILE_DIR)

    coverage_info_file = os.path.join(SUMMARY_FOLDER_DIR, "coverage.info")

    try:
        # generate coverage report -- coverage.info in build folder
        # Use --ignore-errors to bypass known gcov issues with line number mismatches and other errors
        # Common errors to ignore:
        # - mismatch: mismatched line numbers in debug info
        # - gcov: gcov-related warnings
        # - range: out-of-range line numbers
        # - negative: negative coverage counts (can occur with certain compiler optimizations)
        subprocess.check_call(
            [
                "lcov",
                "--capture",
                "--directory",
                build_folder,
                "--output-file",
                coverage_info_file,
                "--ignore-errors",
                "mismatch,gcov,range,negative",
            ]
        )
        # remove files that are unrelated
        cmd_array = (
            ["lcov", "--remove", coverage_info_file]
            + get_html_ignored_pattern()
            + ["--output-file", coverage_info_file, "--ignore-errors", "unused"]
        )
        try:
            subprocess.check_call(cmd_array)
        except subprocess.CalledProcessError as e:
            # If lcov remove fails, continue anyway - the coverage.info file may still be usable
            print(f"Warning: lcov --remove command failed (exit code {e.returncode}), but continuing with coverage report generation")

        # generate beautiful html page
        html_output_dir = os.path.join(SUMMARY_FOLDER_DIR, "html_report")
        try:
            subprocess.check_call(
                [
                    "genhtml",
                    coverage_info_file,
                    "--output-directory",
                    html_output_dir,
                    "--ignore-errors",
                    "empty,unused",
                ]
            )
        except subprocess.CalledProcessError as e:
            # If genhtml fails, try removing problematic patterns and retrying
            print(f"Warning: genhtml command failed (exit code {e.returncode}), attempting to clean coverage data")
            try:
                # Try to remove problematic patterns that might cause genhtml to crash
                subprocess.check_call(
                    [
                        "lcov",
                        "--remove",
                        coverage_info_file,
                        "/usr/*",
                        "*/third_party/*",
                        "--output-file",
                        coverage_info_file,
                        "--ignore-errors",
                        "unused",
                    ]
                )
                # Retry genhtml with cleaned data
                subprocess.check_call(
                    [
                        "genhtml",
                        coverage_info_file,
                        "--output-directory",
                        html_output_dir,
                        "--ignore-errors",
                        "empty,unused",
                    ]
                )
            except subprocess.CalledProcessError as e2:
                print(f"Error: genhtml failed even after cleanup (exit code {e2.returncode})")
                raise
    except FileNotFoundError as e:
        # lcov or genhtml not found - this is expected if they're not installed
        # The coverage data is still collected in .gcda files, but HTML report generation is skipped
        print(f"Warning: {e.filename} not found. HTML coverage report generation skipped.")
        print("To generate HTML reports, install lcov and genhtml:")
        print("  Ubuntu/Debian: sudo apt-get install lcov")
        print("  macOS: brew install lcov")
        print("  Or use Clang coverage instead for automatic HTML report generation")
    except subprocess.CalledProcessError as e:
        print(f"Error generating coverage report: {e}")
        raise


def generate_multifile_html_report(
    covered_lines: dict[str, set[int]],
    uncovered_lines: dict[str, set[int]],
    output_dir: Optional[str] = None,
    source_root: str = "",
) -> None:
    """Generate a multi-file HTML coverage report.

    Creates an index page with overall statistics and individual file reports
    with line-by-line coverage visualization.

    Args:
        covered_lines: Dict mapping file paths to sets of covered line numbers
        uncovered_lines: Dict mapping file paths to sets of uncovered line numbers
        output_dir: Output directory for HTML reports
                   (defaults to SUMMARY_FOLDER_DIR/html_details)
        source_root: Root directory of source files for reading content
    """
    if output_dir is None:
        output_dir = os.path.join(SUMMARY_FOLDER_DIR, "html_details")

    generator = HtmlReportGenerator(output_dir)
    generator.generate_report(covered_lines, uncovered_lines, source_root)
