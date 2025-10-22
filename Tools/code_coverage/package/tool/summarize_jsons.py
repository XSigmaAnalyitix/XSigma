from __future__ import annotations

import json
import os
import time
from typing import Any, TYPE_CHECKING

from ..util.setting import (
    CompilerType,
    JSON_FOLDER_BASE_DIR,
    TestList,
    TestPlatform,
    TestStatusType,
)
from ..util.utils import (
    detect_compiler_type,
    print_error,
    print_log,
    print_time,
    related_to_test_list,
)
from .coverage_filters import is_interested_file
from .parser.gcov_coverage_parser import GcovCoverageParser
from .parser.llvm_coverage_parser import LlvmCoverageParser
from .print_report import (
    file_oriented_report,
    generate_multifile_html_report,
    line_oriented_report,
)


if TYPE_CHECKING:
    from .parser.coverage_record import CoverageRecord


# coverage_records: dict[str, LineInfo] = {}
covered_lines: dict[str, set[int]] = {}
uncovered_lines: dict[str, set[int]] = {}
tests_type: TestStatusType = {"success": set(), "partial": set(), "fail": set()}


def transform_file_name(
    file_path: str, interested_folders: list[str], platform: TestPlatform
) -> str:
    # Normalize path separators to forward slashes for consistent matching
    normalized_path = file_path.replace("\\", "/")

    remove_patterns: set[str] = {".DEFAULT.cpp", ".AVX.cpp", ".AVX2.cpp"}
    for pattern in remove_patterns:
        normalized_path = normalized_path.replace(pattern, "")
    # if user has specified interested folder
    if interested_folders:
        for folder in interested_folders:
            normalized_folder = folder.replace("\\", "/")
            if normalized_folder in normalized_path:
                return normalized_path[normalized_path.find(normalized_folder) :]
    # remove xsigma base folder path
    if platform == TestPlatform.OSS:
        from package.oss.utils import get_xsigma_folder  # type: ignore[import]

        pytorch_foler = get_xsigma_folder().replace("\\", "/")
        assert normalized_path.startswith(pytorch_foler)
        normalized_path = normalized_path[len(pytorch_foler) + 1 :]
    return normalized_path


def is_interested_file_wrapper(
    file_path: str, interested_folders: list[str], platform: TestPlatform | None = None
) -> bool:
    """Check if file should be included in coverage report.

    Wrapper around the shared is_interested_file function for backward
    compatibility with the old typo'd name.

    Args:
        file_path: Path to file to check
        interested_folders: List of folders to include
        platform: Platform type (OSS or FBCODE), optional

    Returns:
        True if file should be included, False otherwise
    """
    return is_interested_file(file_path, interested_folders, platform)


# Deprecated: Use is_interested_file from coverage_filters module instead
# Kept for backward compatibility with typo'd name
is_intrested_file = is_interested_file_wrapper


def get_json_obj(json_file: str) -> tuple[Any, int]:
    """Parse JSON file, handling partial reads and errors.

    Sometimes at the start of file llvm/gcov will complain "fail to find
    coverage data", so we need to skip these lines.

    Args:
        json_file: Path to JSON file to parse

    Returns:
        Tuple of (json_object, status_code) where status_code is:
        - 0: success read (full JSON coverage information)
        - 1: partial success (starts with error prompt but has coverage info)
        - 2: fail to read (no coverage information)
    """
    read_status = -1
    try:
        with open(json_file) as f:
            # Stream line-by-line instead of loading entire file into memory
            # This is more efficient for large JSON files
            for line in f:
                if not line.strip():
                    continue
                try:
                    json_obj = json.loads(line)
                except json.JSONDecodeError:
                    read_status = 1
                    continue
                else:
                    if read_status == -1:
                        # No JSON decode error encountered before, return success
                        read_status = 0
                    return (json_obj, read_status)
        return None, 2
    except (IOError, OSError) as e:
        print_error(f"Failed to read JSON file {json_file}: {e}")
        return None, 2


def parse_json(json_file: str, platform: TestPlatform) -> list[CoverageRecord]:
    print("start parse:", json_file)
    json_obj, read_status = get_json_obj(json_file)
    if read_status == 0:
        tests_type["success"].add(json_file)
    elif read_status == 1:
        tests_type["partial"].add(json_file)
    else:
        tests_type["fail"].add(json_file)
        raise RuntimeError(
            "Fail to do code coverage! Fail to load json file: ", json_file
        )

    cov_type = detect_compiler_type(platform)

    coverage_records: list[CoverageRecord] = []
    if cov_type == CompilerType.CLANG:
        coverage_records = LlvmCoverageParser(json_obj).parse("fbcode")
        # print(coverage_records)
    elif cov_type == CompilerType.GCC:
        coverage_records = GcovCoverageParser(json_obj).parse()

    return coverage_records


def parse_jsons(
    test_list: TestList, interested_folders: list[str], platform: TestPlatform
) -> None:
    g = os.walk(JSON_FOLDER_BASE_DIR)

    for path, _, file_list in g:
        for file_name in file_list:
            if file_name.endswith(".json"):
                # if compiler is clang, we only analyze related json / when compiler is gcc, we analyze all jsons
                cov_type = detect_compiler_type(platform)
                if cov_type == CompilerType.CLANG and not related_to_test_list(
                    file_name, test_list
                ):
                    continue
                json_file = os.path.join(path, file_name)
                try:
                    coverage_records = parse_json(json_file, platform)
                except RuntimeError:
                    print_error("Fail to load json file: ", json_file)
                    continue
                # collect information from each target's export file and merge them together:
                update_coverage(coverage_records, interested_folders, platform)


def update_coverage(
    coverage_records: list[CoverageRecord],
    interested_folders: list[str],
    platform: TestPlatform,
) -> None:
    for item in coverage_records:
        # extract information for the record
        record = item.to_dict()
        file_path = record["filepath"]
        if not is_intrested_file(file_path, interested_folders, platform):
            continue
        covered_range = record["covered_lines"]
        uncovered_range = record["uncovered_lines"]
        # transform file name: remote/13223/caffe2/aten -> caffe2/aten
        file_path = transform_file_name(file_path, interested_folders, platform)

        # if file not exists, add it into dictionary
        if file_path not in covered_lines:
            covered_lines[file_path] = set()
        if file_path not in uncovered_lines:
            uncovered_lines[file_path] = set()
        # update this file's covered and uncovered lines
        if covered_range is not None:
            covered_lines[file_path].update(covered_range)
        if uncovered_range is not None:
            uncovered_lines[file_path].update(uncovered_range)


def update_set() -> None:
    for file_name in covered_lines:
        # difference_update
        uncovered_lines[file_name].difference_update(covered_lines[file_name])


def summarize_jsons(
    test_list: TestList,
    interested_folders: list[str],
    coverage_only: list[str],
    platform: TestPlatform,
) -> None:
    start_time = time.time()
    # Parse JSON files for both GCC and Clang
    parse_jsons(test_list, interested_folders, platform)
    update_set()

    # Generate reports
    line_oriented_report(
        test_list,
        tests_type,
        interested_folders,
        coverage_only,
        covered_lines,
        uncovered_lines,
    )
    file_oriented_report(
        test_list,
        tests_type,
        interested_folders,
        coverage_only,
        covered_lines,
        uncovered_lines,
    )

    # Generate multi-file HTML report using custom HTML generator
    # This works for both GCC and Clang coverage
    try:
        from package.oss.utils import get_xsigma_folder
        source_root = get_xsigma_folder()
        generate_multifile_html_report(
            covered_lines,
            uncovered_lines,
            source_root=source_root,
        )
    except Exception as e:
        print_error(f"Failed to generate multi-file HTML report: {e}")

    print_time("summary jsons take time: ", start_time)
