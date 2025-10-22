from __future__ import annotations

import os
import subprocess
import time

from ..util.setting import (
    JSON_FOLDER_BASE_DIR,
    MERGED_FOLDER_BASE_DIR,
    TestList,
    TestPlatform,
    TestType,
)
from ..util.utils import (
    check_platform_type,
    convert_to_relative_path,
    create_folder,
    get_raw_profiles_folder,
    get_test_name_from_whole_path,
    print_error,
    print_log,
    print_time,
    related_to_test_list,
    replace_extension,
)
from .utils import get_tool_path_by_platform, run_cpp_test


def get_coverage_filters() -> list[str]:
    """Get list of regex patterns to exclude from coverage analysis.

    Excludes:
    - ThirdParty/ directory
    - Library/Core/Testing/ directory
    - Test files (Test*.cxx, *Test.cxx, etc.)
    """
    filters = [
        # Exclude ThirdParty directory (handle both / and \)
        ".*[/\\\\]ThirdParty[/\\\\].*",
        # Exclude test files in Library/Core/Testing
        ".*[/\\\\]Library[/\\\\]Core[/\\\\]Testing[/\\\\].*",
        # Exclude test files by naming pattern
        ".*[/\\\\]Test[A-Za-z0-9_]*\\.(cxx|cpp|h|hxx)$",
        ".*[/\\\\][A-Za-z0-9_]*Test\\.(cxx|cpp|h|hxx)$",
    ]
    return filters


def build_llvm_cov_filter_args() -> list[str]:
    """Build llvm-cov filter arguments to exclude unwanted files."""
    filters = get_coverage_filters()
    args = []
    for filter_pattern in filters:
        args.extend(["-ignore-filename-regex", filter_pattern])
    return args


def create_corresponding_folder(
    cur_path: str, prefix_cur_path: str, dir_list: list[str], new_base_folder: str
) -> None:
    """Create corresponding folder structure in a new base folder.

    Mirrors the directory hierarchy from the current path into a new base
    folder. Used to create matching folder structures for merged profiles
    and JSON exports.

    Args:
        cur_path: Current path being processed
        prefix_cur_path: Base path to calculate relative path from
        dir_list: List of directory names to create
        new_base_folder: Base folder where new directories should be created
    """
    for dir_name in dir_list:
        relative_path = convert_to_relative_path(
            cur_path, prefix_cur_path
        )  # get folder name like 'aten'
        new_folder_path = os.path.join(new_base_folder, relative_path, dir_name)
        create_folder(new_folder_path)


def run_target(
    binary_file: str,
    raw_file: str,
    test_type: TestType,
    platform_type: TestPlatform,
    build_folder: str = "",
    test_subfolder: str = "bin",
) -> None:
    print_log("start run: ", binary_file)
    # set environment variable -- raw profile output path of the binary run
    os.environ["LLVM_PROFILE_FILE"] = raw_file
    # run binary
    if test_type == TestType.PY and platform_type == TestPlatform.OSS:
        from ..oss.utils import run_oss_python_test

        if not run_oss_python_test(binary_file, build_folder, test_subfolder):
            print_error(f"Python test failed: {binary_file}")
    else:
        run_cpp_test(binary_file)


def merge_target(raw_file: str, merged_file: str, platform_type: TestPlatform) -> None:
    print_log("start to merge target: ", raw_file)
    # run command
    llvm_tool_path = get_tool_path_by_platform(platform_type)
    # Use os.path.join to handle path separators correctly on all platforms
    llvm_profdata = os.path.join(llvm_tool_path, "llvm-profdata")
    # Suppress stderr to avoid "functions have mismatched data" warnings from llvm-profdata
    # These warnings are harmless and occur due to LTO and compiler optimizations
    with open(os.devnull, "w") as devnull:
        subprocess.check_call(
            [
                llvm_profdata,
                "merge",
                "-sparse",
                raw_file,
                "-o",
                merged_file,
            ],
            stderr=devnull,
        )


def export_target(
    merged_file: str,
    json_file: str,
    binary_file: str,
    shared_library_list: list[str],
    platform_type: TestPlatform,
) -> bool:
    """Export coverage data to JSON format.

    Args:
        merged_file: Path to merged profile file
        json_file: Path to output JSON file
        binary_file: Path to binary file with coverage data
        shared_library_list: List of shared library paths
        platform_type: Platform type (OSS or FBCODE)

    Returns:
        True if export succeeded, False otherwise
    """
    if binary_file is None:
        print_error(f"{merged_file} doesn't have corresponding binary!")
        return False
    print_log("start to export: ", merged_file)
    # run export
    llvm_tool_path = get_tool_path_by_platform(platform_type)
    llvm_cov = os.path.join(llvm_tool_path, "llvm-cov")

    # Build command arguments
    cmd_args = [llvm_cov, "export"]

    # Add binary file if provided
    if binary_file:
        cmd_args.extend(["-object", binary_file])

    # Add shared libraries
    for shared_lib in shared_library_list:
        cmd_args.extend(["-object", shared_lib])

    # Add profile file
    cmd_args.extend(["-instr-profile=" + merged_file])

    # NOTE: Do NOT apply filters at export stage!
    # Filters should only be applied at report generation stage (summarize_jsons.py)
    # Applying filters here removes files from JSON export permanently
    # and prevents them from appearing in coverage reports even with 0% coverage

    # Run command and redirect output to json file
    # Suppress stderr to avoid "functions have mismatched data" warnings from llvm-cov
    # These warnings are harmless and occur due to LTO and compiler optimizations
    try:
        with open(json_file, "w") as f:
            with open(os.devnull, "w") as devnull:
                subprocess.check_call(cmd_args, stdout=f, stderr=devnull)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to export coverage data: {e}")
        return False
    except (IOError, OSError) as e:
        print_error(f"Failed to write JSON file {json_file}: {e}")
        return False


def merge(test_list: TestList, platform_type: TestPlatform) -> None:
    print("start merge")
    start_time = time.time()
    # find all raw profile under raw_folder and sub-folders
    raw_folder_path = get_raw_profiles_folder()
    g = os.walk(raw_folder_path)
    for path, dir_list, file_list in g:
        # if there is a folder raw/aten/, create corresponding merged folder profile/merged/aten/ if not exists yet
        create_corresponding_folder(
            path, raw_folder_path, dir_list, MERGED_FOLDER_BASE_DIR
        )
        # check if we can find raw profile under this path's folder
        for file_name in file_list:
            if file_name.endswith(".profraw"):
                if not related_to_test_list(file_name, test_list):
                    continue
                print(f"start merge {file_name}")
                raw_file = os.path.join(path, file_name)
                merged_file_name = replace_extension(file_name, ".merged")
                merged_file = os.path.join(
                    MERGED_FOLDER_BASE_DIR,
                    convert_to_relative_path(path, raw_folder_path),
                    merged_file_name,
                )
                merge_target(raw_file, merged_file, platform_type)
    print_time("merge take time: ", start_time, summary_time=True)


def export(
    test_list: TestList,
    platform_type: TestPlatform,
    build_folder: str = "",
    test_subfolder: str = "bin",
) -> None:
    print("start export")
    start_time = time.time()
    # find all merged profile under merged_folder and sub-folders
    g = os.walk(MERGED_FOLDER_BASE_DIR)
    for path, dir_list, file_list in g:
        # create corresponding merged folder in [json folder] if not exists yet
        create_corresponding_folder(
            path, MERGED_FOLDER_BASE_DIR, dir_list, JSON_FOLDER_BASE_DIR
        )
        # check if we can find merged profile under this path's folder
        for file_name in file_list:
            if file_name.endswith(".merged"):
                if not related_to_test_list(file_name, test_list):
                    continue
                print(f"start export {file_name}")
                # merged file
                merged_file = os.path.join(path, file_name)
                # json file
                json_file_name = replace_extension(file_name, ".json")
                json_file = os.path.join(
                    JSON_FOLDER_BASE_DIR,
                    convert_to_relative_path(path, MERGED_FOLDER_BASE_DIR),
                    json_file_name,
                )
                check_platform_type(platform_type)
                # binary file and shared library
                binary_file = ""
                shared_library_list = []
                if platform_type == TestPlatform.FBCODE:
                    from caffe2.fb.code_coverage.tool.package.fbcode.utils import (  # type: ignore[import]
                        get_fbcode_binary_folder,
                    )

                    binary_file = os.path.join(
                        get_fbcode_binary_folder(path),
                        get_test_name_from_whole_path(merged_file),
                    )
                elif platform_type == TestPlatform.OSS:
                    from ..oss.utils import (
                        get_oss_binary_file,
                        get_oss_shared_library,
                    )

                    test_name = get_test_name_from_whole_path(merged_file)
                    # if it is python test, no need to provide binary
                    binary_file = (
                        ""
                        if test_name.endswith(".py")
                        else get_oss_binary_file(
                            test_name, TestType.CPP, build_folder, test_subfolder
                        )
                    )
                    shared_library_list = get_oss_shared_library(
                        build_folder, test_subfolder
                    )
                if not export_target(
                    merged_file,
                    json_file,
                    binary_file,
                    shared_library_list,
                    platform_type,
                ):
                    print_error(f"Failed to export {merged_file}, skipping...")
    print_time("export take time: ", start_time, summary_time=True)


def show_html(
    test_list: TestList,
    platform_type: TestPlatform,
    build_folder: str = "",
    test_subfolder: str = "bin",
) -> None:
    """Generate HTML coverage report using llvm-cov show.

    Args:
        test_list: List of tests to generate reports for
        platform_type: Platform type (OSS or FBCODE)
        build_folder: Name of the build folder
        test_subfolder: Subfolder within build folder where tests are
    """
    print("start html report generation")
    start_time = time.time()

    # Create HTML output directory
    html_dir = os.path.join(JSON_FOLDER_BASE_DIR, "..", "html")
    create_folder(html_dir)

    # Find all merged profile files
    g = os.walk(MERGED_FOLDER_BASE_DIR)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name.endswith(".merged"):
                if not related_to_test_list(file_name, test_list):
                    continue
                print(f"start html generation for {file_name}")

                # merged file
                merged_file = os.path.join(path, file_name)

                # Get binary and shared libraries
                binary_file = ""
                shared_library_list = []
                if platform_type == TestPlatform.OSS:
                    from ..oss.utils import (
                        get_oss_binary_file,
                        get_oss_shared_library,
                    )

                    test_name = get_test_name_from_whole_path(
                        merged_file
                    )
                    # if it is python test, skip HTML generation
                    if test_name.endswith(".py"):
                        continue
                    binary_file = get_oss_binary_file(
                        test_name,
                        TestType.CPP,
                        build_folder,
                        test_subfolder,
                    )
                    shared_library_list = get_oss_shared_library(
                        build_folder, test_subfolder
                    )

                if not binary_file:
                    continue

                # Build llvm-cov show command
                llvm_tool_path = get_tool_path_by_platform(
                    platform_type
                )
                llvm_cov = os.path.join(llvm_tool_path, "llvm-cov")

                # Create output file for this test
                html_file_name = replace_extension(file_name, ".html")
                html_file = os.path.join(html_dir, html_file_name)

                cmd_args = [
                    llvm_cov,
                    "show",
                    "-format=html",
                    "-object",
                    binary_file,
                ]

                # Add shared libraries
                for shared_lib in shared_library_list:
                    cmd_args.extend(["-object", shared_lib])

                # Add profile file
                cmd_args.extend(["-instr-profile=" + merged_file])

                # Add coverage filters
                cmd_args.extend(build_llvm_cov_filter_args())

                # Run command and redirect output to html file
                with open(html_file, "w") as f:
                    subprocess.check_call(cmd_args, stdout=f)

    print_time("html generation take time: ", start_time,
               summary_time=True)


def show_multifile_html(
    covered_lines: dict[str, set[int]],
    uncovered_lines: dict[str, set[int]],
    source_root: str = "",
) -> None:
    """Generate multi-file HTML coverage report.

    Creates an index page with overall statistics and individual file reports
    with line-by-line coverage visualization.

    Args:
        covered_lines: Dict mapping file paths to sets of covered line numbers
        uncovered_lines: Dict mapping file paths to sets of uncovered line numbers
        source_root: Root directory of source files for reading content
    """
    from .html_report_generator import HtmlReportGenerator
    from ..util.setting import SUMMARY_FOLDER_DIR

    print("start multi-file html report generation")
    start_time = time.time()

    # Create output directory
    html_dir = os.path.join(SUMMARY_FOLDER_DIR, "..", "html_details")

    # Generate report
    generator = HtmlReportGenerator(html_dir)
    generator.generate_report(covered_lines, uncovered_lines, source_root)

    print_time("multi-file html generation take time: ", start_time,
               summary_time=True)
