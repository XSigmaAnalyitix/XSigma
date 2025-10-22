#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

# Set coverage output directory before importing settings
# This allows coverage reports to be generated in the build folder
# If XSIGMA_COVERAGE_DIR is already set (by the caller), use it
# Otherwise, try to construct it from XSIGMA_BUILD_FOLDER and XSIGMA_FOLDER
if not os.environ.get("XSIGMA_COVERAGE_DIR"):
    build_folder = os.environ.get("XSIGMA_BUILD_FOLDER", "build")
    if build_folder:
        xsigma_folder = os.environ.get("XSIGMA_FOLDER", ".")
        coverage_dir = str(
            Path(xsigma_folder).parent / build_folder / "coverage_report"
        )
        os.environ["XSIGMA_COVERAGE_DIR"] = coverage_dir

from package.oss.cov_json import get_json_report  # type: ignore[import]
from package.oss.init import initialization  # type: ignore[import]
from package.tool.summarize_jsons import summarize_jsons  # type: ignore[import]
from package.util.setting import TestPlatform  # type: ignore[import]
from package.util.utils import print_time  # type: ignore[import]


def report_coverage() -> None:
    start_time = time.time()
    (options, test_list, interested_folders) = initialization()
    # Extract build_folder and test_subfolder from environment
    # These are set by the caller (coverage.py helper)
    build_folder = os.environ.get("XSIGMA_BUILD_FOLDER", "build")
    test_subfolder = os.environ.get("XSIGMA_TEST_SUBFOLDER", "bin")

    # Extract interested folders from environment if set
    # This allows the build system to specify which folders to analyze
    env_interested_folders = os.environ.get("XSIGMA_INTERESTED_FOLDERS", "")
    if env_interested_folders:
        # Parse comma-separated list and merge with command-line specified folders
        env_folders = [
            f.strip() for f in env_interested_folders.split(",") if f.strip()
        ]
        if interested_folders:
            interested_folders.extend(env_folders)
        else:
            interested_folders = env_folders

    # run cpp tests
    get_json_report(test_list, options, build_folder, test_subfolder)
    # collect coverage data from json profiles
    if options.need_summary:
        summarize_jsons(test_list, interested_folders, [""], TestPlatform.OSS)
    # print program running time
    print_time("Program Total Time: ", start_time)


if __name__ == "__main__":
    report_coverage()
