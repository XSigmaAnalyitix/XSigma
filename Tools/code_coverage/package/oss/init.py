from __future__ import annotations

import argparse
import os
from typing import cast

from ..util.setting import (
    CompilerType,
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    Option,
    Test,
    TestList,
    TestType,
)
from ..util.utils import (
    clean_up,
    create_folder,
    print_log,
    raise_no_test_found_exception,
    remove_file,
    remove_folder,
)
from ..util.utils_init import add_arguments_utils, create_folders, get_options
from .utils import (
    clean_up_gcda,
    detect_compiler_type,
    get_llvm_tool_path,
    get_oss_binary_folder,
    get_xsigma_folder,
)


BLOCKED_PYTHON_TESTS = {
    "run_test.py",
    "test_dataloader.py",
    "test_multiprocessing.py",
    "test_multiprocessing_spawn.py",
    "test_utils.py",
}


def initialization() -> tuple[Option, TestList, list[str]]:
    # create folder if not exists
    create_folders()
    # add arguments
    parser = argparse.ArgumentParser()
    parser = add_arguments_utils(parser)
    parser = add_arguments_oss(parser)
    # parse arguments
    (
        options,
        args_interested_folder,
        args_run_only,
        arg_clean,
        build_folder,
        test_subfolder,
    ) = parse_arguments(parser)
    # clean up
    if arg_clean:
        clean_up_gcda()
        clean_up()
    # get test lists
    test_list = get_test_list(args_run_only, build_folder, test_subfolder)
    # get interested folder -- final report will only over these folders
    interested_folders = empty_list_if_none(args_interested_folder)
    # print initialization information
    print_init_info(build_folder, test_subfolder)
    # remove last time's log
    remove_file(os.path.join(LOG_DIR, "log.txt"))
    return (options, test_list, interested_folders)


def add_arguments_oss(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--build-folder",
        help="name of the build folder (e.g., 'build_ninja_python')",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--test-subfolder",
        help="subfolder within build folder where tests are located (default: 'bin')",
        default="bin",
        type=str,
    )
    parser.add_argument(
        "--run-only",
        help="only run certain test(s), for example: atest test_nn.py.",
        nargs="*",
        default=None,
    )

    return parser


def parse_arguments(
    parser: argparse.ArgumentParser,
) -> tuple[Option, list[str] | None, list[str] | None, bool | None, str, str]:
    # parse args
    args = parser.parse_args()
    # get option
    options = get_options(args)
    return (
        options,
        args.interest_only,
        args.run_only,
        args.clean,
        args.build_folder,
        args.test_subfolder,
    )


def get_test_list_by_type(
    run_only: list[str] | None,
    test_type: TestType,
    build_folder: str,
    test_subfolder: str = "bin",
    exclude_benchmarks: bool = True,
) -> TestList:
    import platform

    test_list: TestList = []
    binary_folder = get_oss_binary_folder(
        test_type, build_folder, test_subfolder
    )
    g = os.walk(binary_folder)
    for _, _, file_list in g:
        for file_name in file_list:
            # Filter for executable files only
            if test_type == TestType.CPP:
                # For C++ tests, only include .exe on Windows
                if platform.system() == "Windows":
                    if not file_name.endswith(".exe"):
                        continue
                else:
                    # On Unix, skip files with extensions
                    if "." in file_name:
                        continue
            elif test_type == TestType.PY:
                # For Python tests, only include .py files
                if not file_name.endswith(".py"):
                    continue

            # Exclude benchmark tests if requested
            if exclude_benchmarks and "Benchmark" in file_name:
                continue

            if run_only is not None and file_name not in run_only:
                continue
            # target pattern in oss is used in printing report
            test: Test = Test(
                name=file_name,
                target_pattern=file_name,
                test_set="",
                test_type=test_type,
            )
            test_list.append(test)
    return test_list


def get_test_list(
    run_only: list[str] | None, build_folder: str, test_subfolder: str = "bin"
) -> TestList:
    test_list: TestList = []
    # add c++ test list
    test_list.extend(
        get_test_list_by_type(run_only, TestType.CPP, build_folder, test_subfolder)
    )
    # add python test list
    py_run_only = get_python_run_only(run_only, build_folder, test_subfolder)
    test_list.extend(
        get_test_list_by_type(py_run_only, TestType.PY, build_folder, test_subfolder)
    )

    # not find any test to run
    if not test_list:
        raise_no_test_found_exception(
            get_oss_binary_folder(TestType.CPP, build_folder, test_subfolder),
            get_oss_binary_folder(TestType.PY, build_folder, test_subfolder),
        )
    return test_list


def empty_list_if_none(arg_interested_folder: list[str] | None) -> list[str]:
    if arg_interested_folder is None:
        return []
    # if this argument is specified, just return itself
    return arg_interested_folder


def gcc_export_init() -> None:
    remove_folder(JSON_FOLDER_BASE_DIR)
    create_folder(JSON_FOLDER_BASE_DIR)


def get_python_run_only(
    args_run_only: list[str] | None,
    build_folder: str,
    test_subfolder: str = "bin",
) -> list[str]:
    # if user specifies run-only option
    if args_run_only:
        return args_run_only

    # if not specified, use default setting, different for gcc and clang
    if detect_compiler_type() == CompilerType.GCC:
        return ["run_test.py"]
    else:
        # for clang, some tests will result in too large intermediate files
        # that can't be merged by llvm, we need to skip them
        run_only: list[str] = []
        binary_folder = get_oss_binary_folder(
            TestType.PY, build_folder, test_subfolder
        )
        g = os.walk(binary_folder)
        for _, _, file_list in g:
            for file_name in file_list:
                if (
                    file_name in BLOCKED_PYTHON_TESTS
                    or not file_name.endswith(".py")
                ):
                    continue
                run_only.append(file_name)
            # only run tests in the first-level folder in test/
            break
        return run_only


def print_init_info(build_folder: str, test_subfolder: str = "bin") -> None:
    print_log("xsigma folder: ", get_xsigma_folder())
    print_log(
        "cpp test binaries folder: ",
        get_oss_binary_folder(TestType.CPP, build_folder, test_subfolder),
    )
    print_log(
        "python test scripts folder: ",
        get_oss_binary_folder(TestType.PY, build_folder, test_subfolder),
    )
    print_log(
        "compiler type: ",
        cast(CompilerType, detect_compiler_type()).value,
    )
    print_log(
        "llvm tool folder (only for clang, if using gcov ignore): ",
        get_llvm_tool_path(),
    )
