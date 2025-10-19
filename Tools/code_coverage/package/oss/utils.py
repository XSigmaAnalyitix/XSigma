from __future__ import annotations

import os
import subprocess

from ..util.setting import CompilerType, TestType, TOOLS_FOLDER
from ..util.utils import print_error, remove_file


def get_oss_binary_folder(
    test_type: TestType, build_folder: str, test_subfolder: str = "bin"
) -> str:
    """
    Get the binary folder path for tests.

    Args:
        test_type: Type of test (CPP or PY)
        build_folder: Name of the build folder (e.g., "build_ninja_python")
        test_subfolder: Subfolder within build_folder where tests are located.
                       Defaults to "bin" for C++ tests, can be overridden.

    Returns:
        Full path to the binary folder
    """
    assert test_type in {TestType.CPP, TestType.PY}
    project_folder = get_pytorch_folder()

    # Try to find the build folder
    # First, check if it exists relative to project folder
    build_path = os.path.join(project_folder, build_folder)
    if not os.path.isdir(build_path):
        # If not found, try parent directory (for builds outside project)
        parent_folder = os.path.dirname(project_folder)
        build_path = os.path.join(parent_folder, build_folder)

    if test_type == TestType.CPP:
        return os.path.join(build_path, test_subfolder)
    else:
        # For Python tests, use "test" subfolder
        return os.path.join(build_path, "test")


def get_oss_shared_library(
    build_folder: str = "build", test_subfolder: str = "bin"
) -> list[str]:
    """
    Get list of shared libraries from the build folder.

    Args:
        build_folder: Name of the build folder
        test_subfolder: Subfolder within build_folder (used to find lib folder)

    Returns:
        List of shared library paths
    """
    import platform

    project_folder = get_pytorch_folder()

    # Try to find the build folder
    build_path = os.path.join(project_folder, build_folder)
    if not os.path.isdir(build_path):
        # If not found, try parent directory
        parent_folder = os.path.dirname(project_folder)
        build_path = os.path.join(parent_folder, build_folder)

    # Look for lib directory
    lib_dir = os.path.join(build_path, "lib")
    if not os.path.isdir(lib_dir):
        # If lib dir doesn't exist, return empty list
        return []

    # Filter libraries based on platform
    libs = []
    system = platform.system()

    for lib in os.listdir(lib_dir):
        lib_path = os.path.join(lib_dir, lib)
        # Only include files (not directories)
        if not os.path.isfile(lib_path):
            continue

        # Filter by platform-specific extensions
        if system == "Darwin":  # macOS
            if lib.endswith(".dylib"):
                libs.append(lib_path)
        elif system == "Windows":
            if lib.endswith(".dll"):
                libs.append(lib_path)
        else:  # Linux and others
            if lib.endswith(".so"):
                libs.append(lib_path)

    return libs


def get_oss_binary_file(
    test_name: str, test_type: TestType, build_folder: str, test_subfolder: str = "bin"
) -> str:
    """
    Get the full path to a binary test file.

    Args:
        test_name: Name of the test file
        test_type: Type of test (CPP or PY)
        build_folder: Name of the build folder
        test_subfolder: Subfolder within build_folder where tests are located

    Returns:
        Full path to the binary file (with "python " prefix for Python tests)
    """
    assert test_type in {TestType.CPP, TestType.PY}
    binary_folder = get_oss_binary_folder(test_type, build_folder, test_subfolder)
    binary_file = os.path.join(binary_folder, test_name)
    if test_type == TestType.PY:
        # add python to the command so we can directly run the script
        binary_file = "python " + binary_file
    return binary_file


def get_llvm_tool_path() -> str:
    """
    Get the LLVM tool path for the current platform.

    On Windows, tries to find llvm-profdata in PATH.
    On macOS, defaults to /usr/local/opt/llvm/bin.
    On Linux, defaults to /usr/local/opt/llvm/bin.

    Returns:
        Path to LLVM tools directory
    """
    import platform
    import shutil

    # Check if LLVM_TOOL_PATH is explicitly set
    if "LLVM_TOOL_PATH" in os.environ:
        return os.environ["LLVM_TOOL_PATH"]

    # On Windows, try to find llvm-profdata in PATH
    if platform.system() == "Windows":
        llvm_profdata = shutil.which("llvm-profdata")
        if llvm_profdata:
            # Return the directory containing llvm-profdata
            return os.path.dirname(llvm_profdata)
        # Fallback: try common LLVM installation paths on Windows
        common_paths = [
            "C:\\Program Files\\LLVM\\bin",
            "C:\\Program Files (x86)\\LLVM\\bin",
        ]
        for path in common_paths:
            if os.path.isdir(path):
                return path

    # Default paths for macOS and Linux
    return "/usr/local/opt/llvm/bin"


def get_pytorch_folder() -> str:
    # TOOLS_FOLDER in oss: xsigma/tools/code_coverage
    return os.path.abspath(
        os.environ.get("XSIGMA_FOLDER", os.path.dirname(os.path.dirname(TOOLS_FOLDER)))
    )


def detect_compiler_type() -> CompilerType | None:
    # check if user specifies the compiler type
    user_specify = os.environ.get("CXX", None)
    if user_specify:
        if user_specify in ["clang", "clang++"]:
            return CompilerType.CLANG
        elif user_specify in ["gcc", "g++"]:
            return CompilerType.GCC

        raise RuntimeError(f"User specified compiler is not valid {user_specify}")

    # auto detect
    auto_detect_result = subprocess.check_output(
        ["cc", "-v"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    if "clang" in auto_detect_result:
        return CompilerType.CLANG
    elif "gcc" in auto_detect_result:
        return CompilerType.GCC
    raise RuntimeError(f"Auto detected compiler is not valid {auto_detect_result}")


def clean_up_gcda() -> None:
    gcda_files = get_gcda_files()
    for item in gcda_files:
        remove_file(item)


def get_gcda_files() -> list[str]:
    import os

    project_folder = get_pytorch_folder()
    # Try to find build folder with .gcda files
    # First check relative to project folder
    folder_has_gcda = os.path.join(project_folder, "build")
    if not os.path.isdir(folder_has_gcda):
        # If not found, try parent directory (for builds outside project)
        parent_folder = os.path.dirname(project_folder)
        # Look for any build_* folder in parent
        for item in os.listdir(parent_folder):
            potential_build = os.path.join(parent_folder, item)
            if os.path.isdir(potential_build) and item.startswith("build"):
                folder_has_gcda = potential_build
                break

    if os.path.isdir(folder_has_gcda):
        # TODO use glob
        # output = glob.glob(f"{folder_has_gcda}/**/*.gcda")
        output = subprocess.check_output(
            ["find", folder_has_gcda, "-iname", "*.gcda"]
        )
        return output.decode("utf-8").split("\n")
    else:
        return []


def run_oss_python_test(
    binary_file: str, build_folder: str, test_subfolder: str = "bin"
) -> None:
    """
    Run a Python test script.

    Args:
        binary_file: Path to the Python test script
        build_folder: Name of the build folder
        test_subfolder: Subfolder within build_folder where tests are located
    """
    # python test script
    try:
        subprocess.check_call(
            binary_file,
            shell=True,
            cwd=get_oss_binary_folder(TestType.PY, build_folder, test_subfolder),
        )
    except subprocess.CalledProcessError:
        print_error(f"Binary failed to run: {binary_file}")
