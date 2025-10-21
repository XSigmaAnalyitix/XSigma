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
    project_folder = get_xsigma_folder()

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

    project_folder = get_xsigma_folder()

    # Try to find the build folder
    build_path = os.path.join(project_folder, build_folder)
    if not os.path.isdir(build_path):
        # If not found, try parent directory
        parent_folder = os.path.dirname(project_folder)
        build_path = os.path.join(parent_folder, build_folder)

    # Filter libraries based on platform
    libs = []
    system = platform.system()

    # Look for shared libraries in multiple directories
    # On Windows: look in both lib/ and bin/ directories
    # On Unix: look in lib/ directory
    search_dirs = []

    lib_dir = os.path.join(build_path, "lib")
    if os.path.isdir(lib_dir):
        search_dirs.append(lib_dir)

    # On Windows, also search in bin directory for DLLs
    if system == "Windows":
        bin_dir = os.path.join(build_path, "bin")
        if os.path.isdir(bin_dir):
            search_dirs.append(bin_dir)

    for search_dir in search_dirs:
        for lib in os.listdir(search_dir):
            lib_path = os.path.join(search_dir, lib)
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

    Searches in the following order:
    1. LLVM_TOOL_PATH environment variable
    2. System PATH for llvm-profdata
    3. Common platform-specific installation directories

    Returns:
        Path to LLVM tools directory

    Raises:
        FileNotFoundError: If LLVM tools cannot be found on the system
    """
    import os
    import platform
    import shutil
    from pathlib import Path

    tool_name = "llvm-profdata"
    env_var = "LLVM_TOOL_PATH"

    # Check explicit environment variable
    if env_var in os.environ:
        path = os.environ[env_var]
        if os.path.isdir(path):
            return path

    # Try to find tool in PATH
    tool_path = shutil.which(tool_name)
    if tool_path:
        return str(Path(tool_path).parent)

    # Platform-specific search directories
    system = platform.system()
    search_paths = {
        "Windows": [
            "C:\\Program Files\\LLVM\\bin",
            "C:\\Program Files (x86)\\LLVM\\bin",
            "C:\\tools\\llvm\\bin",
        ],
        "Darwin": [  # macOS
            "/opt/homebrew/opt/llvm/bin",  # Apple Silicon
            "/usr/local/opt/llvm/bin",      # Intel Homebrew
            "/opt/llvm/bin",
            "/usr/local/llvm/bin",
        ],
        "Linux": [
            "/usr/bin",
            "/usr/local/bin",
            "/opt/llvm/bin",
            "/usr/lib/llvm-*/bin",
        ],
    }

    # Search platform-specific directories
    for path in search_paths.get(system, []):
        if os.path.isdir(path):
            return path

    # If nothing found, raise error
    raise FileNotFoundError(
        f"Could not find {tool_name} on {system}. "
        f"Set {env_var} environment variable to specify location."
    )


def get_xsigma_folder() -> str:
    # TOOLS_FOLDER in oss: xsigma/tools/code_coverage
    return os.path.abspath(
        os.environ.get("XSIGMA_FOLDER", os.path.dirname(os.path.dirname(TOOLS_FOLDER)))
    )


def detect_compiler_type() -> CompilerType | None:
    # check if user specifies the compiler type
    user_specify = os.environ.get("CXX", None)
    if user_specify:
        # Normalize the compiler name
        compiler_lower = user_specify.lower()
        if "clang" in compiler_lower:
            return CompilerType.CLANG
        elif "gcc" in compiler_lower or "g++" in compiler_lower:
            return CompilerType.GCC

        raise RuntimeError(f"User specified compiler is not valid {user_specify}")

    # Check CC environment variable as fallback
    cc_specify = os.environ.get("CC", None)
    if cc_specify:
        compiler_lower = cc_specify.lower()
        if "clang" in compiler_lower:
            return CompilerType.CLANG
        elif "gcc" in compiler_lower:
            return CompilerType.GCC

    # auto detect (only on Unix-like systems)
    import platform
    if platform.system() != "Windows":
        try:
            auto_detect_result = subprocess.check_output(
                ["cc", "-v"], stderr=subprocess.STDOUT
            ).decode("utf-8")
            if "clang" in auto_detect_result:
                return CompilerType.CLANG
            elif "gcc" in auto_detect_result:
                return CompilerType.GCC
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    # Default to Clang on Windows if no compiler is detected
    if platform.system() == "Windows":
        return CompilerType.CLANG

    raise RuntimeError("Could not detect compiler type. Please set CXX environment variable.")


def clean_up_gcda() -> None:
    gcda_files = get_gcda_files()
    for item in gcda_files:
        remove_file(item)


def get_gcda_files() -> list[str]:
    import os

    project_folder = get_xsigma_folder()
    parent_folder = os.path.dirname(project_folder)

    # First, try to use XSIGMA_BUILD_FOLDER environment variable if set
    build_folder_name = os.environ.get("XSIGMA_BUILD_FOLDER", "")
    if build_folder_name:
        # Check if it's an absolute path
        if os.path.isabs(build_folder_name):
            folder_has_gcda = build_folder_name
        else:
            # Try relative to project folder first (most common case)
            folder_has_gcda = os.path.join(project_folder, build_folder_name)
            if not os.path.isdir(folder_has_gcda):
                # Try relative to parent directory (for builds outside project)
                folder_has_gcda = os.path.join(parent_folder, build_folder_name)
    else:
        # Fallback: Try to find build folder with .gcda files
        # First check relative to project folder
        folder_has_gcda = os.path.join(project_folder, "build")
        if not os.path.isdir(folder_has_gcda):
            # If not found, try parent directory (for builds outside project)
            # Look for the most recent build_* folder
            build_folders = []
            for item in os.listdir(parent_folder):
                potential_build = os.path.join(parent_folder, item)
                if os.path.isdir(potential_build) and item.startswith("build"):
                    build_folders.append((potential_build, os.path.getmtime(potential_build)))

            if build_folders:
                # Sort by modification time (most recent first)
                build_folders.sort(key=lambda x: x[1], reverse=True)
                folder_has_gcda = build_folders[0][0]

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
