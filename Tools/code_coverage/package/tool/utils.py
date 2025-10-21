import subprocess

from ..util.setting import TestPlatform
from ..util.utils import print_error


def run_cpp_test(binary_file: str) -> None:
    # cpp test binary
    import os
    import platform
    import stat

    # Convert path to absolute path and normalize for the platform
    binary_file = os.path.abspath(binary_file)

    # On Windows, ensure the path uses backslashes
    if platform.system() == "Windows":
        binary_file = binary_file.replace("/", "\\")

    # Verify the file exists and is executable (on Unix systems)
    if not os.path.isfile(binary_file):
        print_error(f"Binary not found: {binary_file}")
        return

    if platform.system() != "Windows":
        # On Unix systems, check if the file has execute permissions
        try:
            file_stat = os.stat(binary_file)
            if not (file_stat.st_mode & stat.S_IXUSR):
                print_error(f"Binary is not executable (missing execute permissions): {binary_file}")
                return
        except OSError as e:
            print_error(f"Failed to check file permissions: {binary_file} - {e}")
            return

    try:
        # Run the binary and capture output to filter out harmless warnings
        # "functions have mismatched data" warnings are harmless and occur due to LTO
        result = subprocess.run(
            [binary_file],
            capture_output=True,
            text=True,
            check=True,
        )
        # Filter stdout to remove harmless warnings (including ANSI color codes)
        if result.stdout:
            for line in result.stdout.split("\n"):
                # Skip lines containing the mismatched data warning
                # The warning may contain ANSI color codes like \033[0;31m
                if "functions have mismatched data" not in line and "mismatched data" not in line:
                    print(line)
        # Filter stderr to remove harmless warnings
        if result.stderr:
            for line in result.stderr.split("\n"):
                if "functions have mismatched data" not in line and "mismatched data" not in line and line.strip():
                    print(line, file=__import__("sys").stderr)
    except subprocess.CalledProcessError as e:
        # Print any output before the error, filtering out harmless warnings
        if e.stdout:
            for line in e.stdout.split("\n"):
                if "functions have mismatched data" not in line and "mismatched data" not in line:
                    print(line)
        if e.stderr:
            for line in e.stderr.split("\n"):
                if "functions have mismatched data" not in line and "mismatched data" not in line and line.strip():
                    print(line, file=__import__("sys").stderr)
        print_error(f"Binary failed to run: {binary_file}")
    except FileNotFoundError:
        print_error(f"Binary not found: {binary_file}")
    except PermissionError:
        print_error(f"Permission denied when running binary: {binary_file}")


def get_tool_path_by_platform(platform: TestPlatform) -> str:
    if platform == TestPlatform.FBCODE:
        from caffe2.fb.code_coverage.tool.package.fbcode.utils import (  # type: ignore[import]
            get_llvm_tool_path,
        )

        return get_llvm_tool_path()  # type: ignore[no-any-return]
    else:
        from ..oss.utils import get_llvm_tool_path  # type: ignore[no-redef]

        return get_llvm_tool_path()  # type: ignore[no-any-return]
