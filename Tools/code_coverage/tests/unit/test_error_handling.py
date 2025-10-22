"""
Tests for error handling and return values.

Tests that functions properly return error codes instead of raising exceptions,
and that errors are properly propagated to callers.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExportTargetErrorHandling:
    """Test export_target() error handling and return values."""

    def test_export_target_success(self, temp_dir):
        """Test export_target() returns True on success."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform

        merged_file = str(temp_dir / "test.merged")
        json_file = str(temp_dir / "test.json")
        binary_file = str(temp_dir / "test_binary")

        # Create dummy files
        Path(merged_file).touch()
        Path(binary_file).touch()

        with patch("subprocess.check_call") as mock_call:
            mock_call.return_value = 0
            with patch("package.tool.clang_coverage.get_tool_path_by_platform", return_value="/usr/bin"):
                result = export_target(merged_file, json_file, binary_file, [], TestPlatform.OSS)
                assert result is True

    def test_export_target_missing_binary(self, temp_dir):
        """Test export_target() returns False when binary is missing."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform

        merged_file = str(temp_dir / "test.merged")
        json_file = str(temp_dir / "test.json")
        binary_file = None  # Missing binary

        Path(merged_file).touch()

        result = export_target(merged_file, json_file, binary_file, [], TestPlatform.OSS)
        assert result is False

    def test_export_target_subprocess_error(self, temp_dir):
        """Test export_target() returns False on subprocess error."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform

        merged_file = str(temp_dir / "test.merged")
        json_file = str(temp_dir / "test.json")
        binary_file = str(temp_dir / "test_binary")

        Path(merged_file).touch()
        Path(binary_file).touch()

        with patch("subprocess.check_call") as mock_call:
            mock_call.side_effect = subprocess.CalledProcessError(1, "llvm-cov")
            with patch("package.tool.clang_coverage.get_tool_path_by_platform", return_value="/usr/bin"):
                result = export_target(merged_file, json_file, binary_file, [], TestPlatform.OSS)
                assert result is False

    def test_export_target_file_write_error(self, temp_dir):
        """Test export_target() returns False on file write error."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform

        merged_file = str(temp_dir / "test.merged")
        json_file = "/invalid/path/test.json"  # Invalid path
        binary_file = str(temp_dir / "test_binary")

        Path(merged_file).touch()
        Path(binary_file).touch()

        with patch("subprocess.check_call") as mock_call:
            with patch("package.tool.clang_coverage.get_tool_path_by_platform", return_value="/usr/bin"):
                result = export_target(merged_file, json_file, binary_file, [], TestPlatform.OSS)
                # Should return False due to file write error
                assert result is False


class TestRunOssPythonTestErrorHandling:
    """Test run_oss_python_test() error handling and return values."""

    def test_run_oss_python_test_success(self, temp_dir):
        """Test run_oss_python_test() returns True on success."""
        from package.oss.utils import run_oss_python_test

        binary_file = str(temp_dir / "test.py")
        Path(binary_file).write_text("print('test')")

        with patch("subprocess.check_call") as mock_call:
            mock_call.return_value = 0
            result = run_oss_python_test(binary_file, "build")
            assert result is True

    def test_run_oss_python_test_failure(self, temp_dir):
        """Test run_oss_python_test() returns False on failure."""
        from package.oss.utils import run_oss_python_test

        binary_file = str(temp_dir / "test.py")
        Path(binary_file).write_text("print('test')")

        with patch("subprocess.check_call") as mock_call:
            mock_call.side_effect = subprocess.CalledProcessError(1, "python")
            result = run_oss_python_test(binary_file, "build")
            assert result is False

    def test_run_oss_python_test_file_not_found(self):
        """Test run_oss_python_test() returns False when file not found."""
        from package.oss.utils import run_oss_python_test

        binary_file = "/nonexistent/test.py"

        with patch("subprocess.check_call") as mock_call:
            mock_call.side_effect = FileNotFoundError()
            result = run_oss_python_test(binary_file, "build")
            assert result is False


class TestInputValidation:
    """Test input validation in functions."""

    def test_get_oss_shared_library_empty_build_folder(self, temp_dir):
        """Test get_oss_shared_library() with empty build folder."""
        from package.oss.utils import get_oss_shared_library
        from package.util.setting import TestType
        
        result = get_oss_shared_library("", TestType.CPP)
        # Should return empty list for empty build folder
        assert result == []

    def test_get_oss_shared_library_nonexistent_folder(self):
        """Test get_oss_shared_library() with nonexistent folder."""
        from package.oss.utils import get_oss_shared_library
        from package.util.setting import TestType
        
        result = get_oss_shared_library("/nonexistent/folder", TestType.CPP)
        # Should return empty list for nonexistent folder
        assert result == []

    def test_get_oss_shared_library_valid_folder(self, temp_dir):
        """Test get_oss_shared_library() with valid folder."""
        from package.oss.utils import get_oss_shared_library
        from package.util.setting import TestType
        
        # Create lib folder with shared library
        lib_folder = temp_dir / "lib"
        lib_folder.mkdir()
        
        if os.name == "nt":  # Windows
            lib_file = lib_folder / "test.dll"
        else:  # Unix
            lib_file = lib_folder / "libtest.so"
        
        lib_file.touch()
        
        result = get_oss_shared_library(str(temp_dir), TestType.CPP)
        # Should find the shared library
        assert len(result) > 0

    def test_get_oss_shared_library_windows_bin_folder(self, temp_dir, mock_platform_windows):
        """Test get_oss_shared_library() finds DLLs in bin folder on Windows."""
        from package.oss.utils import get_oss_shared_library
        from package.util.setting import TestType
        
        # Create bin folder with DLL
        bin_folder = temp_dir / "bin"
        bin_folder.mkdir()
        dll_file = bin_folder / "test.dll"
        dll_file.touch()
        
        with patch("platform.system", return_value="Windows"):
            result = get_oss_shared_library(str(temp_dir), TestType.CPP)
            # Should find the DLL
            assert len(result) > 0


class TestErrorPropagation:
    """Test that errors are properly propagated through call chain."""

    def test_run_target_handles_export_failure(self, temp_dir):
        """Test that run_target() handles export_target() failure."""
        from package.tool.clang_coverage import run_target
        from package.util.setting import TestType, TestPlatform

        binary_file = str(temp_dir / "test_binary")
        raw_file = str(temp_dir / "test.profraw")
        Path(binary_file).touch()

        with patch("package.tool.clang_coverage.run_cpp_test"):
            with patch("package.oss.utils.run_oss_python_test", return_value=True):
                # Should handle the False return value gracefully
                # (function doesn't raise exception)
                try:
                    run_target(binary_file, raw_file, TestType.CPP, TestPlatform.OSS, "build")
                except Exception as e:
                    pytest.fail(f"run_target() raised exception: {e}")

    def test_run_target_handles_test_failure(self, temp_dir):
        """Test that run_target() handles run_oss_python_test() failure."""
        from package.tool.clang_coverage import run_target
        from package.util.setting import TestType, TestPlatform

        binary_file = str(temp_dir / "test.py")
        raw_file = str(temp_dir / "test.profraw")
        Path(binary_file).touch()

        with patch("package.oss.utils.run_oss_python_test", return_value=False):
            # Should handle the False return value gracefully
            try:
                run_target(binary_file, raw_file, TestType.PY, TestPlatform.OSS, "build")
            except Exception as e:
                pytest.fail(f"run_target() raised exception: {e}")

    def test_gcc_run_target_handles_test_failure(self, temp_dir):
        """Test that gcc run_target() handles run_oss_python_test() failure."""
        from package.tool.gcc_coverage import run_target
        from package.util.setting import TestType

        binary_file = str(temp_dir / "test.py")
        Path(binary_file).touch()

        with patch("package.oss.utils.run_oss_python_test", return_value=False):
            # Should handle the False return value gracefully
            try:
                run_target(binary_file, TestType.PY, "build")
            except Exception as e:
                pytest.fail(f"run_target() raised exception: {e}")


class TestNoExceptionRaising:
    """Test that functions don't raise exceptions (XSigma standard)."""

    def test_export_target_no_exception(self, temp_dir):
        """Test that export_target() doesn't raise exceptions."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform

        merged_file = str(temp_dir / "test.merged")
        json_file = str(temp_dir / "test.json")
        binary_file = None

        # Should not raise exception, should return False
        try:
            result = export_target(merged_file, json_file, binary_file, [], TestPlatform.OSS)
            assert result is False
        except Exception as e:
            pytest.fail(f"export_target() raised exception: {e}")

    def test_run_oss_python_test_no_exception(self):
        """Test that run_oss_python_test() doesn't raise exceptions."""
        from package.oss.utils import run_oss_python_test

        binary_file = "/nonexistent/test.py"

        # Should not raise exception, should return False
        try:
            with patch("subprocess.check_call") as mock_call:
                mock_call.side_effect = subprocess.CalledProcessError(1, "python")
                result = run_oss_python_test(binary_file, "build")
                assert result is False
        except Exception as e:
            pytest.fail(f"run_oss_python_test() raised exception: {e}")

