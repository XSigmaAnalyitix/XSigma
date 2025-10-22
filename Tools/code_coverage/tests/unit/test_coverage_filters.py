"""
Tests for coverage filter logic.

Tests the is_interested_file() function to ensure correct file inclusion/exclusion
for coverage reports.
"""

import pytest

from package.tool.coverage_filters import is_interested_file
from package.util.setting import TestPlatform


class TestIsInterestedFile:
    """Test the is_interested_file() filter function."""

    def test_include_library_file(self):
        """Test that Library files are included."""
        file_path = "Library/Core/test.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is True

    def test_exclude_testing_folder(self):
        """Test that Testing folder files are excluded."""
        file_path = "Testing/test_core.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_exclude_test_subfolder(self):
        """Test that /test/ subfolder files are excluded."""
        file_path = "Library/Core/test/test_utils.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_exclude_tests_subfolder(self):
        """Test that /tests/ subfolder files are excluded."""
        file_path = "Library/Core/tests/test_utils.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_exclude_cuda_files(self):
        """Test that CUDA files are excluded."""
        file_path = "Library/cuda/kernel.cu"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_exclude_thirdparty_files(self):
        """Test that ThirdParty files are excluded."""
        file_path = "Library/ThirdParty/fmt/format.cpp"
        interested_folders = []

        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_exclude_build_files(self):
        """Test that build/ folder files are excluded."""
        file_path = "build/CMakeFiles/test.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_exclude_aten_generated_files(self):
        """Test that aten generated files are excluded."""
        file_path = "Library/aten/gen_aten/ops.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_exclude_aten_prefix_files(self):
        """Test that aten_* files are excluded."""
        file_path = "Library/aten/aten_ops.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_interested_folder_inclusion(self):
        """Test that interested folders are included."""
        file_path = "Library/Core/test.cpp"
        interested_folders = ["Core"]
        
        result = is_interested_file(file_path, interested_folders)
        assert result is True

    def test_interested_folder_exclusion(self):
        """Test that files outside interested folders are excluded."""
        file_path = "Library/Util/utils.cpp"
        interested_folders = ["Core"]
        
        result = is_interested_file(file_path, interested_folders)
        assert result is False

    def test_multiple_interested_folders(self):
        """Test with multiple interested folders."""
        interested_folders = ["Core", "Util"]
        
        # Should include Core files
        assert is_interested_file("Library/Core/test.cpp", interested_folders) is True
        
        # Should include Util files
        assert is_interested_file("Library/Util/utils.cpp", interested_folders) is True
        
        # Should exclude other files
        assert is_interested_file("Library/Other/other.cpp", interested_folders) is False

    def test_windows_path_with_backslashes(self):
        """Test that Windows paths with backslashes are handled correctly."""
        file_path = "Library\\Core\\test.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is True

    def test_mixed_path_separators(self):
        """Test that mixed path separators are handled correctly."""
        file_path = "Library/Core\\test.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is True

    def test_empty_file_path(self):
        """Test with empty file path."""
        file_path = ""
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        # Empty path should be included (no exclusion patterns match)
        assert result is True

    def test_file_path_with_special_characters(self):
        """Test file paths with special characters."""
        file_path = "Library/Core/test-utils_v2.cpp"
        interested_folders = []
        
        result = is_interested_file(file_path, interested_folders)
        assert result is True

    def test_case_sensitivity(self):
        """Test that filter is case-sensitive."""
        # Lowercase "testing" should be included (pattern is "Testing" with capital T)
        assert is_interested_file("testing/test.cpp", []) is True

        # Uppercase "TESTING" should be included (no match for "Testing" pattern)
        assert is_interested_file("TESTING/test.cpp", []) is True

    def test_interested_folder_with_trailing_slash(self):
        """Test interested folder with trailing slash."""
        file_path = "Library/Core/test.cpp"
        interested_folders = ["Core/"]
        
        result = is_interested_file(file_path, interested_folders)
        assert result is True

    def test_interested_folder_without_trailing_slash(self):
        """Test interested folder without trailing slash."""
        file_path = "Library/Core/test.cpp"
        interested_folders = ["Core"]
        
        result = is_interested_file(file_path, interested_folders)
        assert result is True

    def test_nested_interested_folder(self):
        """Test nested interested folder."""
        file_path = "Library/Core/Utils/test.cpp"
        interested_folders = ["Core/Utils"]
        
        result = is_interested_file(file_path, interested_folders)
        assert result is True

    def test_oss_platform_xsigma_folder_check(self, monkeypatch, temp_dir):
        """Test OSS platform checks for XSigma folder."""
        from unittest.mock import patch
        from package.oss.utils import get_xsigma_folder

        # Mock get_xsigma_folder to return temp_dir
        with patch("package.oss.utils.get_xsigma_folder", return_value=str(temp_dir)):
            file_path = str(temp_dir / "Library" / "Core" / "test.cpp")
            interested_folders = []

            result = is_interested_file(file_path, interested_folders, platform=TestPlatform.OSS)
            assert result is True

    def test_oss_platform_outside_xsigma_folder(self, monkeypatch, temp_dir):
        """Test OSS platform excludes files outside XSigma folder."""
        from unittest.mock import patch

        # Mock get_xsigma_folder to return temp_dir
        with patch("package.oss.utils.get_xsigma_folder", return_value=str(temp_dir)):
            # File outside XSigma folder
            file_path = "/other/location/test.cpp"
            interested_folders = []

            result = is_interested_file(file_path, interested_folders, platform=TestPlatform.OSS)
            assert result is False


class TestBackwardCompatibility:
    """Test backward compatibility with old typo'd function name."""

    def test_is_intrested_file_wrapper_exists(self):
        """Test that old typo'd function name still works."""
        from package.tool import summarize_jsons

        file_path = "Library/Core/test.cpp"
        interested_folders = []

        # Should work with old name (without platform parameter)
        result = summarize_jsons.is_intrested_file(file_path, interested_folders)
        assert result is True

    def test_is_intrested_file_in_print_report(self):
        """Test that old typo'd function name works in print_report."""
        from package.tool import print_report

        file_path = "Library/Core/test.cpp"
        interested_folders = []

        # Should work with old name
        result = print_report.is_intrested_file(file_path, interested_folders)
        assert result is True

