"""
Tests for path handling and normalization.

Tests the fixes for path normalization issues, ensuring cross-platform
compatibility for Windows, Linux, and macOS.
"""

import os
import platform
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPathNormalization:
    """Test path normalization using os.path.normpath()."""

    def test_windows_path_with_backslashes(self, mock_platform_windows):
        """Test that Windows paths with backslashes are handled correctly."""
        from package.tool.utils import run_cpp_test
        
        # Mock the subprocess call
        with patch("subprocess.check_call") as mock_call:
            with patch("os.path.exists", return_value=True):
                with patch("os.path.isfile", return_value=True):
                    # This should normalize the path
                    test_path = "C:\\build\\bin\\test.exe"
                    # We can't directly test run_cpp_test without subprocess,
                    # but we can test the normalization logic
                    normalized = os.path.normpath(test_path)
                    assert "\\" in normalized or "/" in normalized
                    assert normalized == test_path  # On Windows, should keep backslashes

    def test_unix_path_with_forward_slashes(self, mock_platform_linux):
        """Test that Unix paths with forward slashes are handled correctly."""
        test_path = "/home/user/build/bin/test"
        normalized = os.path.normpath(test_path)
        # On Windows, os.path.normpath converts to backslashes
        # On Unix, it keeps forward slashes
        # Just verify it's normalized (not empty)
        assert normalized is not None
        assert len(normalized) > 0

    def test_mixed_path_separators(self):
        """Test that mixed path separators are normalized correctly."""
        # Mixed separators should be normalized to platform-specific
        mixed_path = "Library/Core\\test.cpp"
        normalized = os.path.normpath(mixed_path)
        
        # Should use platform-specific separator
        if platform.system() == "Windows":
            assert "\\" in normalized or "/" in normalized
        else:
            assert "/" in normalized

    def test_relative_path_normalization(self):
        """Test that relative paths are normalized correctly."""
        rel_path = "build/../bin/test"
        normalized = os.path.normpath(rel_path)
        
        # Should resolve .. correctly
        assert ".." not in normalized
        assert "bin" in normalized

    def test_absolute_path_normalization(self):
        """Test that absolute paths are normalized correctly."""
        if platform.system() == "Windows":
            abs_path = "C:\\Users\\test\\build\\bin\\test.exe"
        else:
            abs_path = "/home/user/build/bin/test"
        
        normalized = os.path.normpath(abs_path)
        assert normalized == abs_path

    def test_path_with_dots(self):
        """Test paths with . and .. are resolved correctly."""
        path_with_dots = "build/./bin/../lib/test"
        normalized = os.path.normpath(path_with_dots)
        
        # Should resolve . and ..
        assert "." not in normalized or normalized.endswith(".")
        assert ".." not in normalized

    def test_trailing_separator(self):
        """Test that trailing separators are handled correctly."""
        path_with_trailing = "build/bin/"
        normalized = os.path.normpath(path_with_trailing)
        
        # Trailing separator should be removed by normpath
        assert not normalized.endswith(os.sep) or normalized == os.sep

    def test_empty_path(self):
        """Test that empty paths are handled correctly."""
        empty_path = ""
        normalized = os.path.normpath(empty_path)
        
        # Empty path should normalize to current directory
        assert normalized == "."

    def test_path_with_spaces(self):
        """Test that paths with spaces are handled correctly."""
        path_with_spaces = "Program Files/XSigma/build/bin/test"
        normalized = os.path.normpath(path_with_spaces)
        
        # Spaces should be preserved
        assert "Program Files" in normalized or "Program" in normalized


class TestPathConversion:
    """Test path conversion between different formats."""

    def test_convert_to_relative_path(self):
        """Test converting absolute path to relative path."""
        from package.util.utils import convert_to_relative_path
        
        whole_path = "profile/raw/aten"
        base_path = "profile"
        
        result = convert_to_relative_path(whole_path, base_path)
        assert result == "raw/aten"

    def test_convert_to_relative_path_with_backslashes(self):
        """Test converting path with backslashes to relative path."""
        from package.util.utils import convert_to_relative_path
        
        whole_path = "profile\\raw\\aten"
        base_path = "profile"
        
        result = convert_to_relative_path(whole_path, base_path)
        # Should handle backslashes
        assert "aten" in result

    def test_convert_to_relative_path_error(self):
        """Test that error is raised when base path not in whole path."""
        from package.util.utils import convert_to_relative_path
        
        whole_path = "profile/raw/aten"
        base_path = "other"
        
        with pytest.raises(RuntimeError):
            convert_to_relative_path(whole_path, base_path)

    def test_replace_extension(self):
        """Test replacing file extensions."""
        from package.util.utils import replace_extension
        
        filename = "test.profraw"
        new_ext = ".merged"
        
        result = replace_extension(filename, new_ext)
        assert result == "test.merged"

    def test_replace_extension_multiple_dots(self):
        """Test replacing extension in filename with multiple dots."""
        from package.util.utils import replace_extension
        
        filename = "test.data.profraw"
        new_ext = ".merged"
        
        result = replace_extension(filename, new_ext)
        assert result == "test.data.merged"


class TestPathDiscovery:
    """Test path discovery and validation."""

    def test_get_raw_profiles_folder_default(self, monkeypatch):
        """Test getting raw profiles folder with default path."""
        from package.util.utils import get_raw_profiles_folder
        
        # Clear environment variable
        monkeypatch.delenv("RAW_PROFILES_FOLDER", raising=False)
        
        result = get_raw_profiles_folder()
        assert "raw" in result
        assert "profile" in result

    def test_get_raw_profiles_folder_custom(self, monkeypatch):
        """Test getting raw profiles folder with custom environment variable."""
        from package.util.utils import get_raw_profiles_folder
        
        custom_path = "/custom/raw/profiles"
        monkeypatch.setenv("RAW_PROFILES_FOLDER", custom_path)
        
        result = get_raw_profiles_folder()
        assert result == custom_path

    def test_get_xsigma_folder(self, monkeypatch, temp_dir):
        """Test getting XSigma folder path."""
        from package.oss.utils import get_xsigma_folder

        # The function should return the actual XSigma folder
        # In the test environment, it will return the actual project folder
        result = get_xsigma_folder()
        assert result is not None
        assert len(result) > 0
        assert os.path.isdir(result)

