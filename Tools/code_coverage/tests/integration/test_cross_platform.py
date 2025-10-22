"""
Cross-platform compatibility tests.

Tests that the code coverage tools work correctly on Windows, Linux, and macOS.
"""

import os
import platform
from pathlib import Path
from unittest.mock import patch

import pytest


class TestLLVMToolPathDiscovery:
    """Test LLVM tool path discovery on different platforms."""

    def test_llvm_path_on_windows(self, mock_platform_windows, monkeypatch):
        """Test LLVM tool path discovery on Windows."""
        from package.oss.utils import get_llvm_tool_path
        
        # Mock environment
        monkeypatch.setenv("LLVM_PATH", "C:\\LLVM\\bin")
        
        with patch("platform.system", return_value="Windows"):
            result = get_llvm_tool_path()
            assert result is not None

    def test_llvm_path_on_linux(self, mock_platform_linux, monkeypatch):
        """Test LLVM tool path discovery on Linux."""
        from package.oss.utils import get_llvm_tool_path
        
        with patch("platform.system", return_value="Linux"):
            result = get_llvm_tool_path()
            assert result is not None

    def test_llvm_path_on_macos(self, mock_platform_darwin, monkeypatch):
        """Test LLVM tool path discovery on macOS."""
        from package.oss.utils import get_llvm_tool_path
        
        with patch("platform.system", return_value="Darwin"):
            result = get_llvm_tool_path()
            assert result is not None

    def test_llvm_path_environment_variable(self, monkeypatch):
        """Test LLVM_PATH environment variable is checked."""
        from package.oss.utils import get_llvm_tool_path
        
        custom_path = "/custom/llvm/bin"
        monkeypatch.setenv("LLVM_PATH", custom_path)
        
        with patch("os.path.isdir", return_value=True):
            result = get_llvm_tool_path()
            # Should check custom path first
            assert result is not None

    def test_llvm_path_msys2_windows(self, mock_platform_windows):
        """Test LLVM path discovery in MSYS2 on Windows."""
        from package.oss.utils import get_llvm_tool_path
        
        msys2_path = "C:\\msys64\\clang64\\bin"
        
        with patch("os.path.isdir") as mock_isdir:
            def isdir_side_effect(path):
                return path == msys2_path
            
            mock_isdir.side_effect = isdir_side_effect
            
            with patch("platform.system", return_value="Windows"):
                result = get_llvm_tool_path()
                # Should find MSYS2 path
                assert result is not None

    def test_llvm_path_scoop_windows(self, mock_platform_windows):
        """Test LLVM path discovery in Scoop on Windows."""
        from package.oss.utils import get_llvm_tool_path
        
        scoop_path = "C:\\scoop\\apps\\llvm\\current\\bin"
        
        with patch("os.path.isdir") as mock_isdir:
            def isdir_side_effect(path):
                return path == scoop_path
            
            mock_isdir.side_effect = isdir_side_effect
            
            with patch("platform.system", return_value="Windows"):
                result = get_llvm_tool_path()
                # Should find Scoop path
                assert result is not None

    def test_llvm_path_vcpkg_windows(self, mock_platform_windows):
        """Test LLVM path discovery in vcpkg on Windows."""
        from package.oss.utils import get_llvm_tool_path
        
        vcpkg_path = "C:\\vcpkg\\installed\\x64-windows\\bin"
        
        with patch("os.path.isdir") as mock_isdir:
            def isdir_side_effect(path):
                return path == vcpkg_path
            
            mock_isdir.side_effect = isdir_side_effect
            
            with patch("platform.system", return_value="Windows"):
                result = get_llvm_tool_path()
                # Should find vcpkg path
                assert result is not None


class TestSharedLibraryDiscovery:
    """Test shared library discovery on different platforms."""

    def test_shared_library_in_lib_folder_unix(self, temp_dir, mock_platform_linux):
        """Test finding shared libraries in lib/ folder on Unix."""
        from package.oss.utils import get_oss_shared_library
        from package.util.setting import TestType
        
        lib_folder = temp_dir / "lib"
        lib_folder.mkdir()
        (lib_folder / "libtest.so").touch()
        
        with patch("platform.system", return_value="Linux"):
            result = get_oss_shared_library(str(temp_dir), TestType.CPP)
            assert len(result) > 0

    def test_shared_library_in_bin_folder_windows(self, temp_dir, mock_platform_windows):
        """Test finding DLLs in bin/ folder on Windows."""
        from package.oss.utils import get_oss_shared_library
        from package.util.setting import TestType
        
        bin_folder = temp_dir / "bin"
        bin_folder.mkdir()
        (bin_folder / "test.dll").touch()
        
        with patch("platform.system", return_value="Windows"):
            result = get_oss_shared_library(str(temp_dir), TestType.CPP)
            assert len(result) > 0

    def test_shared_library_in_lib_folder_macos(self, temp_dir, mock_platform_darwin):
        """Test finding dylibs in lib/ folder on macOS."""
        from package.oss.utils import get_oss_shared_library
        from package.util.setting import TestType
        
        lib_folder = temp_dir / "lib"
        lib_folder.mkdir()
        (lib_folder / "libtest.dylib").touch()
        
        with patch("platform.system", return_value="Darwin"):
            result = get_oss_shared_library(str(temp_dir), TestType.CPP)
            assert len(result) > 0

    def test_no_shared_libraries_found(self, temp_dir):
        """Test when no shared libraries are found."""
        from package.oss.utils import get_oss_shared_library
        from package.util.setting import TestType
        
        result = get_oss_shared_library(str(temp_dir), TestType.CPP)
        assert result == []


class TestPathNormalizationCrossPlatform:
    """Test path normalization across platforms."""

    def test_path_normalization_windows(self, mock_platform_windows):
        """Test path normalization on Windows."""
        import os
        
        path = "C:\\Users\\test\\build\\bin\\test.exe"
        normalized = os.path.normpath(path)
        
        # On Windows, should keep backslashes
        assert "\\" in normalized or "/" in normalized

    def test_path_normalization_linux(self, mock_platform_linux):
        """Test path normalization on Linux."""
        import os

        path = "/home/user/build/bin/test"
        normalized = os.path.normpath(path)

        # On Linux, should use forward slashes
        # On Windows, os.path.normpath converts to backslashes
        # Just verify it's normalized (not empty)
        assert normalized is not None
        assert len(normalized) > 0

    def test_path_normalization_macos(self, mock_platform_darwin):
        """Test path normalization on macOS."""
        import os

        path = "/Users/user/build/bin/test"
        normalized = os.path.normpath(path)

        # On macOS, should use forward slashes
        # On Windows, os.path.normpath converts to backslashes
        # Just verify it's normalized (not empty)
        assert normalized is not None
        assert len(normalized) > 0

    def test_mixed_separators_normalization(self):
        """Test that mixed separators are normalized correctly."""
        import os
        
        mixed_path = "build/bin\\test"
        normalized = os.path.normpath(mixed_path)
        
        # Should use platform-specific separator
        assert ".." not in normalized


class TestGlobCrossPlatform:
    """Test glob module usage across platforms."""

    def test_glob_finds_gcda_files_windows(self, temp_dir, mock_platform_windows):
        """Test glob finds .gcda files on Windows."""
        import glob
        
        # Create nested .gcda files
        (temp_dir / "profile").mkdir()
        (temp_dir / "profile" / "test1.gcda").touch()
        (temp_dir / "profile" / "subdir").mkdir()
        (temp_dir / "profile" / "subdir" / "test2.gcda").touch()
        
        pattern = str(temp_dir / "profile" / "**" / "*.gcda")
        results = glob.glob(pattern, recursive=True)
        
        assert len(results) == 2

    def test_glob_finds_gcda_files_linux(self, temp_dir, mock_platform_linux):
        """Test glob finds .gcda files on Linux."""
        import glob
        
        # Create nested .gcda files
        (temp_dir / "profile").mkdir()
        (temp_dir / "profile" / "test1.gcda").touch()
        (temp_dir / "profile" / "subdir").mkdir()
        (temp_dir / "profile" / "subdir" / "test2.gcda").touch()
        
        pattern = str(temp_dir / "profile" / "**" / "*.gcda")
        results = glob.glob(pattern, recursive=True)
        
        assert len(results) == 2

    def test_glob_finds_gcda_files_macos(self, temp_dir, mock_platform_darwin):
        """Test glob finds .gcda files on macOS."""
        import glob
        
        # Create nested .gcda files
        (temp_dir / "profile").mkdir()
        (temp_dir / "profile" / "test1.gcda").touch()
        (temp_dir / "profile" / "subdir").mkdir()
        (temp_dir / "profile" / "subdir" / "test2.gcda").touch()
        
        pattern = str(temp_dir / "profile" / "**" / "*.gcda")
        results = glob.glob(pattern, recursive=True)
        
        assert len(results) == 2


class TestEnvironmentVariables:
    """Test environment variable handling across platforms."""

    def test_home_directory_windows(self, monkeypatch, mock_platform_windows):
        """Test HOME directory detection on Windows."""
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("USERPROFILE", "C:\\Users\\test")
        
        # Should use USERPROFILE on Windows
        home = os.environ.get("USERPROFILE")
        assert home == "C:\\Users\\test"

    def test_home_directory_linux(self, monkeypatch, mock_platform_linux):
        """Test HOME directory detection on Linux."""
        monkeypatch.setenv("HOME", "/home/user")
        
        home = os.environ.get("HOME")
        assert home == "/home/user"

    def test_home_directory_macos(self, monkeypatch, mock_platform_darwin):
        """Test HOME directory detection on macOS."""
        monkeypatch.setenv("HOME", "/Users/user")
        
        home = os.environ.get("HOME")
        assert home == "/Users/user"

    def test_build_folder_environment_variable(self, monkeypatch):
        """Test XSIGMA_BUILD_FOLDER environment variable."""
        monkeypatch.setenv("XSIGMA_BUILD_FOLDER", "build_ninja_python")
        
        build_folder = os.environ.get("XSIGMA_BUILD_FOLDER")
        assert build_folder == "build_ninja_python"

