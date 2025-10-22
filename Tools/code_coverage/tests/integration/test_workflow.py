"""
End-to-end workflow integration tests.

Tests the complete coverage collection and report generation workflow.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCoverageWorkflow:
    """Test complete coverage workflow."""

    def test_workflow_initialization(self, temp_dir):
        """Test workflow initialization creates required folders."""
        from package.util.utils import create_folder

        # Create folders directly
        profile_dir = temp_dir / "profile"
        merged_dir = temp_dir / "merged"
        json_dir = temp_dir / "json"

        create_folder(str(profile_dir))
        create_folder(str(merged_dir))
        create_folder(str(json_dir))

        # Verify folders were created
        assert profile_dir.exists()
        assert merged_dir.exists()
        assert json_dir.exists()

    def test_workflow_cleanup(self, temp_dir):
        """Test workflow cleanup removes generated files."""
        from package.util.utils import remove_folder, create_folder
        
        # Create test folder
        test_folder = temp_dir / "test_cleanup"
        create_folder(str(test_folder))
        
        assert test_folder.exists()
        
        # Clean up
        remove_folder(str(test_folder))
        
        assert not test_folder.exists()

    def test_gcda_file_collection(self, temp_dir):
        """Test collecting .gcda files from build folder."""
        from package.oss.utils import get_gcda_files
        
        # Create mock build folder with .gcda files
        build_folder = temp_dir / "build"
        build_folder.mkdir()
        (build_folder / "test1.gcda").touch()
        (build_folder / "subdir").mkdir()
        (build_folder / "subdir" / "test2.gcda").touch()
        
        with patch("package.oss.utils.get_xsigma_folder", return_value=str(temp_dir)):
            with patch.dict("os.environ", {"XSIGMA_BUILD_FOLDER": "build"}):
                result = get_gcda_files()
                
                # Should find both .gcda files
                assert len(result) >= 2

    def test_json_export_workflow(self, temp_dir):
        """Test JSON export workflow."""
        from package.tool.summarize_jsons import get_json_obj
        
        # Create sample JSON file
        json_file = temp_dir / "coverage.json"
        sample_data = {
            "files": [
                {
                    "filename": "Library/Core/test.cpp",
                    "summary": {"lines": {"count": 100, "covered": 85}},
                }
            ]
        }
        
        with open(json_file, "w") as f:
            json.dump(sample_data, f)
        
        # Parse JSON
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert status == 0
        assert len(json_obj["files"]) == 1

    def test_report_generation_workflow(self, temp_dir):
        """Test report generation workflow."""
        from package.tool.print_report import is_intrested_file
        
        # Test file filtering
        library_file = "Library/Core/test.cpp"
        testing_file = "Testing/test_core.cpp"
        
        assert is_intrested_file(library_file, []) is True
        assert is_intrested_file(testing_file, []) is False


class TestMergeWorkflow:
    """Test profile merging workflow."""

    def test_merge_target_workflow(self, temp_dir):
        """Test merging raw profiles."""
        from package.tool.clang_coverage import merge_target
        from package.util.setting import TestPlatform
        
        raw_file = str(temp_dir / "test.profraw")
        merged_file = str(temp_dir / "test.merged")
        
        Path(raw_file).touch()
        
        with patch("subprocess.check_call") as mock_call:
            with patch("package.tool.clang_coverage.get_tool_path_by_platform", return_value="/usr/bin"):
                merge_target(raw_file, merged_file, TestPlatform.OSS)
                
                # Should call subprocess
                mock_call.assert_called_once()

    def test_merge_creates_output_folder(self, temp_dir):
        """Test that merge creates output folder if needed."""
        from package.util.utils import create_folder
        
        output_folder = temp_dir / "output" / "nested" / "folder"
        
        create_folder(str(output_folder))
        
        assert output_folder.exists()


class TestExportWorkflow:
    """Test coverage export workflow."""

    def test_export_target_workflow(self, temp_dir):
        """Test exporting coverage data."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform

        merged_file = str(temp_dir / "test.merged")
        json_file = str(temp_dir / "test.json")
        binary_file = str(temp_dir / "test_binary")

        Path(merged_file).touch()
        Path(binary_file).touch()

        with patch("subprocess.check_call") as mock_call:
            with patch("package.tool.clang_coverage.get_tool_path_by_platform", return_value="/usr/bin"):
                result = export_target(merged_file, json_file, binary_file, [], TestPlatform.OSS)

                assert result is True

    def test_export_with_shared_libraries(self, temp_dir):
        """Test exporting with shared libraries."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform

        merged_file = str(temp_dir / "test.merged")
        json_file = str(temp_dir / "test.json")
        binary_file = str(temp_dir / "test_binary")
        shared_libs = [str(temp_dir / "lib1.so"), str(temp_dir / "lib2.so")]

        Path(merged_file).touch()
        Path(binary_file).touch()

        with patch("subprocess.check_call") as mock_call:
            with patch("package.tool.clang_coverage.get_tool_path_by_platform", return_value="/usr/bin"):
                result = export_target(merged_file, json_file, binary_file, shared_libs, TestPlatform.OSS)

                assert result is True


class TestFilteringWorkflow:
    """Test file filtering in workflow."""

    def test_filter_interested_files(self):
        """Test filtering files by interested folders."""
        from package.tool.coverage_filters import is_interested_file
        
        interested_folders = ["Core", "Util"]
        
        # Should include
        assert is_interested_file("Library/Core/test.cpp", interested_folders) is True
        assert is_interested_file("Library/Util/utils.cpp", interested_folders) is True
        
        # Should exclude
        assert is_interested_file("Library/Other/other.cpp", interested_folders) is False
        assert is_interested_file("Testing/test.cpp", interested_folders) is False

    def test_filter_excluded_patterns(self):
        """Test filtering excluded patterns."""
        from package.tool.coverage_filters import is_interested_file

        # Should exclude
        assert is_interested_file("Testing/test.cpp", []) is False
        assert is_interested_file("Library/cuda/kernel.cu", []) is False
        assert is_interested_file("Library/ThirdParty/fmt/format.cpp", []) is False
        assert is_interested_file("build/CMakeFiles/test.cpp", []) is False


class TestErrorRecovery:
    """Test error recovery in workflow."""

    def test_workflow_handles_missing_binary(self, temp_dir):
        """Test workflow handles missing binary gracefully."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform

        merged_file = str(temp_dir / "test.merged")
        json_file = str(temp_dir / "test.json")
        binary_file = None

        Path(merged_file).touch()

        result = export_target(merged_file, json_file, binary_file, [], TestPlatform.OSS)

        # Should return False, not raise exception
        assert result is False

    def test_workflow_handles_subprocess_error(self, temp_dir):
        """Test workflow handles subprocess errors gracefully."""
        from package.tool.clang_coverage import export_target
        from package.util.setting import TestPlatform
        import subprocess

        merged_file = str(temp_dir / "test.merged")
        json_file = str(temp_dir / "test.json")
        binary_file = str(temp_dir / "test_binary")

        Path(merged_file).touch()
        Path(binary_file).touch()

        with patch("subprocess.check_call") as mock_call:
            mock_call.side_effect = subprocess.CalledProcessError(1, "llvm-cov")
            with patch("package.tool.clang_coverage.get_tool_path_by_platform", return_value="/usr/bin"):
                result = export_target(merged_file, json_file, binary_file, [], TestPlatform.OSS)

                # Should return False, not raise exception
                assert result is False

    def test_workflow_handles_file_errors(self, temp_dir):
        """Test workflow handles file errors gracefully."""
        from package.oss.utils import run_oss_python_test

        binary_file = "/nonexistent/test.py"

        with patch("subprocess.check_call") as mock_call:
            mock_call.side_effect = FileNotFoundError()
            result = run_oss_python_test(binary_file, "build")

            # Should return False, not raise exception
            assert result is False

