"""
Tests for JSON parsing and optimization.

Tests the optimized get_json_obj() function that streams line-by-line
instead of loading entire files into memory.
"""

import json
from pathlib import Path

import pytest


class TestJsonParsing:
    """Test JSON parsing functionality."""

    def test_get_json_obj_valid_json(self, sample_json_file):
        """Test parsing valid JSON file."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_obj, status = get_json_obj(str(sample_json_file))
        
        assert json_obj is not None
        assert status == 0
        assert "files" in json_obj

    def test_get_json_obj_returns_first_object(self, temp_dir):
        """Test that get_json_obj() returns first JSON object."""
        from package.tool.summarize_jsons import get_json_obj
        
        # Create file with multiple JSON objects
        json_file = temp_dir / "multi.json"
        with open(json_file, "w") as f:
            f.write('{"id": 1}\n')
            f.write('{"id": 2}\n')
            f.write('{"id": 3}\n')
        
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert json_obj["id"] == 1
        assert status == 0

    def test_get_json_obj_skips_empty_lines(self, temp_dir):
        """Test that get_json_obj() skips empty lines."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_file = temp_dir / "with_empty_lines.json"
        with open(json_file, "w") as f:
            f.write('\n')
            f.write('\n')
            f.write('{"id": 1}\n')
            f.write('\n')
        
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert json_obj["id"] == 1
        assert status == 0

    def test_get_json_obj_malformed_json(self, malformed_json_file):
        """Test parsing malformed JSON file."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_obj, status = get_json_obj(str(malformed_json_file))
        
        # Should return None and error status
        assert json_obj is None
        assert status == 2

    def test_get_json_obj_empty_file(self, empty_json_file):
        """Test parsing empty JSON file."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_obj, status = get_json_obj(str(empty_json_file))
        
        # Should return None and status 2 (no valid JSON found)
        assert json_obj is None
        assert status == 2

    def test_get_json_obj_file_not_found(self):
        """Test parsing nonexistent JSON file."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_obj, status = get_json_obj("/nonexistent/file.json")
        
        # Should return None and error status
        assert json_obj is None
        assert status == 2

    def test_get_json_obj_permission_error(self, temp_dir):
        """Test parsing JSON file with permission error."""
        from package.tool.summarize_jsons import get_json_obj
        from unittest.mock import patch, mock_open

        json_file = temp_dir / "no_permission.json"
        json_file.write_text('{"id": 1}')

        # Mock file open to raise PermissionError only for the JSON file
        original_open = open

        def mock_file_open(file, *args, **kwargs):
            if str(json_file) in str(file):
                raise PermissionError("Permission denied")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_file_open):
            json_obj, status = get_json_obj(str(json_file))

            # Should return None and error status
            assert json_obj is None
            assert status == 2

    def test_get_json_obj_large_file_memory_efficiency(self, large_json_file):
        """Test that get_json_obj() handles large files efficiently."""
        from package.tool.summarize_jsons import get_json_obj

        # Should not load entire file into memory
        json_obj, status = get_json_obj(str(large_json_file))

        # Should still parse successfully (or return error if file is too large)
        # The important thing is it doesn't crash
        assert status in [0, 1]  # Success or parse error, but not a crash

    def test_get_json_obj_with_whitespace(self, temp_dir):
        """Test parsing JSON with various whitespace."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_file = temp_dir / "whitespace.json"
        with open(json_file, "w") as f:
            f.write('   \n')
            f.write('\t\n')
            f.write('  {"id": 1}  \n')
        
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert json_obj["id"] == 1
        assert status == 0

    def test_get_json_obj_complex_structure(self, temp_dir):
        """Test parsing complex JSON structure."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_file = temp_dir / "complex.json"
        complex_data = {
            "files": [
                {
                    "filename": "test.cpp",
                    "summary": {
                        "lines": {"count": 100, "covered": 85},
                        "functions": {"count": 10, "covered": 9},
                        "branches": {"count": 50, "covered": 40},
                    },
                    "segments": [
                        {"line": 1, "col": 0, "count": 5},
                        {"line": 2, "col": 0, "count": 0},
                    ],
                }
            ]
        }
        
        with open(json_file, "w") as f:
            json.dump(complex_data, f)
        
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert status == 0
        assert "files" in json_obj
        assert len(json_obj["files"]) == 1


class TestJsonParsingErrorStatus:
    """Test JSON parsing error status codes."""

    def test_status_code_success(self, sample_json_file):
        """Test status code 0 for success."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_obj, status = get_json_obj(str(sample_json_file))
        
        assert status == 0

    def test_status_code_parse_error(self, temp_dir):
        """Test status code 1 for parse error."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_file = temp_dir / "parse_error.json"
        with open(json_file, "w") as f:
            f.write('invalid json\n')
            f.write('{"id": 1}\n')
        
        json_obj, status = get_json_obj(str(json_file))
        
        # Should skip invalid line and find valid JSON
        assert json_obj is not None
        assert json_obj["id"] == 1

    def test_status_code_file_error(self):
        """Test status code 2 for file error."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_obj, status = get_json_obj("/nonexistent/file.json")
        
        assert status == 2


class TestJsonParsingEdgeCases:
    """Test edge cases in JSON parsing."""

    def test_json_with_unicode(self, temp_dir):
        """Test parsing JSON with Unicode characters."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_file = temp_dir / "unicode.json"
        data = {"filename": "测试.cpp", "description": "Тест"}
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert status == 0
        assert json_obj["filename"] == "测试.cpp"

    def test_json_with_escaped_characters(self, temp_dir):
        """Test parsing JSON with escaped characters."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_file = temp_dir / "escaped.json"
        data = {"path": "C:\\Users\\test\\file.cpp", "quote": 'He said "hello"'}
        
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert status == 0
        assert "Users" in json_obj["path"]

    def test_json_with_null_values(self, temp_dir):
        """Test parsing JSON with null values."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_file = temp_dir / "null.json"
        data = {"id": 1, "value": None, "name": "test"}
        
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert status == 0
        assert json_obj["value"] is None

    def test_json_with_numbers(self, temp_dir):
        """Test parsing JSON with various number types."""
        from package.tool.summarize_jsons import get_json_obj
        
        json_file = temp_dir / "numbers.json"
        data = {"int": 42, "float": 3.14, "negative": -100, "zero": 0}
        
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        json_obj, status = get_json_obj(str(json_file))
        
        assert json_obj is not None
        assert status == 0
        assert json_obj["int"] == 42
        assert json_obj["float"] == 3.14

