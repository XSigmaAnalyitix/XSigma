"""
Pytest configuration and fixtures for code coverage tests.

Provides common fixtures for:
- Temporary directories
- Mock file systems
- Mock subprocess calls
- Test data
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest


# Add package to path for imports
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests.
    
    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_xsigma_folder(temp_dir: Path) -> Path:
    """Create a mock XSigma project folder structure.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to mock XSigma folder
    """
    xsigma_folder = temp_dir / "XSigma"
    xsigma_folder.mkdir()
    
    # Create typical folder structure
    (xsigma_folder / "Library").mkdir()
    (xsigma_folder / "Library" / "Core").mkdir()
    (xsigma_folder / "Library" / "Util").mkdir()
    (xsigma_folder / "Testing").mkdir()
    (xsigma_folder / "ThirdParty").mkdir()
    
    return xsigma_folder


@pytest.fixture
def mock_build_folder(temp_dir: Path) -> Path:
    """Create a mock build folder with typical structure.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to mock build folder
    """
    build_folder = temp_dir / "build_ninja_python"
    build_folder.mkdir()
    
    # Create typical build structure
    (build_folder / "bin").mkdir()
    (build_folder / "lib").mkdir()
    (build_folder / "profile").mkdir()
    (build_folder / "profile" / "raw").mkdir()
    (build_folder / "profile" / "merged").mkdir()
    
    return build_folder


@pytest.fixture
def sample_json_file(temp_dir: Path) -> Path:
    """Create a sample JSON file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to sample JSON file
    """
    json_file = temp_dir / "sample.json"
    sample_data = {
        "files": [
            {
                "filename": "Library/Core/test.cpp",
                "summary": {"lines": {"count": 100, "covered": 85}},
            },
            {
                "filename": "Library/Util/utils.cpp",
                "summary": {"lines": {"count": 50, "covered": 45}},
            },
        ]
    }
    
    with open(json_file, "w") as f:
        json.dump(sample_data, f)
    
    return json_file


@pytest.fixture
def large_json_file(temp_dir: Path) -> Path:
    """Create a large JSON file for memory efficiency testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to large JSON file
    """
    json_file = temp_dir / "large.json"
    
    # Create a large JSON file with many entries
    with open(json_file, "w") as f:
        f.write('{"files": [\n')
        for i in range(1000):
            entry = {
                "filename": f"Library/Core/file_{i}.cpp",
                "summary": {"lines": {"count": 100, "covered": 85}},
            }
            f.write(json.dumps(entry))
            if i < 999:
                f.write(",\n")
        f.write("\n]}")
    
    return json_file


@pytest.fixture
def malformed_json_file(temp_dir: Path) -> Path:
    """Create a malformed JSON file for error testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to malformed JSON file
    """
    json_file = temp_dir / "malformed.json"
    
    with open(json_file, "w") as f:
        f.write('{"files": [{"filename": "test.cpp"')  # Missing closing braces
    
    return json_file


@pytest.fixture
def empty_json_file(temp_dir: Path) -> Path:
    """Create an empty JSON file for edge case testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to empty JSON file
    """
    json_file = temp_dir / "empty.json"
    json_file.touch()
    return json_file


@pytest.fixture
def mock_subprocess() -> Generator[MagicMock, None, None]:
    """Mock subprocess module for testing.
    
    Yields:
        Mock subprocess object
    """
    with patch("subprocess.check_call") as mock_call:
        mock_call.return_value = 0
        yield mock_call


@pytest.fixture
def mock_environment() -> Generator[Dict[str, str], None, None]:
    """Mock environment variables for testing.
    
    Yields:
        Dictionary of environment variables
    """
    original_env = os.environ.copy()
    
    # Set test environment
    os.environ["XSIGMA_BUILD_FOLDER"] = "build_ninja_python"
    os.environ["CXX"] = "clang++"
    os.environ["CC"] = "clang"
    
    yield os.environ.copy()
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_file_paths() -> Dict[str, str]:
    """Provide test file paths for filter testing.
    
    Returns:
        Dictionary of test file paths
    """
    return {
        "library_file": "Library/Core/test.cpp",
        "library_header": "Library/Core/test.h",
        "testing_file": "Testing/test_core.cpp",
        "thirdparty_file": "ThirdParty/fmt/format.cpp",
        "cuda_file": "Library/cuda/kernel.cu",
        "build_file": "build/CMakeFiles/test.cpp",
        "aten_file": "Library/aten/aten_ops.cpp",
        "windows_path": "Library\\Core\\test.cpp",
        "mixed_path": "Library/Core\\test.cpp",
    }


@pytest.fixture
def mock_platform_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock platform.system() to return Windows.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Windows")


@pytest.fixture
def mock_platform_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock platform.system() to return Linux.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")


@pytest.fixture
def mock_platform_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock platform.system() to return Darwin (macOS).
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

