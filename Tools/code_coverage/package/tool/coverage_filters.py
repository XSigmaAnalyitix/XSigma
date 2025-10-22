"""Shared coverage filtering utilities.

This module provides common filtering functions used across the coverage
reporting pipeline to determine which files should be included in coverage
reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..util.setting import TestPlatform


def is_interested_file(
    file_path: str,
    interested_folders: list[str],
    platform: TestPlatform | None = None,
) -> bool:
    """Check if a file should be included in coverage report.

    Filters out test files, build artifacts, and files outside interested
    folders. Handles cross-platform path separators.

    Args:
        file_path: Path to file to check
        interested_folders: List of folders to include (e.g., ["Library"])
        platform: Platform type (OSS or FBCODE). If OSS, also checks that
                 file belongs to XSigma folder.

    Returns:
        True if file should be included in coverage report, False otherwise
    """
    # Normalize path separators to forward slashes for consistent matching
    normalized_path = file_path.replace("\\", "/")

    # Patterns to exclude from coverage (test code, build artifacts, etc.)
    ignored_patterns = [
        "cuda",
        "aten/gen_aten",
        "aten/aten_",
        "build/",
        "Testing",  # Exclude XSigma Testing folder
        "ThirdParty",  # Exclude ThirdParty directory
        "/test/",   # Exclude test directories
        "/tests/",  # Exclude tests directories
    ]
    if any(pattern in normalized_path for pattern in ignored_patterns):
        return False

    # Check if file belongs to XSigma (for OSS platform)
    if platform is not None:
        from ..util.setting import TestPlatform
        if platform == TestPlatform.OSS:
            from ..oss.utils import get_xsigma_folder

            xsigma_folder = get_xsigma_folder().replace("\\", "/")
            if not normalized_path.startswith(xsigma_folder):
                return False

    # Check interested folders
    if interested_folders:
        for folder in interested_folders:
            # Normalize folder path to forward slashes
            normalized_folder = folder.replace("\\", "/")
            interested_folder_path = (
                normalized_folder
                if normalized_folder.endswith("/")
                else f"{normalized_folder}/"
            )
            if interested_folder_path in normalized_path:
                return True
        return False

    return True

