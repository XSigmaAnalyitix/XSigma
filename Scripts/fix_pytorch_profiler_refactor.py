#!/usr/bin/env python3
"""
Fix issues in the pytorch_profiler refactoring.

This script:
1. Fixes incorrect include paths (removes profiler/pytorch_profiler prefix for local includes)
2. Reverts incorrect namespace replacements (c10:: should stay as c10::)
3. Ensures proper include ordering
"""

import re
from pathlib import Path
from typing import List

def fix_includes(file_path: Path) -> None:
    """Fix include paths to be relative to pytorch_profiler directory."""
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    original_content = content

    # Fix includes that were incorrectly prefixed with profiler/pytorch_profiler/
    # These should just be the filename since they're in the same directory
    content = re.sub(
        r'#include\s+"profiler/pytorch_profiler/([^/"]+)"',
        r'#include "\1"',
        content
    )

    # Fix nested includes that still have the old path structure
    content = re.sub(
        r'#include\s+"profiler/pytorch_profiler/unwind/([^"]+)"',
        r'#include "unwind/\1"',
        content
    )

    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"Fixed includes in: {file_path.name}")

def fix_namespaces(file_path: Path) -> None:
    """Revert incorrect namespace replacements."""
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    original_content = content

    # Revert c10:: replacements (should stay as c10::)
    content = re.sub(r'\bxsigma::GatheredContext\b', 'c10::GatheredContext', content)
    content = re.sub(r'\bxsigma::OperatorHandle\b', 'c10::OperatorHandle', content)

    # Fix enum class declarations that lost their visibility macro
    content = re.sub(
        r'enum class\s+(\s+)RecordScope',
        r'enum class RecordScope',
        content
    )

    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"Fixed namespaces in: {file_path.name}")

def get_all_files(base_dir: str) -> List[Path]:
    """Get all .h and .cxx files in pytorch_profiler."""
    base_path = Path(base_dir)
    files = []
    for ext in ['*.h', '*.cxx']:
        files.extend(base_path.glob(ext))
    return sorted(files)

def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    base_dir = repo_root / "Library/Core/profiler/pytroch_profiler"

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        return

    base_dir = str(base_dir)

    print("=" * 60)
    print("Fixing include paths...")
    print("=" * 60)
    files = get_all_files(base_dir)
    for file_path in files:
        fix_includes(file_path)

    print("\n" + "=" * 60)
    print("Fixing namespace replacements...")
    print("=" * 60)
    for file_path in files:
        fix_namespaces(file_path)

    print("\n" + "=" * 60)
    print("Fixes complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

