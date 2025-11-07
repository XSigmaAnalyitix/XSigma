#!/usr/bin/env python3
"""
Final fix for include paths in pytorch_profiler files.

This script ensures all includes are properly formatted as relative paths
within the pytorch_profiler directory.
"""

import re
from pathlib import Path
from typing import List

def fix_all_includes(file_path: Path) -> bool:
    """Fix all include paths to be relative within pytorch_profiler."""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    original_content = content

    # Fix includes that still have profiler/pytorch_profiler/ prefix
    # These should just be the filename or relative path
    content = re.sub(
        r'#include\s+"profiler/pytorch_profiler/([^/"]+)"',
        r'#include "\1"',
        content
    )

    # Fix nested includes like profiler/pytorch_profiler/orchestration/
    content = re.sub(
        r'#include\s+"profiler/pytorch_profiler/([^"]+)"',
        r'#include "\1"',
        content
    )

    if content != original_content:
        try:
            file_path.write_text(content, encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False

    return False

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
    print("Final fix for include paths...")
    print("=" * 60)
    files = get_all_files(base_dir)
    updated_count = 0
    for file_path in files:
        if fix_all_includes(file_path):
            print(f"Fixed: {file_path.name}")
            updated_count += 1

    print("\n" + "=" * 60)
    print(f"Fixed {updated_count} files")
    print("=" * 60)

if __name__ == "__main__":
    main()

