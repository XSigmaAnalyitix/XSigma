#!/usr/bin/env python3
"""Fix quoted includes in C++ files by converting them to angle bracket includes."""

import re
import sys
from pathlib import Path


def fix_quoted_includes(file_path: Path) -> bool:
    """Convert quoted includes to angle bracket includes.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to match #include "xxx" but not system headers
        # We want to convert project includes from quotes to angle brackets
        # Match: #include "path/to/file.h"
        # Convert to: #include <path/to/file.h>
        
        # This regex matches #include "..." where the path doesn't start with /
        pattern = r'#include\s+"([^"]+)"'
        
        def replace_include(match):
            include_path = match.group(1)
            # Convert to angle brackets
            return f'#include <{include_path}>'
        
        content = re.sub(pattern, replace_include, content)
        
        # Check if anything changed
        if content == original_content:
            return False
        
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    # Read list of files from /tmp/include_files.txt
    files_list = Path('/tmp/include_files.txt')
    
    if not files_list.exists():
        print("Error: /tmp/include_files.txt not found", file=sys.stderr)
        sys.exit(1)
    
    with open(files_list, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    fixed_count = 0
    for file_path_str in files:
        file_path = Path(file_path_str)
        if file_path.exists():
            if fix_quoted_includes(file_path):
                print(f"Fixed: {file_path}")
                fixed_count += 1
        else:
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
    
    print(f"\nTotal files fixed: {fixed_count}/{len(files)}")


if __name__ == '__main__':
    main()

