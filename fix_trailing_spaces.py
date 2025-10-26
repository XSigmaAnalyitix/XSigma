#!/usr/bin/env python3
"""Fix trailing whitespace in files."""

import sys
from pathlib import Path


def fix_trailing_spaces(file_path: Path) -> bool:
    """Remove trailing whitespace from a file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Remove trailing whitespace from each line
        fixed_lines = [line.rstrip() + '\n' if line.endswith('\n') else line.rstrip() 
                       for line in lines]
        
        # Check if anything changed
        if lines == fixed_lines:
            return False
        
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    # Read list of files from /tmp/spaces_files.txt
    files_list = Path('/tmp/spaces_files.txt')
    
    if not files_list.exists():
        print("Error: /tmp/spaces_files.txt not found", file=sys.stderr)
        sys.exit(1)
    
    with open(files_list, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    fixed_count = 0
    for file_path_str in files:
        file_path = Path(file_path_str)
        if file_path.exists():
            if fix_trailing_spaces(file_path):
                print(f"Fixed: {file_path}")
                fixed_count += 1
        else:
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
    
    print(f"\nTotal files fixed: {fixed_count}/{len(files)}")


if __name__ == '__main__':
    main()

