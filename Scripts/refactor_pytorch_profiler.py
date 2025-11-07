#!/usr/bin/env python3
"""
Refactor pytorch_profiler directory structure and update includes/macros.

This script:
1. Flattens the directory structure by moving all .h and .cxx files to base pytorch_profiler
2. Updates include paths to follow XSigma conventions
3. Replaces PyTorch macros with XSigma equivalents
"""

import re
import shutil
from pathlib import Path
from typing import Dict, List

# Macro replacements: PyTorch -> XSigma
MACRO_REPLACEMENTS = {
    r'\bTORCH_API\b': 'XSIGMA_API',
    r'\bTORCH_PYTHON_API\b': 'XSIGMA_API',
    r'\bTORCH_INTERNAL_ASSERT\b': 'XSIGMA_CHECK',
    r'\bTORCH_INTERNAL_ASSERT_DEBUG_ONLY\b': 'XSIGMA_CHECK_DEBUG',
    r'\bTORCH_CHECK\b': 'XSIGMA_CHECK',
    r'\bC10_API_ENUM\b': '',  # Remove, use enum class directly
    r'\bC10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED\b': 'XSIGMA_DIAGNOSTIC_PUSH',
    r'\bC10_DIAGNOSTIC_POP\b': 'XSIGMA_DIAGNOSTIC_POP',
}

# Namespace replacements
NAMESPACE_REPLACEMENTS = {
    r'\baten::\b': 'xsigma::',
    r'\bc10::\b': 'xsigma::',
}

def get_all_files(base_dir: str) -> List[Path]:
    """Get all .h and .cxx files in pytorch_profiler."""
    base_path = Path(base_dir)
    files = []
    for ext in ['*.h', '*.cxx']:
        files.extend(base_path.rglob(ext))
    return sorted(files)

def flatten_structure(base_dir: str) -> Dict[str, Path]:
    """Move all files to base pytorch_profiler directory."""
    base_path = Path(base_dir)
    file_mapping = {}  # old_path -> new_path
    
    files = get_all_files(base_dir)
    for file_path in files:
        if file_path.parent == base_path:
            continue  # Already in base directory
        
        new_path = base_path / file_path.name
        if new_path.exists() and new_path != file_path:
            print(f"Warning: {new_path} already exists, skipping {file_path}")
            continue
        
        print(f"Moving: {file_path.relative_to(base_path)} -> {new_path.name}")
        shutil.move(str(file_path), str(new_path))
        file_mapping[str(file_path)] = str(new_path)
    
    return file_mapping

def update_includes(file_path: Path, file_mapping: Dict[str, Path]) -> None:
    """Update include paths in a file."""
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    original_content = content
    
    # Replace torch/csrc/profiler includes
    content = re.sub(
        r'#include\s+[<"]torch/csrc/profiler/([^>"]+)[>"]',
        r'#include "profiler/pytorch_profiler/\1"',
        content
    )
    
    # Replace aten includes
    content = re.sub(
        r'#include\s+[<"]aten/src/ATen/([^>"]+)[>"]',
        r'#include "profiler/pytorch_profiler/\1"',
        content
    )
    
    # Replace torch/autograd includes
    content = re.sub(
        r'#include\s+[<"]torch/autograd/([^>"]+)[>"]',
        r'#include "profiler/pytorch_profiler/\1"',
        content
    )
    
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"Updated includes in: {file_path.name}")

def replace_macros(file_path: Path) -> None:
    """Replace PyTorch macros with XSigma equivalents."""
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    original_content = content
    
    for torch_macro, xsigma_macro in MACRO_REPLACEMENTS.items():
        content = re.sub(torch_macro, xsigma_macro, content)
    
    for torch_ns, xsigma_ns in NAMESPACE_REPLACEMENTS.items():
        content = re.sub(torch_ns, xsigma_ns, content)
    
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"Updated macros in: {file_path.name}")

def add_visibility_macros(file_path: Path) -> None:
    """Add XSIGMA_VISIBILITY and XSIGMA_API macros."""
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    original_content = content
    
    # Add XSIGMA_VISIBILITY before class declarations
    content = re.sub(
        r'(\n)(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\s*([:{])',
        r'\1class XSIGMA_VISIBILITY \3 \4',
        content
    )
    
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"Added visibility macros in: {file_path.name}")

def main():
    # Get the repository root (parent of Scripts directory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    base_dir = repo_root / "Library/Core/profiler/pytroch_profiler"

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        return

    base_dir = str(base_dir)
    
    print("=" * 60)
    print("Step 1: Flattening directory structure...")
    print("=" * 60)
    file_mapping = flatten_structure(base_dir)
    
    print("\n" + "=" * 60)
    print("Step 2: Updating include paths...")
    print("=" * 60)
    files = get_all_files(base_dir)
    for file_path in files:
        update_includes(file_path, file_mapping)
    
    print("\n" + "=" * 60)
    print("Step 3: Replacing PyTorch macros...")
    print("=" * 60)
    for file_path in files:
        replace_macros(file_path)
    
    print("\n" + "=" * 60)
    print("Step 4: Adding visibility macros...")
    print("=" * 60)
    for file_path in files:
        add_visibility_macros(file_path)
    
    print("\n" + "=" * 60)
    print("Refactoring complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

