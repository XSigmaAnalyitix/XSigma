#!/usr/bin/env python
"""
XSigma Git Hooks Test Script
=============================

This script tests the git hooks to ensure they work correctly.

Tests:
1. Pre-commit hook: Formats C++ files
2. Commit-msg hook: Validates commit messages

Cross-platform compatible: Works on Windows, Linux, and macOS.

Usage:
    python test_git_hooks.py

Author: XSigma Development Team
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional


# ============================================================================
# Utility Functions
# ============================================================================

def print_error(message: str) -> None:
    """Print error message in red."""
    print(f"\033[91m✗ {message}\033[0m", file=sys.stderr)


def print_success(message: str) -> None:
    """Print success message in green."""
    print(f"\033[92m✓ {message}\033[0m")


def print_info(message: str) -> None:
    """Print info message in blue."""
    print(f"\033[94mℹ {message}\033[0m")


def print_header(message: str) -> None:
    """Print header message."""
    print()
    print("=" * 70)
    print(f"  {message}")
    print("=" * 70)
    print()


def run_command(cmd: list, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """
    Run a command and return exit code, stdout, and stderr.
    
    Args:
        cmd: Command and arguments as a list
        cwd: Working directory (optional)
    
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def get_repo_root() -> Optional[Path]:
    """
    Get the root directory of the git repository.
    
    Returns:
        Path to repository root or None if not in a git repository
    """
    exit_code, stdout, stderr = run_command(['git', 'rev-parse', '--show-toplevel'])
    
    if exit_code != 0:
        print_error(f"Not in a git repository: {stderr}")
        return None
    
    return Path(stdout.strip())


# ============================================================================
# Test Functions
# ============================================================================

def test_pre_commit_hook(repo_root: Path) -> bool:
    """
    Test the pre-commit hook.
    
    Args:
        repo_root: Path to repository root
    
    Returns:
        True if test passed
    """
    print_header("Testing Pre-Commit Hook")
    
    # Create a temporary C++ file with bad formatting
    test_file = repo_root / 'test_formatting.cxx'
    
    unformatted_code = """#include <iostream>
int main(){int x=5;int y=10;
if(x<y){std::cout<<"x is less than y"<<std::endl;}
return 0;}
"""
    
    try:
        # Write unformatted code
        print_info(f"Creating test file: {test_file.name}")
        with open(test_file, 'w') as f:
            f.write(unformatted_code)
        
        # Stage the file
        print_info("Staging test file...")
        exit_code, _, stderr = run_command(['git', 'add', str(test_file)], cwd=str(repo_root))
        if exit_code != 0:
            print_error(f"Failed to stage file: {stderr}")
            return False
        
        # Run pre-commit hook manually
        print_info("Running pre-commit hook...")
        hook_path = repo_root / '.git' / 'hooks' / 'pre-commit'
        
        if sys.platform == 'win32':
            exit_code, stdout, stderr = run_command(['python', str(hook_path)], cwd=str(repo_root))
        else:
            exit_code, stdout, stderr = run_command([str(hook_path)], cwd=str(repo_root))
        
        print()
        print("Hook output:")
        print(stdout)
        if stderr:
            print("Errors:")
            print(stderr)
        print()
        
        if exit_code != 0:
            print_error("Pre-commit hook failed")
            return False
        
        # Check if file was formatted
        with open(test_file, 'r') as f:
            formatted_code = f.read()
        
        if formatted_code == unformatted_code:
            print_error("File was not formatted")
            return False
        
        print_success("Pre-commit hook test passed!")
        print_info("File was successfully formatted")
        
        return True
    
    except Exception as e:
        print_error(f"Test failed with exception: {e}")
        return False
    
    finally:
        # Clean up
        print_info("Cleaning up test file...")
        if test_file.exists():
            # Unstage and remove
            run_command(['git', 'reset', 'HEAD', str(test_file)], cwd=str(repo_root))
            test_file.unlink()


def test_commit_msg_hook(repo_root: Path) -> bool:
    """
    Test the commit-msg hook.
    
    Args:
        repo_root: Path to repository root
    
    Returns:
        True if test passed
    """
    print_header("Testing Commit-Msg Hook")
    
    hook_path = repo_root / '.git' / 'hooks' / 'commit-msg'
    
    # Test cases: (message, should_pass, description)
    test_cases = [
        ("", False, "Empty message"),
        ("   ", False, "Whitespace only"),
        ("wip", False, "Placeholder message"),
        ("test", False, "Too short"),
        ("Fix teh bug", False, "Spelling mistake"),
        ("Add new feature for vector normalization", True, "Valid message"),
        ("Implement matrix multiplication algorithm", True, "Valid message"),
    ]
    
    all_passed = True
    
    for message, should_pass, description in test_cases:
        print_info(f"Testing: {description}")
        print(f"  Message: '{message}'")
        
        # Create temporary commit message file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(message)
            temp_file = f.name
        
        try:
            # Run commit-msg hook
            if sys.platform == 'win32':
                exit_code, stdout, stderr = run_command(
                    ['python', str(hook_path), temp_file],
                    cwd=str(repo_root)
                )
            else:
                exit_code, stdout, stderr = run_command(
                    [str(hook_path), temp_file],
                    cwd=str(repo_root)
                )
            
            passed = (exit_code == 0) == should_pass
            
            if passed:
                print_success(f"  Result: {'Passed' if exit_code == 0 else 'Failed'} (as expected)")
            else:
                print_error(f"  Result: {'Passed' if exit_code == 0 else 'Failed'} (expected {'pass' if should_pass else 'fail'})")
                all_passed = False
            
            if stderr and not passed:
                print(f"  Output: {stderr[:200]}")
            
            print()
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    if all_passed:
        print_success("All commit-msg hook tests passed!")
    else:
        print_error("Some commit-msg hook tests failed")
    
    return all_passed


def test_hook_installation(repo_root: Path) -> bool:
    """
    Test that hooks are properly installed.
    
    Args:
        repo_root: Path to repository root
    
    Returns:
        True if hooks are installed
    """
    print_header("Testing Hook Installation")
    
    hooks_dir = repo_root / '.git' / 'hooks'
    required_hooks = ['pre-commit', 'commit-msg']
    
    all_installed = True
    
    for hook_name in required_hooks:
        hook_path = hooks_dir / hook_name
        
        if not hook_path.exists():
            print_error(f"Hook not found: {hook_name}")
            all_installed = False
            continue
        
        # Check if executable (on Unix)
        if sys.platform != 'win32':
            if not os.access(hook_path, os.X_OK):
                print_error(f"Hook not executable: {hook_name}")
                all_installed = False
                continue
        
        print_success(f"Hook installed: {hook_name}")
    
    return all_installed


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """
    Main function.
    
    Returns:
        0 if all tests passed, 1 if any test failed
    """
    print_header("XSigma Git Hooks Test Suite")
    
    # Get repository root
    repo_root = get_repo_root()
    if not repo_root:
        return 1
    
    print_info(f"Repository root: {repo_root}")
    print()
    
    # Run tests
    tests = [
        ("Hook Installation", lambda: test_hook_installation(repo_root)),
        ("Pre-Commit Hook", lambda: test_pre_commit_hook(repo_root)),
        ("Commit-Msg Hook", lambda: test_commit_msg_hook(repo_root)),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("Test Summary")
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print_success("All tests passed!")
        return 0
    else:
        print_error("Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

