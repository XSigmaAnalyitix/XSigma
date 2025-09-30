#!/usr/bin/env python3
"""
Test script to verify setup.py integration with analyze_coverage.py

This script tests various usage patterns of the analyze command.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nExit Code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Run integration tests."""
    script_dir = Path(__file__).parent
    
    print("="*80)
    print("Coverage Analysis Integration Tests")
    print("="*80)
    
    tests = [
        {
            "cmd": [sys.executable, "setup.py", "--help"],
            "description": "Help command shows analyze option",
            "cwd": script_dir,
        },
        {
            "cmd": [sys.executable, "setup.py", "analyze"],
            "description": "Standalone analyze command",
            "cwd": script_dir,
        },
    ]
    
    results = []
    for test in tests:
        success = run_command(test["cmd"], test["description"])
        results.append((test["description"], success))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {description}")
    
    # Overall result
    all_passed = all(success for _, success in results)
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())

