#!/usr/bin/env python3
"""
Simple GPU allocator benchmark runner for current build configuration.
This script runs the GPU allocator benchmarks without rebuilding.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def find_test_executable(xsigma_root: Path):
    """Find the XSigma test executable"""
    possible_paths = [
        xsigma_root / "build_ninja_python" / "Library" / "Core" / "Testing" / "Cxx" / "xsigmaTest",
        xsigma_root / "build_ninja_python" / "xsigmaTest",
        xsigma_root / "build" / "xsigmaTest",
        xsigma_root / "build" / "Library" / "Core" / "Testing" / "Cxx" / "xsigmaTest"
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def run_gpu_benchmarks(xsigma_root: str):
    """Run GPU allocator benchmarks with current build"""
    root_path = Path(xsigma_root)
    
    print("=== XSigma GPU Memory Allocator Benchmark Runner ===")
    print(f"XSigma Root: {root_path}")
    
    # Find test executable
    test_executable = find_test_executable(root_path)
    if not test_executable:
        print("❌ Error: Could not find test executable")
        print("Please build XSigma first:")
        print("  cd Scripts")
        print("  python setup.py config.ninja.clang.python.build.test")
        print("  python setup.py ninja.clang.python.build.test")
        return False
    
    print(f"✅ Found test executable: {test_executable}")
    
    # Check current allocation method
    print("\n=== Current Build Configuration ===")
    
    # Run benchmark tests
    benchmark_tests = [
        ("GpuAllocatorBenchmark.ComprehensiveBenchmark", "Comprehensive allocator comparison"),
        ("GpuAllocatorBenchmark.AllocationMethodComparison", "Current allocation method analysis")
    ]
    
    print(f"\n=== Running GPU Allocator Benchmarks ===")
    
    all_passed = True
    for test_filter, description in benchmark_tests:
        print(f"\n--- {description} ---")
        print(f"Running: {test_filter}")
        
        cmd = [
            str(test_executable),
            f"--gtest_filter={test_filter}",
            "--gtest_color=yes"
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, cwd=root_path, timeout=300)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✅ {test_filter} PASSED ({end_time - start_time:.1f}s)")
            else:
                print(f"❌ {test_filter} FAILED (return code: {result.returncode})")
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_filter} TIMED OUT (>300s)")
            all_passed = False
        except Exception as e:
            print(f"💥 {test_filter} ERROR: {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All GPU allocator benchmarks completed successfully!")
    else:
        print("⚠️  Some benchmarks failed. Check output above for details.")
    
    print("\n📊 Benchmark files generated:")
    report_files = [
        root_path / "gpu_allocator_benchmark_report.txt",
        root_path / "test_results.json"
    ]
    
    for report_file in report_files:
        if report_file.exists():
            print(f"  - {report_file}")
        else:
            print(f"  - {report_file} (not found)")
    
    print(f"\n💡 Tips:")
    print("  - Check the benchmark report files for detailed performance analysis")
    print("  - To test different allocation methods, rebuild with:")
    print("    -DXSIGMA_CUDA_ALLOC=SYNC|ASYNC|POOL_ASYNC")
    print("  - Use the comprehensive benchmark script for full method comparison")
    print(f"{'='*60}")
    
    return all_passed

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_gpu_benchmarks.py <xsigma_root_path>")
        print("\nExample:")
        print("  python run_gpu_benchmarks.py /path/to/xsigma")
        print("  python run_gpu_benchmarks.py .")
        sys.exit(1)
    
    xsigma_root = sys.argv[1]
    if not os.path.exists(xsigma_root):
        print(f"❌ Error: XSigma root path does not exist: {xsigma_root}")
        sys.exit(1)
    
    try:
        success = run_gpu_benchmarks(xsigma_root)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Benchmark failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
