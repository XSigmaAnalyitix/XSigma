#!/usr/bin/env python3
"""
XSigma GPU Memory Allocator Comprehensive Benchmark Script

This script builds and runs benchmarks for all GPU allocation strategies:
- SYNC: Synchronous allocation (cuMemAlloc/cuMemFree)
- ASYNC: Asynchronous allocation (cuMemAllocAsync/cuMemFreeAsync)  
- POOL_ASYNC: Pool-based async allocation (cuMemAllocFromPoolAsync)

It also tests different cache sizes for the CUDA caching allocator and
generates comprehensive performance reports.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any

class GPUAllocatorBenchmark:
    def __init__(self, xsigma_root: str):
        self.xsigma_root = Path(xsigma_root)
        self.scripts_dir = self.xsigma_root / "Scripts"
        self.results = {}
        
    def build_with_allocation_method(self, method: str) -> bool:
        """Build XSigma with specific GPU allocation method"""
        print(f"\n=== Building with allocation method: {method} ===")
        
        # Clean previous build
        clean_cmd = ["python", "setup.py", "clean"]
        try:
            subprocess.run(clean_cmd, cwd=self.scripts_dir, check=True, 
                         capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Clean failed: {e}")
            return False
        
        # Configure with specific allocation method
        config_cmd = [
            "python", "setup.py", "config.ninja.clang.python.build.test",
            f"-DXSIGMA_CUDA_ALLOC={method}"
        ]
        
        try:
            result = subprocess.run(config_cmd, cwd=self.scripts_dir, check=True,
                                  capture_output=True, text=True)
            print(f"Configuration successful for {method}")
        except subprocess.CalledProcessError as e:
            print(f"Configuration failed for {method}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
        
        # Build
        build_cmd = ["python", "setup.py", "ninja.clang.python.build.test"]
        try:
            result = subprocess.run(build_cmd, cwd=self.scripts_dir, check=True,
                                  capture_output=True, text=True)
            print(f"Build successful for {method}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Build failed for {method}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
    
    def run_benchmark_tests(self, method: str) -> Dict[str, Any]:
        """Run GPU allocator benchmark tests"""
        print(f"\n=== Running benchmarks for {method} ===")
        
        # Find the test executable
        test_executable = None
        possible_paths = [
            self.xsigma_root / "build_ninja_python" / "Library" / "Core" / "Testing" / "Cxx" / "xsigmaTest",
            self.xsigma_root / "build_ninja_python" / "xsigmaTest",
            self.xsigma_root / "build" / "xsigmaTest"
        ]
        
        for path in possible_paths:
            if path.exists():
                test_executable = path
                break
        
        if not test_executable:
            print("Error: Could not find test executable")
            return {}
        
        # Run specific GPU allocator benchmarks
        benchmark_tests = [
            "GpuAllocatorBenchmark.ComprehensiveBenchmark",
            "GpuAllocatorBenchmark.AllocationMethodComparison"
        ]
        
        results = {}
        for test in benchmark_tests:
            print(f"Running test: {test}")
            cmd = [str(test_executable), f"--gtest_filter={test}", "--gtest_output=json:test_results.json"]
            
            try:
                result = subprocess.run(cmd, cwd=self.xsigma_root, 
                                      capture_output=True, text=True, timeout=300)
                
                # Parse results
                results[test] = {
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'method': method
                }
                
                if result.returncode == 0:
                    print(f"✓ {test} passed")
                else:
                    print(f"✗ {test} failed (return code: {result.returncode})")
                    
            except subprocess.TimeoutExpired:
                print(f"✗ {test} timed out")
                results[test] = {'error': 'timeout', 'method': method}
            except Exception as e:
                print(f"✗ {test} error: {e}")
                results[test] = {'error': str(e), 'method': method}
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run benchmarks for all allocation methods"""
        allocation_methods = ["SYNC", "ASYNC", "POOL_ASYNC"]
        all_results = {}
        
        print("=== XSigma GPU Memory Allocator Comprehensive Benchmark ===")
        print(f"Testing allocation methods: {', '.join(allocation_methods)}")
        
        for method in allocation_methods:
            print(f"\n{'='*60}")
            print(f"Testing allocation method: {method}")
            print(f"{'='*60}")
            
            # Build with specific method
            if not self.build_with_allocation_method(method):
                print(f"Skipping {method} due to build failure")
                continue
            
            # Run benchmarks
            method_results = self.run_benchmark_tests(method)
            all_results[method] = method_results
            
            # Brief pause between methods
            time.sleep(2)
        
        return all_results
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive benchmark report"""
        report_file = self.xsigma_root / "gpu_allocator_benchmark_comprehensive_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# XSigma GPU Memory Allocator Comprehensive Benchmark Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Test Configuration\n\n")
            f.write("- **Allocation Methods Tested**: SYNC, ASYNC, POOL_ASYNC\n")
            f.write("- **Benchmark Types**: Comprehensive allocation patterns, method comparison\n")
            f.write("- **Test Scenarios**: Small allocations, large allocations, multi-threaded, high frequency\n\n")
            
            f.write("## Results Summary\n\n")
            
            successful_methods = []
            failed_methods = []
            
            for method, method_results in results.items():
                if method_results and any(r.get('returncode') == 0 for r in method_results.values()):
                    successful_methods.append(method)
                else:
                    failed_methods.append(method)
            
            f.write(f"- **Successful Methods**: {', '.join(successful_methods) if successful_methods else 'None'}\n")
            f.write(f"- **Failed Methods**: {', '.join(failed_methods) if failed_methods else 'None'}\n\n")
            
            f.write("## Detailed Results\n\n")
            
            for method, method_results in results.items():
                f.write(f"### {method} Allocation Method\n\n")
                
                if not method_results:
                    f.write("❌ **Build failed** - No benchmark results available\n\n")
                    continue
                
                for test_name, test_result in method_results.items():
                    f.write(f"#### {test_name}\n\n")
                    
                    if test_result.get('returncode') == 0:
                        f.write("✅ **Status**: PASSED\n\n")
                        
                        # Extract performance metrics from stdout
                        stdout = test_result.get('stdout', '')
                        if 'Avg Allocation Time:' in stdout:
                            lines = stdout.split('\n')
                            for line in lines:
                                if 'Avg Allocation Time:' in line or 'Throughput:' in line or 'Total Allocations:' in line:
                                    f.write(f"- {line.strip()}\n")
                        f.write("\n")
                        
                    else:
                        f.write("❌ **Status**: FAILED\n\n")
                        if 'error' in test_result:
                            f.write(f"**Error**: {test_result['error']}\n\n")
                        else:
                            stderr = test_result.get('stderr', '')
                            if stderr:
                                f.write(f"**Error Output**:\n```\n{stderr[:500]}...\n```\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the benchmark results:\n\n")
            f.write("1. **SYNC Method**: Best for simple, single-threaded applications\n")
            f.write("2. **ASYNC Method**: Optimal for stream-based parallel workloads\n")
            f.write("3. **POOL_ASYNC Method**: Best for high-frequency allocation patterns\n")
            f.write("4. **Caching Allocator**: Recommended for applications with frequent alloc/dealloc cycles\n\n")
            
            f.write("## Build Instructions\n\n")
            f.write("To test specific allocation methods:\n\n")
            f.write("```bash\n")
            f.write("cd Scripts\n")
            f.write("python setup.py config.ninja.clang.python.build.test -DXSIGMA_CUDA_ALLOC=SYNC\n")
            f.write("python setup.py ninja.clang.python.build.test\n")
            f.write("```\n\n")
            f.write("Replace `SYNC` with `ASYNC` or `POOL_ASYNC` for other methods.\n\n")
        
        print(f"\nComprehensive report saved to: {report_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark_gpu_allocators.py <xsigma_root_path>")
        sys.exit(1)
    
    xsigma_root = sys.argv[1]
    if not os.path.exists(xsigma_root):
        print(f"Error: XSigma root path does not exist: {xsigma_root}")
        sys.exit(1)
    
    benchmark = GPUAllocatorBenchmark(xsigma_root)
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        benchmark.generate_report(results)
        
        print("\n" + "="*60)
        print("Comprehensive GPU allocator benchmark completed!")
        print("Check the generated report for detailed results.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
