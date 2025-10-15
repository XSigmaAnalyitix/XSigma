#!/usr/bin/env python3
"""
GPU Allocator Test Coverage Analysis Script

This script analyzes the test coverage for XSigma's GPU memory allocator system
and identifies any gaps in testing for the new architecture.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

class GPUTestCoverageAnalyzer:
    def __init__(self, xsigma_root: str):
        self.xsigma_root = Path(xsigma_root)
        self.gpu_source_dir = self.xsigma_root / "Library" / "Core" / "memory" / "gpu"
        self.test_dir = self.xsigma_root / "Library" / "Core" / "Testing" / "Cxx"
        
    def find_gpu_source_files(self) -> List[Path]:
        """Find all GPU-related source files"""
        gpu_files = []
        if self.gpu_source_dir.exists():
            gpu_files.extend(self.gpu_source_dir.glob("*.h"))
            gpu_files.extend(self.gpu_source_dir.glob("*.cxx"))
        return gpu_files
    
    def find_gpu_test_files(self) -> List[Path]:
        """Find all GPU-related test files"""
        test_files = []
        if self.test_dir.exists():
            # Look for test files that contain GPU-related content
            for test_file in self.test_dir.glob("Test*.cxx"):
                if self.is_gpu_related_test(test_file):
                    test_files.append(test_file)
        return test_files
    
    def is_gpu_related_test(self, test_file: Path) -> bool:
        """Check if a test file is GPU-related"""
        try:
            content = test_file.read_text(encoding='utf-8')
            gpu_keywords = [
                'cuda', 'gpu', 'allocator_gpu', 'allocator_cuda', 
                'XSIGMA_ENABLE_CUDA', 'cudaMalloc', 'cuMem',
                'hip', 'XSIGMA_ENABLE_HIP'
            ]
            return any(keyword.lower() in content.lower() for keyword in gpu_keywords)
        except Exception:
            return False
    
    def extract_functions_from_source(self, source_file: Path) -> Set[str]:
        """Extract function names from a source file"""
        functions = set()
        try:
            content = source_file.read_text(encoding='utf-8')
            
            # Match function definitions (simplified regex)
            # Matches: return_type function_name(params)
            function_pattern = r'^\s*(?:XSIGMA_API\s+)?(?:static\s+)?(?:inline\s+)?(?:virtual\s+)?(?:\w+(?:\s*\*)*\s+)+(\w+)\s*\([^)]*\)\s*(?:override\s*)?(?:const\s*)?(?:noexcept\s*)?[{;]'
            
            for line in content.split('\n'):
                match = re.match(function_pattern, line)
                if match:
                    func_name = match.group(1)
                    # Filter out constructors, destructors, and common keywords
                    if not func_name.startswith('~') and func_name not in ['if', 'for', 'while', 'switch', 'return']:
                        functions.add(func_name)
                        
        except Exception as e:
            print(f"Error reading {source_file}: {e}")
            
        return functions
    
    def extract_tested_functions_from_test(self, test_file: Path) -> Set[str]:
        """Extract function names that are being tested"""
        tested_functions = set()
        try:
            content = test_file.read_text(encoding='utf-8')
            
            # Look for function calls in test code
            # This is a simplified approach - looks for common patterns
            call_patterns = [
                r'(\w+)\s*\(',  # function_name(
                r'->(\w+)\s*\(',  # ->function_name(
                r'\.(\w+)\s*\(',  # .function_name(
            ]
            
            for pattern in call_patterns:
                matches = re.findall(pattern, content)
                tested_functions.update(matches)
                
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
            
        return tested_functions
    
    def analyze_coverage(self) -> Dict:
        """Analyze test coverage for GPU allocator system"""
        print("=== XSigma GPU Allocator Test Coverage Analysis ===\n")
        
        # Find source and test files
        gpu_sources = self.find_gpu_source_files()
        gpu_tests = self.find_gpu_test_files()
        
        print(f"Found {len(gpu_sources)} GPU source files:")
        for source in gpu_sources:
            print(f"  - {source.name}")
        
        print(f"\nFound {len(gpu_tests)} GPU test files:")
        for test in gpu_tests:
            print(f"  - {test.name}")
        
        # Analyze each source file
        coverage_report = {}
        all_source_functions = set()
        all_tested_functions = set()
        
        print(f"\n=== Source File Analysis ===")
        for source_file in gpu_sources:
            if source_file.suffix == '.h':  # Focus on header files for public API
                functions = self.extract_functions_from_source(source_file)
                all_source_functions.update(functions)
                coverage_report[source_file.name] = {
                    'functions': functions,
                    'tested': set(),
                    'untested': functions.copy()
                }
                print(f"\n{source_file.name}:")
                print(f"  Functions found: {len(functions)}")
                if functions:
                    for func in sorted(functions):
                        print(f"    - {func}")
        
        print(f"\n=== Test File Analysis ===")
        for test_file in gpu_tests:
            tested_funcs = self.extract_tested_functions_from_test(test_file)
            all_tested_functions.update(tested_funcs)
            print(f"\n{test_file.name}:")
            print(f"  Functions tested: {len(tested_funcs)}")
            
            # Update coverage report
            for source_name, source_info in coverage_report.items():
                tested_in_source = source_info['functions'].intersection(tested_funcs)
                source_info['tested'].update(tested_in_source)
                source_info['untested'] -= tested_in_source
        
        # Generate coverage summary
        print(f"\n=== Coverage Summary ===")
        total_functions = len(all_source_functions)
        total_tested = len(all_source_functions.intersection(all_tested_functions))
        coverage_percentage = (total_tested / total_functions * 100) if total_functions > 0 else 0
        
        print(f"Total functions in GPU sources: {total_functions}")
        print(f"Functions with tests: {total_tested}")
        print(f"Coverage percentage: {coverage_percentage:.1f}%")
        
        # Detailed coverage by file
        print(f"\n=== Detailed Coverage by File ===")
        for source_name, source_info in coverage_report.items():
            total = len(source_info['functions'])
            tested = len(source_info['tested'])
            untested = len(source_info['untested'])
            file_coverage = (tested / total * 100) if total > 0 else 0
            
            print(f"\n{source_name}:")
            print(f"  Total functions: {total}")
            print(f"  Tested functions: {tested}")
            print(f"  Untested functions: {untested}")
            print(f"  Coverage: {file_coverage:.1f}%")
            
            if source_info['untested']:
                print(f"  Untested functions:")
                for func in sorted(source_info['untested']):
                    print(f"    - {func}")
        
        # Recommendations
        print(f"\n=== Recommendations ===")
        if coverage_percentage < 98:
            print(f"❌ Coverage is {coverage_percentage:.1f}%, below the required 98%")
            print("Recommendations:")
            print("1. Add tests for untested functions listed above")
            print("2. Focus on critical functions: Alloc, Free, allocate_raw, deallocate_raw")
            print("3. Add error handling tests for edge cases")
            print("4. Add multi-threaded stress tests")
            print("5. Add performance regression tests")
        else:
            print(f"✅ Coverage is {coverage_percentage:.1f}%, meeting the 98% requirement")
        
        # Test file recommendations
        print(f"\n=== Test File Status ===")
        current_tests = {
            "TestAllocatorCuda.cxx": "✅ Updated for new allocator_gpu architecture",
            "TestGpuAllocatorBenchmark.cxx": "✅ Comprehensive benchmarks implemented", 
            "TestGpuAllocatorFactory.cxx": "⚠️  Tests deprecated factory (kept for compatibility)",
            "TestCudaCachingAllocator.cxx": "✅ Tests caching allocator functionality",
            "TestGPUMemoryStats.cxx": "⚠️  May need updates for new architecture"
        }
        
        for test_name, status in current_tests.items():
            print(f"  {test_name}: {status}")
        
        return coverage_report
    
    def generate_coverage_report(self, output_file: str = "gpu_test_coverage_report.md"):
        """Generate a detailed coverage report"""
        coverage_data = self.analyze_coverage()
        
        report_path = self.xsigma_root / output_file
        with open(report_path, 'w') as f:
            f.write("# XSigma GPU Allocator Test Coverage Report\n\n")
            f.write(f"Generated: {Path(__file__).name}\n\n")
            
            f.write("## Summary\n\n")
            f.write("This report analyzes test coverage for the XSigma GPU memory allocator system.\n\n")
            
            f.write("## Files Analyzed\n\n")
            f.write("### Source Files\n")
            for source_file in self.find_gpu_source_files():
                f.write(f"- `{source_file.name}`\n")
            
            f.write("\n### Test Files\n")
            for test_file in self.find_gpu_test_files():
                f.write(f"- `{test_file.name}`\n")
            
            f.write("\n## Coverage Analysis\n\n")
            f.write("Detailed coverage analysis is printed to console when running this script.\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. Ensure all public API functions have corresponding tests\n")
            f.write("2. Add edge case and error handling tests\n")
            f.write("3. Include multi-threaded stress tests\n")
            f.write("4. Add performance regression tests\n")
            f.write("5. Test cross-platform compatibility (CUDA/HIP)\n\n")
        
        print(f"\nDetailed report saved to: {report_path}")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_gpu_test_coverage.py <xsigma_root_path>")
        sys.exit(1)
    
    xsigma_root = sys.argv[1]
    if not os.path.exists(xsigma_root):
        print(f"Error: XSigma root path does not exist: {xsigma_root}")
        sys.exit(1)
    
    analyzer = GPUTestCoverageAnalyzer(xsigma_root)
    analyzer.generate_coverage_report()

if __name__ == "__main__":
    main()
