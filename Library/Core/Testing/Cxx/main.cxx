/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include <iostream>
#include <exception>
#include "xsigmaTest.h"

// Forward declarations of all test functions
int TestAtomic(int argc, char* argv[]);
int TestCPUMemory(int argc, char* argv[]);
int TestCPUMemoryStats(int argc, char* argv[]);
int TestCPUinfo(int argc, char* argv[]);
int TestEnhancedProfiler(int argc, char* argv[]);
int TestException(int argc, char* argv[]);
int TestFMT(int argc, char* argv[]);
int TestGPUMemory(int argc, char* argv[]);
int TestGPUMemoryStats(int argc, char* argv[]);
int TestLogger(int argc, char* argv[]);
int TestLoggerThreadName(int argc, char* argv[]);
int TestMultiProcessStream(int argc, char* argv[]);
int TestProfiler(int argc, char* argv[]);
int TestSMP(int argc, char* argv[]);
int TestStringUtil(int argc, char* argv[]);
int TestTimerLog(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    int failed_tests = 0;
    int total_tests = 0;
    
    std::cout << "Running XSigma Core C++ Tests..." << std::endl;
    
    // Define test functions and their names
    struct TestInfo {
        const char* name;
        int (*func)(int, char**);
    };
    
    TestInfo tests[] = {
        {"Atomic", TestAtomic},
        {"CPUMemory", TestCPUMemory},
        {"CPUMemoryStats", TestCPUMemoryStats},
        {"CPUinfo", TestCPUinfo},
        {"EnhancedProfiler", TestEnhancedProfiler},
        {"Exception", TestException},
        {"FMT", TestFMT},
        {"GPUMemory", TestGPUMemory},
        {"GPUMemoryStats", TestGPUMemoryStats},
        {"Logger", TestLogger},
        {"LoggerThreadName", TestLoggerThreadName},
        {"MultiProcessStream", TestMultiProcessStream},
        {"Profiler", TestProfiler},
        {"SMP", TestSMP},
        {"StringUtil", TestStringUtil},
        {"TimerLog", TestTimerLog}
    };
    
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    for (int i = 0; i < num_tests; ++i) {
        total_tests++;
        std::cout << "Running test: " << tests[i].name << "..." << std::endl;
        
        try {
            int result = tests[i].func(argc, argv);
            if (result != 0) {
                std::cout << "FAILED: " << tests[i].name << " (returned " << result << ")" << std::endl;
                failed_tests++;
            } else {
                std::cout << "PASSED: " << tests[i].name << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << tests[i].name << " (exception: " << e.what() << ")" << std::endl;
            failed_tests++;
        } catch (...) {
            std::cout << "FAILED: " << tests[i].name << " (unknown exception)" << std::endl;
            failed_tests++;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Test Results:" << std::endl;
    std::cout << "  Total tests: " << total_tests << std::endl;
    std::cout << "  Passed: " << (total_tests - failed_tests) << std::endl;
    std::cout << "  Failed: " << failed_tests << std::endl;
    
    if (failed_tests == 0) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed." << std::endl;
        return 1;
    }
}
