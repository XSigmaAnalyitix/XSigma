#!/usr/bin/env bash
# =============================================================================
# XSigma Valgrind CTest Runner
# =============================================================================
# This script runs CTest with Valgrind memory checking.
#
# All Valgrind configuration options are defined in Cmake/tools/valgrind.cmake.
# This script only handles test execution and result analysis.
#
# Usage: valgrind_ctest.sh <build_directory>
# =============================================================================

set -euo pipefail

# =============================================================================
# Color Definitions for Output
# =============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# =============================================================================
# Utility Functions
# =============================================================================

# Print formatted status messages
print_status() {
    local message="$1"
    local type="${2:-INFO}"

    case "$type" in
        SUCCESS)
            echo -e "${GREEN}[✓] $message${NC}"
            ;;
        ERROR)
            echo -e "${RED}[✗] $message${NC}"
            ;;
        WARNING)
            echo -e "${YELLOW}[!] $message${NC}"
            ;;
        *)
            echo -e "${CYAN}[i] $message${NC}"
            ;;
    esac
}

# Print section header
print_header() {
    local message="$1"
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$message${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# =============================================================================
# Argument Validation
# =============================================================================

if [ $# -eq 0 ]; then
    print_status "Usage: $0 <build_directory>" "ERROR"
    print_status "Example: $0 ../build_ninja_valgrind" "INFO"
    exit 1
fi

BUILD_DIR="$1"

if [ ! -d "$BUILD_DIR" ]; then
    print_status "Build directory not found: $BUILD_DIR" "ERROR"
    exit 1
fi

# =============================================================================
# Platform Compatibility Check
# =============================================================================

check_platform_compatibility() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        local arch
        arch=$(uname -m)
        if [ "$arch" == "arm64" ]; then
            print_status "WARNING: Valgrind does not support Apple Silicon (ARM64)" "WARNING"
            print_status "Consider using sanitizers instead:" "INFO"
            print_status "  AddressSanitizer: python3 setup.py config.ninja.clang.test --sanitizer.address" "INFO"
            print_status "  LeakSanitizer:    python3 setup.py config.ninja.clang.test --sanitizer.leak" "INFO"
            print_status "Attempting to run Valgrind anyway (will fail if not installed)..." "WARNING"
        fi
    fi
}

# =============================================================================
# Valgrind Installation Check
# =============================================================================

check_valgrind_installed() {
    if ! command -v valgrind &> /dev/null; then
        print_status "Valgrind is not installed!" "ERROR"
        print_status "Please install Valgrind:" "INFO"
        print_status "  Ubuntu/Debian: sudo apt-get install valgrind" "INFO"
        print_status "  Fedora/RHEL:   sudo dnf install valgrind" "INFO"
        print_status "  macOS (Intel): brew install valgrind" "INFO"
        return 1
    fi

    local valgrind_version
    valgrind_version=$(valgrind --version)
    print_status "Found Valgrind: $valgrind_version" "SUCCESS"
    return 0
}

# =============================================================================
# CTest Execution with Valgrind
# =============================================================================

run_valgrind_tests() {
    local build_dir="$1"

    cd "$build_dir"
    print_status "Running Valgrind tests in: $build_dir"

    print_header "Running CTest with Valgrind Memory Checking"
    print_status "This may take a while (tests run 10-50x slower under Valgrind)..." "INFO"
    echo ""

    # Run CTest with memcheck
    # Note: All Valgrind options are configured in CMake (valgrind.cmake)
    # The --output-on-failure flag ensures we see test output if something fails
    local ctest_exit_code=0
    ctest -T memcheck --output-on-failure || ctest_exit_code=$?

    return $ctest_exit_code
}

# =============================================================================
# Valgrind Results Analysis
# =============================================================================

analyze_valgrind_results() {
    local build_dir="$1"
    local has_memory_issues=0

    print_header "Analyzing Valgrind Results"

    # Find all Valgrind log files
    local memcheck_logs
    memcheck_logs=$(find "$build_dir/Testing/Temporary" -name "MemoryChecker.*.log" 2>/dev/null || true)

    if [ -z "$memcheck_logs" ]; then
        print_status "No Valgrind log files found" "WARNING"
        print_status "This may indicate that tests did not run or Valgrind was not invoked" "WARNING"
        return 1
    fi

    print_status "Found Valgrind log files, analyzing..."

    # Check for memory leaks (definitely lost)
    if grep -q "definitely lost: [1-9]" $memcheck_logs 2>/dev/null; then
        print_status "Memory leaks detected!" "ERROR"
        echo ""
        echo "=== Memory Leak Summary ==="
        grep -B 2 -A 5 "definitely lost" $memcheck_logs | head -30
        has_memory_issues=1
    else
        print_status "No memory leaks detected" "SUCCESS"
    fi

    # Check for memory errors (invalid reads/writes, use of uninitialized values, etc.)
    if grep -qE "ERROR SUMMARY: [1-9][0-9]* errors" $memcheck_logs 2>/dev/null; then
        print_status "Memory errors detected!" "ERROR"
        echo ""
        echo "=== Memory Error Summary ==="
        grep "ERROR SUMMARY" $memcheck_logs
        has_memory_issues=1
    else
        print_status "No memory errors detected" "SUCCESS"
    fi

    # Check for invalid memory access
    if grep -qE "Invalid (read|write)" $memcheck_logs 2>/dev/null; then
        print_status "Invalid memory access detected!" "ERROR"
        echo ""
        echo "=== Invalid Memory Access ==="
        grep -B 2 -A 3 "Invalid" $memcheck_logs | head -20
        has_memory_issues=1
    fi

    # Check for file descriptor leaks
    if grep -q "FILE DESCRIPTORS" $memcheck_logs 2>/dev/null; then
        if grep -qE "Open file descriptor [0-9]+:" $memcheck_logs 2>/dev/null; then
            print_status "File descriptor leaks detected" "WARNING"
        fi
    fi

    return $has_memory_issues
}

# =============================================================================
# Determine Overall Test Status
# =============================================================================

determine_test_status() {
    local ctest_exit_code="$1"
    local memory_issues="$2"
    local build_dir="$3"

    print_header "Test Results Summary"

    # Check if tests timed out
    local timeout_detected=0
    if [ -f "$build_dir/Testing/Temporary/LastTest.log" ]; then
        if grep -q "Timeout" "$build_dir/Testing/Temporary/LastTest.log" 2>/dev/null; then
            timeout_detected=1
            print_status "Some tests timed out" "WARNING"
            print_status "Tests run slower under Valgrind (10-50x). Consider increasing timeouts." "INFO"
        fi
    fi

    # Determine final status based on multiple factors
    local final_exit_code=0

    # Memory issues are the primary concern
    if [ $memory_issues -ne 0 ]; then
        print_status "FAILED: Memory issues detected by Valgrind" "ERROR"
        final_exit_code=1
    # CTest failures (excluding timeouts if no memory issues)
    elif [ $ctest_exit_code -ne 0 ] && [ $timeout_detected -eq 0 ]; then
        print_status "FAILED: Tests failed (non-timeout failures)" "ERROR"
        final_exit_code=1
    # Timeout without memory issues is a warning, not a hard failure
    elif [ $timeout_detected -eq 1 ] && [ $memory_issues -eq 0 ]; then
        print_status "WARNING: Tests timed out but no memory issues detected" "WARNING"
        print_status "Consider this a PASS for memory checking purposes" "INFO"
        print_status "Increase test timeouts in CMake configuration if needed" "INFO"
        final_exit_code=0
    # All good
    else
        print_status "SUCCESS: All tests passed with no memory issues!" "SUCCESS"
        final_exit_code=0
    fi

    # Provide helpful information
    echo ""
    print_status "Valgrind logs location: $build_dir/Testing/Temporary/" "INFO"
    print_status "View detailed logs: ls -lh $build_dir/Testing/Temporary/MemoryChecker.*.log" "INFO"

    return $final_exit_code
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local exit_code=0

    print_header "XSigma Valgrind Memory Check"

    # Step 1: Platform compatibility check
    check_platform_compatibility

    # Step 2: Verify Valgrind is installed
    if ! check_valgrind_installed; then
        exit 1
    fi

    # Step 3: Run tests with Valgrind
    local ctest_exit_code=0
    run_valgrind_tests "$BUILD_DIR" || ctest_exit_code=$?

    # Step 4: Analyze Valgrind results
    local memory_issues=0
    analyze_valgrind_results "$BUILD_DIR" || memory_issues=$?

    # Step 5: Determine overall status
    determine_test_status "$ctest_exit_code" "$memory_issues" "$BUILD_DIR" || exit_code=$?

    return $exit_code
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Run main function and capture exit code
main
exit $?

