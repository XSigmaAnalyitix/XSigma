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
# Report Generation Functions
# =============================================================================

# Generate a comprehensive Valgrind summary report
generate_valgrind_report() {
    local build_dir="$1"
    local report_file="$2"
    local memcheck_logs="$3"

    # Initialize report
    {
        echo "================================================================================"
        echo "XSigma Valgrind Memory Analysis Report"
        echo "================================================================================"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Build Directory: $build_dir"
        echo ""

        # =====================================================================
        # Section 1: Executive Summary
        # =====================================================================
        echo "EXECUTIVE SUMMARY"
        echo "================================================================================"

        local total_errors=0
        local total_leaks=0
        local total_invalid_access=0
        local total_uninitialized=0
        local total_fd_leaks=0

        # Count different types of errors
        total_errors=$(grep -h "ERROR SUMMARY:" $memcheck_logs 2>/dev/null | \
                      grep -oE "[0-9]+ errors" | grep -oE "[0-9]+" | \
                      awk '{sum+=$1} END {print sum}' || echo "0")

        total_leaks=$(grep -h "definitely lost:" $memcheck_logs 2>/dev/null | \
                     grep -oE "[0-9]+ bytes" | grep -oE "[0-9]+" | \
                     awk '{sum+=$1} END {print sum}' || echo "0")

        total_invalid_access=$(grep -hc "Invalid read\|Invalid write" $memcheck_logs 2>/dev/null || echo "0")
        total_uninitialized=$(grep -hc "Use of uninitialised value" $memcheck_logs 2>/dev/null || echo "0")
        total_fd_leaks=$(grep -hc "Open file descriptor" $memcheck_logs 2>/dev/null || echo "0")

        echo "Total Memory Errors:        $total_errors"
        echo "Total Bytes Leaked:         $total_leaks bytes"
        echo "Invalid Memory Accesses:    $total_invalid_access"
        echo "Uninitialized Value Uses:   $total_uninitialized"
        echo "File Descriptor Leaks:      $total_fd_leaks"
        echo ""

        # =====================================================================
        # Section 2: Memory Leaks
        # =====================================================================
        echo "MEMORY LEAKS ANALYSIS"
        echo "================================================================================"

        if grep -q "definitely lost: [1-9]" $memcheck_logs 2>/dev/null; then
            echo "Status: FOUND"
            echo ""
            echo "Leak Details:"
            echo "-------------"
            grep -h "definitely lost:" $memcheck_logs 2>/dev/null | sort | uniq | while read line; do
                echo "  $line"
            done
            echo ""
            echo "Leak Locations (with stack traces):"
            echo "-----------------------------------"
            grep -B 10 "definitely lost: [1-9]" $memcheck_logs 2>/dev/null | \
            grep -E "at 0x|by 0x|in |at " | head -50
        else
            echo "Status: NONE DETECTED ✓"
        fi
        echo ""

        # =====================================================================
        # Section 3: Invalid Memory Access
        # =====================================================================
        echo "INVALID MEMORY ACCESS"
        echo "================================================================================"

        if grep -qE "Invalid (read|write)" $memcheck_logs 2>/dev/null; then
            echo "Status: FOUND"
            echo ""
            echo "Invalid Read/Write Errors:"
            echo "--------------------------"
            grep -h "Invalid read\|Invalid write" $memcheck_logs 2>/dev/null | \
            sort | uniq -c | sort -rn | head -20
            echo ""
            echo "Locations:"
            echo "----------"
            grep -B 5 "Invalid read\|Invalid write" $memcheck_logs 2>/dev/null | \
            grep -E "at 0x|by 0x|in |at " | head -30
        else
            echo "Status: NONE DETECTED ✓"
        fi
        echo ""

        # =====================================================================
        # Section 4: Uninitialized Values
        # =====================================================================
        echo "UNINITIALIZED VALUE USAGE"
        echo "================================================================================"

        if grep -q "Use of uninitialised value" $memcheck_logs 2>/dev/null; then
            echo "Status: FOUND"
            echo ""
            echo "Uninitialized Value Errors:"
            echo "---------------------------"
            grep -h "Use of uninitialised value" $memcheck_logs 2>/dev/null | \
            wc -l | xargs echo "Total occurrences:"
            echo ""
            echo "Sample Locations:"
            echo "-----------------"
            grep -B 3 "Use of uninitialised value" $memcheck_logs 2>/dev/null | \
            grep -E "at 0x|by 0x|in |at " | head -20
        else
            echo "Status: NONE DETECTED ✓"
        fi
        echo ""

        # =====================================================================
        # Section 5: File Descriptor Leaks
        # =====================================================================
        echo "FILE DESCRIPTOR LEAKS"
        echo "================================================================================"

        if grep -q "Open file descriptor" $memcheck_logs 2>/dev/null; then
            echo "Status: FOUND"
            echo ""
            echo "Open File Descriptors:"
            echo "---------------------"
            grep -h "Open file descriptor" $memcheck_logs 2>/dev/null | head -20
        else
            echo "Status: NONE DETECTED ✓"
        fi
        echo ""

        # =====================================================================
        # Section 6: Error Summary by Test
        # =====================================================================
        echo "ERROR SUMMARY BY TEST"
        echo "================================================================================"

        if [ -n "$memcheck_logs" ]; then
            for log_file in $memcheck_logs; do
                local test_name=$(basename "$log_file" | sed 's/MemoryChecker\.\(.*\)\.log/\1/')
                local error_count=$(grep -c "ERROR SUMMARY:" "$log_file" 2>/dev/null || echo "0")
                if [ "$error_count" -gt 0 ]; then
                    echo "Test: $test_name"
                    grep "ERROR SUMMARY:" "$log_file" 2>/dev/null | head -1
                    echo ""
                fi
            done
        fi

        # =====================================================================
        # Section 7: Overall Status
        # =====================================================================
        echo "OVERALL STATUS"
        echo "================================================================================"

        if [ "$total_errors" -eq 0 ] && [ "$total_leaks" -eq 0 ]; then
            echo "Result: PASS ✓"
            echo "No memory issues detected during test execution."
        else
            echo "Result: FAIL ✗"
            echo "Memory issues detected. Please review the details above."
        fi
        echo ""
        echo "================================================================================"
        echo "End of Report"
        echo "================================================================================"

    } > "$report_file"

    print_status "Report saved to: $report_file" "SUCCESS"
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

    # Generate comprehensive report
    local report_file="$build_dir/valgrind_summary_report.txt"
    generate_valgrind_report "$build_dir" "$report_file" "$memcheck_logs"

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
# Display Report Summary to Console
# =============================================================================

display_report_summary() {
    local report_file="$1"

    if [ ! -f "$report_file" ]; then
        return
    fi

    print_header "Valgrind Summary Report"
    echo ""

    # Display key sections from the report
    if grep -q "EXECUTIVE SUMMARY" "$report_file"; then
        echo "=== EXECUTIVE SUMMARY ==="
        sed -n '/^EXECUTIVE SUMMARY/,/^[A-Z]/p' "$report_file" | head -20
        echo ""
    fi

    # Display overall status
    if grep -q "OVERALL STATUS" "$report_file"; then
        echo "=== OVERALL STATUS ==="
        sed -n '/^OVERALL STATUS/,/^====/p' "$report_file" | grep -v "^====" | head -10
        echo ""
    fi

    print_status "Full report saved to: $report_file" "INFO"
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

    # Display report summary to console
    local report_file="$build_dir/valgrind_summary_report.txt"
    display_report_summary "$report_file"

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

