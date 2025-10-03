#!/usr/bin/env bash
# XSigma Code Coverage Computation Script
# This script is called by setup.py during the coverage build process
# It runs tests and generates coverage reports using the CMake-based coverage system

set -e  # Exit on error

# Parse arguments from setup.py
BUILD_PATH="$1"
OUTPUT_DIR="$2"
SUFFIX="$3"
EXTENSION="$4"
EXE_EXTENSION="$5"
GTEST="$6"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

function write_status {
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

function write_section {
    local title="$1"
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA} $title${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""
}

write_section "XSigma Code Coverage Computation"

write_status "Build path: $BUILD_PATH"
write_status "Output directory: $OUTPUT_DIR"
write_status "Using GTest: ${GTEST:-no}"

# Change to build directory
cd "$BUILD_PATH" || {
    write_status "Failed to change to build directory: $BUILD_PATH" "ERROR"
    exit 1
}

# Detect compiler type from build directory name
COMPILER_TYPE="unknown"
if [[ "$BUILD_PATH" == *"clang"* ]]; then
    COMPILER_TYPE="clang"
elif [[ "$BUILD_PATH" == *"gcc"* ]]; then
    COMPILER_TYPE="gcc"
fi

write_status "Detected compiler type: $COMPILER_TYPE"

# Step 1: Set up environment for LLVM coverage (if using Clang)
if [[ "$COMPILER_TYPE" == "clang" ]]; then
    write_section "Step 1: Configure LLVM Coverage Environment"
    
    COVERAGE_DIR="$BUILD_PATH/coverage"
    mkdir -p "$COVERAGE_DIR"
    
    export LLVM_PROFILE_FILE="coverage/default-%p.profraw"
    write_status "Set LLVM_PROFILE_FILE=$LLVM_PROFILE_FILE" "SUCCESS"
fi

# Step 2: Run tests
write_section "Step 2: Run Tests with Coverage"

write_status "Running CTest..."

if ctest --verbose; then
    write_status "All tests passed" "SUCCESS"
else
    TEST_RESULT=$?
    write_status "Some tests failed (exit code: $TEST_RESULT)" "WARNING"
    write_status "Continuing with coverage report generation..." "WARNING"
fi

# Verify coverage data was generated
if [[ "$COMPILER_TYPE" == "clang" ]]; then
    PROFRAW_COUNT=$(find coverage -name "*.profraw" 2>/dev/null | wc -l)
    if [ "$PROFRAW_COUNT" -eq 0 ]; then
        write_status "No .profraw files generated!" "ERROR"
        write_status "Coverage report will be empty or fail." "ERROR"
        write_status "Make sure tests actually executed." "WARNING"
    else
        write_status "Generated $PROFRAW_COUNT profraw file(s)" "SUCCESS"
    fi
elif [[ "$COMPILER_TYPE" == "gcc" ]]; then
    GCDA_COUNT=$(find . -name "*.gcda" 2>/dev/null | wc -l)
    if [ "$GCDA_COUNT" -eq 0 ]; then
        write_status "No .gcda files generated!" "ERROR"
        write_status "Coverage report will be empty or fail." "ERROR"
    else
        write_status "Generated $GCDA_COUNT gcda file(s)" "SUCCESS"
    fi
fi

# Step 3: Generate coverage reports using CMake targets
write_section "Step 3: Generate Coverage Reports"

# Check if coverage targets exist
if cmake --build . --target help 2>/dev/null | grep -q "coverage-report"; then
    write_status "Using CMake coverage targets..."
    
    # Generate text report
    write_status "Generating text coverage report..."
    if cmake --build . --target coverage-report; then
        write_status "Text coverage report generated successfully" "SUCCESS"
    else
        write_status "Text coverage report generation failed" "WARNING"
    fi
    
    # Generate HTML report
    write_status "Generating HTML coverage report..."
    if cmake --build . --target coverage-html; then
        write_status "HTML coverage report generated successfully" "SUCCESS"
        
        HTML_REPORT="$BUILD_PATH/coverage_report/html/index.html"
        if [ -f "$HTML_REPORT" ]; then
            write_status "HTML report location: $HTML_REPORT" "SUCCESS"
        fi
    else
        write_status "HTML coverage report generation failed" "WARNING"
    fi
else
    write_status "CMake coverage targets not found, using fallback method..." "WARNING"
    
    # Fallback: Use LLVM tools directly
    if [[ "$COMPILER_TYPE" == "clang" ]]; then
        write_status "Using LLVM tools directly..."
        
        # Find LLVM tools
        LLVM_PROFDATA=$(command -v llvm-profdata 2>/dev/null || echo "")
        LLVM_COV=$(command -v llvm-cov 2>/dev/null || echo "")
        
        if [ -z "$LLVM_PROFDATA" ] || [ -z "$LLVM_COV" ]; then
            write_status "LLVM coverage tools not found in PATH" "ERROR"
            write_status "Please ensure llvm-profdata and llvm-cov are installed" "ERROR"
            exit 1
        fi
        
        # Merge profraw files
        write_status "Merging profraw files..."
        mkdir -p coverage_report
        
        if $LLVM_PROFDATA merge -sparse coverage/*.profraw -o coverage_report/coverage.profdata; then
            write_status "Profraw files merged successfully" "SUCCESS"
        else
            write_status "Failed to merge profraw files" "ERROR"
            exit 1
        fi
        
        # Find test executables
        TEST_EXES=$(find "$OUTPUT_DIR" -name "*Tests$EXE_EXTENSION" -o -name "*Test$EXE_EXTENSION" 2>/dev/null)
        
        if [ -z "$TEST_EXES" ]; then
            write_status "No test executables found in $OUTPUT_DIR" "ERROR"
            exit 1
        fi
        
        # Generate text report
        write_status "Generating coverage report..."
        $LLVM_COV report $TEST_EXES \
            -instr-profile=coverage_report/coverage.profdata \
            -ignore-filename-regex="ThirdParty|Testing|Test"
        
        write_status "Coverage report generated successfully" "SUCCESS"
        
    elif [[ "$COMPILER_TYPE" == "gcc" ]]; then
        write_status "Using gcov directly..."
        
        GCOV=$(command -v gcov 2>/dev/null || echo "")
        if [ -z "$GCOV" ]; then
            write_status "gcov not found in PATH" "ERROR"
            exit 1
        fi
        
        # Generate gcov reports
        find . -name "*.gcda" -exec $GCOV -r -p {} \; > /dev/null 2>&1
        write_status "gcov reports generated" "SUCCESS"
        
        # Try to use lcov if available
        LCOV=$(command -v lcov 2>/dev/null || echo "")
        GENHTML=$(command -v genhtml 2>/dev/null || echo "")
        
        if [ -n "$LCOV" ] && [ -n "$GENHTML" ]; then
            write_status "Generating HTML report with lcov..."
            mkdir -p coverage_report
            
            $LCOV --capture --directory . --output-file coverage_report/coverage.info
            $LCOV --remove coverage_report/coverage.info \
                '*/ThirdParty/*' '*/Testing/*' '*/Test*' \
                --output-file coverage_report/coverage_filtered.info
            $GENHTML coverage_report/coverage_filtered.info \
                --output-directory coverage_report/html
            
            write_status "HTML report: coverage_report/html/index.html" "SUCCESS"
        fi
    fi
fi

# Summary
write_section "Coverage Computation Complete"

write_status "Build directory: $BUILD_PATH"
write_status "Coverage data: $BUILD_PATH/coverage/"
write_status "Coverage reports: $BUILD_PATH/coverage_report/"

if [ -f "$BUILD_PATH/coverage_report/html/index.html" ]; then
    echo ""
    write_status "To view HTML report, open:" "SUCCESS"
    echo -e "  ${GREEN}$BUILD_PATH/coverage_report/html/index.html${NC}"
fi

echo ""
write_status "Coverage computation completed successfully!" "SUCCESS"
echo ""

exit 0

