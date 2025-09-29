#!/usr/bin/env bash
# XSigma Code Coverage Runner Script
# This script automates the complete coverage workflow:
# 1. Configure and build with coverage enabled
# 2. Run tests with proper environment variables
# 3. Generate coverage reports

set -e  # Exit on error

# Default values
COMPILER="clang"
REPORT_TYPE="both"
SKIP_BUILD=false
SKIP_TESTS=false
OPEN_HTML=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

function show_help {
    cat << EOF
XSigma Code Coverage Runner

Usage: ./run_coverage.sh [OPTIONS]

Options:
  -c, --compiler <clang|gcc>    Compiler to use (default: clang)
  -r, --report <text|html|both> Type of report to generate (default: both)
  -b, --skip-build              Skip the build step
  -t, --skip-tests              Skip running tests
  -o, --open-html               Open HTML report in browser after generation
  -h, --help                    Show this help message

Examples:
  # Full workflow with Clang (default)
  ./run_coverage.sh

  # Generate only HTML report with GCC
  ./run_coverage.sh --compiler gcc --report html

  # Skip build, just run tests and generate reports
  ./run_coverage.sh --skip-build

  # Generate report and open in browser
  ./run_coverage.sh --open-html

EOF
    exit 0
}

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--compiler)
            COMPILER="$2"
            shift 2
            ;;
        -r|--report)
            REPORT_TYPE="$2"
            shift 2
            ;;
        -b|--skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -t|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -o|--open-html)
            OPEN_HTML=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate compiler
if [[ "$COMPILER" != "clang" && "$COMPILER" != "gcc" ]]; then
    write_status "Invalid compiler: $COMPILER (must be 'clang' or 'gcc')" "ERROR"
    exit 1
fi

# Validate report type
if [[ "$REPORT_TYPE" != "text" && "$REPORT_TYPE" != "html" && "$REPORT_TYPE" != "both" ]]; then
    write_status "Invalid report type: $REPORT_TYPE (must be 'text', 'html', or 'both')" "ERROR"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

write_section "XSigma Code Coverage Runner"

# Step 1: Configure and Build
if [ "$SKIP_BUILD" = false ]; then
    write_section "Step 1: Configure and Build with Coverage"
    
    cd "$SCRIPT_DIR"
    
    BUILD_CONFIG="ninja.$COMPILER.config.tbb.build.coverage"
    write_status "Running: python setup.py $BUILD_CONFIG"
    
    if python setup.py "$BUILD_CONFIG"; then
        write_status "Build completed successfully" "SUCCESS"
    else
        write_status "Build failed with exit code $?" "ERROR"
        exit 1
    fi
else
    write_status "Skipping build step (as requested)" "WARNING"
fi

# Determine build directory
BUILD_DIR_NAME="build_ninja_${COMPILER}_coverage"
BUILD_DIR="$PROJECT_ROOT/$BUILD_DIR_NAME"

if [ ! -d "$BUILD_DIR" ]; then
    write_status "Build directory not found: $BUILD_DIR" "ERROR"
    write_status "Please run without --skip-build first" "ERROR"
    exit 1
fi

write_status "Using build directory: $BUILD_DIR"

# Step 2: Run Tests
if [ "$SKIP_TESTS" = false ]; then
    write_section "Step 2: Run Tests with Coverage"
    
    cd "$BUILD_DIR"
    
    # Set LLVM_PROFILE_FILE environment variable for Clang
    if [ "$COMPILER" = "clang" ]; then
        COVERAGE_DIR="$BUILD_DIR/coverage"
        mkdir -p "$COVERAGE_DIR"
        
        export LLVM_PROFILE_FILE="coverage/default-%p.profraw"
        write_status "Set LLVM_PROFILE_FILE=$LLVM_PROFILE_FILE"
    fi
    
    write_status "Running tests with CTest..."
    
    if ctest --verbose; then
        write_status "All tests passed" "SUCCESS"
    else
        write_status "Some tests failed, but continuing with coverage report" "WARNING"
    fi
    
    # Verify profraw files were generated (for Clang)
    if [ "$COMPILER" = "clang" ]; then
        PROFRAW_COUNT=$(find coverage -name "*.profraw" 2>/dev/null | wc -l)
        if [ "$PROFRAW_COUNT" -eq 0 ]; then
            write_status "No .profraw files generated! Coverage report may be empty." "WARNING"
            write_status "Make sure tests actually executed." "WARNING"
        else
            write_status "Generated $PROFRAW_COUNT profraw file(s)" "SUCCESS"
        fi
    fi
else
    write_status "Skipping test execution (as requested)" "WARNING"
fi

# Step 3: Generate Coverage Reports
write_section "Step 3: Generate Coverage Reports"

cd "$BUILD_DIR"

# Generate text report
if [ "$REPORT_TYPE" = "text" ] || [ "$REPORT_TYPE" = "both" ]; then
    write_status "Generating text coverage report..."
    
    if cmake --build . --target coverage-report; then
        write_status "Text report generated successfully" "SUCCESS"
    else
        write_status "Text report generation failed" "ERROR"
    fi
fi

# Generate HTML report
if [ "$REPORT_TYPE" = "html" ] || [ "$REPORT_TYPE" = "both" ]; then
    write_status "Generating HTML coverage report..."
    
    if cmake --build . --target coverage-html; then
        write_status "HTML report generated successfully" "SUCCESS"
        
        HTML_REPORT="$BUILD_DIR/coverage_report/html/index.html"
        if [ -f "$HTML_REPORT" ]; then
            write_status "HTML report location: $HTML_REPORT" "SUCCESS"
            
            if [ "$OPEN_HTML" = true ]; then
                write_status "Opening HTML report in browser..."
                
                # Detect OS and open browser accordingly
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    open "$HTML_REPORT"
                elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                    xdg-open "$HTML_REPORT" 2>/dev/null || write_status "Could not open browser automatically" "WARNING"
                else
                    write_status "Automatic browser opening not supported on this OS" "WARNING"
                fi
            fi
        fi
    else
        write_status "HTML report generation failed" "ERROR"
    fi
fi

# Summary
write_section "Coverage Workflow Complete"

write_status "Build directory: $BUILD_DIR"
write_status "Coverage data: $BUILD_DIR/coverage/"
write_status "Coverage reports: $BUILD_DIR/coverage_report/"

if [ "$REPORT_TYPE" = "html" ] || [ "$REPORT_TYPE" = "both" ]; then
    HTML_REPORT="$BUILD_DIR/coverage_report/html/index.html"
    if [ -f "$HTML_REPORT" ]; then
        echo ""
        write_status "To view HTML report, open:" "SUCCESS"
        echo -e "  ${GREEN}$HTML_REPORT${NC}"
        
        if [ "$OPEN_HTML" = false ]; then
            echo ""
            write_status "Tip: Use --open-html to automatically open the report" "INFO"
        fi
    fi
fi

echo ""
write_status "Coverage workflow completed successfully!" "SUCCESS"
echo ""

