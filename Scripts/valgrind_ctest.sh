#!/usr/bin/env bash
# XSigma Valgrind CTest Runner
# This script runs CTest with Valgrind memory checking

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

function print_status {
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

# Check if build directory is provided
if [ $# -eq 0 ]; then
    print_status "Usage: $0 <build_directory>" "ERROR"
    exit 1
fi

BUILD_DIR="$1"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    print_status "Build directory not found: $BUILD_DIR" "ERROR"
    exit 1
fi

# Check platform compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
    ARCH=$(uname -m)
    if [ "$ARCH" == "arm64" ]; then
        print_status "WARNING: Valgrind does not support Apple Silicon (ARM64)" "WARNING"
        print_status "Consider using sanitizers instead:" "INFO"
        print_status "  AddressSanitizer: python3 setup.py config.ninja.clang.test --sanitizer.address" "INFO"
        print_status "  LeakSanitizer:    python3 setup.py config.ninja.clang.test --sanitizer.leak" "INFO"
        print_status "Attempting to run Valgrind anyway (will fail if not installed)..." "WARNING"
    fi
fi

# Check if Valgrind is installed
if ! command -v valgrind &> /dev/null; then
    print_status "Valgrind is not installed!" "ERROR"
    print_status "Please install Valgrind:" "INFO"
    print_status "  Ubuntu/Debian: sudo apt-get install valgrind" "INFO"
    print_status "  macOS (Intel): brew install valgrind" "INFO"
    exit 1
fi

# Get Valgrind version
VALGRIND_VERSION=$(valgrind --version)
print_status "Found Valgrind: $VALGRIND_VERSION" "SUCCESS"

# Change to build directory
cd "$BUILD_DIR"
print_status "Running Valgrind tests in: $BUILD_DIR"

# Get script directory for suppression file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SUPPRESSION_FILE="$PROJECT_ROOT/Cmake/xsigmaValgrindSuppression.txt"

# Configure Valgrind options
VALGRIND_OPTS=(
    "--tool=memcheck"
    "--leak-check=full"
    "--show-leak-kinds=all"
    "--track-origins=yes"
    "--verbose"
    "--gen-suppressions=all"
    "--trace-children=yes"
    "--show-reachable=yes"
    "--num-callers=50"
    "--error-exitcode=1"
)

# Add suppression file if it exists
if [ -f "$SUPPRESSION_FILE" ]; then
    VALGRIND_OPTS+=("--suppressions=$SUPPRESSION_FILE")
    print_status "Using suppression file: $SUPPRESSION_FILE"
else
    print_status "Suppression file not found: $SUPPRESSION_FILE" "WARNING"
fi

# Run CTest with Valgrind
print_status "Running tests with Valgrind (this may take a while)..."
echo ""

# Run CTest with memcheck
if ctest -T memcheck --output-on-failure; then
    print_status "All tests passed with Valgrind" "SUCCESS"
    EXIT_CODE=0
else
    print_status "Some tests failed or memory errors detected" "ERROR"
    EXIT_CODE=1
fi

echo ""
print_status "Checking Valgrind results..."

# Check for memory check log files
MEMCHECK_LOGS=$(find Testing/Temporary -name "MemoryChecker.*.log" 2>/dev/null || true)

if [ -z "$MEMCHECK_LOGS" ]; then
    print_status "No Valgrind log files found" "WARNING"
else
    print_status "Analyzing Valgrind logs..."
    
    # Check for memory leaks
    if grep -q "definitely lost" $MEMCHECK_LOGS; then
        print_status "Memory leaks detected!" "ERROR"
        echo ""
        echo "=== Memory Leak Summary ==="
        grep -A 5 "definitely lost" $MEMCHECK_LOGS | head -20
        EXIT_CODE=1
    else
        print_status "No memory leaks detected" "SUCCESS"
    fi
    
    # Check for memory errors
    if grep -q "ERROR SUMMARY: [1-9]" $MEMCHECK_LOGS; then
        print_status "Memory errors detected!" "ERROR"
        echo ""
        echo "=== Memory Error Summary ==="
        grep "ERROR SUMMARY" $MEMCHECK_LOGS
        EXIT_CODE=1
    else
        print_status "No memory errors detected" "SUCCESS"
    fi
    
    # Check for invalid reads/writes
    if grep -q "Invalid read\|Invalid write" $MEMCHECK_LOGS; then
        print_status "Invalid memory access detected!" "ERROR"
        EXIT_CODE=1
    fi
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    print_status "Valgrind memory check completed successfully!" "SUCCESS"
else
    print_status "Valgrind memory check failed!" "ERROR"
    print_status "Check the logs in: $BUILD_DIR/Testing/Temporary/" "INFO"
fi

exit $EXIT_CODE

