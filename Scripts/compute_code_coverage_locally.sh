#!/usr/bin/env bash
# XSigma Code Coverage Computation Script
# Modern wrapper around Tools/code_coverage/oss_coverage.py to produce
# detailed HTML coverage reports (line, file, and multi-file dashboards).

set -euo pipefail

BUILD_PATH="${1:-}"
OUTPUT_DIR="${2:-}"
SUFFIX="${3:-}"
EXTENSION="${4:-}"
EXE_EXTENSION="${5:-}"
GTEST="${6:-}"

# -----------------------------------------------------------------------------
# Pretty printing helpers
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

write_status() {
    local message="$1"
    local type="${2:-INFO}"

    case "$type" in
        SUCCESS) printf "%b[✓] %s%b\n" "${GREEN}" "${message}" "${NC}" ;;
        ERROR)   printf "%b[✗] %s%b\n" "${RED}" "${message}" "${NC}" ;;
        WARNING) printf "%b[!] %s%b\n" "${YELLOW}" "${message}" "${NC}" ;;
        *)       printf "%b[i] %s%b\n" "${CYAN}" "${message}" "${NC}" ;;
    esac
}

write_section() {
    local title="$1"
    printf "\n%b========================================%b\n" "${MAGENTA}" "${NC}"
    printf "%b %s%b\n" "${MAGENTA}" "${title}" "${NC}"
    printf "%b========================================%b\n\n" "${MAGENTA}" "${NC}"
}

abort() {
    write_status "$1" "ERROR"
    exit "${2:-1}"
}

# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
abspath() {
    local target="$1"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$target"
    else
        python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$target"
    fi
}

relpath_or_empty() {
    local path="$1"
    local start="$2"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c '
import os, sys
try:
    print(os.path.relpath(sys.argv[1], sys.argv[2]))
except ValueError:
    print(sys.argv[1])
' "$path" "$start"
    else
        python -c '
import os, sys
try:
    print(os.path.relpath(sys.argv[1], sys.argv[2]))
except ValueError:
    print(sys.argv[1])
' "$path" "$start"
    fi
}

# -----------------------------------------------------------------------------
# Validate arguments
# -----------------------------------------------------------------------------
write_section "XSigma Code Coverage Computation"

[[ -z "${BUILD_PATH}" ]] && abort "Missing BUILD_PATH argument"
[[ -z "${OUTPUT_DIR}" ]] && abort "Missing OUTPUT_DIR argument"

BUILD_PATH="$(abspath "${BUILD_PATH}")"
OUTPUT_DIR="$(abspath "${OUTPUT_DIR}")"

write_status "Build path: ${BUILD_PATH}"
write_status "Test output directory: ${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

write_status "Repository root: ${REPO_ROOT}"

COVERAGE_SCRIPT="${REPO_ROOT}/Tools/code_coverage/oss_coverage.py"
[[ -f "${COVERAGE_SCRIPT}" ]] || abort "Coverage driver not found at ${COVERAGE_SCRIPT}"

# -----------------------------------------------------------------------------
# Compute compiler type (informational only)
# -----------------------------------------------------------------------------
COMPILER_TYPE="unknown"
case "${BUILD_PATH}" in
    *clang*) COMPILER_TYPE="clang" ;;
    *gcc*)   COMPILER_TYPE="gcc" ;;
esac
write_status "Detected compiler type: ${COMPILER_TYPE}"

# -----------------------------------------------------------------------------
# Resolve test subfolder (relative to build dir where possible)
# -----------------------------------------------------------------------------
TEST_SUBFOLDER_REL="$(relpath_or_empty "${OUTPUT_DIR}" "${BUILD_PATH}")"
if [[ "${TEST_SUBFOLDER_REL}" == "." ]]; then
    TEST_SUBFOLDER_REL="bin"
elif [[ "${TEST_SUBFOLDER_REL}" == "${OUTPUT_DIR}" ]]; then
    # relpath failed (likely different roots); fall back to absolute path
    write_status "Unable to compute relative test folder; using absolute path" "WARNING"
fi

# -----------------------------------------------------------------------------
# Prepare environment for the Python coverage driver
# -----------------------------------------------------------------------------
export XSIGMA_FOLDER="${REPO_ROOT}"
export XSIGMA_BUILD_FOLDER="${BUILD_PATH}"
export XSIGMA_TEST_SUBFOLDER="${TEST_SUBFOLDER_REL}"
export XSIGMA_ENABLE_COVERAGE="ON"

write_status "XSIGMA_BUILD_FOLDER=${XSIGMA_BUILD_FOLDER}"
write_status "XSIGMA_TEST_SUBFOLDER=${XSIGMA_TEST_SUBFOLDER}"

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python || true)"
fi
[[ -n "${PYTHON_BIN}" ]] || abort "Python interpreter not found in PATH"

write_section "Step 1: Execute coverage pipeline (tests + reports)"
write_status "Running ${COVERAGE_SCRIPT} via ${PYTHON_BIN}"

coverage_args=( "--build-folder" "${XSIGMA_BUILD_FOLDER}" )
if [[ -n "${XSIGMA_TEST_SUBFOLDER}" && "${XSIGMA_TEST_SUBFOLDER}" != "." ]]; then
    coverage_args+=( "--test-subfolder" "${XSIGMA_TEST_SUBFOLDER}" )
fi

# Propagate GTest selection (if provided) as --run-only
if [[ -n "${GTEST}" && "${GTEST}" != "no" ]]; then
    coverage_args+=( "--run-only" "${GTEST}" )
fi

if "${PYTHON_BIN}" "${COVERAGE_SCRIPT}" "${coverage_args[@]}"; then
    write_status "Coverage pipeline completed" "SUCCESS"
else
    abort "Coverage pipeline failed"
fi

# -----------------------------------------------------------------------------
# Summaries and output pointers
# -----------------------------------------------------------------------------
SUMMARY_ROOT="${XSIGMA_BUILD_FOLDER}/coverage_report"
HTML_PRIMARY="${SUMMARY_ROOT}/html/index.html"
HTML_DETAILS="${SUMMARY_ROOT}/summary/html_details/index.html"

write_section "Coverage Report Summary"
write_status "Coverage artifacts: ${SUMMARY_ROOT}"

if [[ -f "${HTML_PRIMARY}" ]]; then
    write_status "Primary HTML report: ${HTML_PRIMARY}" "SUCCESS"
else
    write_status "Primary HTML report not found at ${HTML_PRIMARY}" "WARNING"
fi

if [[ -f "${HTML_DETAILS}" ]]; then
    write_status "Rich multi-file HTML dashboard: ${HTML_DETAILS}" "SUCCESS"
else
    write_status "Multi-file HTML dashboard not found (clang-only feature)" "WARNING"
fi

echo ""
write_status "Coverage computation completed successfully!" "SUCCESS"
echo ""

exit 0
