if(NOT XSIGMA_ENABLE_VALGRIND)
    return()
endif()

# XSigma Valgrind Configuration
message(STATUS "Configuring Valgrind memory checking...")

# Detect platform
cmake_host_system_information(RESULT PLATFORM_ARCH QUERY OS_PLATFORM)
message(STATUS "Platform architecture: ${PLATFORM_ARCH}")

# Check for Apple Silicon (ARM64) - Valgrind not supported
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(WARNING "Valgrind does not support Apple Silicon (ARM64) architecture")
    message(WARNING "Consider using sanitizers instead:")
    message(WARNING "  AddressSanitizer: -DXSIGMA_ENABLE_SANITIZER=ON -DXSIGMA_SANITIZER_TYPE=address")
    message(WARNING "  LeakSanitizer:    -DXSIGMA_ENABLE_SANITIZER=ON -DXSIGMA_SANITIZER_TYPE=leak")
    message(WARNING "Continuing with Valgrind configuration (will fail if Valgrind is not installed)...")
endif()

# Find Valgrind executable
find_program(CMAKE_MEMORYCHECK_COMMAND valgrind)

if(NOT CMAKE_MEMORYCHECK_COMMAND)
    message(FATAL_ERROR "Valgrind not found! Please install Valgrind or disable XSIGMA_ENABLE_VALGRIND")
endif()

message(STATUS "Found Valgrind: ${CMAKE_MEMORYCHECK_COMMAND}")

# Get Valgrind version
execute_process(
    COMMAND ${CMAKE_MEMORYCHECK_COMMAND} --version
    OUTPUT_VARIABLE VALGRIND_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Valgrind version: ${VALGRIND_VERSION_OUTPUT}")

# Configure Valgrind options
set(CMAKE_MEMORYCHECK_COMMAND_OPTIONS
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

# Set up suppression file
set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE
    "${PROJECT_SOURCE_DIR}/Cmake/xsigmaValgrindSuppression.txt"
)

if(EXISTS "${CTEST_MEMORYCHECK_SUPPRESSIONS_FILE}")
    message(STATUS "Using Valgrind suppression file: ${CTEST_MEMORYCHECK_SUPPRESSIONS_FILE}")
    list(APPEND CMAKE_MEMORYCHECK_COMMAND_OPTIONS
        "--suppressions=${CTEST_MEMORYCHECK_SUPPRESSIONS_FILE}"
    )
else()
    message(WARNING "Valgrind suppression file not found: ${CTEST_MEMORYCHECK_SUPPRESSIONS_FILE}")
endif()

# Build the complete memcheck command
set(memcheck_command "${CMAKE_MEMORYCHECK_COMMAND}")
foreach(opt ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS})
    list(APPEND memcheck_command ${opt})
endforeach()

message(STATUS "Valgrind command: ${memcheck_command}")
message(STATUS "Executing test suite with Valgrind enabled")

