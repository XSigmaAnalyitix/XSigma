# =============================================================================
# XSigma Code Coverage Configuration Module
# =============================================================================
# This module configures code coverage instrumentation and automated report generation.
# Supports LLVM (Clang), GCC (gcov), and MSVC (OpenCppCoverage) coverage workflows.
# Generates coverage reports in text and HTML formats.
# =============================================================================

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# Code Coverage Flag
# Controls whether code coverage instrumentation is enabled during compilation.
# When enabled, generates coverage data for LLVM (Clang), GCC (gcov), or MSVC (OpenCppCoverage) analysis.
# Supports automated report generation in text and HTML formats.
option(XSIGMA_ENABLE_COVERAGE "Build XSIGMA with coverage" OFF)
mark_as_advanced(XSIGMA_ENABLE_COVERAGE)

if(XSIGMA_ENABLE_COVERAGE)
set(XSIGMA_ENABLE_LTO OFF)
set(CMAKE_BUILD_TYPE "Debug")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    string(APPEND CMAKE_C_FLAGS " --coverage -g -O0  -fprofile-arcs -ftest-coverage")
    string(APPEND CMAKE_CXX_FLAGS " --coverage -g -O0  -fprofile-arcs -ftest-coverage")
    message(STATUS "Enabling GCC code coverage")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_C_FLAGS " -g -O0  -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND CMAKE_CXX_FLAGS " -g -O0  -fprofile-instr-generate -fcoverage-mapping")
    message(STATUS "Enabling Clang code coverage")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # For MSVC with OpenCppCoverage, we need debug information
    # OpenCppCoverage instruments binaries at runtime, so we just need debug symbols
    string(APPEND CMAKE_C_FLAGS " /Zi /Od")
    string(APPEND CMAKE_CXX_FLAGS " /Zi /Od")
    # Ensure debug info is included in the linker output
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /DEBUG")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " /DEBUG")
    message(STATUS "Enabling MSVC code coverage (OpenCppCoverage mode)")
  else()
    message(
      WARNING
      "Code coverage for compiler ${CMAKE_CXX_COMPILER_ID} is unsupported natively.")
  endif()
endif()

