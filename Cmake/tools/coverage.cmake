# =============================================================================
# XSigma Code Coverage Configuration Module
# =============================================================================
# This module configures code coverage instrumentation and automated report generation.
# Supports both LLVM (Clang) and GCC (gcov) coverage workflows.
# Generates coverage reports in text and HTML formats.
# =============================================================================

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# Code Coverage Flag
# Controls whether code coverage instrumentation is enabled during compilation.
# When enabled, generates coverage data for LLVM (Clang) or GCC (gcov) analysis.
# Supports automated report generation in text and HTML formats.
option(XSIGMA_ENABLE_COVERAGE "Build XSIGMA with coverage" OFF)
mark_as_advanced(XSIGMA_ENABLE_COVERAGE)

if(XSIGMA_ENABLE_COVERAGE)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    string(APPEND CMAKE_C_FLAGS " --coverage -fprofile-abs-path")
    string(APPEND CMAKE_CXX_FLAGS " --coverage -fprofile-abs-path")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_C_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND CMAKE_CXX_FLAGS
            " -fprofile-instr-generate -fcoverage-mapping")
  else()
    message(
      ERROR
      "Code coverage for compiler ${CMAKE_CXX_COMPILER_ID} is unsupported")
  endif()
endif()

