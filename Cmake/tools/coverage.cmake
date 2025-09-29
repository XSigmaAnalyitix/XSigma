# XSigma Code Coverage Configuration
# This module configures code coverage instrumentation and automated report generation
# Supports both LLVM (Clang) and GCC (gcov) coverage workflows

if(NOT XSIGMA_ENABLE_COVERAGE)
    return()
endif()

# Detect Clang compiler (including AppleClang)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(CMAKE_COMPILER_IS_CLANGXX 1)
endif()

# =============================================================================
# Coverage Instrumentation Configuration
# =============================================================================

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
  set(xsigma_coverage_compile_args)
  set(xsigma_coverage_link_args)

  if(CMAKE_COMPILER_IS_CLANGXX)
    # Clang coverage using LLVM instrumentation (works on all platforms including Windows)
    string(APPEND xsigma_coverage_compile_args
            "-g -O0 -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND xsigma_coverage_link_args
            "-fprofile-instr-generate -fcoverage-mapping")

    # Set environment variable for profraw file output location
    set(XSIGMA_COVERAGE_PROFRAW_DIR "${CMAKE_BINARY_DIR}/coverage" CACHE PATH
        "Directory for LLVM profraw coverage files")
    file(MAKE_DIRECTORY "${XSIGMA_COVERAGE_PROFRAW_DIR}")

    # Configure profraw file pattern (unique per process)
    # %p = process ID, %m = module signature
    set(XSIGMA_LLVM_PROFILE_FILE "${XSIGMA_COVERAGE_PROFRAW_DIR}/default-%p.profraw")
    set(ENV{LLVM_PROFILE_FILE} "${XSIGMA_LLVM_PROFILE_FILE}")

    # Set environment variable for CTest to use during test execution
    # This ensures profraw files are generated when tests run via CTest
    if(XSIGMA_BUILD_TESTING)
      set_property(TEST PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}")
    endif()

  elseif(CMAKE_COMPILER_IS_GNUCXX AND UNIX)
    # GCC coverage using gcov (Unix/Linux only)
    string(APPEND xsigma_coverage_compile_args "-g -O0  -fprofile-arcs -ftest-coverage")
    string(APPEND xsigma_coverage_link_args "-lgcov --coverage")
  endif()

  # Apply coverage flags to all build configurations
  # We're setting the CXX flags and C flags because they're propagated down
  # independent of build type.
  string(APPEND CMAKE_CXX_FLAGS " ${xsigma_coverage_compile_args}")
  string(APPEND CMAKE_C_FLAGS " ${xsigma_coverage_compile_args}")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " ${xsigma_coverage_link_args}")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " ${xsigma_coverage_link_args}")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS " ${xsigma_coverage_link_args}")

  message(STATUS "XSigma: Coverage instrumentation enabled")
  if(CMAKE_COMPILER_IS_CLANGXX)
    message(STATUS "  Coverage type: LLVM (Clang)")
    message(STATUS "  Profraw directory: ${XSIGMA_COVERAGE_PROFRAW_DIR}")
  else()
    message(STATUS "  Coverage type: GCC (gcov)")
  endif()
endif()

# =============================================================================
# Automated Coverage Report Generation Targets
# =============================================================================

# Create coverage report output directory
set(XSIGMA_COVERAGE_REPORT_DIR "${CMAKE_BINARY_DIR}/coverage_report" CACHE PATH
    "Directory for coverage reports")
file(MAKE_DIRECTORY "${XSIGMA_COVERAGE_REPORT_DIR}")

if(CMAKE_COMPILER_IS_CLANGXX)
  # ---------------------------------------------------------------------------
  # LLVM Coverage Targets (Clang/AppleClang)
  # ---------------------------------------------------------------------------

  # Find LLVM coverage tools
  find_program(LLVM_PROFDATA llvm-profdata
    HINTS
      "${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}/bin"
      "${CMAKE_CXX_COMPILER}/../"
    DOC "Path to llvm-profdata tool")

  find_program(LLVM_COV llvm-cov
    HINTS
      "${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}/bin"
      "${CMAKE_CXX_COMPILER}/../"
    DOC "Path to llvm-cov tool")

  if(LLVM_PROFDATA AND LLVM_COV)
    message(STATUS "  LLVM coverage tools found:")
    message(STATUS "    llvm-profdata: ${LLVM_PROFDATA}")
    message(STATUS "    llvm-cov: ${LLVM_COV}")

    # Create helper script for merging profraw files (handles wildcards properly)
    set(COVERAGE_MERGE_SCRIPT "${CMAKE_BINARY_DIR}/coverage_merge.cmake")
    file(WRITE "${COVERAGE_MERGE_SCRIPT}" "
# Coverage merge helper script
file(GLOB PROFRAW_FILES \"${XSIGMA_COVERAGE_PROFRAW_DIR}/*.profraw\")
list(LENGTH PROFRAW_FILES PROFRAW_COUNT)

if(PROFRAW_COUNT EQUAL 0)
  message(WARNING \"No .profraw files found in ${XSIGMA_COVERAGE_PROFRAW_DIR}\")
  message(WARNING \"Make sure to run tests before generating coverage reports.\")
  message(FATAL_ERROR \"Coverage merge failed: no profraw files found\")
endif()

message(STATUS \"Found \${PROFRAW_COUNT} profraw file(s) to merge\")

# Build the merge command
set(MERGE_CMD \"${LLVM_PROFDATA}\" merge -sparse)
foreach(PROFRAW_FILE \${PROFRAW_FILES})
  list(APPEND MERGE_CMD \"\${PROFRAW_FILE}\")
endforeach()
list(APPEND MERGE_CMD -o \"${XSIGMA_COVERAGE_REPORT_DIR}/coverage.profdata\")

# Execute the merge
execute_process(
  COMMAND \${MERGE_CMD}
  RESULT_VARIABLE MERGE_RESULT
  OUTPUT_VARIABLE MERGE_OUTPUT
  ERROR_VARIABLE MERGE_ERROR
)

if(NOT MERGE_RESULT EQUAL 0)
  message(FATAL_ERROR \"Coverage merge failed: \${MERGE_ERROR}\")
endif()

message(STATUS \"Coverage data merged successfully: ${XSIGMA_COVERAGE_REPORT_DIR}/coverage.profdata\")
")

    # Determine the test executable directory
    if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
      set(TEST_EXE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    else()
      set(TEST_EXE_DIR "${CMAKE_BINARY_DIR}/bin")
    endif()

    # Create helper script for generating coverage reports (handles test executable discovery)
    set(COVERAGE_REPORT_SCRIPT "${CMAKE_BINARY_DIR}/coverage_report.cmake")
    file(WRITE "${COVERAGE_REPORT_SCRIPT}" "
# Coverage report helper script
file(GLOB TEST_EXECUTABLES
  \"${TEST_EXE_DIR}/*Tests${CMAKE_EXECUTABLE_SUFFIX}\"
  \"${TEST_EXE_DIR}/*Test${CMAKE_EXECUTABLE_SUFFIX}\"
)

list(LENGTH TEST_EXECUTABLES TEST_COUNT)
if(TEST_COUNT EQUAL 0)
  message(WARNING \"No test executables found in ${TEST_EXE_DIR}\")
  message(FATAL_ERROR \"Coverage report failed: no test executables found\")
endif()

message(STATUS \"Generating coverage report for \${TEST_COUNT} test executable(s)\")

# Find shared libraries (DLLs/SOs) that contain instrumented code
file(GLOB SHARED_LIBS
  \"${TEST_EXE_DIR}/*${CMAKE_SHARED_LIBRARY_SUFFIX}\"
)

# Build the report command
set(REPORT_CMD \"${LLVM_COV}\" report)
# Add test executables first
foreach(TEST_EXE \${TEST_EXECUTABLES})
  list(APPEND REPORT_CMD \"\${TEST_EXE}\")
endforeach()
# Add shared libraries with -object flag
foreach(SHARED_LIB \${SHARED_LIBS})
  list(APPEND REPORT_CMD \"-object\" \"\${SHARED_LIB}\")
endforeach()
list(APPEND REPORT_CMD
  \"-instr-profile=${XSIGMA_COVERAGE_REPORT_DIR}/coverage.profdata\"
  \"-ignore-filename-regex=ThirdParty|Testing|Test\"
)

# Execute the report
execute_process(
  COMMAND \${REPORT_CMD}
  RESULT_VARIABLE REPORT_RESULT
  OUTPUT_VARIABLE REPORT_OUTPUT
  ERROR_VARIABLE REPORT_ERROR
)

if(NOT REPORT_RESULT EQUAL 0)
  message(FATAL_ERROR \"Coverage report failed: \${REPORT_ERROR}\")
endif()

message(\"\${REPORT_OUTPUT}\")
")

    # Create helper script for HTML coverage reports
    set(COVERAGE_HTML_SCRIPT "${CMAKE_BINARY_DIR}/coverage_html.cmake")
    file(WRITE "${COVERAGE_HTML_SCRIPT}" "
# Coverage HTML report helper script
file(GLOB TEST_EXECUTABLES
  \"${TEST_EXE_DIR}/*Tests${CMAKE_EXECUTABLE_SUFFIX}\"
  \"${TEST_EXE_DIR}/*Test${CMAKE_EXECUTABLE_SUFFIX}\"
)

list(LENGTH TEST_EXECUTABLES TEST_COUNT)
if(TEST_COUNT EQUAL 0)
  message(WARNING \"No test executables found in ${TEST_EXE_DIR}\")
  message(FATAL_ERROR \"Coverage HTML report failed: no test executables found\")
endif()

message(STATUS \"Generating HTML coverage report for \${TEST_COUNT} test executable(s)\")

# Find shared libraries (DLLs/SOs) that contain instrumented code
file(GLOB SHARED_LIBS
  \"${TEST_EXE_DIR}/*${CMAKE_SHARED_LIBRARY_SUFFIX}\"
)

# Build the HTML report command
set(HTML_CMD \"${LLVM_COV}\" show)
# Add test executables first
foreach(TEST_EXE \${TEST_EXECUTABLES})
  list(APPEND HTML_CMD \"\${TEST_EXE}\")
endforeach()
# Add shared libraries with -object flag
foreach(SHARED_LIB \${SHARED_LIBS})
  list(APPEND HTML_CMD \"-object\" \"\${SHARED_LIB}\")
endforeach()
list(APPEND HTML_CMD
  \"-instr-profile=${XSIGMA_COVERAGE_REPORT_DIR}/coverage.profdata\"
  \"-format=html\"
  \"-output-dir=${XSIGMA_COVERAGE_REPORT_DIR}/html\"
  \"-ignore-filename-regex=ThirdParty|Testing|Test\"
)

# Execute the HTML report
execute_process(
  COMMAND \${HTML_CMD}
  RESULT_VARIABLE HTML_RESULT
  OUTPUT_VARIABLE HTML_OUTPUT
  ERROR_VARIABLE HTML_ERROR
)

if(NOT HTML_RESULT EQUAL 0)
  message(FATAL_ERROR \"Coverage HTML report failed: \${HTML_ERROR}\")
endif()

message(STATUS \"HTML coverage report generated: ${XSIGMA_COVERAGE_REPORT_DIR}/html/index.html\")
")

    # Target: coverage-merge
    # Merges all .profraw files into a single .profdata file
    add_custom_target(coverage-merge
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      COMMAND ${CMAKE_COMMAND} -E echo "Merging coverage profraw files..."
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      COMMAND ${CMAKE_COMMAND} -P "${COVERAGE_MERGE_SCRIPT}"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      COMMENT "Merging LLVM coverage profraw files into profdata"
    )

    # Target: coverage-report
    # Generates text-based coverage report
    add_custom_target(coverage-report
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      COMMAND ${CMAKE_COMMAND} -E echo "Generating coverage report..."
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      COMMAND ${CMAKE_COMMAND} -P "${COVERAGE_REPORT_SCRIPT}"
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated successfully!"
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      DEPENDS coverage-merge
      COMMENT "Generating LLVM coverage report"
    )

    # Target: coverage-html
    # Generates HTML coverage report
    add_custom_target(coverage-html
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      COMMAND ${CMAKE_COMMAND} -E echo "Generating HTML coverage report..."
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      COMMAND ${CMAKE_COMMAND} -P "${COVERAGE_HTML_SCRIPT}"
      COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      DEPENDS coverage-merge
      COMMENT "Generating LLVM HTML coverage report"
    )

    message(STATUS "  Coverage targets created:")
    message(STATUS "    - coverage-merge: Merge profraw files")
    message(STATUS "    - coverage-report: Generate text report")
    message(STATUS "    - coverage-html: Generate HTML report")

  else()
    message(WARNING "LLVM coverage tools not found. Coverage report targets will not be available.")
    if(NOT LLVM_PROFDATA)
      message(WARNING "  llvm-profdata not found")
    endif()
    if(NOT LLVM_COV)
      message(WARNING "  llvm-cov not found")
    endif()
  endif()

elseif(CMAKE_COMPILER_IS_GNUCXX AND UNIX)
  # ---------------------------------------------------------------------------
  # GCC Coverage Targets (gcov/lcov)
  # ---------------------------------------------------------------------------

  # Find GCC coverage tools
  find_program(GCOV gcov DOC "Path to gcov tool")
  find_program(LCOV lcov DOC "Path to lcov tool")
  find_program(GENHTML genhtml DOC "Path to genhtml tool")

  if(GCOV)
    message(STATUS "  GCC coverage tools found:")
    message(STATUS "    gcov: ${GCOV}")
    if(LCOV)
      message(STATUS "    lcov: ${LCOV}")
    endif()
    if(GENHTML)
      message(STATUS "    genhtml: ${GENHTML}")
    endif()

    # Target: coverage-report (GCC/gcov)
    add_custom_target(coverage-report
      COMMAND ${CMAKE_COMMAND} -E echo "Generating GCC coverage report..."
      COMMAND ${GCOV} -r -p "${CMAKE_BINARY_DIR}/**/*.gcda"
      COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated successfully!"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      COMMENT "Generating GCC coverage report with gcov"
      VERBATIM
    )

    # Target: coverage-html (if lcov and genhtml are available)
    if(LCOV AND GENHTML)
      add_custom_target(coverage-html
        COMMAND ${CMAKE_COMMAND} -E echo "Generating HTML coverage report with lcov..."
        COMMAND ${LCOV} --capture --directory "${CMAKE_BINARY_DIR}"
                --output-file "${XSIGMA_COVERAGE_REPORT_DIR}/coverage.info"
        COMMAND ${LCOV} --remove "${XSIGMA_COVERAGE_REPORT_DIR}/coverage.info"
                '*/ThirdParty/*' '*/Testing/*' '*/Test*'
                --output-file "${XSIGMA_COVERAGE_REPORT_DIR}/coverage_filtered.info"
        COMMAND ${GENHTML} "${XSIGMA_COVERAGE_REPORT_DIR}/coverage_filtered.info"
                --output-directory "${XSIGMA_COVERAGE_REPORT_DIR}/html"
        COMMAND ${CMAKE_COMMAND} -E echo "HTML coverage report: ${XSIGMA_COVERAGE_REPORT_DIR}/html/index.html"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating GCC HTML coverage report with lcov/genhtml"
        VERBATIM
      )

      message(STATUS "  Coverage targets created:")
      message(STATUS "    - coverage-report: Generate gcov report")
      message(STATUS "    - coverage-html: Generate HTML report with lcov")
    else()
      message(STATUS "  Coverage targets created:")
      message(STATUS "    - coverage-report: Generate gcov report")
      message(STATUS "  Note: Install lcov and genhtml for HTML coverage reports")
    endif()

  else()
    message(WARNING "gcov not found. Coverage report targets will not be available.")
  endif()
endif()

# =============================================================================
# Helper Function: xsigma_enable_coverage_for_test
# =============================================================================
# This function configures a test target to generate coverage data properly
# Usage: xsigma_enable_coverage_for_test(test_target_name)
#
# For LLVM coverage, this sets the LLVM_PROFILE_FILE environment variable
# For GCC coverage, no additional configuration is needed (automatic)

function(xsigma_enable_coverage_for_test TEST_TARGET)
  if(NOT XSIGMA_ENABLE_COVERAGE)
    return()
  endif()

  if(NOT TARGET ${TEST_TARGET})
    message(WARNING "xsigma_enable_coverage_for_test: Target '${TEST_TARGET}' does not exist")
    return()
  endif()

  if(CMAKE_COMPILER_IS_CLANGXX)
    # For LLVM coverage, set the environment variable for the test
    # This ensures profraw files are generated when the test runs
    set_property(TARGET ${TEST_TARGET}
      PROPERTY
      VS_DEBUGGER_ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}"
    )

    # Also set it for CTest if this is a test executable
    get_target_property(TEST_NAME ${TEST_TARGET} NAME)
    if(TEST ${TEST_NAME})
      set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}"
      )
    endif()

    message(STATUS "Coverage enabled for test target: ${TEST_TARGET}")
  endif()
endfunction()

message(STATUS "XSigma coverage configuration completed successfully")
message(STATUS "Coverage report directory: ${XSIGMA_COVERAGE_REPORT_DIR}")
