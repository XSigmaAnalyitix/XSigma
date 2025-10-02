# =============================================================================
# Test Source Files Discovery
# =============================================================================
file(
  GLOB_RECURSE test_sources
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cu")

set(TestFiles)
foreach(_test_source IN ITEMS ${test_sources})
  get_filename_component(file_name "${_test_source}" NAME)
  if(NOT file_name MATCHES "TestLoggerDisableSignalHandler")
    list(APPEND TestFiles "${_test_source}") 
  endif()
endforeach()

enable_testing()

# =============================================================================
# CoreCxxTests Executable Configuration
# =============================================================================
add_executable(CoreCxxTests ${TestFiles})

# Compile definitions for Google Test
target_compile_definitions(CoreCxxTests PRIVATE
    XSIGMA_GOOGLE_TEST
    USE_GTEST
    _VARIADIC_MAX=10
)

# Link libraries
target_link_libraries(CoreCxxTests PRIVATE
    gtest
    gtest_main
    XSigma::Core
    ${XSIGMA_CUDA_LIBRARIES}
)

# Set target properties including Visual Studio debugger working directory
set_target_properties(CoreCxxTests PROPERTIES
    FOLDER "Tests"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# Apply include-what-you-use if enabled
if(COMMAND xsigma_apply_iwyu)
    xsigma_apply_iwyu(CoreCxxTests)
endif()

# =============================================================================
# Google Test Discovery for CTest and Visual Studio Test Explorer
# =============================================================================

# XSigma uses third-party GoogleTest from ThirdParty/googletest
# The GoogleTest CMake module (built-in to CMake 3.10+) provides gtest_discover_tests()
# which enables automatic test discovery for Visual Studio Test Explorer

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.9")
    # We need to include the CMake built-in GoogleTest module, NOT the XSigma googletest.cmake
    # The XSigma googletest.cmake (in Cmake/tools/) sets up the third-party GoogleTest library
    # The CMake GoogleTest module (in CMake's Modules/) provides gtest_discover_tests()
    #
    # Problem: CMAKE_MODULE_PATH includes Cmake/tools which has googletest.cmake
    # Solution: Temporarily modify CMAKE_MODULE_PATH to prioritize CMake's modules

    set(_original_module_path ${CMAKE_MODULE_PATH})

    # Clear the module path and only use CMake's built-in module path
    set(CMAKE_MODULE_PATH "")

    # Now include the CMake built-in GoogleTest module
    include(GoogleTest OPTIONAL RESULT_VARIABLE _gtest_module_path)

    # Restore the original module path
    set(CMAKE_MODULE_PATH ${_original_module_path})

    if(_gtest_module_path)
        message(STATUS "CMake GoogleTest module loaded from: ${_gtest_module_path}")
    else()
        message(WARNING "Failed to load CMake GoogleTest module")
    endif()

    # Check if the command is now available
    # Note: gtest_discover_tests has a bug in CMake 3.31.0-rc1 with Visual Studio generator
    # The path to GoogleTestAddTests.cmake gets corrupted. Use gtest_add_tests instead.
    if(COMMAND gtest_add_tests)
        message(STATUS "✓ gtest_add_tests command is available")
        message(STATUS "Configuring Google Test discovery for CoreCxxTests with Visual Studio Test Explorer support")

        # Use gtest_add_tests which scans source files for tests
        # This is more reliable with Visual Studio generator
        gtest_add_tests(
            TARGET CoreCxxTests
            WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            TEST_LIST _discovered_tests
        )

        message(STATUS "✓ Google Test discovery configured successfully with gtest_add_tests")
        message(STATUS "  - Tests will be discovered by scanning source files")
        message(STATUS "  - Tests will appear in Visual Studio Test Explorer")
        message(STATUS "  - Working directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        message(STATUS "  - Note: Re-run CMake if you add/remove/rename tests")

        # Set coverage environment variable for all discovered tests
        if(XSIGMA_ENABLE_COVERAGE AND DEFINED XSIGMA_LLVM_PROFILE_FILE)
            foreach(_test_name ${_discovered_tests})
                set_tests_properties(${_test_name} PROPERTIES
                    ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}"
                )
            endforeach()
            message(STATUS "  - Coverage environment configured for ${CMAKE_CURRENT_LIST_LENGTH} tests")
        endif()
    elseif(COMMAND gtest_discover_tests)
        # Fallback to gtest_add_tests (older method, requires source scanning)
        message(STATUS "gtest_discover_tests not available, using gtest_add_tests instead")
        message(STATUS "Note: This requires re-running CMake when tests change")

        gtest_add_tests(
            TARGET CoreCxxTests
            WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            TEST_PREFIX "Core."
            TEST_LIST _test_list
        )

        message(STATUS "✓ Google Test discovery configured with gtest_add_tests")

        # Set coverage environment variable for all discovered tests
        if(XSIGMA_ENABLE_COVERAGE AND DEFINED XSIGMA_LLVM_PROFILE_FILE)
            foreach(_test_name ${_test_list})
                set_tests_properties(${_test_name} PROPERTIES
                    ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}"
                )
            endforeach()
            message(STATUS "  - Coverage environment configured for tests")
        endif()
    else()
        message(WARNING "Neither gtest_discover_tests nor gtest_add_tests available")
        message(WARNING "GoogleTest module may not have loaded correctly")
        message(WARNING "Module path: ${_gtest_module_path}")
        message(WARNING "Using manual test registration - individual tests won't appear in Test Explorer")

        # Fallback to manual test registration
        add_test(NAME CoreCxxTests COMMAND CoreCxxTests WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        set_tests_properties(CoreCxxTests PROPERTIES TIMEOUT 300 LABELS "Core;Unit")

        # Set coverage environment variable for manual test
        if(XSIGMA_ENABLE_COVERAGE AND DEFINED XSIGMA_LLVM_PROFILE_FILE)
            set_tests_properties(CoreCxxTests PROPERTIES
                ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}"
            )
        endif()
    endif()
else()
    message(WARNING "CMake ${CMAKE_VERSION} is too old for GoogleTest module (requires 3.9+)")
    message(WARNING "Using manual test registration - individual tests won't appear in Test Explorer")

    # Fallback to manual test registration
    add_test(NAME CoreCxxTests COMMAND CoreCxxTests WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    set_tests_properties(CoreCxxTests PROPERTIES TIMEOUT 300 LABELS "Core;Unit")

    # Set coverage environment variable for manual test
    if(XSIGMA_ENABLE_COVERAGE AND DEFINED XSIGMA_LLVM_PROFILE_FILE)
        set_tests_properties(CoreCxxTests PROPERTIES
            ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}"
        )
    endif()
endif()

# Skip sanitizer flags for CoreCxxTests when using MSVC due to third-party library conflicts
# Third-party libraries (gtest, fmt) are not built with sanitizer support
if(DEFINED XSIGMA_SANITIZER_COMPILE_FLAGS AND XSIGMA_SANITIZER_COMPILE_FLAGS)
    target_compile_options(CoreCxxTests PRIVATE ${XSIGMA_SANITIZER_COMPILE_FLAGS})
endif()

if(DEFINED XSIGMA_SANITIZER_LINK_FLAGS AND XSIGMA_SANITIZER_LINK_FLAGS)
    target_link_options(CoreCxxTests PRIVATE ${XSIGMA_SANITIZER_LINK_FLAGS})
endif()

# Apply XSigma output directory configuration
if(COMMAND xsigma_set_target_output_directories)
    xsigma_set_target_output_directories(CoreCxxTests)
endif()


# Add TBB support for tests
if(XSIGMA_ENABLE_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(CoreCxxTests PRIVATE TBB::tbb TBB::tbbmalloc)
endif()

# Add logger benchmark executable if benchmarking is enabled
if(XSIGMA_ENABLE_BENCHMARK AND TARGET XSigma::benchmark)
    add_executable(BenchmarkLogger BenchmarkLogger.cxx)

    if(COMMAND xsigma_apply_iwyu)
        xsigma_apply_iwyu(BenchmarkLogger)
    endif()

    target_link_libraries(BenchmarkLogger PRIVATE
        XSigma::benchmark
        XSigma::Core
        ${XSIGMA_CUDA_LIBRARIES}
    )

    set_target_properties(BenchmarkLogger PROPERTIES FOLDER "Benchmarks")

    # Apply sanitizer flags if enabled
    if(DEFINED XSIGMA_SANITIZER_COMPILE_FLAGS AND XSIGMA_SANITIZER_COMPILE_FLAGS)
        target_compile_options(BenchmarkLogger PRIVATE ${XSIGMA_SANITIZER_COMPILE_FLAGS})
    endif()

    if(DEFINED XSIGMA_SANITIZER_LINK_FLAGS AND XSIGMA_SANITIZER_LINK_FLAGS)
        target_link_options(BenchmarkLogger PRIVATE ${XSIGMA_SANITIZER_LINK_FLAGS})
    endif()

    # Apply XSigma output directory configuration
    if(COMMAND xsigma_set_target_output_directories)
        xsigma_set_target_output_directories(BenchmarkLogger)
    endif()

    # Add TBB support for benchmarks if enabled
    if(XSIGMA_ENABLE_TBB)
        find_package(TBB REQUIRED)
        target_link_libraries(BenchmarkLogger PRIVATE TBB::tbb TBB::tbbmalloc)
    endif()

    message(STATUS "Logger benchmark executable will be built: BenchmarkLogger")
endif()