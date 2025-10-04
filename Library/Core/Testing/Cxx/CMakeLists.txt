# =============================================================================
# Core Library C++ Tests
# =============================================================================
# Following cpuinfo's simple pattern for test integration

# =============================================================================
# Test Source Files Discovery
# =============================================================================
file(
  GLOB_RECURSE test_sources
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cu"
)

# Exclude special test files that need separate handling
set(TestFiles)
foreach(_test_source IN ITEMS ${test_sources})
  get_filename_component(file_name "${_test_source}" NAME)
  if(NOT file_name MATCHES "TestLoggerDisableSignalHandler")
    list(APPEND TestFiles "${_test_source}")
  endif()
endforeach()

# =============================================================================
# CoreCxxTests Executable
# =============================================================================
add_executable(CoreCxxTests ${TestFiles})

# Compile definitions
target_compile_definitions(CoreCxxTests PRIVATE
    XSIGMA_GOOGLE_TEST
    USE_GTEST
    _VARIADIC_MAX=10
)

# Link libraries (following cpuinfo pattern: simple target_link_libraries)
target_link_libraries(CoreCxxTests PRIVATE
    gtest
    gtest_main
    XSigma::Core
    ${XSIGMA_CUDA_LIBRARIES}
)

# Set target properties
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
# Following cpuinfo pattern: simple add_test() calls

# Register test with CTest (following cpuinfo pattern)
add_test(NAME CoreCxxTests COMMAND CoreCxxTests WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# Set test properties
set_tests_properties(CoreCxxTests PROPERTIES
    TIMEOUT 300
    LABELS "Core;Unit"
)

# Set coverage environment variable if enabled
if(XSIGMA_ENABLE_COVERAGE AND DEFINED XSIGMA_LLVM_PROFILE_FILE)
    set_tests_properties(CoreCxxTests PROPERTIES
        ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}"
    )
endif()

message(STATUS "✓ CoreCxxTests registered with CTest")
message(STATUS "  - Working directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# =============================================================================
# Sanitizer Support
# =============================================================================
# Apply sanitizer flags if enabled
if(DEFINED XSIGMA_SANITIZER_COMPILE_FLAGS AND XSIGMA_SANITIZER_COMPILE_FLAGS)
    target_compile_options(CoreCxxTests PRIVATE ${XSIGMA_SANITIZER_COMPILE_FLAGS})
endif()

if(DEFINED XSIGMA_SANITIZER_LINK_FLAGS AND XSIGMA_SANITIZER_LINK_FLAGS)
    target_link_options(CoreCxxTests PRIVATE ${XSIGMA_SANITIZER_LINK_FLAGS})
endif()

# =============================================================================
# Output Directory Configuration
# =============================================================================
if(COMMAND xsigma_set_target_output_directories)
    xsigma_set_target_output_directories(CoreCxxTests)
endif()

# =============================================================================
# TBB Support
# =============================================================================
if(XSIGMA_ENABLE_TBB)
    # TBB targets should be available from the tbb.cmake module
    if(TARGET TBB::tbb)
        target_link_libraries(CoreCxxTests PRIVATE TBB::tbb)
        if(TARGET TBB::tbbmalloc)
            target_link_libraries(CoreCxxTests PRIVATE TBB::tbbmalloc)
        endif()
        message(STATUS "CoreCxxTests linked with TBB")
    else()
        message(WARNING "XSIGMA_ENABLE_TBB is ON but TBB::tbb target is not available for tests")
    endif()
endif()

# =============================================================================
# Mimalloc Support
# =============================================================================
if(XSIGMA_ENABLE_MIMALLOC)
    if(TARGET mimalloc-static)
        target_link_libraries(CoreCxxTests PRIVATE mimalloc-static)
        message(STATUS "CoreCxxTests linked with mimalloc-static")
    else()
        message(WARNING "XSIGMA_ENABLE_MIMALLOC is ON but mimalloc-static target is not available for tests")
    endif()
endif()

# =============================================================================
# BenchmarkSimple Executable (following cpuinfo pattern)
# =============================================================================
if(XSIGMA_ENABLE_BENCHMARK)

file(
  GLOB_RECURSE bench_sources
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cu"
)
    add_executable(CoreCxxBenchmark ${bench_sources})

    # Link libraries (following cpuinfo pattern: simple target_link_libraries)
    target_link_libraries(CoreCxxBenchmark PRIVATE benchmark XSigma::Core ${XSIGMA_CUDA_LIBRARIES})

    # Add mimalloc support for benchmarks
    if(XSIGMA_ENABLE_MIMALLOC AND TARGET mimalloc-static)
        target_link_libraries(CoreCxxBenchmark PRIVATE mimalloc-static)
        message(STATUS "CoreCxxBenchmark linked with mimalloc-static")
    endif()

    # Set target properties
    set_target_properties(CoreCxxBenchmark PROPERTIES
        FOLDER "Benchmarks"
    )

    # Apply sanitizer flags if enabled
    if(DEFINED XSIGMA_SANITIZER_COMPILE_FLAGS AND XSIGMA_SANITIZER_COMPILE_FLAGS)
        target_compile_options(CoreCxxBenchmark PRIVATE ${XSIGMA_SANITIZER_COMPILE_FLAGS})
    endif()

    if(DEFINED XSIGMA_SANITIZER_LINK_FLAGS AND XSIGMA_SANITIZER_LINK_FLAGS)
        target_link_options(CoreCxxBenchmark PRIVATE ${XSIGMA_SANITIZER_LINK_FLAGS})
    endif()

    # Apply XSigma output directory configuration
    if(COMMAND xsigma_set_target_output_directories)
        xsigma_set_target_output_directories(CoreCxxBenchmark)
    endif()

    message(STATUS "✓ BenchmarkSimple executable configured")
endif()