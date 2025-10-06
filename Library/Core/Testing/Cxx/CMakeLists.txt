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

# Disable memory sanitizer for TestFMT.cxx due to compiler crash with fmt library templates
if(XSIGMA_ENABLE_SANITIZER AND XSIGMA_SANITIZER_TYPE STREQUAL "memory")
    set_source_files_properties(TestFMT.cxx PROPERTIES
        COMPILE_FLAGS "-fno-sanitize=memory"
    )
    message(STATUS "Disabled memory sanitizer for TestFMT.cxx due to compiler compatibility issues")
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
    ENVIRONMENT "LSAN_OPTIONS=${XSIGMA_GLOBAL_LSAN_OPTIONS}"
)

# Add MSAN_OPTIONS if memory sanitizer is enabled
if(XSIGMA_ENABLE_SANITIZER AND XSIGMA_SANITIZER_TYPE STREQUAL "memory" AND DEFINED XSIGMA_GLOBAL_MSAN_OPTIONS)
    set_tests_properties(CoreCxxTests PROPERTIES
        ENVIRONMENT "LSAN_OPTIONS=${XSIGMA_GLOBAL_LSAN_OPTIONS};MSAN_OPTIONS=${XSIGMA_GLOBAL_MSAN_OPTIONS}"
        # Memory sanitizer warnings should not cause test failure
        WILL_FAIL FALSE
    )
endif()

# Add TSAN_OPTIONS if thread sanitizer is enabled
if(XSIGMA_ENABLE_SANITIZER AND XSIGMA_SANITIZER_TYPE STREQUAL "thread")
    set_tests_properties(CoreCxxTests PROPERTIES
        ENVIRONMENT "LSAN_OPTIONS=${XSIGMA_GLOBAL_LSAN_OPTIONS};TSAN_OPTIONS=halt_on_error=0:abort_on_error=0:report_bugs=1:history_size=7:detect_thread_leaks=false:force_seq_cst_atomics=0"
        # Thread sanitizer warnings should not cause test failure
        WILL_FAIL FALSE
    )
endif()

# Set coverage environment variable if enabled
if(XSIGMA_ENABLE_COVERAGE AND DEFINED XSIGMA_LLVM_PROFILE_FILE)
    if(XSIGMA_ENABLE_SANITIZER AND XSIGMA_SANITIZER_TYPE STREQUAL "memory" AND DEFINED XSIGMA_GLOBAL_MSAN_OPTIONS)
        set_tests_properties(CoreCxxTests PROPERTIES
            ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE};LSAN_OPTIONS=${XSIGMA_GLOBAL_LSAN_OPTIONS};MSAN_OPTIONS=${XSIGMA_GLOBAL_MSAN_OPTIONS}"
            # Memory sanitizer warnings should not cause test failure
            WILL_FAIL FALSE
        )
    elseif(XSIGMA_ENABLE_SANITIZER AND XSIGMA_SANITIZER_TYPE STREQUAL "thread")
        set_tests_properties(CoreCxxTests PROPERTIES
            ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE};LSAN_OPTIONS=${XSIGMA_GLOBAL_LSAN_OPTIONS};TSAN_OPTIONS=halt_on_error=0:abort_on_error=0:report_bugs=1:history_size=7:detect_thread_leaks=false:force_seq_cst_atomics=0"
            # Thread sanitizer warnings should not cause test failure
            WILL_FAIL FALSE
        )
    else()
        set_tests_properties(CoreCxxTests PROPERTIES
            ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE};LSAN_OPTIONS=${XSIGMA_GLOBAL_LSAN_OPTIONS}"
        )
    endif()
endif()

message(STATUS "✓ CoreCxxTests registered with CTest")
message(STATUS "  - Working directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

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

    # Apply XSigma output directory configuration
    if(COMMAND xsigma_set_target_output_directories)
        xsigma_set_target_output_directories(CoreCxxBenchmark)
    endif()

    message(STATUS "✓ BenchmarkSimple executable configured")
    # =============================================================================
    # TBB Support
    # =============================================================================
    if(XSIGMA_ENABLE_TBB)
        # TBB targets should be available from the tbb.cmake module
        if(TARGET TBB::tbb)
            target_link_libraries(CoreCxxBenchmark PRIVATE TBB::tbb)
            if(TARGET TBB::tbbmalloc)
                target_link_libraries(CoreCxxBenchmark PRIVATE TBB::tbbmalloc)
            endif()
            message(STATUS "CoreCxxBenchmark linked with TBB")
        else()
            message(WARNING "XSIGMA_ENABLE_TBB is ON but TBB::tbb target is not available for tests")
        endif()
    endif()
endif()