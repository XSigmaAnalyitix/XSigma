# ============================================================================= Core Library C++
# Tests
# =============================================================================
# Following cpuinfo's simple pattern for test integration

# ============================================================================= Test Source Files
# Discovery
# =============================================================================
file(GLOB_RECURSE test_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cpp"
                                                      "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cu"
)

# Exclude special test files that need separate handling
set(TestFiles)
foreach(_test_source IN ITEMS ${test_sources})
  get_filename_component(file_name "${_test_source}" NAME)

  # Always exclude TestLoggerDisableSignalHandler
  if(file_name MATCHES "TestLoggerDisableSignalHandler")
    continue()
  endif()

  # Exclude GPU-related test files based on compile flags When both CUDA and HIP are disabled,
  # exclude all GPU tests
  if(NOT XSIGMA_ENABLE_CUDA AND NOT XSIGMA_ENABLE_HIP)
    if(file_name MATCHES "TestGpu" OR file_name MATCHES "TestGPU")
      message(STATUS "Excluding GPU test (no GPU support): ${file_name}")
      continue()
    endif()
  endif()

  # When CUDA is disabled, exclude CUDA-specific tests
  if(NOT XSIGMA_ENABLE_CUDA)
    if(file_name MATCHES "TestCuda")
      message(STATUS "Excluding CUDA test (CUDA disabled): ${file_name}")
      continue()
    endif()
  endif()

  # When HIP is disabled, exclude HIP-specific tests
  if(NOT XSIGMA_ENABLE_HIP)
    if(file_name MATCHES "TestHip")
      message(STATUS "Excluding HIP test (HIP disabled): ${file_name}")
      continue()
    endif()
  endif()

  # When Kineto is disabled, exclude Kineto-specific tests
  if(NOT XSIGMA_ENABLE_KINETO)
    if(file_name MATCHES "TestKineto")
      message(STATUS "Excluding Kineto test (Kineto disabled): ${file_name}")
      continue()
    endif()
  endif()

  list(APPEND TestFiles "${_test_source}")
endforeach()

# ============================================================================= CoreCxxTests
# Executable
# =============================================================================
add_executable(CoreCxxTests ${TestFiles})

# Compile definitions
target_compile_definitions(CoreCxxTests PRIVATE XSIGMA_GOOGLE_TEST USE_GTEST _VARIADIC_MAX=10)

# Link libraries Core library is linked which transitively brings in all centralized dependencies
# via XSIGMA_DEPENDENCY_LIBS
target_link_libraries(CoreCxxTests PRIVATE gtest gtest_main XSigma::Core)

# Set target properties
set_target_properties(
  CoreCxxTests PROPERTIES FOLDER "Tests" VS_DEBUGGER_WORKING_DIRECTORY
                                         "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# ============================================================================= Google Test
# Discovery for CTest and Visual Studio Test Explorer
# =============================================================================

# Register test with CTest
add_test(NAME CoreCxxTests COMMAND CoreCxxTests
         WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# Set test properties
set_tests_properties(CoreCxxTests PROPERTIES TIMEOUT 300 LABELS "Core;Unit")

# Set coverage environment variable if enabled
if(XSIGMA_ENABLE_COVERAGE AND DEFINED XSIGMA_LLVM_PROFILE_FILE)
  set_tests_properties(
    CoreCxxTests PROPERTIES ENVIRONMENT "LLVM_PROFILE_FILE=${XSIGMA_LLVM_PROFILE_FILE}"
  )
endif()

message(STATUS "✓ CoreCxxTests registered with CTest")
message(STATUS "  - Working directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# ============================================================================= Output Directory
# Configuration
# =============================================================================
if(COMMAND xsigma_set_target_output_directories)
  xsigma_set_target_output_directories(CoreCxxTests)
endif()

# ============================================================================= BenchmarkSimple
# Executable (following cpuinfo pattern)
# =============================================================================
if(XSIGMA_ENABLE_BENCHMARK)

  file(GLOB_RECURSE bench_sources LIST_DIRECTORIES false
       "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cu"
  )

  # Exclude CUDA-related benchmarks when GPU support is disabled
  if(NOT XSIGMA_ENABLE_CUDA AND NOT XSIGMA_ENABLE_HIP)
    set(filtered_bench_sources)
    foreach(_bench_source IN ITEMS ${bench_sources})
      get_filename_component(file_name "${_bench_source}" NAME)
      # Exclude CUDA-related benchmarks
      if(NOT file_name MATCHES ".*[Cc]uda.*")
        list(APPEND filtered_bench_sources "${_bench_source}")
      else()
        message(STATUS "Excluding CUDA benchmark (no GPU support): ${file_name}")
      endif()
    endforeach()
    set(bench_sources ${filtered_bench_sources})
  endif()

  add_executable(CoreCxxBenchmark ${bench_sources})

  # Link libraries Core library is linked which transitively brings in all centralized dependencies
  # via XSIGMA_DEPENDENCY_LIBS Link benchmark_main to provide main() function (instead of
  # BENCHMARK_MAIN() in each file)
  target_link_libraries(CoreCxxBenchmark PRIVATE benchmark benchmark_main XSigma::Core)

  # Set target properties
  set_target_properties(CoreCxxBenchmark PROPERTIES FOLDER "Benchmarks")

  # Apply XSigma output directory configuration
  if(COMMAND xsigma_set_target_output_directories)
    xsigma_set_target_output_directories(CoreCxxBenchmark)
  endif()

  message(STATUS "✓ BenchmarkSimple executable configured")
endif()

# ============================================================================= ProfileParallelBackends
# Executable (Profiling Tool)
# =============================================================================
# This executable profiles the parallel backends (OpenMP, TBB, Native) using Kineto
# Build separately for each backend to compare performance

add_executable(ProfileParallelBackends ProfileParallelBackends.cpp)

# Link libraries
target_link_libraries(ProfileParallelBackends PRIVATE XSigma::Core)

# Set target properties
set_target_properties(ProfileParallelBackends PROPERTIES FOLDER "Profiling")

# Apply XSigma output directory configuration
if(COMMAND xsigma_set_target_output_directories)
  xsigma_set_target_output_directories(ProfileParallelBackends)
endif()

message(STATUS "✓ ProfileParallelBackends executable configured")
message(STATUS "  - Backend: ${XSIGMA_PARALLEL_BACKEND}")
message(STATUS "  - OpenMP: ${XSIGMA_ENABLE_OPENMP}")
message(STATUS "  - TBB: ${XSIGMA_ENABLE_TBB}")
