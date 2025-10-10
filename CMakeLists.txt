cmake_minimum_required(VERSION 3.16)

# Set project name and version
project(XSigma
    VERSION 1.0.0
    DESCRIPTION "XSigma High-Performance Computing Library"
    LANGUAGES CXX C
)

# Set version macros
set(XSIGMA_MAJOR_VERSION 1)
set(XSIGMA_MINOR_VERSION 0)
set(XSIGMA_BUILD_VERSION 0)
set(XSIGMA_VERSION_FULL "${XSIGMA_MAJOR_VERSION}.${XSIGMA_MINOR_VERSION}.${XSIGMA_BUILD_VERSION}")

# Add CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/tools")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/flags")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/packages")

# Build optimization settings
option(XSIGMA_ENABLE_LTO "Enable Link Time Optimization" ON)

# Include necessary CMake modules early
include(CMakeDependentOption)

# Set C++ standard configuration
set(XSIGMA_CXX_STANDARD
    "17"
    CACHE STRING
    "C++ standard to use. Options are 17, 20, or 23"
)
mark_as_advanced(XSIGMA_CXX_STANDARD)

# Validate C++ standard
if(NOT XSIGMA_CXX_STANDARD MATCHES "^(17|20|23)$")
    message(FATAL_ERROR "XSIGMA_CXX_STANDARD must be 17, 20, or 23")
endif()

set(CMAKE_CXX_STANDARD ${XSIGMA_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable options
option(XSIGMA_ENABLE_NUMA "Enable numa node" OFF)
option(XSIGMA_ENABLE_MKL "Enable MKL" OFF)
option(XSIGMA_ENABLE_CUDA "Enable CUDA compilation" OFF)
option(XSIGMA_ENABLE_MEMKIND "Build support for extended memory" OFF)

option(XSIGMA_ENABLE_IWYU "Enable include-what-you-use (iwyu) support." OFF)
option(XSIGMA_ENABLE_CLANGTIDY "enable clangtidy check" OFF)

option(XSIGMA_SOBOL_1111 "Build support sobol with 1111 dimension" ON)
option(XSIGMA_LU_PIVOTING "Enable LU PIVOTING" OFF)

# Testing options
set(XSIGMA_BUILD_TESTING
    "ON"
    CACHE STRING "Build module testing directories")
set_property(CACHE XSIGMA_BUILD_TESTING PROPERTY STRINGS "ON;OFF;WANT")
option(XSIGMA_ENABLE_BENCHMARK "Enable google benchmark" OFF)
option(XSIGMA_ENABLE_GTEST "Enable google test" ON)

option(XSIGMA_ENABLE_VALGRIND "Execute test suite with Valgrind" OFF)
option(XSIGMA_ENABLE_COVERAGE "Build XSIGMA with coverage" OFF)
option(XSIGMA_ENABLE_SANITIZER "Build with sanitizer support (Clang only)" OFF)
set(XSIGMA_SANITIZER_TYPE
        "address"
        CACHE STRING "The sanitizer to use. Options are address, undefined, thread, memory, leak")

set_property(CACHE XSIGMA_SANITIZER_TYPE PROPERTY STRINGS address undefined thread memory leak)

option(XSIGMA_ENABLE_SPELL "Enable spell checking with automatic corrections (WARNING: modifies source files)" OFF)

# Vectorization options
set(XSIGMA_VECTORIZATION_TYPE
    "avx2"
    CACHE
      STRING
      "vectorization implementation to use. Options are no, sse, avx, avx2, or avx512"
)
set_property(CACHE XSIGMA_VECTORIZATION_TYPE PROPERTY STRINGS no sse avx avx2 avx512)
cmake_dependent_option(XSIGMA_USE_HIP "Support HIP backend accelerator" OFF
  "CMAKE_VERSION VERSION_GREATER_EQUAL 3.21;NOT XSIGMA_ENABLE_CUDA" OFF)

# Mandatory core dependencies (always included)
# - fmt: Modern C++ formatting library
# - cpuinfo: CPU feature detection library

# Optional third-party library options
option(XSIGMA_ENABLE_EXTERNAL "Use external copies of third party libraries by default" OFF)
option(XSIGMA_ENABLE_MAGICENUM "Enable magic_enum static reflection for enums." ON)
mark_as_advanced(XSIGMA_ENABLE_MAGICENUM)

# Compression support options
option(XSIGMA_ENABLE_COMPRESSION "Enable compression support" OFF)
set(XSIGMA_COMPRESSION_TYPE
    "none"
    CACHE STRING "Compression library to use. Options are  none, snappy")
set_property(CACHE XSIGMA_COMPRESSION_TYPE PROPERTY STRINGS none snappy)
mark_as_advanced(XSIGMA_ENABLE_COMPRESSION XSIGMA_COMPRESSION_TYPE)

# Logging backend selection (mutually exclusive)
set(XSIGMA_LOGGING_BACKEND
    "LOGURU"
    CACHE STRING
    "Logging backend to use. Options are NATIVE, LOGURU, or GLOG"
)
set_property(CACHE XSIGMA_LOGGING_BACKEND PROPERTY STRINGS NATIVE LOGURU GLOG)
mark_as_advanced(XSIGMA_LOGGING_BACKEND)

# Validate logging backend selection
if(NOT XSIGMA_LOGGING_BACKEND MATCHES "^(NATIVE|LOGURU|GLOG)$")
    message(FATAL_ERROR "XSIGMA_LOGGING_BACKEND must be NATIVE, LOGURU, or GLOG (got: ${XSIGMA_LOGGING_BACKEND})")
endif()

# Set preprocessor definitions based on selected backend
if(XSIGMA_LOGGING_BACKEND STREQUAL "NATIVE")
    message(STATUS "Using NATIVE logging backend")
    set(XSIGMA_USE_NATIVE_LOGGING ON)
elseif(XSIGMA_LOGGING_BACKEND STREQUAL "LOGURU")
    message(STATUS "Using LOGURU logging backend")
    set(XSIGMA_USE_LOGURU ON)
elseif(XSIGMA_LOGGING_BACKEND STREQUAL "GLOG")
    message(STATUS "Using GLOG logging backend")
    set(XSIGMA_USE_GLOG ON)
endif()

# Intel TBB (Threading Building Blocks) support with automatic fallback
option(XSIGMA_ENABLE_TBB "Enable Intel TBB (Threading Building Blocks) support with automatic fallback." ON)
mark_as_advanced(XSIGMA_ENABLE_TBB)

option(XSIGMA_ENABLE_MIMALLOC "Enable Microsoft mimalloc high-performance memory allocator support." ON)
mark_as_advanced(XSIGMA_ENABLE_MIMALLOC)

if(NOT UNIX)
  set(XSIGMA_ENABLE_NUMA OFF)
endif()

# Add an option to enable memkind
if (NOT UNIX OR APPLE)
  set(XSIGMA_ENABLE_MEMKIND OFF)
endif()

if(XSIGMA_ENABLE_SANITIZER)
  set(XSIGMA_ENABLE_MIMALLOC OFF)
endif()

# Compression configuration validation and setup
if(XSIGMA_ENABLE_COMPRESSION)
    if(XSIGMA_COMPRESSION_TYPE STREQUAL "SNAPPY")
        set(XSIGMA_COMPRESSION_TYPE_SNAPPY ON)
        message(STATUS "Compression enabled: Snappy")
    elseif(XSIGMA_COMPRESSION_TYPE STREQUAL "NONE")
        set(XSIGMA_ENABLE_COMPRESSION OFF)
        message(STATUS "Compression type set to NONE - disabling compression")
    else()
        message(FATAL_ERROR "Invalid XSIGMA_COMPRESSION_TYPE: ${XSIGMA_COMPRESSION_TYPE}. Valid options are: NONE, SNAPPY")
    endif()
else()
    message(STATUS "Compression disabled")
endif()

# Set global variables
set(XSIGMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(XSIGMA_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(XSIGMA_VERSION "${PROJECT_VERSION}")

# Build shared libraries option
option(XSIGMA_BUILD_SHARED_LIBS "Build shared libraries" ON)
set(BUILD_SHARED_LIBS ${XSIGMA_BUILD_SHARED_LIBS})

# Include necessary CMake modules
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(CheckSymbolExists)

include(build_type)
include(checks)
include(xsigmaUtils)
include(platform)

# Configure optional features
include(cuda)
include(mkl)
include(numa)
include(tbb)
include(clangTidy)
include(spell)
include(iwyu)
include(coverage)
include(valgrind)
include(helperMacros)

# Create build interface target for common settings
add_library(xsigmabuild INTERFACE)
add_library(XSigma::build ALIAS xsigmabuild)

# Include sanitizer configuration after xsigmabuild target is created
include(sanitize)

# Set compile features
target_compile_features(xsigmabuild INTERFACE cxx_std_${XSIGMA_CXX_STANDARD})

# Apply common linker flags
if(DEFINED XSIGMA_LINKER_FLAGS_COMMON)
    target_link_options(xsigmabuild INTERFACE ${XSIGMA_LINKER_FLAGS_COMMON})
endif()




set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${XSIGMA_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${XSIGMA_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${XSIGMA_BINARY_DIR}/lib")

# Configure testing
if(XSIGMA_BUILD_TESTING STREQUAL "ON" OR XSIGMA_BUILD_TESTING STREQUAL "WANT")
    enable_testing()
    include(CTest)


endif()

# Add third-party dependencies
add_subdirectory(ThirdParty)

# Add subdirectories
add_subdirectory(Library)

# Apply Valgrind timeout multiplier to all tests (if Valgrind is enabled)
# This must be called after all tests are registered
if(XSIGMA_ENABLE_VALGRIND AND XSIGMA_BUILD_TESTING)
    xsigma_apply_valgrind_timeouts()
endif()

# Include summary at the end
include(summary)
print_configuration_summary()
