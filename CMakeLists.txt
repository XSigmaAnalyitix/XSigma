cmake_minimum_required(VERSION 3.16)

# Set project name and version
project(XSigma
    VERSION 1.0.0
    DESCRIPTION "XSigma High-Performance Computing Library"
    LANGUAGES CXX C
)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Platform detection
if(WIN32)
    set(XSIGMA_PLATFORM_WINDOWS TRUE)
elseif(APPLE)
    set(XSIGMA_PLATFORM_MACOS TRUE)
elseif(UNIX)
    set(XSIGMA_PLATFORM_LINUX TRUE)
endif()

# Compiler detection
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(XSIGMA_COMPILER_MSVC TRUE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(XSIGMA_COMPILER_GCC TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(XSIGMA_COMPILER_CLANG TRUE)
endif()

# Build optimization settings
option(XSIGMA_ENABLE_LTO "Enable Link Time Optimization" ON)

# Configure compiler-specific optimization flags
if(XSIGMA_COMPILER_MSVC)
    # MSVC optimization flags
    set(XSIGMA_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
    set(XSIGMA_CXX_FLAGS_DEBUG "/Od /Zi /RTC1")
    set(XSIGMA_CXX_FLAGS_RELWITHDEBINFO "/O2 /Ob1 /Zi /DNDEBUG")
    set(XSIGMA_CXX_FLAGS_MINSIZEREL "/O1 /Ob1 /DNDEBUG")

    # Enable parallel compilation
    set(XSIGMA_CXX_FLAGS_COMMON "/MP")

    # LTO for MSVC
    if(XSIGMA_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
        set(XSIGMA_CXX_FLAGS_RELEASE "${XSIGMA_CXX_FLAGS_RELEASE} /GL")
        set(XSIGMA_LINKER_FLAGS_RELEASE "/LTCG")
    endif()

elseif(XSIGMA_COMPILER_GCC OR XSIGMA_COMPILER_CLANG)
    # GCC/Clang optimization flags
    set(XSIGMA_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(XSIGMA_CXX_FLAGS_DEBUG "-O0 -g")
    set(XSIGMA_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(XSIGMA_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

    # Common flags for better performance and debugging
    set(XSIGMA_CXX_FLAGS_COMMON "-fPIC")

    # LTO for GCC/Clang
    if(XSIGMA_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
        if(XSIGMA_COMPILER_GCC)
            set(XSIGMA_CXX_FLAGS_RELEASE "${XSIGMA_CXX_FLAGS_RELEASE} -flto")
            set(XSIGMA_LINKER_FLAGS_RELEASE "-flto")
        elseif(XSIGMA_COMPILER_CLANG)
            set(XSIGMA_CXX_FLAGS_RELEASE "${XSIGMA_CXX_FLAGS_RELEASE} -flto=thin")
            set(XSIGMA_LINKER_FLAGS_RELEASE "-flto=thin")
        endif()
    endif()

    # Platform-specific flags
    if(XSIGMA_PLATFORM_LINUX)
        set(XSIGMA_CXX_FLAGS_COMMON "${XSIGMA_CXX_FLAGS_COMMON} -pthread")
        set(XSIGMA_LINKER_FLAGS_COMMON "-pthread")
    endif()
endif()

# Add CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/tools")

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
option(XSIGMA_LU_PIVOTING "Enable LU PIVOTING" OFF)

option(XSIGMA_ENABLE_CUDA "Enable CUDA compilation" OFF)
option(XSIGMA_ENABLE_IWYU "Enable include-what-you-use (iwyu) support." OFF)
option(XSIGMA_ENABLE_MEMKIND "Build support for extended memory" OFF)
option(XSIGMA_SOBOL_1111 "Build support sobol with 1111 dimension" ON)

# Testing options
set(XSIGMA_BUILD_TESTING
    "OFF"
    CACHE STRING "Build module testing directories")
set_property(CACHE XSIGMA_BUILD_TESTING PROPERTY STRINGS "ON;OFF;WANT")

option(XSIGMA_ENABLE_VALGRIND "Execute test suite with Valgrind" OFF)
option(XSIGMA_ENABLE_COVERAGE "Build XSIGMA with coverage" OFF)
option(XSIGMA_ENABLE_BENCHMARK "Enable google benchmark" OFF)
option(XSIGMA_GOOGLE_TEST "Enable google test" OFF)

option(XSIGMA_ENABLE_CLANGTIDY "enable clangtidy check" OFF)
option(XSIGMA_USE_EXTERNAL "Use external copies of third party libraries by default" OFF)
option(XSIGMA_ENABLE_SANITIZER "Build with sanitizer support." OFF)
set(XSIGMA_SANITIZER_TYPE
        "address"
        CACHE STRING "The sanitizer to use. Options are address, undefined, thread, memory, leak")

set_property(CACHE XSIGMA_SANITIZER_TYPE PROPERTY STRINGS address undefined thread memory leak)

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
option(XSIGMA_ENABLE_MAGIC_ENUM "Enable magic_enum static reflection for enums." ON)
mark_as_advanced(XSIGMA_ENABLE_MAGIC_ENUM)

option(XSIGMA_ENABLE_LOGURU "Enable loguru lightweight C++ logging library." ON)
mark_as_advanced(XSIGMA_ENABLE_LOGURU)

option(XSIGMA_ENABLE_TBB "Enable Intel TBB (Threading Building Blocks) support." OFF)
mark_as_advanced(XSIGMA_ENABLE_TBB)

option(XSIGMA_ENABLE_MIMALLOC "Enable Microsoft mimalloc high-performance memory allocator support." OFF)
mark_as_advanced(XSIGMA_ENABLE_MIMALLOC)

if(NOT UNIX)
  set(XSIGMA_ENABLE_NUMA OFF)
endif()

# Add an option to enable memkind
if (NOT UNIX OR APPLE)
  set(XSIGMA_ENABLE_MEMKIND OFF)
endif()

# Set global variables
set(XSIGMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(XSIGMA_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(XSIGMA_VERSION "${PROJECT_VERSION}")

# Build shared libraries option
option(XSIGMA_BUILD_SHARED_LIBS "Build shared libraries" OFF)
set(BUILD_SHARED_LIBS ${XSIGMA_BUILD_SHARED_LIBS})

# Include necessary CMake modules
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(CheckSymbolExists)

# Create build interface target for common settings
add_library(xsigmabuild INTERFACE)
add_library(XSigma::build ALIAS xsigmabuild)

# Set compile features
target_compile_features(xsigmabuild INTERFACE cxx_std_${XSIGMA_CXX_STANDARD})

# Apply compiler-specific flags to build interface
if(DEFINED XSIGMA_CXX_FLAGS_COMMON)
    # Convert string to list for proper flag handling
    string(REPLACE " " ";" XSIGMA_CXX_FLAGS_COMMON_LIST "${XSIGMA_CXX_FLAGS_COMMON}")
    target_compile_options(xsigmabuild INTERFACE ${XSIGMA_CXX_FLAGS_COMMON_LIST})
endif()

# Apply build-type specific flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND DEFINED XSIGMA_CXX_FLAGS_RELEASE)
    # Convert string to list for proper flag handling
    string(REPLACE " " ";" XSIGMA_CXX_FLAGS_RELEASE_LIST "${XSIGMA_CXX_FLAGS_RELEASE}")
    target_compile_options(xsigmabuild INTERFACE $<$<CONFIG:Release>:${XSIGMA_CXX_FLAGS_RELEASE_LIST}>)
    if(DEFINED XSIGMA_LINKER_FLAGS_RELEASE)
        string(REPLACE " " ";" XSIGMA_LINKER_FLAGS_RELEASE_LIST "${XSIGMA_LINKER_FLAGS_RELEASE}")
        target_link_options(xsigmabuild INTERFACE $<$<CONFIG:Release>:${XSIGMA_LINKER_FLAGS_RELEASE_LIST}>)
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug" AND DEFINED XSIGMA_CXX_FLAGS_DEBUG)
    string(REPLACE " " ";" XSIGMA_CXX_FLAGS_DEBUG_LIST "${XSIGMA_CXX_FLAGS_DEBUG}")
    target_compile_options(xsigmabuild INTERFACE $<$<CONFIG:Debug>:${XSIGMA_CXX_FLAGS_DEBUG_LIST}>)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" AND DEFINED XSIGMA_CXX_FLAGS_RELWITHDEBINFO)
    string(REPLACE " " ";" XSIGMA_CXX_FLAGS_RELWITHDEBINFO_LIST "${XSIGMA_CXX_FLAGS_RELWITHDEBINFO}")
    target_compile_options(xsigmabuild INTERFACE $<$<CONFIG:RelWithDebInfo>:${XSIGMA_CXX_FLAGS_RELWITHDEBINFO_LIST}>)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" AND DEFINED XSIGMA_CXX_FLAGS_MINSIZEREL)
    string(REPLACE " " ";" XSIGMA_CXX_FLAGS_MINSIZEREL_LIST "${XSIGMA_CXX_FLAGS_MINSIZEREL}")
    target_compile_options(xsigmabuild INTERFACE $<$<CONFIG:MinSizeRel>:${XSIGMA_CXX_FLAGS_MINSIZEREL_LIST}>)
endif()

# Apply common linker flags
if(DEFINED XSIGMA_LINKER_FLAGS_COMMON)
    target_link_options(xsigmabuild INTERFACE ${XSIGMA_LINKER_FLAGS_COMMON})
endif()

# Include utility functions and configurations
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/tools/xsigmaUtils.cmake")
    include(xsigmaUtils)
endif()

# Include testing utilities
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/tools/xsigmaTestUtils.cmake")
    include(xsigmaTestUtils)
endif()

# Vectorization is handled by the existing vectorization detection in xsigmaUtils.cmake

# Configure optional features
if(XSIGMA_ENABLE_MKL)
    include(mkl)
endif()

if(XSIGMA_ENABLE_NUMA)
    include(numa)
endif()

if(XSIGMA_ENABLE_IWYU)
    include(iwyu)
endif()

if(XSIGMA_ENABLE_CLANGTIDY)
    include(clangTidy)
endif()

if(XSIGMA_ENABLE_COVERAGE)
    include(coverage)
endif()

if(XSIGMA_ENABLE_SANITIZER)
    include(xsigmaSanitize)
endif()

if(XSIGMA_ENABLE_VALGRIND)
    include(memcheckValgrind)
endif()

# Configure testing
if(XSIGMA_BUILD_TESTING STREQUAL "ON" OR XSIGMA_BUILD_TESTING STREQUAL "WANT")
    enable_testing()
    include(CTest)
    
    if(XSIGMA_GOOGLE_TEST OR XSIGMA_ENABLE_BENCHMARK)
        include(googletest)
    endif()
endif()

# Add third-party dependencies
add_subdirectory(ThirdParty)

# Add subdirectories
add_subdirectory(Library)

# Include summary at the end
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/tools/summary.cmake")
    include(summary)
    print_configuration_summary()
endif()
