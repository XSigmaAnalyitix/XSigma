# Third-party dependencies for XSigma
cmake_minimum_required(VERSION 3.16)
message(STATUS "DEBUG: Processing ThirdParty/CMakeLists.txt")

# Force static libraries for third-party dependencies (except TBB)
# Save the current BUILD_SHARED_LIBS setting
set(_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static build for third-party libraries" FORCE)

# Option to use external libraries instead of bundled ones
option(XSIGMA_ENABLE_EXTERNAL "Use external copies of third party libraries by default" OFF)

# Configure Visual Studio folder structure to match file system
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Disable tests, examples, and benchmarks for all third-party libraries
set(BUILD_TESTING OFF CACHE BOOL "Disable testing for third-party libraries" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable examples for third-party libraries" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable benchmarks for third-party libraries" FORCE)

# fmt-specific options
set(FMT_TEST OFF CACHE BOOL "Disable fmt tests" FORCE)
set(FMT_DOC OFF CACHE BOOL "Disable fmt documentation" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "Disable fmt install" FORCE)

# cpuinfo-specific options
set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "Disable cpuinfo tools" FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable cpuinfo unit tests" FORCE)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Disable cpuinfo mock tests" FORCE)
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Disable cpuinfo benchmarks" FORCE)

# magic_enum-specific options
set(MAGIC_ENUM_OPT_BUILD_EXAMPLES OFF CACHE BOOL "Disable magic_enum examples" FORCE)
set(MAGIC_ENUM_OPT_BUILD_TESTS OFF CACHE BOOL "Disable magic_enum tests" FORCE)

# loguru-specific options
set(LOGURU_BUILD_EXAMPLES OFF CACHE BOOL "Disable loguru examples" FORCE)

# googletest-specific options
set(BUILD_GMOCK OFF CACHE BOOL "Disable Google Mock" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable Google Test install" FORCE)
set(gtest_build_samples OFF CACHE BOOL "Disable Google Test samples" FORCE)
set(gtest_build_tests OFF CACHE BOOL "Disable Google Test tests" FORCE)



# Google Benchmark-specific options
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
set(BENCHMARK_ENABLE_EXCEPTIONS ON CACHE BOOL "Enable benchmark exceptions" FORCE)
set(BENCHMARK_ENABLE_LTO OFF CACHE BOOL "Disable benchmark LTO" FORCE)
set(BENCHMARK_USE_LIBCXX OFF CACHE BOOL "Disable benchmark libcxx" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "Disable benchmark werror" FORCE)
set(BENCHMARK_FORCE_WERROR OFF CACHE BOOL "Disable benchmark force werror" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "Disable benchmark docs install" FORCE)
set(BENCHMARK_ENABLE_DOXYGEN OFF CACHE BOOL "Disable benchmark doxygen" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest tests" FORCE)
set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "Don't use bundled gtest for benchmark" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Don't download dependencies" FORCE)

# mimalloc-specific options
set(MI_BUILD_TESTS OFF CACHE BOOL "Disable mimalloc tests" FORCE)
set(MI_BUILD_OBJECT OFF CACHE BOOL "Disable mimalloc object library" FORCE)
set(MI_INSTALL_TOPLEVEL OFF CACHE BOOL "Disable mimalloc top-level install" FORCE)
set(MI_OVERRIDE OFF CACHE BOOL "Disable mimalloc malloc override" FORCE)
set(MI_BUILD_SHARED OFF CACHE BOOL "Build mimalloc as static library" FORCE)
set(MI_BUILD_STATIC ON CACHE BOOL "Build mimalloc as static library" FORCE)

# Function to add third-party library with external option
function(add_third_party_library name)
    cmake_parse_arguments(TPL "" "EXTERNAL_NAME;CONDITION" "" ${ARGN})

    if(NOT DEFINED TPL_EXTERNAL_NAME)
        set(TPL_EXTERNAL_NAME ${name})
    endif()

    # Check condition if provided
    if(DEFINED TPL_CONDITION)
        if(NOT ${TPL_CONDITION})
            return()
        endif()
    endif()

    # Check if already processed to avoid duplicates
    get_property(XSIGMA_${name}_PROCESSED GLOBAL PROPERTY XSIGMA_${name}_PROCESSED)
    if(XSIGMA_${name}_PROCESSED)
        message(STATUS "Third-party library ${name} already processed, skipping")
        return()
    endif()
    set_property(GLOBAL PROPERTY XSIGMA_${name}_PROCESSED TRUE)

    # Try to find external library first if requested
    if(XSIGMA_ENABLE_EXTERNAL)
        find_package(${TPL_EXTERNAL_NAME} QUIET)
        if(${TPL_EXTERNAL_NAME}_FOUND OR ${name}_FOUND)
            message(STATUS "Using external ${name}")
            return()
        endif()
    endif()

    # Use bundled version
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name}/CMakeLists.txt")
        message(STATUS "Using bundled ${name}")
        # Add the library
        add_subdirectory(${name} ${CMAKE_CURRENT_BINARY_DIR}/${name}_build)
    else()
        message(WARNING "Third-party library ${name} not found in ${CMAKE_CURRENT_SOURCE_DIR}/${name}")
    endif()
endfunction()

# Add third-party libraries conditionally based on XSIGMA_ENABLE_XXX options

# Define default enable options for core libraries if not set
if(NOT DEFINED XSIGMA_ENABLE_FMT)
    set(XSIGMA_ENABLE_FMT ON CACHE BOOL "Enable fmt library")
endif()

if(NOT DEFINED XSIGMA_ENABLE_CPUINFO)
    set(XSIGMA_ENABLE_CPUINFO ON CACHE BOOL "Enable cpuinfo library")
endif()

if(NOT DEFINED XSIGMA_ENABLE_MAGICENUM)
    set(XSIGMA_ENABLE_MAGICENUM ON CACHE BOOL "Enable magic_enum library")
endif()

if(NOT DEFINED XSIGMA_ENABLE_LOGURU)
    set(XSIGMA_ENABLE_LOGURU ON CACHE BOOL "Enable loguru library")
endif()

# fmt - Modern C++ formatting library (mandatory dependency)
add_third_party_library(fmt)
if(TARGET fmt)
    set_target_properties(fmt PROPERTIES FOLDER "ThirdParty/fmt")
endif()
if(TARGET fmt-header-only)
    set_target_properties(fmt-header-only PROPERTIES FOLDER "ThirdParty/fmt")
endif()

# cpuinfo - CPU feature detection library (mandatory dependency)
add_third_party_library(cpuinfo)
if(TARGET cpuinfo)
    set_target_properties(cpuinfo PROPERTIES FOLDER "ThirdParty/cpuinfo")
endif()
if(TARGET cpuinfo_internals)
    set_target_properties(cpuinfo_internals PROPERTIES FOLDER "ThirdParty/cpuinfo")
endif()

# magic_enum - Static reflection for enums in C++
add_third_party_library(magic_enum
    CONDITION XSIGMA_ENABLE_MAGICENUM
)
if(TARGET magic_enum)
    set_target_properties(magic_enum PROPERTIES FOLDER "ThirdParty/magic_enum")
endif()

# loguru - Lightweight C++ logging library
add_third_party_library(loguru
    CONDITION XSIGMA_ENABLE_LOGURU
)
if(TARGET loguru)
    set_target_properties(loguru PROPERTIES FOLDER "ThirdParty/loguru")
endif()

# TBB - Intel Threading Building Blocks (conditional) - Force shared library build
if(XSIGMA_ENABLE_TBB)
    # Use our custom FindTBB.cmake
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Cmake/packages")
    find_package(TBB REQUIRED)
    if(TBB_FOUND)
        message(STATUS "Using external TBB from: ${TBB_INCLUDE_DIR}")
        message(STATUS "TBB libraries: ${TBB_LIBRARIES}")

        # Ensure TBB targets have proper output directory configuration
        if(TARGET TBB::tbb AND COMMAND xsigma_set_target_output_directories)
            xsigma_set_target_output_directories(TBB::tbb)
        endif()
        if(TARGET TBB::tbbmalloc AND COMMAND xsigma_set_target_output_directories)
            xsigma_set_target_output_directories(TBB::tbbmalloc)
        endif()
        if(TARGET TBB::tbb12 AND COMMAND xsigma_set_target_output_directories)
            xsigma_set_target_output_directories(TBB::tbb12)
        endif()
        if(TARGET TBB::tbbmalloc_proxy AND COMMAND xsigma_set_target_output_directories)
            xsigma_set_target_output_directories(TBB::tbbmalloc_proxy)
        endif()
    endif()
endif()

# mimalloc - Microsoft's high-performance memory allocator (conditional)
add_third_party_library(mimalloc
    CONDITION XSIGMA_ENABLE_MIMALLOC
)
if(TARGET mimalloc)
    set_target_properties(mimalloc PROPERTIES FOLDER "ThirdParty/mimalloc")
endif()
if(TARGET mimalloc-static)
    set_target_properties(mimalloc-static PROPERTIES FOLDER "ThirdParty/mimalloc")
endif()

# Google Benchmark and Google Test are handled by Cmake/tools/googletest.cmake
# which is included from the main CMakeLists.txt when XSIGMA_ENABLE_GTEST or XSIGMA_ENABLE_BENCHMARK is enabled
# We only need to set folder properties here if the targets exist

if(TARGET benchmark)
    set_target_properties(benchmark PROPERTIES FOLDER "ThirdParty/benchmark")
endif()
if(TARGET benchmark_main)
    set_target_properties(benchmark_main PROPERTIES FOLDER "ThirdParty/benchmark")
endif()

# Set folder properties for googletest targets
if(TARGET gtest)
    set_target_properties(gtest PROPERTIES FOLDER "ThirdParty/googletest")
endif()
if(TARGET gtest_main)
    set_target_properties(gtest_main PROPERTIES FOLDER "ThirdParty/googletest")
endif()
if(TARGET gmock)
    set_target_properties(gmock PROPERTIES FOLDER "ThirdParty/googletest")
endif()
if(TARGET gmock_main)
    set_target_properties(gmock_main PROPERTIES FOLDER "ThirdParty/googletest")
endif()

# Create interface targets for easier linking
# Mandatory dependencies (always created)
if(TARGET fmt::fmt)
    # fmt::fmt might be an alias, so we need to get the actual target
    get_target_property(fmt_aliased_target fmt::fmt ALIASED_TARGET)
    if(fmt_aliased_target)
        add_library(XSigma::fmt ALIAS ${fmt_aliased_target})
    else()
        add_library(XSigma::fmt ALIAS fmt::fmt)
    endif()
    message(STATUS "Created XSigma::fmt target alias")
elseif(TARGET fmt)
    add_library(XSigma::fmt ALIAS fmt)
    message(STATUS "Created XSigma::fmt target alias")
endif()

if(TARGET cpuinfo)
    add_library(XSigma::cpuinfo ALIAS cpuinfo)
    message(STATUS "Created XSigma::cpuinfo target alias")
endif()

# Optional dependencies (only if enabled)

if(XSIGMA_ENABLE_MAGICENUM)
    if(TARGET magic_enum::magic_enum)
        # magic_enum::magic_enum might be an alias, so we need to get the actual target
        get_target_property(magic_enum_aliased_target magic_enum::magic_enum ALIASED_TARGET)
        if(magic_enum_aliased_target)
            add_library(XSigma::magic_enum ALIAS ${magic_enum_aliased_target})
        else()
            add_library(XSigma::magic_enum ALIAS magic_enum::magic_enum)
        endif()
        message(STATUS "Created XSigma::magic_enum target alias")
    elseif(TARGET magic_enum)
        add_library(XSigma::magic_enum ALIAS magic_enum)
        message(STATUS "Created XSigma::magic_enum target alias")
    endif()
endif()

if(XSIGMA_ENABLE_LOGURU)
    if(TARGET loguru::loguru)
        # loguru::loguru might be an alias, so we need to get the actual target
        get_target_property(loguru_aliased_target loguru::loguru ALIASED_TARGET)
        if(loguru_aliased_target)
            add_library(XSigma::loguru ALIAS ${loguru_aliased_target})
        else()
            add_library(XSigma::loguru ALIAS loguru::loguru)
        endif()
        message(STATUS "Created XSigma::loguru target alias")
    elseif(TARGET loguru)
        add_library(XSigma::loguru ALIAS loguru)
        message(STATUS "Created XSigma::loguru target alias")
    endif()
endif()

if(XSIGMA_ENABLE_MIMALLOC)
    if(TARGET mimalloc)
        add_library(XSigma::mimalloc ALIAS mimalloc)
        message(STATUS "Created XSigma::mimalloc target alias")
    elseif(TARGET mimalloc-static)
        add_library(XSigma::mimalloc ALIAS mimalloc-static)
        message(STATUS "Created XSigma::mimalloc target alias")
    endif()
endif()

if(XSIGMA_ENABLE_BENCHMARK)
    if(TARGET benchmark::benchmark)
        # Get the actual target that benchmark::benchmark points to
        get_target_property(benchmark_aliased_target benchmark::benchmark ALIASED_TARGET)
        if(benchmark_aliased_target)
            add_library(XSigma::benchmark ALIAS ${benchmark_aliased_target})
        else()
            # If it's not an alias, create alias directly
            add_library(XSigma::benchmark ALIAS benchmark::benchmark)
        endif()
        message(STATUS "Created XSigma::benchmark target alias")
    elseif(TARGET benchmark)
        add_library(XSigma::benchmark ALIAS benchmark)
        message(STATUS "Created XSigma::benchmark target alias")
    endif()
endif()

if(XSIGMA_ENABLE_GTEST)
    if(TARGET GTest::gtest)
        # Get the actual targets that GTest aliases point to
        get_target_property(gtest_aliased_target GTest::gtest ALIASED_TARGET)
        get_target_property(gtest_main_aliased_target GTest::gtest_main ALIASED_TARGET)

        if(gtest_aliased_target)
            add_library(XSigma::gtest ALIAS ${gtest_aliased_target})
        else()
            add_library(XSigma::gtest ALIAS GTest::gtest)
        endif()

        if(gtest_main_aliased_target)
            add_library(XSigma::gtest_main ALIAS ${gtest_main_aliased_target})
        else()
            add_library(XSigma::gtest_main ALIAS GTest::gtest_main)
        endif()

        message(STATUS "Created XSigma::gtest target aliases")
    elseif(TARGET gtest)
        add_library(XSigma::gtest ALIAS gtest)
        message(STATUS "Created XSigma::gtest target alias")
        if(TARGET gtest_main)
            add_library(XSigma::gtest_main ALIAS gtest_main)
        endif()
    endif()
endif()

