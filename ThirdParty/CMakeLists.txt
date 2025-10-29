# Third-party dependencies for XSigma
cmake_minimum_required(VERSION 3.16)
message(STATUS "DEBUG: Processing ThirdParty/CMakeLists.txt")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "Type of cpuinfo library (shared, static, or default) to build")

# magic_enum-specific options
set(MAGIC_ENUM_OPT_BUILD_EXAMPLES OFF CACHE BOOL "Disable magic_enum examples" FORCE)
set(MAGIC_ENUM_OPT_BUILD_TESTS OFF CACHE BOOL "Disable magic_enum tests" FORCE)

# loguru-specific options
set(LOGURU_BUILD_EXAMPLES OFF CACHE BOOL "Disable loguru examples" FORCE)
#set(LOGURU_USE_FMTLIB ON CACHE BOOL "Enable fmtlib formatting in loguru" FORCE)
set(LOGURU_WITH_STREAMS ON CACHE BOOL "Enable streams in loguru" FORCE)

# glog-specific options
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable glog examples" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "Disable gflags dependency for glog" FORCE)
set(WITH_GTEST OFF CACHE BOOL "Disable gtest for glog" FORCE)
set(WITH_GMOCK OFF CACHE BOOL "Disable gmock for glog" FORCE)

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

# Intel ITT API-specific options
set(ITT_API_IPT_SUPPORT OFF CACHE BOOL "Disable ITT API ptmarks support" FORCE)
set(ITT_API_FORTRAN_SUPPORT OFF CACHE BOOL "Disable ITT API Fortran support" FORCE)

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

# Logging backend libraries (conditionally added based on XSIGMA_LOGGING_BACKEND)
if(XSIGMA_USE_LOGURU)
    # Save the current BUILD_SHARED_LIBS state
    set(_saved_build_shared_libs ${BUILD_SHARED_LIBS})
    
    # Force loguru to be static
    set(BUILD_SHARED_LIBS OFF)
    # loguru - Lightweight C++ logging library
    add_third_party_library(loguru)
    # Restore the original BUILD_SHARED_LIBS state
    set(BUILD_SHARED_LIBS ${_saved_build_shared_libs})
    set_target_properties(loguru PROPERTIES FOLDER "ThirdParty/loguru")
endif()

# glog - Google logging library
add_third_party_library(glog
    CONDITION XSIGMA_USE_GLOG
)
if(TARGET glog)
    set_target_properties(glog PROPERTIES FOLDER "ThirdParty/glog")
    set_target_properties(glog_internal PROPERTIES FOLDER "ThirdParty/glog")
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

# =============================================================================
# Google Test and Google Benchmark Integration
# =============================================================================

# Google Test - Unit testing framework
if(XSIGMA_ENABLE_GTEST)
    # Check if gtest target already exists (avoid duplicate add_subdirectory)
    if(NOT TARGET gtest)
        message(STATUS "Adding Google Test from ThirdParty/googletest")

        # Set options before adding subdirectory
        # Force shared CRT on MSVC to match XSigma's runtime library settings
        if(MSVC)
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        endif()

        # Add googletest subdirectory
        # This creates targets: gtest, gtest_main, gmock, gmock_main
        add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/googletest"
            "${CMAKE_CURRENT_BINARY_DIR}/googletest"
            EXCLUDE_FROM_ALL
        )

        message(STATUS "✓ Google Test targets created: gtest, gtest_main")
    else()
        message(STATUS "Google Test targets already exist, skipping add_subdirectory")
    endif()
endif()

# Google Benchmark - Microbenchmarking framework
if(XSIGMA_ENABLE_BENCHMARK)
    # Check if benchmark target already exists (avoid duplicate add_subdirectory)
    if(NOT TARGET benchmark)
        message(STATUS "Adding Google Benchmark from ThirdParty/benchmark")

        # Add benchmark subdirectory
        # This creates targets: benchmark, benchmark_main
        add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/benchmark"
            "${CMAKE_CURRENT_BINARY_DIR}/benchmark"
            EXCLUDE_FROM_ALL
        )

        message(STATUS "✓ Google Benchmark targets created: benchmark, benchmark_main")
    else()
        message(STATUS "Google Benchmark targets already exist, skipping add_subdirectory")
    endif()
endif()

# Set folder properties for Visual Studio organization
if(TARGET benchmark)
    set_target_properties(benchmark PROPERTIES FOLDER "ThirdParty/benchmark")
endif()
if(TARGET benchmark_main)
    set_target_properties(benchmark_main PROPERTIES FOLDER "ThirdParty/benchmark")
endif()

# Set folder properties for ITT API targets
if(TARGET ittnotify)
    set_target_properties(ittnotify PROPERTIES FOLDER "ThirdParty/ittapi")
endif()
if(TARGET jitprofiling)
    set_target_properties(jitprofiling PROPERTIES FOLDER "ThirdParty/ittapi")
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

# Kineto - PyTorch profiling library
# Note: Kineto has complex dependencies (CUPTI, ROCm, etc.) and is provided as a wrapper interface.
# Full Kineto integration requires manual setup of dependencies.
if(XSIGMA_ENABLE_KINETO)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kineto/libkineto/CMakeLists.txt")
        message(STATUS "PyTorch Kineto found - wrapper interface enabled")
        message(STATUS "Note: Full Kineto integration requires manual dependency setup")
        # Kineto wrapper is always available; actual Kineto library linking is optional
        # The wrapper provides graceful degradation if Kineto is not fully available
    else()
        message(WARNING "PyTorch Kineto not found in ${CMAKE_CURRENT_SOURCE_DIR}/kineto")
    endif()
endif()

# Intel ITT API - Instrumentation and Tracing Technology API
# Used for task/frame annotations in Intel VTune profiling
if(XSIGMA_ENABLE_ITTAPI)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ittapi/CMakeLists.txt")
        message(STATUS "Intel ITT API found - building as shared library")

        # Save the current BUILD_SHARED_LIBS state
        set(_saved_build_shared_libs ${BUILD_SHARED_LIBS})

        # Force ITT API to be built as shared library (XSigma requirement)
        set(BUILD_SHARED_LIBS ON)

        # Add ITT API subdirectory
        add_subdirectory(ittapi ${CMAKE_CURRENT_BINARY_DIR}/ittapi_build)

        # Restore the original BUILD_SHARED_LIBS state
        set(BUILD_SHARED_LIBS ${_saved_build_shared_libs})

        message(STATUS "✓ Intel ITT API targets created: ittnotify, jitprofiling")
    else()
        message(WARNING "Intel ITT API not found in ${CMAKE_CURRENT_SOURCE_DIR}/ittapi")
    endif()
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

# Logging backend aliases (only create for the selected backend)
if(XSIGMA_USE_LOGURU)
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

if(XSIGMA_USE_GLOG)
    if(TARGET glog::glog)
        # glog::glog might be an alias, so we need to get the actual target
        get_target_property(glog_aliased_target glog::glog ALIASED_TARGET)
        if(glog_aliased_target)
            add_library(XSigma::glog ALIAS ${glog_aliased_target})
        else()
            add_library(XSigma::glog ALIAS glog::glog)
        endif()
        message(STATUS "Created XSigma::glog target alias")
    elseif(TARGET glog)
        add_library(XSigma::glog ALIAS glog)
        message(STATUS "Created XSigma::glog target alias")
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

# Intel ITT API target aliases
if(XSIGMA_ENABLE_ITTAPI)
    if(TARGET ittnotify)
        add_library(XSigma::ittapi ALIAS ittnotify)
        message(STATUS "Created XSigma::ittapi target alias")
    endif()
    if(TARGET jitprofiling)
        add_library(XSigma::jitprofiling ALIAS jitprofiling)
        message(STATUS "Created XSigma::jitprofiling target alias")
    endif()
endif()