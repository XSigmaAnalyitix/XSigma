# Third-party dependencies for XSigma
cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0069 NEW)
message(STATUS "DEBUG: Processing ThirdParty/CMakeLists.txt")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option to use external libraries instead of bundled ones
option(XSIGMA_ENABLE_EXTERNAL "Use external copies of third party libraries by default" OFF)

# Global option to control third-party library type (STATIC, SHARED, or DEFAULT)
# DEFAULT means respect each library's own settings or BUILD_SHARED_LIBS
set(XSIGMA_THIRDPARTY_TYPE "DEFAULT" CACHE STRING "Default library type for third-party libraries (STATIC, SHARED, or DEFAULT)")
set_property(CACHE XSIGMA_THIRDPARTY_TYPE PROPERTY STRINGS "STATIC" "SHARED" "DEFAULT")

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
# Note: MI_BUILD_STATIC is now controlled by add_third_party_library LIBRARY_TYPE parameter

# =============================================================================
# Helper Functions for Third-Party Library Management
# =============================================================================

# -----------------------------------------------------------------------------
# _filter_flags_internal: Remove specified flags from a flag string
# -----------------------------------------------------------------------------
# Internal helper to filter out specific flags from a string of compiler/linker flags
#
# Parameters:
#   input_flags - The flag string to filter
#   flags_to_remove - List of flags to remove (can be regex patterns)
#   output_var - Variable name to store the filtered result
#
function(_filter_flags_internal input_flags flags_to_remove output_var)
    set(result "${input_flags}")

    foreach(flag_pattern ${flags_to_remove})
        # Remove the flag pattern from the string
        string(REGEX REPLACE "${flag_pattern}( |$)" "" result "${result}")
    endforeach()

    # Clean up multiple spaces
    string(REGEX REPLACE "  +" " " result "${result}")
    string(STRIP "${result}" result)

    set(${output_var} "${result}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# _save_third_party_flags: Save and optionally filter compiler/linker flags
# -----------------------------------------------------------------------------
# Saves current CMAKE_CXX_FLAGS, CMAKE_C_FLAGS, and linker flags, then optionally
# filters out problematic flags that shouldn't be applied to third-party libraries
#
# Parameters:
#   name - Library name (for namespacing saved variables)
#   filter_flags - List of flag patterns to remove (optional)
#
function(_save_third_party_flags name filter_flags)
    # Save original flags to parent scope
    set(_SAVED_CMAKE_CXX_FLAGS_${name} "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
    set(_SAVED_CMAKE_C_FLAGS_${name} "${CMAKE_C_FLAGS}" PARENT_SCOPE)
    set(_SAVED_CMAKE_EXE_LINKER_FLAGS_${name} "${CMAKE_EXE_LINKER_FLAGS}" PARENT_SCOPE)
    set(_SAVED_CMAKE_SHARED_LINKER_FLAGS_${name} "${CMAKE_SHARED_LINKER_FLAGS}" PARENT_SCOPE)
    set(_SAVED_CMAKE_MODULE_LINKER_FLAGS_${name} "${CMAKE_MODULE_LINKER_FLAGS}" PARENT_SCOPE)
    set(_SAVED_CMAKE_STATIC_LINKER_FLAGS_${name} "${CMAKE_STATIC_LINKER_FLAGS}" PARENT_SCOPE)

    # If filter flags are provided, filter them out
    if(filter_flags)
        _filter_flags_internal("${CMAKE_CXX_FLAGS}" "${filter_flags}" filtered_cxx_flags)
        _filter_flags_internal("${CMAKE_C_FLAGS}" "${filter_flags}" filtered_c_flags)
        _filter_flags_internal("${CMAKE_EXE_LINKER_FLAGS}" "${filter_flags}" filtered_exe_linker_flags)
        _filter_flags_internal("${CMAKE_SHARED_LINKER_FLAGS}" "${filter_flags}" filtered_shared_linker_flags)
        _filter_flags_internal("${CMAKE_MODULE_LINKER_FLAGS}" "${filter_flags}" filtered_module_linker_flags)
        _filter_flags_internal("${CMAKE_STATIC_LINKER_FLAGS}" "${filter_flags}" filtered_static_linker_flags)

        # Apply filtered flags to parent scope
        set(CMAKE_CXX_FLAGS "${filtered_cxx_flags}" PARENT_SCOPE)
        set(CMAKE_C_FLAGS "${filtered_c_flags}" PARENT_SCOPE)
        set(CMAKE_EXE_LINKER_FLAGS "${filtered_exe_linker_flags}" PARENT_SCOPE)
        set(CMAKE_SHARED_LINKER_FLAGS "${filtered_shared_linker_flags}" PARENT_SCOPE)
        set(CMAKE_MODULE_LINKER_FLAGS "${filtered_module_linker_flags}" PARENT_SCOPE)
        set(CMAKE_STATIC_LINKER_FLAGS "${filtered_static_linker_flags}" PARENT_SCOPE)

        message(STATUS "Filtered flags for ${name}: removed ${filter_flags}")
    endif()
endfunction()

# -----------------------------------------------------------------------------
# _restore_third_party_flags: Restore original compiler/linker flags
# -----------------------------------------------------------------------------
# Restores the compiler and linker flags that were saved before building
# a third-party library
#
# Parameters:
#   name - Library name (for namespacing saved variables)
#
function(_restore_third_party_flags name)
    # Restore original flags to parent scope
    set(CMAKE_CXX_FLAGS "${_SAVED_CMAKE_CXX_FLAGS_${name}}" PARENT_SCOPE)
    set(CMAKE_C_FLAGS "${_SAVED_CMAKE_C_FLAGS_${name}}" PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${_SAVED_CMAKE_EXE_LINKER_FLAGS_${name}}" PARENT_SCOPE)
    set(CMAKE_SHARED_LINKER_FLAGS "${_SAVED_CMAKE_SHARED_LINKER_FLAGS_${name}}" PARENT_SCOPE)
    set(CMAKE_MODULE_LINKER_FLAGS "${_SAVED_CMAKE_MODULE_LINKER_FLAGS_${name}}" PARENT_SCOPE)
    set(CMAKE_STATIC_LINKER_FLAGS "${_SAVED_CMAKE_STATIC_LINKER_FLAGS_${name}}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# _set_third_party_folder_properties: Set FOLDER property for all targets
# -----------------------------------------------------------------------------
# Automatically discovers all targets created by a third-party library and
# sets their FOLDER property for IDE organization
#
# Parameters:
#   name - Library name
#   binary_dir - Binary directory where the library was built
#
function(_set_third_party_folder_properties name binary_dir)
    # Get all targets in the directory
    get_property(all_targets DIRECTORY "${binary_dir}" PROPERTY BUILDSYSTEM_TARGETS)

    if(all_targets)
        message(STATUS "Setting FOLDER property for ${name} targets: ${all_targets}")
        foreach(target ${all_targets})
            if(TARGET ${target})
                set_target_properties(${target} PROPERTIES FOLDER "ThirdParty/${name}")
            endif()
        endforeach()
    endif()
endfunction()

# -----------------------------------------------------------------------------
# _create_third_party_aliases: Create library aliases
# -----------------------------------------------------------------------------
# Creates alias targets for a third-party library to provide consistent
# naming conventions (e.g., fmt::fmt as an alias for fmt)
#
# Parameters:
#   name - Library name
#   aliases - List of alias names to create
#
function(_create_third_party_aliases name aliases)
    if(NOT aliases)
        return()
    endif()

    foreach(alias ${aliases})
        # Check if the alias already exists
        if(NOT TARGET ${alias})
            # Check if the base target exists
            if(TARGET ${name})
                add_library(${alias} ALIAS ${name})
                message(STATUS "Created alias ${alias} for ${name}")
            else()
                message(WARNING "Cannot create alias ${alias}: target ${name} does not exist")
            endif()
        else()
            message(STATUS "Alias ${alias} already exists, skipping")
        endif()
    endforeach()
endfunction()

# -----------------------------------------------------------------------------
# _create_third_party_interface_targets: Create XSigma:: namespace aliases
# -----------------------------------------------------------------------------
# Creates XSigma:: namespace aliases for third-party library targets.
# This provides a consistent interface for linking against third-party libraries
# throughout the XSigma project.
#
# The function handles multiple scenarios:
# 1. Target with namespace alias (e.g., fmt::fmt) - resolves the actual target
# 2. Plain target (e.g., fmt) - creates alias directly
# 3. Multiple possible target names - tries each in order
#
# Parameters:
#   xsigma_targets - List of XSigma target specifications in format:
#                    "XSigma::name=target1|target2|..."
#                    where target1, target2, etc. are tried in order
#
# Example:
#   _create_third_party_interface_targets(
#       "XSigma::fmt=fmt::fmt|fmt"
#       "XSigma::gtest=GTest::gtest|gtest;XSigma::gtest_main=GTest::gtest_main|gtest_main"
#   )
#
function(_create_third_party_interface_targets xsigma_targets)
    if(NOT xsigma_targets)
        return()
    endif()

    foreach(target_spec ${xsigma_targets})
        # Parse the specification: "XSigma::name=target1|target2|..."
        string(REPLACE "=" ";" spec_parts "${target_spec}")
        list(LENGTH spec_parts spec_parts_len)

        if(NOT spec_parts_len EQUAL 2)
            message(WARNING "Invalid XSigma target specification: ${target_spec}")
            continue()
        endif()

        list(GET spec_parts 0 xsigma_name)
        list(GET spec_parts 1 source_targets)

        # Check if XSigma target already exists
        if(TARGET ${xsigma_name})
            message(STATUS "XSigma target ${xsigma_name} already exists, skipping")
            continue()
        endif()

        # Split source targets by pipe (|) to get alternatives
        string(REPLACE "|" ";" source_target_list "${source_targets}")

        set(target_created FALSE)
        foreach(source_target ${source_target_list})
            if(TARGET ${source_target})
                # Check if source target is an alias and resolve it
                get_target_property(aliased_target ${source_target} ALIASED_TARGET)

                if(aliased_target)
                    # Source is an alias, create XSigma alias to the actual target
                    add_library(${xsigma_name} ALIAS ${aliased_target})
                    message(STATUS "Created ${xsigma_name} -> ${source_target} (resolved to ${aliased_target})")
                else()
                    # Source is a real target, create alias directly
                    add_library(${xsigma_name} ALIAS ${source_target})
                    message(STATUS "Created ${xsigma_name} -> ${source_target}")
                endif()

                set(target_created TRUE)
                break()
            endif()
        endforeach()

        if(NOT target_created)
            message(STATUS "Could not create ${xsigma_name}: none of the source targets exist (${source_targets})")
        endif()
    endforeach()
endfunction()

# =============================================================================
# Main Function: add_third_party_library
# =============================================================================
# Enhanced function to add third-party libraries with comprehensive configuration
# options including flag filtering, automatic target organization, and alias creation
#
# Parameters:
#   name - Name of the third-party library (required)
#
# Optional Parameters:
#   EXTERNAL_NAME <name>        - Name to use when searching for external package
#   CONDITION <variable>        - CMake variable that must be true to add the library
#   LIBRARY_TYPE <type>         - Type of library: STATIC, SHARED, or DEFAULT
#                                 If not specified, uses XSIGMA_THIRDPARTY_TYPE
#   LIBRARY_TYPE_VAR <var>      - CMake variable name for library-specific control
#                                 Example: MI_BUILD_STATIC for mimalloc
#   FILTER_FLAGS <flags...>     - List of compiler/linker flags to filter out
#                                 Example: "-fno-exceptions" "-Werror"
#   ALIASES <aliases...>        - List of alias names to create for the library
#                                 Example: "fmt::fmt" for fmt library
#   XSIGMA_TARGETS <specs...>   - List of XSigma:: namespace target specifications
#                                 Format: "XSigma::name=target1|target2|..."
#                                 Example: "XSigma::fmt=fmt::fmt|fmt"
#   AUTO_FOLDER <ON|OFF>        - Automatically set FOLDER property (default: ON)
#
# Example Usage:
#   add_third_party_library(fmt
#       LIBRARY_TYPE STATIC
#       FILTER_FLAGS "-fno-exceptions" "-Werror"
#       ALIASES "fmt::fmt"
#       XSIGMA_TARGETS "XSigma::fmt=fmt::fmt|fmt"
#       AUTO_FOLDER ON
#   )
#
function(add_third_party_library name)
    # Parse function arguments
    cmake_parse_arguments(
        TPL                                                    # Prefix
        "AUTO_FOLDER"                                          # Options (boolean flags)
        "EXTERNAL_NAME;CONDITION;LIBRARY_TYPE;LIBRARY_TYPE_VAR" # One-value keywords
        "FILTER_FLAGS;ALIASES;XSIGMA_TARGETS"                  # Multi-value keywords
        ${ARGN}                                                # Arguments to parse
    )

    # Set default values for optional parameters
    if(NOT DEFINED TPL_EXTERNAL_NAME)
        set(TPL_EXTERNAL_NAME ${name})
    endif()

    # AUTO_FOLDER defaults to ON if not specified
    if(NOT DEFINED TPL_AUTO_FOLDER)
        set(TPL_AUTO_FOLDER ON)
    endif()

    # -------------------------------------------------------------------------
    # Step 1: Check condition (early return pattern)
    # -------------------------------------------------------------------------
    if(DEFINED TPL_CONDITION)
        if(NOT ${TPL_CONDITION})
            message(STATUS "Skipping ${name}: condition ${TPL_CONDITION} is not met")
            return()
        endif()
    endif()

    # -------------------------------------------------------------------------
    # Step 2: Check if already processed to avoid duplicates
    # -------------------------------------------------------------------------
    get_property(XSIGMA_${name}_PROCESSED GLOBAL PROPERTY XSIGMA_${name}_PROCESSED)
    if(XSIGMA_${name}_PROCESSED)
        message(STATUS "Third-party library ${name} already processed, skipping")
        return()
    endif()
    set_property(GLOBAL PROPERTY XSIGMA_${name}_PROCESSED TRUE)

    # -------------------------------------------------------------------------
    # Step 3: Try to find external library first if requested
    # -------------------------------------------------------------------------
    if(XSIGMA_ENABLE_EXTERNAL)
        find_package(${TPL_EXTERNAL_NAME} QUIET)
        if(${TPL_EXTERNAL_NAME}_FOUND OR ${name}_FOUND)
            message(STATUS "Using external ${name}")
            return()
        endif()
    endif()

    # -------------------------------------------------------------------------
    # Step 4: Determine library type
    # -------------------------------------------------------------------------
    # Priority: LIBRARY_TYPE parameter > XSIGMA_THIRDPARTY_TYPE global > DEFAULT
    set(lib_type "DEFAULT")
    if(DEFINED TPL_LIBRARY_TYPE)
        set(lib_type ${TPL_LIBRARY_TYPE})
    elseif(NOT XSIGMA_THIRDPARTY_TYPE STREQUAL "DEFAULT")
        set(lib_type ${XSIGMA_THIRDPARTY_TYPE})
    endif()

    # -------------------------------------------------------------------------
    # Step 5: Save current state (BUILD_SHARED_LIBS and flags)
    # -------------------------------------------------------------------------
    set(_saved_build_shared_libs ${BUILD_SHARED_LIBS})

    # Save and filter compiler/linker flags if requested
    if(TPL_FILTER_FLAGS)
        _save_third_party_flags(${name} "${TPL_FILTER_FLAGS}")
    endif()

    # -------------------------------------------------------------------------
    # Step 6: Configure library type
    # -------------------------------------------------------------------------
    if(lib_type STREQUAL "STATIC")
        set(BUILD_SHARED_LIBS OFF)
        message(STATUS "Configuring ${name} as STATIC library")

        # Set library-specific variable if provided
        if(DEFINED TPL_LIBRARY_TYPE_VAR)
            set(${TPL_LIBRARY_TYPE_VAR} ON CACHE BOOL "Build ${name} as static library" FORCE)
        endif()
    elseif(lib_type STREQUAL "SHARED")
        set(BUILD_SHARED_LIBS ON)
        message(STATUS "Configuring ${name} as SHARED library")

        # Set library-specific variable if provided
        if(DEFINED TPL_LIBRARY_TYPE_VAR)
            set(${TPL_LIBRARY_TYPE_VAR} OFF CACHE BOOL "Build ${name} as shared library" FORCE)
        endif()
    else()
        message(STATUS "Configuring ${name} with DEFAULT library type")
    endif()

    # -------------------------------------------------------------------------
    # Step 7: Add the third-party library subdirectory
    # -------------------------------------------------------------------------
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name}/CMakeLists.txt")
        message(STATUS "Using bundled ${name}")

        # Store the binary directory for later use
        set(lib_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}_build")

        # Add the library
        add_subdirectory(${name} ${lib_binary_dir})

        # -------------------------------------------------------------------------
        # Step 8: Automatically set FOLDER properties for IDE organization
        # -------------------------------------------------------------------------
        if(TPL_AUTO_FOLDER)
            _set_third_party_folder_properties(${name} "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
        endif()

        # -------------------------------------------------------------------------
        # Step 9: Create library aliases if requested
        # -------------------------------------------------------------------------
        if(TPL_ALIASES)
            _create_third_party_aliases(${name} "${TPL_ALIASES}")
        endif()

        # -------------------------------------------------------------------------
        # Step 10: Create XSigma:: namespace interface targets if requested
        # -------------------------------------------------------------------------
        if(TPL_XSIGMA_TARGETS)
            _create_third_party_interface_targets("${TPL_XSIGMA_TARGETS}")
        endif()
    else()
        message(WARNING "Third-party library ${name} not found in ${CMAKE_CURRENT_SOURCE_DIR}/${name}")
    endif()

    # -------------------------------------------------------------------------
    # Step 11: Restore original state
    # -------------------------------------------------------------------------
    # Restore BUILD_SHARED_LIBS
    set(BUILD_SHARED_LIBS ${_saved_build_shared_libs})

    # Restore compiler/linker flags if they were filtered
    if(TPL_FILTER_FLAGS)
        _restore_third_party_flags(${name})
    endif()
endfunction()

# =============================================================================
# Third-Party Library Declarations
# =============================================================================

# Add third-party libraries conditionally based on XSIGMA_ENABLE_XXX options

# fmt - Modern formatting library
add_third_party_library(fmt
    XSIGMA_TARGETS "XSigma::fmt=fmt::fmt|fmt"
)

# cpuinfo - CPU feature detection library (mandatory dependency)
add_third_party_library(cpuinfo
    XSIGMA_TARGETS "XSigma::cpuinfo=cpuinfo"
)

# magic_enum - Static reflection for enums in C++
add_third_party_library(magic_enum
    CONDITION XSIGMA_ENABLE_MAGICENUM
    XSIGMA_TARGETS "XSigma::magic_enum=magic_enum::magic_enum|magic_enum"
)

# loguru - Lightweight C++ logging library (forced to static)
add_third_party_library(loguru
    CONDITION XSIGMA_USE_LOGURU
    LIBRARY_TYPE STATIC
    XSIGMA_TARGETS "XSigma::loguru=loguru::loguru|loguru"
)

# glog - Google logging library
add_third_party_library(glog
    CONDITION XSIGMA_USE_GLOG
    XSIGMA_TARGETS "XSigma::glog=glog::glog|glog"
)

# mimalloc - Microsoft's high-performance memory allocator (conditional)
# By default builds as static library, controlled by LIBRARY_TYPE parameter
add_third_party_library(mimalloc
    CONDITION XSIGMA_ENABLE_MIMALLOC
    LIBRARY_TYPE STATIC
    LIBRARY_TYPE_VAR MI_BUILD_STATIC
    XSIGMA_TARGETS "XSigma::mimalloc=mimalloc|mimalloc-static"
)

# Kineto - PyTorch profiling library
# Aligned with PyTorch's XSIGMA_ENABLE_KINETO configuration for feature parity.
# Supports GPU backends: CUPTI (NVIDIA), ROCtracer (AMD), XPUPTI (Intel XPU), AIUPTI (AI accelerators)

# Configure GPU backend support based on enabled platforms
set(LIBKINETO_NOAIUPTI ON CACHE STRING "" FORCE)
if(NOT XSIGMA_ENABLE_CUDA)
    set(LIBKINETO_NOCUPTI ON CACHE STRING "" FORCE)
else()
    set(LIBKINETO_NOCUPTI OFF CACHE STRING "")
    message(STATUS "Using Kineto with CUPTI support")
endif()

if(NOT USE_ROCM)
    set(LIBKINETO_NOROCTRACER ON CACHE STRING "" FORCE)
else()
    set(LIBKINETO_NOROCTRACER OFF CACHE STRING "")
    message(STATUS "Using Kineto with Roctracer support")
endif()

if(LIBKINETO_NOCUPTI AND LIBKINETO_NOROCTRACER AND LIBKINETO_NOXPUPTI)
    message(STATUS "Using CPU-only version of Kineto")
endif()

# Configure Kineto library
set(KINETO_BUILD_TESTS OFF CACHE BOOL "")
set(KINETO_LIBRARY_TYPE "static" CACHE STRING "")

message(STATUS "Configuring Kineto dependency:")
message(STATUS "  KINETO_BUILD_TESTS = ${KINETO_BUILD_TESTS}")
message(STATUS "  KINETO_LIBRARY_TYPE = ${KINETO_LIBRARY_TYPE}")

add_third_party_library(kineto
    CONDITION XSIGMA_ENABLE_KINETO
    LIBRARY_TYPE STATIC
)

# =============================================================================
# Google Test and Google Benchmark Integration
# =============================================================================

# Note: googletest and benchmark require special handling due to EXCLUDE_FROM_ALL
# and MSVC-specific settings, so they use manual add_subdirectory calls

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

        # Set folder properties for all googletest targets
        _set_third_party_folder_properties("googletest" "${CMAKE_CURRENT_SOURCE_DIR}/googletest")

        message(STATUS "✓ Google Test targets created: gtest, gtest_main")
    else()
        message(STATUS "Google Test targets already exist, skipping add_subdirectory")
    endif()

    # Create XSigma:: namespace aliases for googletest
    _create_third_party_interface_targets(
        "XSigma::gtest=GTest::gtest|gtest"
        "XSigma::gtest_main=GTest::gtest_main|gtest_main"
    )
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

        # Set folder properties for all benchmark targets
        _set_third_party_folder_properties("benchmark" "${CMAKE_CURRENT_SOURCE_DIR}/benchmark")

        message(STATUS "✓ Google Benchmark targets created: benchmark, benchmark_main")
    else()
        message(STATUS "Google Benchmark targets already exist, skipping add_subdirectory")
    endif()

    # Create XSigma:: namespace alias for benchmark
    _create_third_party_interface_targets(
        "XSigma::benchmark=benchmark::benchmark|benchmark"
    )
endif()

# =============================================================================
# XSigma Interface Targets
# =============================================================================
# All XSigma:: namespace interface targets are now created automatically
# by the add_third_party_library() function using the XSIGMA_TARGETS parameter.
# This eliminates code duplication and provides a consistent interface for
# linking against third-party libraries throughout the XSigma project.
#
# See the library declarations above for XSIGMA_TARGETS specifications.
# =============================================================================
