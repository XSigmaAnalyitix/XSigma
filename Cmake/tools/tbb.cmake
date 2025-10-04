# =============================================================================
# XSigma Intel TBB (Threading Building Blocks) Integration Module
# =============================================================================
# This module provides robust TBB integration with automatic fallback support:
# 1. First attempts to find system-installed TBB (vcpkg, apt, homebrew, Intel oneAPI)
# 2. If not found, automatically downloads and builds TBB from source
# 3. Creates consistent TBB::tbb and TBB::tbbmalloc targets regardless of source
# =============================================================================

cmake_minimum_required(VERSION 3.16)

# Guard against multiple inclusions
if(XSIGMA_TBB_CONFIGURED)
    return()
endif()
set(XSIGMA_TBB_CONFIGURED TRUE CACHE INTERNAL "TBB module loaded")

# Only proceed if TBB is enabled
if(NOT XSIGMA_ENABLE_TBB)
    message(STATUS "Intel TBB support is disabled (XSIGMA_ENABLE_TBB=OFF)")
    return()
endif()

message(STATUS "Configuring Intel TBB (Threading Building Blocks) support...")

# =============================================================================
# Configuration Options
# =============================================================================

# Option to force building TBB from source (useful for testing)
option(XSIGMA_TBB_FORCE_BUILD_FROM_SOURCE "Force building TBB from source instead of using system TBB" OFF)
mark_as_advanced(XSIGMA_TBB_FORCE_BUILD_FROM_SOURCE)

# TBB version to use when building from source
set(XSIGMA_TBB_VERSION "v2021.13.0" CACHE STRING "TBB version to build from source")
mark_as_advanced(XSIGMA_TBB_VERSION)

# TBB repository URL
set(XSIGMA_TBB_REPOSITORY "https://github.com/oneapi-src/oneTBB.git" CACHE STRING "TBB repository URL")
mark_as_advanced(XSIGMA_TBB_REPOSITORY)

# =============================================================================
# Step 1: Try to find system-installed TBB
# =============================================================================

set(TBB_FOUND FALSE)
set(TBB_FROM_SOURCE FALSE)

if(NOT XSIGMA_TBB_FORCE_BUILD_FROM_SOURCE)
    message(STATUS "Searching for system-installed Intel TBB...")

    # Try to find TBB using our custom FindTBB module first
    find_package(TBB QUIET)

    if(TBB_FOUND AND TARGET TBB::tbb)
        message(STATUS "✅ Found system-installed Intel TBB")
        message(STATUS "   TBB Include Dir: ${TBB_INCLUDE_DIR}")
        if(TBB_LIBRARY_RELEASE)
            message(STATUS "   TBB Library: ${TBB_LIBRARY_RELEASE}")
        endif()
        if(TBB_MALLOC_LIBRARY_RELEASE)
            message(STATUS "   TBB Malloc Library: ${TBB_MALLOC_LIBRARY_RELEASE}")
        endif()

        # Verify the targets work
        if(TARGET TBB::tbb)
            message(STATUS "   TBB::tbb target available")
        endif()
        if(TARGET TBB::tbbmalloc)
            message(STATUS "   TBB::tbbmalloc target available")
        endif()

        return()
    else()
        message(STATUS "❌ System-installed Intel TBB not found")
        message(STATUS "   Will build TBB from source as fallback")
    endif()
endif()

# =============================================================================
# Step 2: Build TBB from source using FetchContent
# =============================================================================

message(STATUS "Building Intel TBB from source...")
message(STATUS "   Repository: ${XSIGMA_TBB_REPOSITORY}")
message(STATUS "   Version: ${XSIGMA_TBB_VERSION}")

include(FetchContent)

# Configure FetchContent for TBB
FetchContent_Declare(
    oneTBB
    GIT_REPOSITORY ${XSIGMA_TBB_REPOSITORY}
    GIT_TAG ${XSIGMA_TBB_VERSION}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

# Set TBB-specific variables before fetching
set(TBB_TEST OFF CACHE BOOL "Disable TBB tests" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "Disable TBB examples" FORCE)
set(TBB_STRICT OFF CACHE BOOL "Disable strict mode for compatibility" FORCE)

# Set TBB build options before fetching
set(TBB_TEST OFF CACHE BOOL "Build TBB tests" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "Build TBB examples" FORCE)
set(TBB_STRICT OFF CACHE BOOL "Treat compiler warnings as errors" FORCE)

# Handle CMake policy compatibility for TBB
if(POLICY CMP0126)
    cmake_policy(SET CMP0126 NEW)
endif()

# Platform-specific TBB configuration
if(WIN32)
    # Windows-specific TBB settings
    set(TBB_WINDOWS_DRIVER OFF CACHE BOOL "Build TBB for Windows kernel mode" FORCE)
    # Disable problematic flags on Windows
    set(CMAKE_POSITION_INDEPENDENT_CODE OFF CACHE BOOL "Disable PIC on Windows" FORCE)
    # Override TBB's internal PIC settings
    set(TBB_ENABLE_PIC OFF CACHE BOOL "Disable TBB PIC on Windows" FORCE)
    set(CMAKE_CXX_COMPILE_OPTIONS_PIC "" CACHE STRING "Clear PIC options" FORCE)
    set(CMAKE_C_COMPILE_OPTIONS_PIC "" CACHE STRING "Clear PIC options" FORCE)
    # Additional Windows-specific TBB configuration
    set(TBB_STRICT OFF CACHE BOOL "Disable strict mode for Windows compatibility" FORCE)
    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests" FORCE)
    set(TBB_EXAMPLES OFF CACHE BOOL "Disable TBB examples" FORCE)
endif()

# Fetch and build TBB
message(STATUS "Downloading Intel TBB source code...")

# Temporarily save and clear PIC-related variables on Windows to prevent TBB build issues
if(WIN32)
    set(_SAVED_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
    set(_SAVED_CMAKE_CXX_COMPILE_OPTIONS_PIC "${CMAKE_CXX_COMPILE_OPTIONS_PIC}")
    set(_SAVED_CMAKE_C_COMPILE_OPTIONS_PIC "${CMAKE_C_COMPILE_OPTIONS_PIC}")

    set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
    set(CMAKE_CXX_COMPILE_OPTIONS_PIC "")
    set(CMAKE_C_COMPILE_OPTIONS_PIC "")
endif()

FetchContent_MakeAvailable(oneTBB)

# Restore saved variables on Windows
if(WIN32)
    set(CMAKE_POSITION_INDEPENDENT_CODE ${_SAVED_CMAKE_POSITION_INDEPENDENT_CODE})
    set(CMAKE_CXX_COMPILE_OPTIONS_PIC "${_SAVED_CMAKE_CXX_COMPILE_OPTIONS_PIC}")
    set(CMAKE_C_COMPILE_OPTIONS_PIC "${_SAVED_CMAKE_C_COMPILE_OPTIONS_PIC}")
endif()

# Check if TBB was successfully populated
FetchContent_GetProperties(oneTBB)
if(NOT onetbb_POPULATED)
    message(FATAL_ERROR "Failed to download Intel TBB from ${XSIGMA_TBB_REPOSITORY}")
endif()

message(STATUS "✅ Successfully downloaded Intel TBB source")
message(STATUS "   Source directory: ${onetbb_SOURCE_DIR}")
message(STATUS "   Binary directory: ${onetbb_BINARY_DIR}")

# =============================================================================
# Step 3: Verify TBB targets were created
# =============================================================================

# Check if the expected targets were created
if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB::tbb target was not created after building from source")
endif()

if(NOT TARGET TBB::tbbmalloc)
    message(WARNING "TBB::tbbmalloc target was not created - this may be expected on some platforms")
endif()

# Mark that we built TBB from source
set(TBB_FROM_SOURCE TRUE)
set(TBB_FOUND TRUE)

message(STATUS "✅ Successfully built Intel TBB from source")
message(STATUS "   TBB::tbb target available")
if(TARGET TBB::tbbmalloc)
    message(STATUS "   TBB::tbbmalloc target available")
endif()

# =============================================================================
# Step 4: Configure TBB output directories to match XSigma project structure
# =============================================================================

# Configure TBB output directories to match XSigma project structure
# This ensures TBB binaries are placed in the correct locations for runtime linking
if(TBB_FROM_SOURCE)
    # Get the actual TBB targets (not the imported interface targets)
    # TBB creates targets like 'tbb' and 'tbbmalloc' when built from source
    set(_tbb_targets)
    if(TARGET tbb)
        list(APPEND _tbb_targets tbb)
    endif()
    if(TARGET tbbmalloc)
        list(APPEND _tbb_targets tbbmalloc)
    endif()

    # Configure output directories for each TBB target
    foreach(_tbb_target ${_tbb_targets})
        if(TARGET ${_tbb_target})
            # Set output directories for all build configurations
            foreach(config Debug Release RelWithDebInfo MinSizeRel)
                string(TOUPPER ${config} config_upper)
                set_target_properties(${_tbb_target} PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BINARY_DIR}/bin"
                    ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BINARY_DIR}/lib"
                    LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BINARY_DIR}/lib"
                )
            endforeach()

            # Also set the default output directories (for single-config generators)
            set_target_properties(${_tbb_target} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${XSIGMA_BINARY_DIR}/bin"
                ARCHIVE_OUTPUT_DIRECTORY "${XSIGMA_BINARY_DIR}/lib"
                LIBRARY_OUTPUT_DIRECTORY "${XSIGMA_BINARY_DIR}/lib"
            )

            message(STATUS "Configured output directories for TBB target '${_tbb_target}'")
        endif()
    endforeach()
endif()

# =============================================================================
# Step 5: Export TBB information for other parts of the build system
# =============================================================================

# Set variables that other parts of the build system might expect
set(TBB_FOUND TRUE CACHE BOOL "TBB was found or built successfully" FORCE)
set(TBB_FROM_SOURCE ${TBB_FROM_SOURCE} CACHE BOOL "TBB was built from source" FORCE)

# Export the source directory for potential use by other modules
if(TBB_FROM_SOURCE)
    set(TBB_SOURCE_DIR ${onetbb_SOURCE_DIR} CACHE PATH "TBB source directory" FORCE)
    set(TBB_BINARY_DIR ${onetbb_BINARY_DIR} CACHE PATH "TBB binary directory" FORCE)
endif()

message(STATUS "Intel TBB configuration complete")
