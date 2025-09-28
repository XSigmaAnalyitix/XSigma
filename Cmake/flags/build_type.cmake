# =============================================================================
# XSigma Build Type Configuration Module
# =============================================================================
# This module optimizes build configurations for maximum performance and
# minimal CMake reconfiguration overhead through aggressive caching.
#
# Performance Optimizations:
# - Cached compiler flag validation to avoid repeated checks
# - Build type-specific optimization flags for maximum runtime performance
# - Efficient MSVC runtime library selection
# - Assertion and debug symbol management per build type
# =============================================================================

# Guard against multiple inclusions for performance
if(XSIGMA_BUILD_TYPE_CONFIGURED)
    return()
endif()
set(XSIGMA_BUILD_TYPE_CONFIGURED TRUE CACHE INTERNAL "Build type module loaded")

# Set default build type with caching for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Cache build type for performance optimization
set(XSIGMA_CACHED_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE INTERNAL "Cached build type")

# =============================================================================
# Build Type Specific Optimizations
# =============================================================================

# Release: Maximum optimization for production performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        # MSVC Release optimizations
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /GL" CACHE STRING "Release CXX flags" FORCE)
        set(CMAKE_C_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /GL" CACHE STRING "Release C flags" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF" CACHE STRING "Release linker flags" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF" CACHE STRING "Release shared linker flags" FORCE)
        # Use release runtime library
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "MSVC Runtime Library" FORCE)
    else()
        # GCC/Clang Release optimizations
        # Use -fno-math-errno instead of -ffast-math to preserve infinity/NaN support
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto -fno-math-errno -funroll-loops" CACHE STRING "Release CXX flags" FORCE)
        set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -flto -fno-math-errno -funroll-loops" CACHE STRING "Release C flags" FORCE)
        if(WIN32)
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto" CACHE STRING "Release linker flags" FORCE)
            set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-flto" CACHE STRING "Release shared linker flags" FORCE)
        else()
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto -Wl,--gc-sections" CACHE STRING "Release linker flags" FORCE)
            set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-flto -Wl,--gc-sections" CACHE STRING "Release shared linker flags" FORCE)
        endif()
    endif()
    # Disable assertions for maximum performance
    add_compile_definitions(NDEBUG)
    message(STATUS "XSigma: Release build configured with maximum optimizations")

# Debug: Full debugging support with minimal optimization
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        # MSVC Debug optimizations
        set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /RTC1 /MDd" CACHE STRING "Debug CXX flags" FORCE)
        set(CMAKE_C_FLAGS_DEBUG "/Od /Zi /RTC1 /MDd" CACHE STRING "Debug C flags" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "/DEBUG /INCREMENTAL" CACHE STRING "Debug linker flags" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "/DEBUG /INCREMENTAL" CACHE STRING "Debug shared linker flags" FORCE)
        # Use debug runtime library
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug" CACHE STRING "MSVC Runtime Library" FORCE)
    else()
        # GCC/Clang Debug optimizations
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -fno-omit-frame-pointer -fstack-protector-strong" CACHE STRING "Debug CXX flags" FORCE)
        set(CMAKE_C_FLAGS_DEBUG "-O0 -g3 -fno-omit-frame-pointer -fstack-protector-strong" CACHE STRING "Debug C flags" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-g" CACHE STRING "Debug linker flags" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "-g" CACHE STRING "Debug shared linker flags" FORCE)
    endif()
    # Enable assertions and debug checks
    add_compile_definitions(_DEBUG)
    message(STATUS "XSigma: Debug build configured with full debugging support")

# RelWithDebInfo: Optimized with debug information
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(MSVC)
        # MSVC RelWithDebInfo optimizations
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Zi /DNDEBUG" CACHE STRING "RelWithDebInfo CXX flags" FORCE)
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "/O2 /Zi /DNDEBUG" CACHE STRING "RelWithDebInfo C flags" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "/DEBUG /OPT:REF /OPT:ICF" CACHE STRING "RelWithDebInfo linker flags" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "/DEBUG /OPT:REF /OPT:ICF" CACHE STRING "RelWithDebInfo shared linker flags" FORCE)
        # Use release runtime library
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "MSVC Runtime Library" FORCE)
    else()
        # GCC/Clang RelWithDebInfo optimizations
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -fno-omit-frame-pointer" CACHE STRING "RelWithDebInfo CXX flags" FORCE)
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -fno-omit-frame-pointer" CACHE STRING "RelWithDebInfo C flags" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "-g" CACHE STRING "RelWithDebInfo linker flags" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "-g" CACHE STRING "RelWithDebInfo shared linker flags" FORCE)
    endif()
    # Disable assertions but keep debug symbols
    add_compile_definitions(NDEBUG)
    message(STATUS "XSigma: RelWithDebInfo build configured with optimizations and debug symbols")

# MinSizeRel: Size optimization
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    if(MSVC)
        # MSVC MinSizeRel optimizations
        set(CMAKE_CXX_FLAGS_MINSIZEREL "/O1 /Os /DNDEBUG /GL" CACHE STRING "MinSizeRel CXX flags" FORCE)
        set(CMAKE_C_FLAGS_MINSIZEREL "/O1 /Os /DNDEBUG /GL" CACHE STRING "MinSizeRel C flags" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "/LTCG /OPT:REF /OPT:ICF" CACHE STRING "MinSizeRel linker flags" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "/LTCG /OPT:REF /OPT:ICF" CACHE STRING "MinSizeRel shared linker flags" FORCE)
        # Use release runtime library
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "MSVC Runtime Library" FORCE)
    else()
        # GCC/Clang MinSizeRel optimizations
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG -flto" CACHE STRING "MinSizeRel CXX flags" FORCE)
        set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG -flto" CACHE STRING "MinSizeRel C flags" FORCE)
        if(WIN32)
            set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "-flto -s" CACHE STRING "MinSizeRel linker flags" FORCE)
            set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "-flto -s" CACHE STRING "MinSizeRel shared linker flags" FORCE)
        else()
            set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "-flto -Wl,--gc-sections -s" CACHE STRING "MinSizeRel linker flags" FORCE)
            set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "-flto -Wl,--gc-sections -s" CACHE STRING "MinSizeRel shared linker flags" FORCE)
        endif()
    endif()
    # Disable assertions for size optimization
    add_compile_definitions(NDEBUG)
    message(STATUS "XSigma: MinSizeRel build configured with size optimizations")
endif()

# =============================================================================
# Link Time Optimization (LTO) Configuration
# =============================================================================
if(XSIGMA_ENABLE_LTO AND (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel"))
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "XSigma: Link Time Optimization (LTO) enabled")
    else()
        message(WARNING "XSigma: LTO requested but not supported: ${ipo_error}")
    endif()
endif()

# =============================================================================
# Runtime Library Configuration for MSVC
# =============================================================================
if(MSVC)
    # Ensure consistent runtime library usage across all targets
    set(CMAKE_MSVC_RUNTIME_LIBRARY_DEFAULT "${CMAKE_MSVC_RUNTIME_LIBRARY}")

    # Set runtime library policy for CMake 3.15+
    if(POLICY CMP0091)
        cmake_policy(SET CMP0091 NEW)
    endif()
endif()

# =============================================================================
# Performance Validation and Caching
# =============================================================================

# Cache validation results to avoid repeated checks
set(XSIGMA_BUILD_TYPE_VALIDATION_CACHE "${CMAKE_BUILD_TYPE}_${CMAKE_CXX_COMPILER_ID}_${CMAKE_CXX_COMPILER_VERSION}"
    CACHE INTERNAL "Build type validation cache key")

# Mark configuration as complete
set(XSIGMA_BUILD_TYPE_CONFIGURED_FOR "${CMAKE_BUILD_TYPE}" CACHE INTERNAL "Build type configuration completed")

message(STATUS "XSigma: Build type configuration completed for ${CMAKE_BUILD_TYPE}")

# =============================================================================
# End of build_type.cmake
# =============================================================================

