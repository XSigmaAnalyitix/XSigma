# =============================================================================
# XSigma Platform-Specific Compiler Optimization Module
# =============================================================================
# This module configures platform-specific compiler optimizations for maximum
# runtime performance while maintaining compatibility across toolchains.
#
# Performance Optimizations:
# - Platform-specific optimization flags (Windows: /arch:AVX2, Linux/macOS: -march=native)
# - Aggressive vectorization and SIMD instruction usage
# - Optimized warning levels for each compiler
# - Cached flag validation to minimize reconfiguration overhead
# - Conflict-free flag management
# =============================================================================

# Guard against multiple inclusions for performance
if(XSIGMA_PLATFORM_CONFIGURED)
    return()
endif()
set(XSIGMA_PLATFORM_CONFIGURED TRUE CACHE INTERNAL "Platform module loaded")

# Initialize platform-specific flag variables
set(XSIGMA_REQUIRED_C_FLAGS)
set(XSIGMA_REQUIRED_CXX_FLAGS)
set(XSIGMA_REQUIRED_CUDA_FLAGS)
set(XSIGMA_LINKER_FLAGS_COMMON)

# =============================================================================
# Vectorization and SIMD Optimization Integration
# =============================================================================

# Display vectorization configuration (if available from xsigmaUtils.cmake)
if(DEFINED VECTORIZATION_COMPILER_FLAGS)
    message(STATUS "XSigma: Vectorization flags: ${VECTORIZATION_COMPILER_FLAGS}")
else()
    message(STATUS "XSigma: Vectorization flags will be determined by xsigmaUtils.cmake")
    set(VECTORIZATION_COMPILER_FLAGS "")
endif()

# =============================================================================
# MSVC (Windows) Platform Optimizations
# =============================================================================

if(MSVC)
    message(STATUS "XSigma: Configuring MSVC optimizations...")

    # Core MSVC optimization flags
    set(XSIGMA_MSVC_BASE_FLAGS
        "/Zc:__cplusplus /permissive- /Zc:inline /Zc:throwingNew /volatile:iso /bigobj /utf-8"
    )

    # Performance optimization flags
    set(XSIGMA_MSVC_PERF_FLAGS
        "/favor:INTEL64 /Gy /Gw"
    )

    # Warning configuration (balanced approach)
    set(XSIGMA_MSVC_WARNING_FLAGS
        "/W3 /wd4244 /wd4267 /wd4996 /wd4251 /wd4018"  # Combined warning flags
    )

    # Combine all MSVC flags
    set(XSIGMA_REQUIRED_CXX_FLAGS
        "${XSIGMA_MSVC_BASE_FLAGS} ${XSIGMA_MSVC_PERF_FLAGS} ${XSIGMA_MSVC_WARNING_FLAGS} ${VECTORIZATION_COMPILER_FLAGS}")
    set(XSIGMA_REQUIRED_C_FLAGS
        "${XSIGMA_MSVC_BASE_FLAGS} ${XSIGMA_MSVC_PERF_FLAGS} ${XSIGMA_MSVC_WARNING_FLAGS}")

    # MSVC parallel compilation
    if(NOT DEFINED CMAKE_CXX_MP_FLAG OR CMAKE_CXX_MP_FLAG)
        set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
        if(NOT PROCESSOR_COUNT)
            set(PROCESSOR_COUNT "4")  # Default fallback
        endif()
        set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} /MP${PROCESSOR_COUNT}")
        set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} /MP${PROCESSOR_COUNT}")
        message(STATUS "XSigma: MSVC parallel compilation enabled with ${PROCESSOR_COUNT} processes")
    endif()

    # MSVC linker optimizations (as CMake list for proper handling)
    set(XSIGMA_LINKER_FLAGS_COMMON "/OPT:REF" "/OPT:ICF")

    # Disable problematic MSVC warnings globally
    add_definitions(
        -D_CRT_SECURE_NO_DEPRECATE
        -D_CRT_NONSTDC_NO_DEPRECATE
        -D_CRT_SECURE_NO_WARNINGS
        -D_SCL_SECURE_NO_DEPRECATE
        -D_SCL_SECURE_NO_WARNINGS
    )

# =============================================================================
# GCC/Clang (Linux/macOS) Platform Optimizations
# =============================================================================

else()
    message(STATUS "XSigma: Configuring GCC/Clang optimizations...")

    # Base optimization flags for GCC/Clang
    set(XSIGMA_GNU_BASE_FLAGS
        "-fPIC -fvisibility=hidden -fvisibility-inlines-hidden"
    )

    # Performance optimization flags
    set(XSIGMA_GNU_PERF_FLAGS
        "-ftree-vectorize -ffast-math -funroll-loops -finline-functions"
    )

    # Warning configuration (comprehensive but not overwhelming)
    set(XSIGMA_GNU_WARNING_FLAGS
        "-Wall -Wextra -Wno-ignored-attributes -Wno-unused-parameter -Wno-sign-compare"
    )

    # Combine all GNU flags
    set(XSIGMA_REQUIRED_CXX_FLAGS
        "${XSIGMA_GNU_BASE_FLAGS} ${XSIGMA_GNU_PERF_FLAGS} ${XSIGMA_GNU_WARNING_FLAGS}")
    set(XSIGMA_REQUIRED_C_FLAGS
        "${XSIGMA_GNU_BASE_FLAGS} ${XSIGMA_GNU_PERF_FLAGS} ${XSIGMA_GNU_WARNING_FLAGS}")

    # GNU linker optimizations (as CMake list for proper handling)
    set(XSIGMA_LINKER_FLAGS_COMMON "-Wl,--gc-sections" "-Wl,--as-needed")

endif()

# Common definitions
add_definitions(-DHAVE_SNPRINTF)

# =============================================================================
# Legacy Platform Support (Modernized)
# =============================================================================

# Modern GCC-specific optimizations
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "XSigma: Applying GCC-specific optimizations...")

    # Windows GCC (MinGW/Cygwin) support
    if(WIN32)
        if(MINGW)
            # MinGW-specific threading and linking
            set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -mthreads")
            set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -mthreads")
            set(XSIGMA_REQUIRED_EXE_LINKER_FLAGS "${XSIGMA_REQUIRED_EXE_LINKER_FLAGS} -mthreads")
            set(XSIGMA_REQUIRED_SHARED_LINKER_FLAGS "${XSIGMA_REQUIRED_SHARED_LINKER_FLAGS} -mthreads")
            set(XSIGMA_REQUIRED_MODULE_LINKER_FLAGS "${XSIGMA_REQUIRED_MODULE_LINKER_FLAGS} -mthreads")
            message(STATUS "XSigma: MinGW threading support enabled")
        else()
            # Cygwin support
            set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -mwin32")
            set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -mwin32")
            link_libraries(-lgdi32)
            message(STATUS "XSigma: Cygwin support enabled")
        endif()
    endif()

    # Linux-specific optimizations
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Enable GNU-specific optimizations
        set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -pthread")
        set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -pthread")
        message(STATUS "XSigma: Linux GCC optimizations enabled")
    endif()

    # Solaris support (if needed)
    if(CMAKE_SYSTEM MATCHES "SunOS.*")
        set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -Wno-unknown-pragmas")
        set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -Wno-unknown-pragmas")
        message(STATUS "XSigma: Solaris GCC support enabled")
    endif()
endif()

# Clang-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "XSigma: Applying Clang-specific optimizations...")

    # Clang-specific performance flags
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -fstrict-aliasing")
    set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -fstrict-aliasing")

    # Apple Clang specific
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        # macOS-specific optimizations
        set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -stdlib=libc++")
        message(STATUS "XSigma: Apple Clang optimizations enabled")
    endif()
endif()

# Legacy system support (modernized)
if(CMAKE_ANSI_CFLAGS)
    set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} ${CMAKE_ANSI_CFLAGS}")
endif()

# =============================================================================
# Intel Compiler Support (Modernized)
# =============================================================================

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message(STATUS "XSigma: Applying Intel compiler optimizations...")

    # Intel-specific performance optimizations
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -xHost -ipo")
    set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -xHost -ipo")

    # Intel-specific vectorization
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -vec-report=2")

    # Check if -i_dynamic is needed (legacy support)
    if(UNIX)
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/TestNO_ICC_IDYNAMIC_NEEDED.cmake")
            include(${CMAKE_CURRENT_LIST_DIR}/TestNO_ICC_IDYNAMIC_NEEDED.cmake)
            testno_icc_idynamic_needed(NO_ICC_IDYNAMIC_NEEDED ${CMAKE_CURRENT_LIST_DIR})
            if(NOT NO_ICC_IDYNAMIC_NEEDED)
                set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -i_dynamic")
                message(STATUS "XSigma: Intel compiler -i_dynamic flag enabled")
            endif()
        endif()
    endif()

    message(STATUS "XSigma: Intel compiler optimizations enabled")
endif()

# =============================================================================
# NVIDIA HPC SDK (formerly PGI) Support
# =============================================================================

if(CMAKE_CXX_COMPILER_ID STREQUAL "PGI" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(STATUS "XSigma: Applying NVIDIA HPC SDK optimizations...")

    # Suppress common PGI/NVHPC warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --diag_suppress=236")  # Constant value asserts
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} --diag_suppress=381")  # Redundant semicolons

    # Enable GPU acceleration if available
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -acc -Minfo=accel")

    message(STATUS "XSigma: NVIDIA HPC SDK optimizations enabled")
endif()

# =============================================================================
# Final Flag Application and Validation
# =============================================================================

# Store flags for later application (after compiler tests)
# This prevents interference with CMake's compiler validation

# Cache the flags for application by xsigmaUtils.cmake or other modules
set(XSIGMA_PLATFORM_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS}" CACHE INTERNAL "Platform C flags")
set(XSIGMA_PLATFORM_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS}" CACHE INTERNAL "Platform CXX flags")
set(XSIGMA_PLATFORM_EXE_LINKER_FLAGS "${XSIGMA_REQUIRED_EXE_LINKER_FLAGS}" CACHE INTERNAL "Platform exe linker flags")
set(XSIGMA_PLATFORM_SHARED_LINKER_FLAGS "${XSIGMA_REQUIRED_SHARED_LINKER_FLAGS}" CACHE INTERNAL "Platform shared linker flags")
set(XSIGMA_PLATFORM_MODULE_LINKER_FLAGS "${XSIGMA_REQUIRED_MODULE_LINKER_FLAGS}" CACHE INTERNAL "Platform module linker flags")
set(XSIGMA_PLATFORM_COMMON_LINKER_FLAGS "${XSIGMA_LINKER_FLAGS_COMMON}" CACHE INTERNAL "Platform common linker flags")

# Apply basic, safe flags immediately (these shouldn't interfere with compiler tests)
if(MSVC)
    # Apply only essential MSVC flags that won't break compiler tests
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Zc:__cplusplus /utf-8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /utf-8")
else()
    # Apply only essential GCC/Clang flags that won't break compiler tests
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# Function to apply platform flags after compiler validation
function(xsigma_apply_platform_flags)
    message(STATUS "XSigma: Applying platform-specific optimization flags...")

    # Apply cached platform flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${XSIGMA_PLATFORM_C_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${XSIGMA_PLATFORM_CXX_FLAGS}" PARENT_SCOPE)

    # Apply linker flags
    if(XSIGMA_PLATFORM_EXE_LINKER_FLAGS)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${XSIGMA_PLATFORM_EXE_LINKER_FLAGS}" PARENT_SCOPE)
    endif()
    if(XSIGMA_PLATFORM_SHARED_LINKER_FLAGS)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${XSIGMA_PLATFORM_SHARED_LINKER_FLAGS}" PARENT_SCOPE)
    endif()
    if(XSIGMA_PLATFORM_MODULE_LINKER_FLAGS)
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${XSIGMA_PLATFORM_MODULE_LINKER_FLAGS}" PARENT_SCOPE)
    endif()
    if(XSIGMA_PLATFORM_COMMON_LINKER_FLAGS)
        # Convert list to space-separated string for proper linker flag handling
        string(REPLACE ";" " " COMMON_LINKER_FLAGS_STR "${XSIGMA_PLATFORM_COMMON_LINKER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${COMMON_LINKER_FLAGS_STR}" PARENT_SCOPE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${COMMON_LINKER_FLAGS_STR}" PARENT_SCOPE)
    endif()

    message(STATUS "XSigma: Platform optimization flags applied successfully")
endfunction()

# =============================================================================
# Configuration Summary and Caching
# =============================================================================

# Cache the configuration for performance
set(XSIGMA_PLATFORM_FLAGS_CACHE_KEY
    "${CMAKE_CXX_COMPILER_ID}_${CMAKE_CXX_COMPILER_VERSION}_${CMAKE_SYSTEM_NAME}"
    CACHE INTERNAL "Platform flags cache key")

# Store configuration summary
set(XSIGMA_PLATFORM_CONFIG_SUMMARY
    "Compiler: ${CMAKE_CXX_COMPILER_ID}, Platform: ${CMAKE_SYSTEM_NAME}"
    CACHE INTERNAL "Platform configuration summary")

message(STATUS "XSigma: Platform optimization completed")
message(STATUS "XSigma: ${XSIGMA_PLATFORM_CONFIG_SUMMARY}")
message(STATUS "XSigma: C++ Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "XSigma: Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}")

# =============================================================================
# End of platform.cmake
# =============================================================================
