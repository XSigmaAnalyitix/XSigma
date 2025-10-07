# =============================================================================
# XSigma Simplified Sanitizer Configuration
# =============================================================================
# Simplified sanitizer support for memory debugging and analysis
# Supports: AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer,
#          MemorySanitizer, and LeakSanitizer
# Compatible with: Clang only (cross-platform: Linux, macOS, Windows)
# =============================================================================

# Early exit if sanitizers are disabled
if(NOT XSIGMA_ENABLE_SANITIZER)
    return()
endif()

# =============================================================================
# Compiler Validation - Clang Only
# =============================================================================

function(validate_sanitizer_compiler)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "Sanitizers are only supported with Clang compiler.\n"
            "Current compiler: ${CMAKE_CXX_COMPILER_ID}\n"
            "Please use Clang to enable sanitizers.")
    endif()

    message(STATUS "Sanitizer compiler validation passed: ${CMAKE_CXX_COMPILER_ID}")
endfunction()

validate_sanitizer_compiler()

# =============================================================================
# Build Configuration Enforcement
# =============================================================================

function(enforce_debug_build)
    # On Windows, use RelWithDebInfo to match sanitizer runtime library
    if(WIN32)
        if(NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            message(STATUS "Sanitizers on Windows require RelWithDebInfo build - forcing CMAKE_BUILD_TYPE to RelWithDebInfo")
            set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
        endif()
    else()
        # On other platforms, use Debug as before
        if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "Sanitizers require Debug build - forcing CMAKE_BUILD_TYPE to Debug")
            set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
        endif()
    endif()

    message(STATUS "Build type enforced for sanitizers: ${CMAKE_BUILD_TYPE}")
endfunction()

enforce_debug_build()

# =============================================================================
# Sanitizer Configuration
# =============================================================================

# Validate sanitizer type
set(VALID_SANITIZERS "address;undefined;thread;memory;leak")
if(NOT XSIGMA_SANITIZER_TYPE IN_LIST VALID_SANITIZERS)
    message(FATAL_ERROR
        "Invalid sanitizer type: ${XSIGMA_SANITIZER_TYPE}\n"
        "Valid options: ${VALID_SANITIZERS}")
endif()

# Platform-specific sanitizer validation
if(XSIGMA_SANITIZER_TYPE STREQUAL "memory" AND WIN32)
    message(FATAL_ERROR "MemorySanitizer is not supported on Windows")
endif()

if(XSIGMA_SANITIZER_TYPE STREQUAL "thread" AND WIN32)
    message(FATAL_ERROR "ThreadSanitizer is not supported on Windows with Clang. Use AddressSanitizer instead.")
endif()

if(XSIGMA_SANITIZER_TYPE STREQUAL "undefined" AND WIN32)
    message(WARNING "UndefinedBehaviorSanitizer may have linker issues on Windows with Clang. Consider using AddressSanitizer instead.")
endif()

if(XSIGMA_SANITIZER_TYPE STREQUAL "leak" AND APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(FATAL_ERROR "LeakSanitizer is not supported on Apple Silicon (ARM64)")
endif()

message(STATUS "Configuring ${XSIGMA_SANITIZER_TYPE} sanitizer with Clang")

# =============================================================================
# Sanitizer Flags Configuration
# =============================================================================

function(configure_sanitizer_flags)
    # Base sanitizer flags
    set(SANITIZER_COMPILE_FLAGS "-fsanitize=${XSIGMA_SANITIZER_TYPE}")
    set(SANITIZER_LINK_FLAGS "-fsanitize=${XSIGMA_SANITIZER_TYPE}")

    # Disable ALL optimizations for sanitizer builds
    list(APPEND SANITIZER_COMPILE_FLAGS "-O0")

    # Essential debugging flags
    list(APPEND SANITIZER_COMPILE_FLAGS "-g")
    list(APPEND SANITIZER_COMPILE_FLAGS "-fno-omit-frame-pointer")

    # Sanitizer-specific flags
    if(XSIGMA_SANITIZER_TYPE STREQUAL "address")
        list(APPEND SANITIZER_COMPILE_FLAGS "-fno-optimize-sibling-calls")
    elseif(XSIGMA_SANITIZER_TYPE STREQUAL "memory")
        list(APPEND SANITIZER_COMPILE_FLAGS "-fsanitize-memory-track-origins=2")
    endif()

    # Export flags to parent scope
    set(XSIGMA_SANITIZER_COMPILE_FLAGS ${SANITIZER_COMPILE_FLAGS} PARENT_SCOPE)
    set(XSIGMA_SANITIZER_LINK_FLAGS ${SANITIZER_LINK_FLAGS} PARENT_SCOPE)

    message(STATUS "Sanitizer compile flags: ${SANITIZER_COMPILE_FLAGS}")
    message(STATUS "Sanitizer link flags: ${SANITIZER_LINK_FLAGS}")
endfunction()

configure_sanitizer_flags()

# =============================================================================
# Apply Sanitizer Configuration
# =============================================================================

# Apply sanitizer configuration to build target only (exclude third-party)
if(TARGET xsigmabuild)
    # Apply sanitizer compile flags
    target_compile_options(xsigmabuild INTERFACE ${XSIGMA_SANITIZER_COMPILE_FLAGS})

    # Apply sanitizer link flags
    if(XSIGMA_SANITIZER_LINK_FLAGS)
        target_link_options(xsigmabuild INTERFACE ${XSIGMA_SANITIZER_LINK_FLAGS})
    endif()

    message(STATUS "Applied sanitizer configuration to xsigmabuild target")
    message(STATUS "  Third-party dependencies excluded to prevent linker mismatches")
else()
    message(FATAL_ERROR "xsigmabuild target not found - cannot apply sanitizer configuration")
endif()

# Configure suppression files for sanitizers
function(configure_sanitizer_suppressions)
    set(SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/Scripts")

    if(XSIGMA_SANITIZER_TYPE STREQUAL "address")
        set(SUPPRESSION_FILE "${SCRIPTS_DIR}/asan_suppressions.txt")
        if(EXISTS ${SUPPRESSION_FILE})
            set(ENV{ASAN_OPTIONS} "suppressions=${SUPPRESSION_FILE}:halt_on_error=1:abort_on_error=1")
            message(STATUS "AddressSanitizer suppressions: ${SUPPRESSION_FILE}")
        endif()
    elseif(XSIGMA_SANITIZER_TYPE STREQUAL "undefined")
        set(SUPPRESSION_FILE "${SCRIPTS_DIR}/ubsan_suppressions.txt")
        if(EXISTS ${SUPPRESSION_FILE})
            set(ENV{UBSAN_OPTIONS} "suppressions=${SUPPRESSION_FILE}:halt_on_error=1:abort_on_error=1")
            message(STATUS "UndefinedBehaviorSanitizer suppressions: ${SUPPRESSION_FILE}")
        endif()
    elseif(XSIGMA_SANITIZER_TYPE STREQUAL "thread")
        set(SUPPRESSION_FILE "${SCRIPTS_DIR}/tsan_suppressions.txt")
        if(EXISTS ${SUPPRESSION_FILE})
            set(ENV{TSAN_OPTIONS} "suppressions=${SUPPRESSION_FILE}:halt_on_error=1:abort_on_error=1")
            message(STATUS "ThreadSanitizer suppressions: ${SUPPRESSION_FILE}")
        endif()
    elseif(XSIGMA_SANITIZER_TYPE STREQUAL "memory")
        set(SUPPRESSION_FILE "${SCRIPTS_DIR}/msan_suppressions.txt")
        if(EXISTS ${SUPPRESSION_FILE})
            set(ENV{MSAN_OPTIONS} "suppressions=${SUPPRESSION_FILE}:halt_on_error=1:abort_on_error=1")
            message(STATUS "MemorySanitizer suppressions: ${SUPPRESSION_FILE}")
        endif()
    elseif(XSIGMA_SANITIZER_TYPE STREQUAL "leak")
        set(SUPPRESSION_FILE "${SCRIPTS_DIR}/lsan_suppressions.txt")
        if(EXISTS ${SUPPRESSION_FILE})
            set(ENV{LSAN_OPTIONS} "suppressions=${SUPPRESSION_FILE}:halt_on_error=1:abort_on_error=1")
            message(STATUS "LeakSanitizer suppressions: ${SUPPRESSION_FILE}")
        endif()
    endif()
endfunction()

configure_sanitizer_suppressions()

# =============================================================================
# Windows-specific Linker Compatibility Fixes
# =============================================================================

# On Windows, ensure consistent debug settings to prevent linker mismatches
if(WIN32)
    # Force consistent _ITERATOR_DEBUG_LEVEL across all targets
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)

    # Force consistent runtime library settings to match sanitizer runtime
    # Use release runtime library to match sanitizer runtime
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

    # Override debug build to use release runtime
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type for sanitizer compatibility" FORCE)

    # Apply minimal sanitizer flags globally to prevent linker mismatches
    # Only apply the sanitizer flag itself, not the debug flags
    string(JOIN " " SANITIZER_FLAG_STR "-fsanitize=${XSIGMA_SANITIZER_TYPE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_FLAG_STR}" CACHE STRING "C flags with sanitizer" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAG_STR}" CACHE STRING "CXX flags with sanitizer" FORCE)

    if(XSIGMA_SANITIZER_LINK_FLAGS)
        string(JOIN " " SANITIZER_LINK_FLAGS_STR ${XSIGMA_SANITIZER_LINK_FLAGS})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_LINK_FLAGS_STR}" CACHE STRING "Exe linker flags with sanitizer" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_LINK_FLAGS_STR}" CACHE STRING "Shared linker flags with sanitizer" FORCE)
    endif()

    message(STATUS "Applied Windows-specific sanitizer compatibility fixes")
    message(STATUS "  _ITERATOR_DEBUG_LEVEL=0 to prevent runtime library mismatches")
    message(STATUS "  Runtime library: MultiThreadedDLL (/MD) to match sanitizer runtime")
    message(STATUS "  Build type: RelWithDebInfo for sanitizer compatibility")
    message(STATUS "  Minimal global sanitizer flags to ensure linker compatibility")
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "=== Sanitizer Configuration Summary ===")
message(STATUS "Sanitizer Type: ${XSIGMA_SANITIZER_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
if(WIN32)
    message(STATUS "Target Scope: All code (Windows compatibility mode)")
    message(STATUS "Third-party Dependencies: Minimal instrumentation for linker compatibility")
    message(STATUS "Windows Fixes: _ITERATOR_DEBUG_LEVEL=0, minimal global flags")
else()
    message(STATUS "Target Scope: Main library code only")
    message(STATUS "Third-party Dependencies: Excluded from instrumentation")
endif()
message(STATUS "Suppression Files: Configured for runtime error filtering")
message(STATUS "========================================")