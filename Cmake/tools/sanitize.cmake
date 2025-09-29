# =============================================================================
# XSigma Sanitizer Configuration
# =============================================================================
# Comprehensive sanitizer support for memory debugging and analysis
# Supports: AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer,
#          MemorySanitizer, and LeakSanitizer
# Compatible with: GCC, Clang, Apple Clang, and MSVC (where supported)
# =============================================================================

if(NOT XSIGMA_ENABLE_SANITIZER)
    return()
endif()

# Validate sanitizer type
set(VALID_SANITIZER_TYPES "address;undefined;thread;memory;leak")
if(NOT XSIGMA_SANITIZER_TYPE IN_LIST VALID_SANITIZER_TYPES)
    message(FATAL_ERROR "Invalid XSIGMA_SANITIZER_TYPE: ${XSIGMA_SANITIZER_TYPE}. "
                        "Valid options are: ${VALID_SANITIZER_TYPES}")
endif()

# Compiler detection
set(CMAKE_COMPILER_IS_CLANGXX FALSE)
set(CMAKE_COMPILER_IS_MSVC FALSE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_COMPILER_IS_CLANGXX TRUE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_COMPILER_IS_MSVC TRUE)
endif()

# =============================================================================
# Sanitizer Compatibility Matrix
# =============================================================================
function(check_sanitizer_compatibility)
    set(SANITIZER_SUPPORTED FALSE)

    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
        # GCC and Clang support most sanitizers
        if(XSIGMA_SANITIZER_TYPE STREQUAL "address" OR
           XSIGMA_SANITIZER_TYPE STREQUAL "undefined" OR
           XSIGMA_SANITIZER_TYPE STREQUAL "thread" OR
           XSIGMA_SANITIZER_TYPE STREQUAL "leak")
            set(SANITIZER_SUPPORTED TRUE)
        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "memory")
            # MemorySanitizer is Clang-only
            if(CMAKE_COMPILER_IS_CLANGXX)
                set(SANITIZER_SUPPORTED TRUE)
            else()
                message(WARNING "MemorySanitizer is only supported with Clang. "
                               "Falling back to AddressSanitizer.")
                set(XSIGMA_SANITIZER_TYPE "address" CACHE STRING
                    "The sanitizer to use" FORCE)
                set(SANITIZER_SUPPORTED TRUE)
            endif()
        endif()
    elseif(CMAKE_COMPILER_IS_MSVC)
        # MSVC only supports AddressSanitizer
        if(XSIGMA_SANITIZER_TYPE STREQUAL "address")
            set(SANITIZER_SUPPORTED TRUE)
        else()
            message(WARNING "MSVC only supports AddressSanitizer. "
                           "Falling back to AddressSanitizer.")
            set(XSIGMA_SANITIZER_TYPE "address" CACHE STRING
                "The sanitizer to use" FORCE)
            set(SANITIZER_SUPPORTED TRUE)
        endif()
    endif()

    if(NOT SANITIZER_SUPPORTED)
        message(FATAL_ERROR "Sanitizer ${XSIGMA_SANITIZER_TYPE} is not supported "
                           "with compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    # Export to parent scope
    set(XSIGMA_SANITIZER_TYPE ${XSIGMA_SANITIZER_TYPE} PARENT_SCOPE)
endfunction()

check_sanitizer_compatibility()

# =============================================================================
# Sanitizer Configuration Functions
# =============================================================================

function(configure_sanitizer_flags)
    set(xsigma_sanitize_args)
    set(xsigma_sanitize_link_args)

    if(CMAKE_COMPILER_IS_MSVC)
        # MSVC sanitizer configuration
        if(XSIGMA_SANITIZER_TYPE STREQUAL "address")
            list(APPEND xsigma_sanitize_args "/fsanitize=address")
            # MSVC AddressSanitizer requires specific runtime library
            list(APPEND xsigma_sanitize_args "/MD")
            # Suppress ASAN warning about debug info for third-party libraries
            list(APPEND xsigma_sanitize_args "/wd5072")
        endif()
    else()
        # GCC/Clang sanitizer configuration
        list(APPEND xsigma_sanitize_args "-fsanitize=${XSIGMA_SANITIZER_TYPE}")
        list(APPEND xsigma_sanitize_link_args "-fsanitize=${XSIGMA_SANITIZER_TYPE}")

        # Common flags for all sanitizers
        list(APPEND xsigma_sanitize_args "-fno-omit-frame-pointer")
        list(APPEND xsigma_sanitize_args "-g")
        list(APPEND xsigma_sanitize_args "-O1")

        # Windows-specific library requirements for sanitizers
        if(WIN32 AND CMAKE_COMPILER_IS_CLANGXX)
            if(XSIGMA_SANITIZER_TYPE STREQUAL "undefined")
                # UBSan on Windows requires additional system libraries
                list(APPEND xsigma_sanitize_link_args "-ldbghelp")
            endif()
        endif()

        # Additional flags for specific sanitizers
        if(XSIGMA_SANITIZER_TYPE STREQUAL "address")
            list(APPEND xsigma_sanitize_args "-fno-optimize-sibling-calls")

            # Enable stack-use-after-return detection (Clang and newer GCC)
            if(CMAKE_COMPILER_IS_CLANGXX OR
               (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "8.0"))
                list(APPEND xsigma_sanitize_args "-fsanitize-address-use-after-scope")
            endif()

            # GCC-specific AddressSanitizer optimizations
            if(CMAKE_COMPILER_IS_GNUCXX)
                list(APPEND xsigma_sanitize_args "-fsanitize-recover=address")
            endif()

        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "undefined")
            # Enable additional UBSan checks
            list(APPEND xsigma_sanitize_args "-fno-sanitize-recover=undefined")

            # GCC-specific UBSan flags
            if(CMAKE_COMPILER_IS_GNUCXX)
                # Enable additional undefined behavior checks in GCC
                list(APPEND xsigma_sanitize_args "-fsanitize=float-divide-by-zero")
                list(APPEND xsigma_sanitize_args "-fsanitize=float-cast-overflow")
            endif()

        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "thread")
            # ThreadSanitizer requires position-independent code
            list(APPEND xsigma_sanitize_args "-fPIC")

            # GCC-specific ThreadSanitizer optimizations
            if(CMAKE_COMPILER_IS_GNUCXX)
                list(APPEND xsigma_sanitize_args "-ftsan-instrument-func-entry-exit")
            endif()

        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "memory")
            # MemorySanitizer specific flags (Clang only)
            if(CMAKE_COMPILER_IS_CLANGXX)
                list(APPEND xsigma_sanitize_args "-fsanitize-memory-track-origins=2")
            endif()

        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "leak")
            # LeakSanitizer flags
            # GCC-specific leak detection optimizations
            if(CMAKE_COMPILER_IS_GNUCXX)
                list(APPEND xsigma_sanitize_args "-fsanitize-recover=leak")
            endif()
        endif()
    endif()

    # Export to parent scope and cache for global access
    set(xsigma_sanitize_args ${xsigma_sanitize_args} PARENT_SCOPE)
    set(xsigma_sanitize_link_args ${xsigma_sanitize_link_args} PARENT_SCOPE)
    set(XSIGMA_SANITIZER_COMPILE_FLAGS ${xsigma_sanitize_args} CACHE INTERNAL "Sanitizer compile flags")
    set(XSIGMA_SANITIZER_LINK_FLAGS ${xsigma_sanitize_link_args} CACHE INTERNAL "Sanitizer link flags")
endfunction()

function(configure_sanitizer_runtime_libraries)
    if(UNIX AND NOT APPLE AND NOT CMAKE_COMPILER_IS_MSVC)
        # Find sanitizer runtime libraries for preloading
        set(sanitizer_libraries)

        # Get compiler-specific library paths
        if(CMAKE_COMPILER_IS_GNUCXX)
            # GCC sanitizer libraries
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE GCC_ASAN_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-search-dirs
                OUTPUT_VARIABLE GCC_SEARCH_DIRS
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif()

        if(XSIGMA_SANITIZER_TYPE STREQUAL "address")
            if(CMAKE_COMPILER_IS_GNUCXX)
                # Try GCC-specific path first
                find_library(XSIGMA_ASAN_LIBRARY
                    NAMES libasan.so.8 libasan.so.7 libasan.so.6 libasan.so.5 libasan.so
                    PATHS ${GCC_ASAN_PATH}
                    DOC "AddressSanitizer runtime library (GCC)")
            else()
                # Clang paths
                find_library(XSIGMA_ASAN_LIBRARY
                    NAMES libclang_rt.asan-x86_64.so libasan.so.8 libasan.so.7 libasan.so.6 libasan.so.5
                    DOC "AddressSanitizer runtime library (Clang)")
            endif()
            if(XSIGMA_ASAN_LIBRARY)
                list(APPEND sanitizer_libraries ${XSIGMA_ASAN_LIBRARY})
            endif()

        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "thread")
            if(CMAKE_COMPILER_IS_GNUCXX)
                find_library(XSIGMA_TSAN_LIBRARY
                    NAMES libtsan.so.2 libtsan.so.1 libtsan.so.0 libtsan.so
                    DOC "ThreadSanitizer runtime library (GCC)")
            else()
                find_library(XSIGMA_TSAN_LIBRARY
                    NAMES libclang_rt.tsan-x86_64.so libtsan.so.2 libtsan.so.1 libtsan.so.0
                    DOC "ThreadSanitizer runtime library (Clang)")
            endif()
            if(XSIGMA_TSAN_LIBRARY)
                list(APPEND sanitizer_libraries ${XSIGMA_TSAN_LIBRARY})
            endif()

        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "undefined")
            if(CMAKE_COMPILER_IS_GNUCXX)
                find_library(XSIGMA_UBSAN_LIBRARY
                    NAMES libubsan.so.1 libubsan.so.0 libubsan.so
                    DOC "UndefinedBehaviorSanitizer runtime library (GCC)")
            else()
                find_library(XSIGMA_UBSAN_LIBRARY
                    NAMES libclang_rt.ubsan_standalone-x86_64.so libubsan.so.1 libubsan.so.0
                    DOC "UndefinedBehaviorSanitizer runtime library (Clang)")
            endif()
            if(XSIGMA_UBSAN_LIBRARY)
                list(APPEND sanitizer_libraries ${XSIGMA_UBSAN_LIBRARY})
            endif()

        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "memory")
            # MemorySanitizer is Clang-only
            if(CMAKE_COMPILER_IS_CLANGXX)
                find_library(XSIGMA_MSAN_LIBRARY
                    NAMES libclang_rt.msan-x86_64.so libmsan.so.0
                    DOC "MemorySanitizer runtime library (Clang)")
                if(XSIGMA_MSAN_LIBRARY)
                    list(APPEND sanitizer_libraries ${XSIGMA_MSAN_LIBRARY})
                endif()
            endif()

        elseif(XSIGMA_SANITIZER_TYPE STREQUAL "leak")
            if(CMAKE_COMPILER_IS_GNUCXX)
                find_library(XSIGMA_LSAN_LIBRARY
                    NAMES liblsan.so.0 liblsan.so
                    DOC "LeakSanitizer runtime library (GCC)")
            else()
                find_library(XSIGMA_LSAN_LIBRARY
                    NAMES libclang_rt.lsan-x86_64.so liblsan.so.0
                    DOC "LeakSanitizer runtime library (Clang)")
            endif()
            if(XSIGMA_LSAN_LIBRARY)
                list(APPEND sanitizer_libraries ${XSIGMA_LSAN_LIBRARY})
            endif()
        endif()

        if(sanitizer_libraries)
            mark_as_advanced(XSIGMA_ASAN_LIBRARY XSIGMA_TSAN_LIBRARY
                           XSIGMA_UBSAN_LIBRARY XSIGMA_MSAN_LIBRARY XSIGMA_LSAN_LIBRARY)
            string(REPLACE ";" ":" sanitizer_preload "${sanitizer_libraries}")
            set(_xsigma_testing_ld_preload "${sanitizer_preload}" PARENT_SCOPE)
            message(STATUS "Found sanitizer runtime libraries: ${sanitizer_libraries}")
        else()
            message(STATUS "No sanitizer runtime libraries found for preloading")
        endif()
    endif()
endfunction()
function(configure_sanitizer_ignore_file)
    # Configure sanitizer ignore file for Clang
    if(CMAKE_COMPILER_IS_CLANGXX)
        set(SCRIPTS_IGNORE "${XSIGMA_SOURCE_DIR}/Scripts/sanitizer_ignore.txt")

        if(EXISTS "${SCRIPTS_IGNORE}")
            message(STATUS "Using sanitizer ignore file: ${SCRIPTS_IGNORE}")
            # Add ignore file to sanitizer arguments
            list(APPEND xsigma_sanitize_args
                 "SHELL:-fsanitize-ignorelist=${SCRIPTS_IGNORE}")
            set(xsigma_sanitize_args ${xsigma_sanitize_args} PARENT_SCOPE)
        else()
            message(WARNING "Sanitizer ignore file not found: ${SCRIPTS_IGNORE}")
            message(WARNING "Sanitizer will run without ignore list")
        endif()
    endif()
endfunction()

# =============================================================================
# Main Sanitizer Configuration
# =============================================================================

message(STATUS "Configuring ${XSIGMA_SANITIZER_TYPE} sanitizer for ${CMAKE_CXX_COMPILER_ID}")

# Configure sanitizer flags
configure_sanitizer_flags()

# Configure runtime libraries
configure_sanitizer_runtime_libraries()

# Configure ignore file
configure_sanitizer_ignore_file()

# Apply sanitizer configuration to build target
if(TARGET xsigmabuild)
    target_compile_options(xsigmabuild INTERFACE
        "$<BUILD_INTERFACE:${xsigma_sanitize_args}>")

    if(xsigma_sanitize_link_args)
        target_link_options(xsigmabuild INTERFACE
            "$<BUILD_INTERFACE:${xsigma_sanitize_link_args}>")
    endif()

    message(STATUS "Applied sanitizer configuration to xsigmabuild target")
else()
    message(WARNING "xsigmabuild target not found - sanitizer flags not applied")
endif()

# Function to apply sanitizer flags to third-party libraries
# This ensures consistent annotations between main project and third-party libraries
function(xsigma_apply_sanitizer_to_third_party_targets)
    if(NOT xsigma_sanitize_args)
        return()
    endif()

    # List of third-party targets that need sanitizer flags for consistency
    set(THIRD_PARTY_TARGETS
        gtest
        gtest_main
        fmt
        benchmark
        benchmark_main
    )

    foreach(target_name ${THIRD_PARTY_TARGETS})
        if(TARGET ${target_name})
            target_compile_options(${target_name} PRIVATE ${xsigma_sanitize_args})
            if(xsigma_sanitize_link_args)
                target_link_options(${target_name} PRIVATE ${xsigma_sanitize_link_args})
            endif()
            message(STATUS "Applied sanitizer flags to third-party target: ${target_name}")
        endif()
    endforeach()
endfunction()

# Store sanitizer flags in cache for use by other parts of the build system
set(XSIGMA_SANITIZER_COMPILE_FLAGS "${xsigma_sanitize_args}" CACHE INTERNAL "Sanitizer compile flags")
set(XSIGMA_SANITIZER_LINK_FLAGS "${xsigma_sanitize_link_args}" CACHE INTERNAL "Sanitizer link flags")

# Set environment variables for testing
if(_xsigma_testing_ld_preload)
    set(ENV{LD_PRELOAD} "${_xsigma_testing_ld_preload}")
    message(STATUS "Set LD_PRELOAD for testing: ${_xsigma_testing_ld_preload}")
endif()

# Print configuration summary
message(STATUS "Sanitizer configuration complete:")
message(STATUS "  Type: ${XSIGMA_SANITIZER_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Compile flags: ${xsigma_sanitize_args}")
if(xsigma_sanitize_link_args)
    message(STATUS "  Link flags: ${xsigma_sanitize_link_args}")
endif()
if(_xsigma_testing_ld_preload)
    message(STATUS "  Runtime preload: ${_xsigma_testing_ld_preload}")
endif()
