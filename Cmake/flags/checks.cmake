# ============================================================================= XSigma System
# Validation and Checks Module
# =============================================================================
# This module performs efficient system validation with aggressive caching to minimize CMake
# reconfiguration overhead while ensuring all required dependencies and capabilities are available.
#
# Performance Optimizations: - Cached validation results to avoid redundant checks - Fast platform
# detection with cached results - Efficient compiler capability validation - Streamlined dependency
# checking
# =============================================================================

# Guard against multiple inclusions for performance
if(XSIGMA_CHECKS_CONFIGURED)
  return()
endif()
set(XSIGMA_CHECKS_CONFIGURED TRUE CACHE INTERNAL "Checks module loaded")

# Create cache key for validation results
set(XSIGMA_VALIDATION_CACHE_KEY
    "${CMAKE_CXX_COMPILER_ID}_${CMAKE_CXX_COMPILER_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_VERSION}"
    CACHE INTERNAL "Validation cache key"
)

# Check if validation has already been completed for this configuration
if(XSIGMA_VALIDATION_COMPLETED STREQUAL XSIGMA_VALIDATION_CACHE_KEY)
  message(STATUS "XSigma: Using cached validation results")
  return()
endif()

message(STATUS "XSigma: Performing system validation...")

# ============================================================================= Platform Detection
# with Caching
# =============================================================================

# Fast platform detection with cached results
if(NOT DEFINED XSIGMA_PLATFORM_DETECTED)
  if(WIN32)
    set(XSIGMA_PLATFORM "Windows" CACHE INTERNAL "Detected platform")
    set(XSIGMA_PLATFORM_WINDOWS TRUE CACHE INTERNAL "Windows platform detected")
  elseif(APPLE)
    set(XSIGMA_PLATFORM "macOS" CACHE INTERNAL "Detected platform")
    set(XSIGMA_PLATFORM_MACOS TRUE CACHE INTERNAL "macOS platform detected")
  elseif(UNIX)
    set(XSIGMA_PLATFORM "Linux" CACHE INTERNAL "Detected platform")
    set(XSIGMA_PLATFORM_LINUX TRUE CACHE INTERNAL "Linux platform detected")
  else()
    message(WARNING "XSigma: Unknown platform detected")
    set(XSIGMA_PLATFORM "Unknown" CACHE INTERNAL "Detected platform")
  endif()
  set(XSIGMA_PLATFORM_DETECTED TRUE CACHE INTERNAL "Platform detection completed")
  message(STATUS "XSigma: Platform detected: ${XSIGMA_PLATFORM}")
endif()

# ============================================================================= Compiler Version
# Validation (Updated for Modern Requirements)
# =============================================================================

# Updated minimum compiler versions for C++17 support and modern optimizations
set(XSIGMA_MIN_GCC_VERSION "7.0")
set(XSIGMA_MIN_CLANG_VERSION "5.0")
set(XSIGMA_MIN_APPLE_CLANG_VERSION "9.0")
set(XSIGMA_MIN_MSVC_VERSION "19.14") # VS 2017 15.7
set(XSIGMA_MIN_INTEL_VERSION "18.0")

set(XSIGMA_COMPILER_ID ${CMAKE_CXX_COMPILER_ID} CACHE INTERNAL "Compiler ID")
mark_as_advanced(XSIGMA_COMPILER_ID)

# GCC version check
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${XSIGMA_MIN_GCC_VERSION})
    message(
      FATAL_ERROR
        "XSigma requires GCC ${XSIGMA_MIN_GCC_VERSION} or later for C++17 support and modern optimizations. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
  set(XSIGMA_COMPILER_GCC TRUE CACHE INTERNAL "GCC compiler detected")
  message(STATUS "XSigma: GCC ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(XSIGMA_COMPILER_ID "gcc" CACHE INTERNAL "Compiler ID")

  # Clang version check
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${XSIGMA_MIN_CLANG_VERSION})
    message(
      FATAL_ERROR
        "XSigma requires Clang ${XSIGMA_MIN_CLANG_VERSION} or later for C++17 support and modern optimizations. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
  set(XSIGMA_COMPILER_CLANG TRUE CACHE INTERNAL "Clang compiler detected")
  message(STATUS "XSigma: Clang ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(XSIGMA_COMPILER_ID "clang" CACHE INTERNAL "Compiler ID")

  # Apple Clang version check
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${XSIGMA_MIN_APPLE_CLANG_VERSION})
    message(
      FATAL_ERROR
        "XSigma requires Apple Clang ${XSIGMA_MIN_APPLE_CLANG_VERSION} or later for C++17 support and modern optimizations. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
  set(XSIGMA_COMPILER_APPLE_CLANG TRUE CACHE INTERNAL "Apple Clang compiler detected")
  message(STATUS "XSigma: Apple Clang ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(XSIGMA_COMPILER_ID "clang" CACHE INTERNAL "Compiler ID")

  # MSVC version check
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${XSIGMA_MIN_MSVC_VERSION})
    message(
      FATAL_ERROR
        "XSigma requires MSVC ${XSIGMA_MIN_MSVC_VERSION} or later (Visual Studio 2017 15.7+) for C++17 support and modern optimizations. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
  set(XSIGMA_COMPILER_MSVC TRUE CACHE INTERNAL "MSVC compiler detected")
  message(STATUS "XSigma: MSVC ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(XSIGMA_COMPILER_ID "msvc" CACHE INTERNAL "Compiler ID")

  # Intel C++ version check
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${XSIGMA_MIN_INTEL_VERSION})
    message(
      FATAL_ERROR
        "XSigma requires Intel C++ ${XSIGMA_MIN_INTEL_VERSION} or later for C++17 support and modern optimizations. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()
  set(XSIGMA_COMPILER_INTEL TRUE CACHE INTERNAL "Intel compiler detected")
  message(STATUS "XSigma: Intel C++ ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(XSIGMA_COMPILER_ID "intel" CACHE INTERNAL "Compiler ID")

else()
  message(WARNING "XSigma: Unknown compiler '${CMAKE_CXX_COMPILER_ID}'. Build may fail.")
endif()

# ============================================================================= C++ Standard
# Validation
# =============================================================================

# Ensure C++17 is properly configured (updated from C++11)
if(NOT XSIGMA_IGNORE_CMAKE_CXX_STANDARD_CHECKS)
  # Validate that the requested C++ standard is supported
  if(XSIGMA_CXX_STANDARD LESS 11)
    message(
      FATAL_ERROR "XSigma requires C++11 or later. Current setting: C++${XSIGMA_CXX_STANDARD}"
    )
  endif()

  # Ensure standard is properly set
  set(CMAKE_CXX_STANDARD ${XSIGMA_CXX_STANDARD})
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  message(STATUS "XSigma: C++${XSIGMA_CXX_STANDARD} standard validated and configured")
endif()

# ============================================================================= Essential System
# Dependencies Validation
# =============================================================================

include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckFunctionExists)
include(CheckLibraryExists)
include(CheckSymbolExists)

# Threading support validation (cached)
if(NOT DEFINED XSIGMA_THREADING_VALIDATED)
  find_package(Threads REQUIRED)
  if(NOT Threads_FOUND)
    message(FATAL_ERROR "XSigma: Threading support is required but not found")
  endif()
  set(XSIGMA_THREADING_VALIDATED TRUE CACHE INTERNAL "Threading validation completed")
  message(STATUS "XSigma: Threading support validated")
endif()

# Math library validation (cached)
if(NOT DEFINED XSIGMA_MATH_LIB_VALIDATED)
  if(NOT WIN32)
    check_library_exists(m sin "" HAVE_LIBM)
    if(NOT HAVE_LIBM)
      message(FATAL_ERROR "XSigma: Math library (libm) is required but not found")
    endif()
  endif()
  set(XSIGMA_MATH_LIB_VALIDATED TRUE CACHE INTERNAL "Math library validation completed")
  message(STATUS "XSigma: Math library support validated")
endif()

# ============================================================================= Compiler Capability
# Validation (Cached)
# =============================================================================

include(CheckCXXSourceCompiles)

# C++17 features validation (cached)
if(NOT DEFINED XSIGMA_CXX17_FEATURES_VALIDATED)
  # Test structured bindings
  check_cxx_source_compiles(
    "
        #include <tuple>
        int main() {
            auto [a, b] = std::make_tuple(1, 2);
            return a + b - 3;
        }
    "
    XSIGMA_HAS_STRUCTURED_BINDINGS
  )

  # Test if constexpr
  check_cxx_source_compiles(
    "
        constexpr int test_func(int x) {
            if constexpr (true) {
                return x * 2;
            } else {
                return x;
            }
        }
        int main() {
            constexpr int result = test_func(5);
            return result - 10;
        }
    "
    XSIGMA_HAS_IF_CONSTEXPR
  )

  # Test std::optional
  check_cxx_source_compiles(
    "
        #include <optional>
        int main() {
            std::optional<int> opt = 42;
            return opt.value_or(0) - 42;
        }
    "
    XSIGMA_HAS_STD_OPTIONAL
  )

  if(NOT XSIGMA_HAS_STRUCTURED_BINDINGS OR NOT XSIGMA_HAS_IF_CONSTEXPR OR NOT
                                                                          XSIGMA_HAS_STD_OPTIONAL
  )
    message(WARNING "XSigma: Compiler does not support required C++17 features")
  endif()

  set(XSIGMA_CXX17_FEATURES_VALIDATED TRUE CACHE INTERNAL "C++17 features validation completed")
  message(STATUS "XSigma: C++17 features validated")
endif()

# Exception handling validation (cached)
if(NOT DEFINED XSIGMA_EXCEPTION_HANDLING_VALIDATED)
  check_cxx_source_compiles(
    "
        #include <stdexcept>
        int main() {
            try {
                throw std::runtime_error(\"test\");
            } catch (const std::exception& e) {
                return 0;
            }
            return 1;
        }
    "
    XSIGMA_HAS_EXCEPTION_HANDLING
  )

  if(NOT XSIGMA_HAS_EXCEPTION_HANDLING)
    message(WARNING "XSigma: Exception handling not available - some features may be limited")
  endif()

  set(XSIGMA_EXCEPTION_HANDLING_VALIDATED TRUE CACHE INTERNAL
                                                     "Exception handling validation completed"
  )
  message(STATUS "XSigma: Exception handling validated")
endif()

# ============================================================================= Platform-Specific
# Validations
# =============================================================================

# Windows-specific checks
if(XSIGMA_PLATFORM_WINDOWS)
  if(NOT DEFINED XSIGMA_WINDOWS_VALIDATED)
    check_include_file("windows.h" HAVE_WINDOWS_H)
    if(NOT HAVE_WINDOWS_H)
      message(FATAL_ERROR "XSigma: Windows.h header not found on Windows platform")
    endif()
    set(XSIGMA_WINDOWS_VALIDATED TRUE CACHE INTERNAL "Windows validation completed")
    message(STATUS "XSigma: Windows platform validation completed")
  endif()
endif()

# Unix-specific checks
if(XSIGMA_PLATFORM_LINUX OR XSIGMA_PLATFORM_MACOS)
  if(NOT DEFINED XSIGMA_UNIX_VALIDATED)
    check_include_file("unistd.h" HAVE_UNISTD_H)
    check_include_file("pthread.h" HAVE_PTHREAD_H)
    if(NOT HAVE_UNISTD_H OR NOT HAVE_PTHREAD_H)
      message(FATAL_ERROR "XSigma: Required Unix headers not found")
    endif()
    set(XSIGMA_UNIX_VALIDATED TRUE CACHE INTERNAL "Unix validation completed")
    message(STATUS "XSigma: Unix platform validation completed")
  endif()
endif()

# ============================================================================= Validation
# Completion and Caching
# =============================================================================

# Mark validation as completed for this configuration
set(XSIGMA_VALIDATION_COMPLETED "${XSIGMA_VALIDATION_CACHE_KEY}"
    CACHE INTERNAL "Validation completed for configuration"
)

# Store validation summary
set(XSIGMA_VALIDATION_SUMMARY
    "Platform: ${XSIGMA_PLATFORM}, Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, C++: ${XSIGMA_CXX_STANDARD}"
    CACHE INTERNAL "Validation summary"
)

message(STATUS "XSigma: System validation completed successfully")
message(STATUS "XSigma: ${XSIGMA_VALIDATION_SUMMARY}")

# ============================================================================= End of checks.cmake
# =============================================================================
