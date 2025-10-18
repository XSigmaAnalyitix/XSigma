# Link Time Optimization (LTO) Configuration Module
# Enables Link Time Optimization for improved runtime performance
# Supports GCC, Clang, and MSVC on Linux, macOS, and Windows
#
# NOTE: This module applies LTO flags ONLY to the xsigmabuild interface target,
# ensuring that third-party dependencies are not affected by LTO configuration.

# Guard against multiple inclusions
if(XSIGMA_LTO_CONFIGURED)
  return()
endif()
set(XSIGMA_LTO_CONFIGURED TRUE CACHE INTERNAL "LTO module loaded")

# Check if LTO is enabled
if(NOT XSIGMA_ENABLE_LTO)
  message(STATUS "Link Time Optimization (LTO) is disabled")
  return()
endif()

message(STATUS "Configuring Link Time Optimization (LTO)...")

# ============================================================================
# LTO Configuration for Different Compilers
# ============================================================================

# Store LTO flags to be applied to xsigmabuild target later
set(XSIGMA_LTO_COMPILE_FLAGS)
set(XSIGMA_LTO_LINK_FLAGS)

# Determine compiler and prepare appropriate LTO flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # GCC LTO Configuration
  message(STATUS "Configuring LTO for GCC compiler")

  # Prepare -flto flag for compilation and linking
  list(APPEND XSIGMA_LTO_COMPILE_FLAGS -flto)
  list(APPEND XSIGMA_LTO_LINK_FLAGS -flto)

  # For GCC, we need to use gcc-ar and gcc-ranlib for static libraries
  find_program(GCC_AR gcc-ar)
  find_program(GCC_RANLIB gcc-ranlib)

  if(GCC_AR AND GCC_RANLIB)
    set(CMAKE_AR "${GCC_AR}" CACHE FILEPATH "GCC ar wrapper for LTO" FORCE)
    set(CMAKE_RANLIB "${GCC_RANLIB}" CACHE FILEPATH "GCC ranlib wrapper for LTO" FORCE)
    message(STATUS "Using gcc-ar and gcc-ranlib for LTO")
  else()
    message(WARNING "gcc-ar or gcc-ranlib not found - LTO may not work correctly with static libraries")
  endif()

  # Optional: Add -Wno-lto-type-mismatch to suppress LTO-related warnings
  list(APPEND XSIGMA_LTO_COMPILE_FLAGS -Wno-lto-type-mismatch)

  message(STATUS "GCC LTO enabled with -flto flag")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  # Clang/Apple Clang LTO Configuration
  message(STATUS "Configuring LTO for Clang compiler")

  # Prepare -flto flag for compilation and linking
  list(APPEND XSIGMA_LTO_COMPILE_FLAGS -flto)
  list(APPEND XSIGMA_LTO_LINK_FLAGS -flto)

  # Optional: Add -Wno-lto-type-mismatch to suppress LTO-related warnings (not on Windows)
  if(NOT WIN32)
    list(APPEND XSIGMA_LTO_COMPILE_FLAGS -Wno-lto-type-mismatch)
  endif()

  # For Clang with LTO, we may need to use llvm-ar and llvm-ranlib
  # This is optional but recommended for better LTO support
  find_program(LLVM_AR llvm-ar)
  find_program(LLVM_RANLIB llvm-ranlib)

  if(LLVM_AR AND LLVM_RANLIB)
    set(CMAKE_AR "${LLVM_AR}" CACHE FILEPATH "LLVM ar wrapper for LTO" FORCE)
    set(CMAKE_RANLIB "${LLVM_RANLIB}" CACHE FILEPATH "LLVM ranlib wrapper for LTO" FORCE)
    message(STATUS "Using llvm-ar and llvm-ranlib for LTO")
  endif()

  message(STATUS "Clang LTO enabled with -flto flag")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # MSVC LTO Configuration
  message(STATUS "Configuring LTO for MSVC compiler")

  # MSVC uses /GL for compilation and /LTCG for linking
  # Prepare flags to be applied to xsigmabuild target
  list(APPEND XSIGMA_LTO_COMPILE_FLAGS /GL)
  list(APPEND XSIGMA_LTO_LINK_FLAGS /LTCG)

  message(STATUS "MSVC LTO enabled with /GL (compilation) and /LTCG (linking)")

else()
  message(WARNING "LTO configuration not available for compiler: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "Supported compilers: GCC, Clang, AppleClang, MSVC")
endif()

# ============================================================================
# Verify LTO Support
# ============================================================================

# Check if the compiler supports LTO by attempting a simple compilation
include(CheckCXXSourceCompiles)

cmake_push_check_state(RESET)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # MSVC: Use /GL flag
  set(CMAKE_REQUIRED_FLAGS "/GL")
else()
  # GCC/Clang: Use -flto flag
  set(CMAKE_REQUIRED_FLAGS "-flto")
endif()

check_cxx_source_compiles(
  "int main() { return 0; }"
  XSIGMA_COMPILER_SUPPORTS_LTO
)

cmake_pop_check_state()

if(XSIGMA_COMPILER_SUPPORTS_LTO)
  message(STATUS "LTO support verified for ${CMAKE_CXX_COMPILER_ID}")
else()
  message(WARNING "LTO may not be fully supported by ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  message(WARNING "Build may fail or produce suboptimal results")
endif()

# ============================================================================
# Apply LTO Flags to xsigmabuild Target
# ============================================================================
#
# NOTE: LTO flags are applied ONLY to the xsigmabuild interface target.
# This ensures that third-party dependencies are not affected by LTO configuration.
# The xsigmabuild target is created in CMakeLists.txt and is linked by all XSigma targets.

if(TARGET xsigmabuild)
  if(XSIGMA_LTO_COMPILE_FLAGS)
    target_compile_options(xsigmabuild INTERFACE ${XSIGMA_LTO_COMPILE_FLAGS})
    message(STATUS "Applied LTO compile flags to xsigmabuild target")
  endif()

  if(XSIGMA_LTO_LINK_FLAGS)
    target_link_options(xsigmabuild INTERFACE ${XSIGMA_LTO_LINK_FLAGS})
    message(STATUS "Applied LTO link flags to xsigmabuild target")
  endif()
else()
  message(WARNING "xsigmabuild target not found - LTO flags will not be applied")
  message(WARNING "Ensure this module is included AFTER xsigmabuild target is created in CMakeLists.txt")
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "Link Time Optimization (LTO) configuration complete")
message(STATUS "  - Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  - LTO Status: ENABLED")
message(STATUS "  - Target: xsigmabuild (XSigma targets only)")
message(STATUS "  - Expected build time increase: 10-30%")
message(STATUS "  - Expected runtime performance improvement: 5-15%")

