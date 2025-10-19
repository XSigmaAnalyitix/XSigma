# =============================================================================
# XSigma Build Speed Optimization Configuration Module
# =============================================================================
# Enables ccache (compiler cache) and faster linkers for improved build performance.
# Supports GCC, Clang, and MSVC on Linux, macOS, and Windows.
#
# NOTE: This module applies faster linker configuration ONLY to the xsigmabuild interface target,
# ensuring that third-party dependencies are not affected by linker choices.
# =============================================================================

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# Build Speed Optimization Flag
# Controls whether ccache and faster linker optimizations are enabled.
# When enabled, uses ccache for compiler caching and selects faster linkers when available.
option(XSIGMA_ENABLE_CCACHE "Enable ccache and faster linker for faster builds" ON)
mark_as_advanced(XSIGMA_ENABLE_CCACHE)

if(NOT XSIGMA_ENABLE_CCACHE)
  message("Build speed optimization disabled")
  return()
endif()

message("Configuring build speed optimizations...")

# Note: When LTO is enabled, faster linkers (especially gold) may run out of memory
# during the linking phase. In such cases, it's better to use the default linker.
# This is a known limitation of LTO with certain linkers.

# ============================================================================
# CCACHE Configuration
# ============================================================================
#
# NOTE: ccache is configured globally as a compiler launcher because it needs to
# intercept all compilation commands, including those for third-party dependencies.
# This is safe because ccache only caches compilation results and doesn't affect
# the actual compilation flags or behavior.

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message("Found ccache: ${CCACHE_PROGRAM}")

  # Set ccache as the compiler launcher for C and CXX
  # This is applied globally because ccache needs to intercept all compilation
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")

  # Also set for CUDA if enabled
  if(XSIGMA_ENABLE_CUDA)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
  endif()

  message("ccache enabled for C/C++ compilation")
else()
  message("ccache not found - skipping compiler caching")
endif()
