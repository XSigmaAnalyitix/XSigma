# Build Speed Optimization Configuration
# Enables ccache (compiler cache) and faster linkers for improved build performance
# Supports GCC, Clang, and MSVC on Linux, macOS, and Windows
#
# NOTE: This module applies faster linker configuration ONLY to the xsigmabuild interface target,
# ensuring that third-party dependencies are not affected by linker choices.

# Option to enable/disable build speed optimizations
option(XSIGMA_ENABLE_CCACH "Enable ccache and faster linker for faster builds" ON)
mark_as_advanced(XSIGMA_ENABLE_CCACH)

if(NOT XSIGMA_ENABLE_CCACH)
  message(STATUS "Build speed optimization disabled")
  return()
endif()

message(STATUS "Configuring build speed optimizations...")

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
  message(STATUS "Found ccache: ${CCACHE_PROGRAM}")

  # Set ccache as the compiler launcher for C and CXX
  # This is applied globally because ccache needs to intercept all compilation
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")

  # Also set for CUDA if enabled
  if(XSIGMA_ENABLE_CUDA)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
  endif()

  message(STATUS "ccache enabled for C/C++ compilation")
else()
  message(STATUS "ccache not found - skipping compiler caching")
endif()

# ============================================================================
# Faster Linker Configuration
# ============================================================================
#
# NOTE: Linker flags are applied ONLY to the xsigmabuild interface target.
# This ensures that third-party dependencies use their default linker settings
# and are not affected by XSigma's linker choices.

# Determine which linker to use based on platform and compiler
set(XSIGMA_LINKER_CHOICE "default" CACHE STRING "Linker to use: default, lld, mold, gold, lld-link")
set_property(CACHE XSIGMA_LINKER_CHOICE PROPERTY STRINGS default lld mold gold lld-link)

function(xsigma_find_linker)
  set(LINKER_FOUND FALSE)
  set(LINKER_NAME "")
  set(LINKER_FLAGS)

  # Platform and compiler specific linker selection
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # Linux + Clang: prefer lld or mold
      find_program(LLD_LINKER lld)
      find_program(MOLD_LINKER mold)

      if(MOLD_LINKER)
        set(LINKER_NAME "${MOLD_LINKER}")
        set(LINKER_FOUND TRUE)
        message(STATUS "Linux/Clang: Using mold linker for faster linking")
      elseif(LLD_LINKER)
        set(LINKER_NAME "${LLD_LINKER}")
        set(LINKER_FOUND TRUE)
        message(STATUS "Linux/Clang: Using lld linker for faster linking")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # Linux + GCC: prefer mold or gold
      find_program(MOLD_LINKER mold)
      find_program(GOLD_LINKER ld.gold)

      if(MOLD_LINKER)
        set(LINKER_NAME "${MOLD_LINKER}")
        set(LINKER_FOUND TRUE)
        message(STATUS "Linux/GCC: Using mold linker for faster linking")
      elseif(GOLD_LINKER)
        set(LINKER_NAME "${GOLD_LINKER}")
        set(LINKER_FOUND TRUE)
        message(STATUS "Linux/GCC: Using gold linker for faster linking")
      endif()
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS: use system linker or lld if available
    find_program(LLD_LINKER lld)
    if(LLD_LINKER)
      set(LINKER_NAME "${LLD_LINKER}")
      set(LINKER_FOUND TRUE)
      message(STATUS "macOS: Using lld linker for faster linking")
    else()
      message(STATUS "macOS: Using system linker (lld not found)")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # Windows + Clang: prefer lld-link
      find_program(LLD_LINK_LINKER lld-link)
      if(LLD_LINK_LINKER)
        set(LINKER_NAME "${LLD_LINK_LINKER}")
        set(LINKER_FOUND TRUE)
        message(STATUS "Windows/Clang: Using lld-link linker for faster linking")
      else()
        message(STATUS "Windows/Clang: lld-link not found - using default linker")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      # Windows + MSVC: use default linker with optimizations
      message(STATUS "Windows/MSVC: Using default linker with optimizations")
    endif()
  endif()

  # Apply linker flags to xsigmabuild target if found
  if(LINKER_FOUND)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # For Clang and GCC, use -fuse-ld flag
      set(LINKER_FLAGS "-fuse-ld=${LINKER_NAME}")

      if(TARGET xsigmabuild)
        target_link_options(xsigmabuild INTERFACE ${LINKER_FLAGS})
        message(STATUS "Applied linker flag to xsigmabuild: ${LINKER_FLAGS}")
      else()
        message(WARNING "xsigmabuild target not found - linker flag will not be applied")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND LINKER_NAME MATCHES "lld-link")
      # For MSVC with lld-link, set the linker directly
      set(CMAKE_LINKER "${LINKER_NAME}" CACHE FILEPATH "Linker executable")
      message(STATUS "Linker configured: ${LINKER_NAME}")
    endif()
  else()
    message(STATUS "No faster linker found - using default linker")
  endif()
endfunction()

# Call the linker detection function
xsigma_find_linker()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "Build speed optimization configuration complete")
if(CCACHE_PROGRAM)
  message(STATUS "  - ccache: ENABLED (global compiler launcher)")
else()
  message(STATUS "  - ccache: NOT FOUND")
endif()
message(STATUS "  - Faster linker: Automatically detected and applied to xsigmabuild target")
message(STATUS "  - Third-party dependencies: Not affected by linker configuration")

