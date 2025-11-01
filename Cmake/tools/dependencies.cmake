# ============================================================================= XSigma Centralized
# Dependency Management Module
# =============================================================================
# This module centralizes all third-party and system dependency management for XSigma. It populates
# three cache variables based on enabled feature flags:
#
# 1. XSIGMA_DEPENDENCY_LIBS: Libraries to link against
# 2. XSIGMA_DEPENDENCY_INCLUDE_DIRS: Include directories for compilation
# 3. XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS: Compile definitions for feature availability
#
# This ensures consistent dependency management across all targets (Core library, tests, benchmarks,
# etc.). The module is included after all feature modules (CUDA, HIP, TBB, etc.) are loaded,
# allowing it to aggregate dependencies from all enabled features.
# =============================================================================

include_guard(GLOBAL)

# Initialize the dependency lists if not already done
if(NOT DEFINED XSIGMA_DEPENDENCY_LIBS)
  set(XSIGMA_DEPENDENCY_LIBS "")
endif()

if(NOT DEFINED XSIGMA_DEPENDENCY_INCLUDE_DIRS)
  set(XSIGMA_DEPENDENCY_INCLUDE_DIRS "")
endif()

if(NOT DEFINED XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS)
  set(XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS "")
endif()

# ============================================================================= Mandatory Core
# Dependencies
# =============================================================================
# These libraries are always linked regardless of feature flags

if(TARGET XSigma::fmt)
  list(APPEND XSIGMA_DEPENDENCY_LIBS XSigma::fmt)
  list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/fmt/include")
  message(STATUS "Dependency: XSigma::fmt added to XSIGMA_DEPENDENCY_LIBS")
endif()

if(TARGET XSigma::cpuinfo)
  list(APPEND XSIGMA_DEPENDENCY_LIBS XSigma::cpuinfo)
  list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS
       "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/cpuinfo/include"
  )
  message(STATUS "Dependency: XSigma::cpuinfo added to XSIGMA_DEPENDENCY_LIBS")
endif()

# ============================================================================= Optional Feature
# Dependencies
# =============================================================================
# These libraries are conditionally linked based on feature flags

# Magic Enum support
if(XSIGMA_ENABLE_MAGICENUM AND TARGET XSigma::magic_enum)
  list(APPEND XSIGMA_DEPENDENCY_LIBS XSigma::magic_enum)
  list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS
       "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/magic_enum/include"
  )
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_USE_MAGICENUM)
  message(STATUS "Dependency: XSigma::magic_enum added to XSIGMA_DEPENDENCY_LIBS")
endif()

# Logging backend dependencies (mutually exclusive)
if(XSIGMA_USE_LOGURU AND TARGET XSigma::loguru)
  list(APPEND XSIGMA_DEPENDENCY_LIBS XSigma::loguru)
  list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/loguru")
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_USE_LOGURU)
  message(STATUS "Dependency: XSigma::loguru added to XSIGMA_DEPENDENCY_LIBS")
elseif(XSIGMA_USE_GLOG AND TARGET XSigma::glog)
  list(APPEND XSIGMA_DEPENDENCY_LIBS XSigma::glog)
  list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/glog/src")
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_USE_GLOG)
  message(STATUS "Dependency: XSigma::glog added to XSIGMA_DEPENDENCY_LIBS")
elseif(XSIGMA_USE_NATIVE_LOGGING)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_USE_NATIVE_LOGGING)
  message(STATUS "Dependency: NATIVE logging backend selected")
endif()

# TBB (Threading Building Blocks) support
if(XSIGMA_ENABLE_TBB)
  if(TARGET TBB::tbb)
    list(APPEND XSIGMA_DEPENDENCY_LIBS TBB::tbb)
    message(STATUS "Dependency: TBB::tbb added to XSIGMA_DEPENDENCY_LIBS")
  endif()

  if(TARGET TBB::tbbmalloc)
    list(APPEND XSIGMA_DEPENDENCY_LIBS TBB::tbbmalloc)
    message(STATUS "Dependency: TBB::tbbmalloc added to XSIGMA_DEPENDENCY_LIBS")
  endif()
endif()

# Mimalloc support
if(XSIGMA_ENABLE_MIMALLOC AND TARGET XSigma::mimalloc)
  list(APPEND XSIGMA_DEPENDENCY_LIBS XSigma::mimalloc)
  list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS
       "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/mimalloc/include"
  )
  message(STATUS "Dependency: XSigma::mimalloc added to XSIGMA_DEPENDENCY_LIBS")
endif()

# PyTorch Kineto profiling library support
if(XSIGMA_ENABLE_KINETO)
  if(TARGET XSigma::kineto)
    list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_KINETO)
    list(APPEND XSIGMA_DEPENDENCY_LIBS XSigma::kineto)
    list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS
         "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/kineto/libkineto/include"
    )
    message(STATUS "Dependency: XSigma::kineto added to XSIGMA_DEPENDENCY_LIBS")
  else()
    message(
      STATUS "Kineto enabled but library not found - Kineto code will compile but may not link"
    )
  endif()
endif()

# Intel ITT API support
if(XSIGMA_ENABLE_ITTAPI)
  find_package(ITT)
  if(ITT_FOUND)
    list(APPEND XSIGMA_DEPENDENCY_LIBS ${ITT_LIBRARIES})
    list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS ${ITT_INCLUDE_DIR})
    list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_ITT)
    message(STATUS "Dependency: ITT libraries added to XSIGMA_DEPENDENCY_LIBS")
  endif()
endif()

# GPU support (CUDA or HIP) Note: CUDA and HIP libraries are already added by cuda.cmake and
# hip.cmake via list(APPEND XSIGMA_DEPENDENCY_LIBS ...) calls, so they're already in the list

# Compression support
if(XSIGMA_ENABLE_COMPRESSION)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_ENABLE_COMPRESSION)
  if(XSIGMA_COMPRESSION_TYPE_SNAPPY)
    list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_COMPRESSION_TYPE_SNAPPY)
  endif()
endif()

# ============================================================================= Summary
# =============================================================================
message(STATUS "XSIGMA_DEPENDENCY_LIBS populated with: ${XSIGMA_DEPENDENCY_LIBS}")
message(STATUS "XSIGMA_DEPENDENCY_INCLUDE_DIRS populated with: ${XSIGMA_DEPENDENCY_INCLUDE_DIRS}")
message(
  STATUS
    "XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS populated with: ${XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS}"
)
