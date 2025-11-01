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
#
# NAMING CONVENTION: - CMake options and variables use XSIGMA_ENABLE_XXX (e.g., XSIGMA_ENABLE_CUDA)
# - C++ preprocessor macros use XSIGMA_HAS_XXX (e.g., XSIGMA_HAS_CUDA) - This module maps
# XSIGMA_ENABLE_XXX to XSIGMA_HAS_XXX for configure_file() processing
# =============================================================================

include_guard(GLOBAL)

# ============================================================================= Feature Flag Mapping
# =============================================================================
# Map CMake XSIGMA_ENABLE_* variables to XSIGMA_HAS_* compile definitions This ensures consistent
# naming: CMake uses ENABLE, C++ code uses HAS All feature flags are defined as compile definitions
# (1 or 0) rather than using configure_file()

# MEMKIND support
if(XSIGMA_ENABLE_MEMKIND)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_MEMKIND=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_MEMKIND=0)
endif()

# MKL support
if(XSIGMA_ENABLE_MKL)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_MKL=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_MKL=0)
endif()

# TBB support
if(XSIGMA_ENABLE_TBB)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_TBB=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_TBB=0)
endif()

# Loguru support
if(XSIGMA_ENABLE_LOGURU)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_LOGURU=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_LOGURU=0)
endif()

# Mimalloc support
if(XSIGMA_ENABLE_MIMALLOC)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_MIMALLOC=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_MIMALLOC=0)
endif()

# NUMA support
if(XSIGMA_ENABLE_NUMA)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_NUMA=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_NUMA=0)
endif()

# SVML support
if(XSIGMA_ENABLE_SVML)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_SVML=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_SVML=0)
endif()

# CUDA support
if(XSIGMA_ENABLE_CUDA)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_CUDA=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_CUDA=0)
endif()

# HIP support
if(XSIGMA_ENABLE_HIP)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_HIP=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_HIP=0)
endif()

# Google Test support
if(XSIGMA_ENABLE_GTEST)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_GTEST=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_GTEST=0)
endif()

# Kineto profiling support
if(XSIGMA_ENABLE_KINETO)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_KINETO=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_KINETO=0)
endif()

# Intel ITT API support Note: CMake uses XSIGMA_ENABLE_ITTAPI, but C++ code uses XSIGMA_HAS_ITT
if(XSIGMA_ENABLE_ITTAPI)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_ITT=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_ITT=0)
endif()

# Exception pointer support (detected by compiler checks in utils.cmake)
if(XSIGMA_USE_EXCEPTION_PTR)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_USE_EXCEPTION_PTR=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_USE_EXCEPTION_PTR=0)
endif()

# Vectorization support (SSE, AVX, AVX2, AVX512 - detected by compiler checks in utils.cmake)
if(XSIGMA_SSE)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_SSE=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_SSE=0)
endif()

if(XSIGMA_AVX)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_AVX=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_AVX=0)
endif()

if(XSIGMA_AVX2)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_AVX2=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_AVX2=0)
endif()

if(XSIGMA_AVX512)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_AVX512=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_AVX512=0)
endif()

# Optional feature flags (not always set)
if(XSIGMA_SOBOL_1111)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_SOBOL_1111=1)
endif()

if(XSIGMA_LU_PIVOTING)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_LU_PIVOTING=1)
endif()

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

# PyTorch Kineto profiling library support Note: XSIGMA_HAS_KINETO is defined via configure_file()
# in xsigma_features.h
if(XSIGMA_ENABLE_KINETO)
  if(TARGET XSigma::kineto)
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

# Intel ITT API support Note: XSIGMA_HAS_ITTAPI is defined via configure_file() in xsigma_features.h
if(XSIGMA_ENABLE_ITTAPI)
  find_package(ITT)
  if(ITT_FOUND)
    list(APPEND XSIGMA_DEPENDENCY_LIBS ${ITT_LIBRARIES})
    list(APPEND XSIGMA_DEPENDENCY_INCLUDE_DIRS ${ITT_INCLUDE_DIR})
    message(STATUS "Dependency: ITT libraries added to XSIGMA_DEPENDENCY_LIBS")
  endif()
endif()

# GPU support (CUDA or HIP) Note: CUDA and HIP libraries are already added by cuda.cmake and
# hip.cmake via list(APPEND XSIGMA_DEPENDENCY_LIBS ...) calls, so they're already in the list

# Compression support
if(XSIGMA_ENABLE_COMPRESSION)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_COMPRESSION=1)
  if(XSIGMA_COMPRESSION_TYPE_SNAPPY)
    list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_COMPRESSION_TYPE_SNAPPY=1)
  else()
    list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_COMPRESSION_TYPE_SNAPPY=0)
  endif()
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_COMPRESSION=0)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_COMPRESSION_TYPE_SNAPPY=0)
endif()

# Allocation statistics support (optional feature flag)
if(XSIGMA_ENABLE_ALLOCATION_STATS)
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_ALLOCATION_STATS=1)
else()
  list(APPEND XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS XSIGMA_HAS_ALLOCATION_STATS=0)
endif()

# ============================================================================= Summary
# =============================================================================
message(STATUS "XSIGMA_DEPENDENCY_LIBS populated with: ${XSIGMA_DEPENDENCY_LIBS}")
message(STATUS "XSIGMA_DEPENDENCY_INCLUDE_DIRS populated with: ${XSIGMA_DEPENDENCY_INCLUDE_DIRS}")
message(
  STATUS
    "XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS populated with: ${XSIGMA_DEPENDENCY_COMPILE_DEFINITIONS}"
)
