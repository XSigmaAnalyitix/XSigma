# SMP Backend Selection based on XSIGMA_ENABLE_TBB flag
# If XSIGMA_ENABLE_TBB is ON, use TBB backend, otherwise use STDThread backend

if(XSIGMA_ENABLE_TBB)
    set(XSIGMA_SMP_IMPLEMENTATION_TYPE "TBB")
    set(XSIGMA_SMP_ENABLE_TBB ON)
    set(XSIGMA_SMP_ENABLE_STDTHREAD OFF)
    message(STATUS "SMP Backend: Using TBB (Intel Threading Building Blocks)")
else()
    set(XSIGMA_SMP_IMPLEMENTATION_TYPE "STDThread")
    set(XSIGMA_SMP_ENABLE_TBB OFF)
    set(XSIGMA_SMP_ENABLE_STDTHREAD ON)
    message(STATUS "SMP Backend: Using STDThread (Standard C++ Threading)")
endif()

# Cache the implementation type for reference
set(XSIGMA_SMP_IMPLEMENTATION_TYPE
    "${XSIGMA_SMP_IMPLEMENTATION_TYPE}"
    CACHE
      STRING
      "Multi-threaded parallelism implementation (automatically set based on XSIGMA_ENABLE_TBB)"
      FORCE
)
set_property(CACHE XSIGMA_SMP_IMPLEMENTATION_TYPE PROPERTY STRINGS TBB STDThread)

set(xsigma_defines)
set(xsigma_use_default_atomics ON)

set(xsigma_backends)

if(XSIGMA_SMP_ENABLE_TBB)
  xsigma_module_find_package(PACKAGE TBB)
  list(APPEND xsigma_libraries TBB::tbb)
  list(APPEND xsigma_libraries "${TBB_LIBRARIES}")
  set(xsigma_enable_tbb 1)
  list(APPEND xsigma_backends "TBB")

  set(xsigma_use_default_atomics OFF)
  set(xsigma_implementation_dir smp/TBB)
  list(APPEND xsigma_sources
       "${xsigma_implementation_dir}/tools_impl.cxx")
  list(APPEND xsigma_nowrap_headers
       "${xsigma_implementation_dir}/thread_local_impl.h")
  list(APPEND xsigma_templates
       "${xsigma_implementation_dir}/tools_impl.hxx")
endif()

if(XSIGMA_SMP_ENABLE_STDTHREAD)
  set(xsigma_enable_stdthread 1)
  set(xsigma_implementation_dir smp/STDThread)
  list(APPEND xsigma_backends "STDThread")

  list(
    APPEND
    xsigma_sources
    "${xsigma_implementation_dir}/tools_impl.cxx"
    "${xsigma_implementation_dir}/thread_local_backend.cxx"
    "${xsigma_implementation_dir}/thread_pool.cxx")
  list(
    APPEND
    xsigma_nowrap_headers
    "${xsigma_implementation_dir}/thread_local_impl.h"
    "${xsigma_implementation_dir}/thread_local_backend.h"
    "${xsigma_implementation_dir}/thread_pool.h")
  list(APPEND xsigma_templates
       "${xsigma_implementation_dir}/tools_impl.hxx")
endif()

set_property(GLOBAL PROPERTY _xsigma_backends "${xsigma_backends}")

if(xsigma_use_default_atomics)
  include(CheckSymbolExists)

  # include("${CMAKE_CURRENT_SOURCE_DIR}/xsigmaTestBuiltins.cmake")

  set(xsigma_atomics_default_impl_dir
      "${CMAKE_CURRENT_SOURCE_DIR}/smp/STDThread")
endif()

set(xsigma_common_dir smp/Common)
list(APPEND xsigma_sources "${xsigma_common_dir}/tools_api.cxx")
list(
  APPEND
  xsigma_headers
  "${xsigma_common_dir}/thread_local_api.h"
  "${xsigma_common_dir}/thread_local_impl_abstract.h"
  "${xsigma_common_dir}/tools_api.h"
  "${xsigma_common_dir}/tools_impl.h"
  "${xsigma_common_dir}/tools_internal.h")

list(APPEND xsigma_sources tools.cxx)
list(APPEND xsigma_headers tools.h xsigma_thread_local.h)
