# Core module CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Configure header files
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xsigma_features.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_features.h"
    @ONLY
)

# Configure threading
if(WIN32)
    set(XSIGMA_USE_WIN32_THREADS ON)
    set(XSIGMA_MAX_THREADS 64)
else()
    set(XSIGMA_USE_PTHREADS ON)
    set(XSIGMA_MAX_THREADS 64)
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xsigma_version_macros.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_version_macros.h"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xsigma_threads.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_threads.h"
    @ONLY
)
set(xsigma_headers)
set(xsigma_sources)
set(xsigma_templates)

list(APPEND xsigma_headers
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_features.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_version_macros.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_threads.h"
)

file(
  GLOB_RECURSE headers
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/util/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/logging/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/experimental/profiler/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/smp/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/c17/*.h"
  ${xsigma_headers})

file(
  GLOB_RECURSE sources
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/util/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/logging/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/experimental/profiler/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/smp/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/c17/*.h"
  ${xsigma_sources})

file(
  GLOB_RECURSE templates
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/util/*.hxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/*.hxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/logging/*.hxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/*.hxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/smp/*.hxx"
  ${xsigma_templates})

# Remove GPU files if CUDA is not enabled
if(NOT XSIGMA_ENABLE_CUDA)
    file(GLOB_RECURSE GPU_SOURCE_FILES memory/gpu/*.cxx)
    list(REMOVE_ITEM sources ${GPU_SOURCE_FILES})
    file(GLOB_RECURSE GPU_HEADER_FILES memory/gpu/*.h)
    list(REMOVE_ITEM headers ${GPU_HEADER_FILES})
endif()

# Remove logging.cxx and logging.h for NATIVE backend (uses simplified implementation in logger.cxx)
if(XSIGMA_USE_NATIVE_LOGGING)
    list(REMOVE_ITEM sources "${CMAKE_CURRENT_SOURCE_DIR}/logging/logging.cxx")
    list(REMOVE_ITEM headers "${CMAKE_CURRENT_SOURCE_DIR}/logging/logging.h")
    message(STATUS "NATIVE logging backend: Excluding logging.cxx and logging.h (using simplified fmt-based implementation)")
endif()

if(XSIGMA_ENABLE_TBB)
    # If TBB is enabled, remove STDThread files
    file(GLOB_RECURSE TBB_FILES smp/STDThread/*.cxx)
    list(REMOVE_ITEM sources ${TBB_FILES})
    file(GLOB_RECURSE TBB_HEADER_FILES smp/STDThread/*.h smp/STDThread/*.hxx)
    list(REMOVE_ITEM headers ${TBB_HEADER_FILES})
    list(REMOVE_ITEM templates ${TBB_HEADER_FILES})
else()
    file(GLOB_RECURSE TBB_FILES smp/TBB/*.cxx)
    list(REMOVE_ITEM sources ${TBB_FILES})
    file(GLOB_RECURSE TBB_HEADER_FILES smp/TBB/*.h  smp/TBB/*.hxx)
    list(REMOVE_ITEM headers ${TBB_HEADER_FILES})
    list(REMOVE_ITEM templates ${TBB_HEADER_FILES})
endif()

# Create the Core library
# Respect BUILD_SHARED_LIBS setting for library type
add_library(Core ${sources} ${headers} ${templates})

# Set appropriate compile definitions based on library type
if(BUILD_SHARED_LIBS)
    target_compile_definitions(Core
        PUBLIC XSIGMA_SHARED_DEFINE
        PRIVATE XSIGMA_BUILDING_DLL
    )
    # Set output directories using XSigma's standardized approach
    if(COMMAND xsigma_set_target_output_directories)
        xsigma_set_target_output_directories(Core)
    endif()
else()
    target_compile_definitions(Core PUBLIC XSIGMA_STATIC_DEFINE)
endif()

# Create alias for consistent naming
add_library(XSigma::Core ALIAS Core)

# Apply IWYU to the Core target (if available)
if(COMMAND xsigma_apply_iwyu)
    xsigma_apply_iwyu(Core)
endif()

# Configure Visual Studio source groups to preserve directory structure
# This ensures the Solution Explorer shows the actual filesystem hierarchy
xsigma_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")

# Set target properties
set_target_properties(Core PROPERTIES
    CXX_STANDARD ${XSIGMA_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(Core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Testing>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link with build interface target
target_link_libraries(Core PUBLIC XSigma::build)

# Link with third-party dependencies
# Mandatory core dependencies (always linked)
if(TARGET XSigma::fmt)
    target_link_libraries(Core PUBLIC XSigma::fmt)
endif()

if(TARGET XSigma::cpuinfo)
    target_link_libraries(Core PRIVATE XSigma::cpuinfo)
endif()

# Optional dependencies (based on XSIGMA_ENABLE_XXX options)

if(XSIGMA_ENABLE_MAGICENUM AND TARGET XSigma::magic_enum)
    target_link_libraries(Core PUBLIC XSigma::magic_enum)
    target_compile_definitions(Core PUBLIC XSIGMA_USE_MAGICENUM)
endif()

# Logging backend dependencies (mutually exclusive based on XSIGMA_LOGGING_BACKEND)
# Note: Compile definitions are PUBLIC so they propagate to all targets that link to Core
# This ensures logging macros are available in test files and other dependent code
if(XSIGMA_USE_NATIVE_LOGGING)
    target_compile_definitions(Core PUBLIC XSIGMA_USE_NATIVE_LOGGING)
    message(STATUS "Core library using NATIVE logging backend")
elseif(XSIGMA_USE_LOGURU AND TARGET XSigma::loguru)
    target_link_libraries(Core PRIVATE XSigma::loguru)
    target_compile_definitions(Core PUBLIC
        XSIGMA_USE_LOGURU
        XSIGMA_MODULE_ENABLE_XSIGMA_loguru=1
    )
    message(STATUS "Core library using LOGURU logging backend")
elseif(XSIGMA_USE_GLOG AND TARGET XSigma::glog)
    target_link_libraries(Core PRIVATE XSigma::glog)
    target_compile_definitions(Core PUBLIC XSIGMA_USE_GLOG)
    message(STATUS "Core library using GLOG logging backend")
endif()

# Optional dependencies
if(XSIGMA_ENABLE_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(Core PRIVATE TBB::tbb TBB::tbbmalloc)
endif()

if(XSIGMA_ENABLE_MIMALLOC)
    target_link_libraries(Core PRIVATE "mimalloc-static")
    add_dependencies(Core mimalloc-static)
endif()

# CUDA support
if(XSIGMA_ENABLE_CUDA)
    target_link_libraries(Core PRIVATE ${XSIGMA_CUDA_LIBRARIES})
endif()

# Install targets
install(TARGETS Core
    EXPORT XSigmaTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(FILES ${headers}
    DESTINATION include/XSigma/Core
    COMPONENT Development
)

# Install configured headers
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_features.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_version_macros.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_threads.h"
    DESTINATION include/XSigma/Core
    COMPONENT Development
)

# Configure testing
if(XSIGMA_BUILD_TESTING STREQUAL "ON" OR XSIGMA_BUILD_TESTING STREQUAL "WANT")
    add_subdirectory(Testing)
endif()
