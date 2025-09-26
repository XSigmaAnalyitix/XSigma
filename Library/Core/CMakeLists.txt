# Core module CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Include the selection configuration for SMP backends
include(${CMAKE_CURRENT_SOURCE_DIR}/selection.cmake)

# Configure header files
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xsigma_features.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_features.h"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xsigma_build.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_build.h"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xsigma_smp.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_smp.h"
    @ONLY
)

# Set version macros
set(XSIGMA_MAJOR_VERSION 1)
set(XSIGMA_MINOR_VERSION 0)
set(XSIGMA_BUILD_VERSION 0)
set(XSIGMA_VERSION_FULL "${XSIGMA_MAJOR_VERSION}.${XSIGMA_MINOR_VERSION}.${XSIGMA_BUILD_VERSION}")

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

# Collect source files
set(CORE_SOURCES
    # Memory management
    memory/allocator.cxx
    memory/data_ptr.cxx
    memory/device.cxx
    memory/numa.cxx
    memory/unified_memory_stats.cxx
    
    # SMP (Symmetric Multi-Processing) sources
    smp/multi_threader.cxx
    smp/threaded_callback_queue.cxx
    smp/tools.cxx
    
    # Utility sources
    util/back_trace.cxx
    util/cpu_info.cxx
    util/exception.cxx
    util/logger.cxx
    util/logging.cxx
    util/multi_process_stream.cxx
    util/strcat.cxx
    util/string_util.cxx
    util/timer_log.cxx
    util/version.cxx
    util/xtea.cxx
    
    # Experimental profiler sources
    experimental/profiler/env_time.cxx
    experimental/profiler/env_var.cxx
    experimental/profiler/memory_tracker.cxx
    experimental/profiler/profiler.cxx
    experimental/profiler/profiler_collection.cxx
    experimental/profiler/profiler_controller.cxx
    experimental/profiler/profiler_factory.cxx
    experimental/profiler/profiler_lock.cxx
    experimental/profiler/profiler_report.cxx
    experimental/profiler/scoped_memory_debug_annotation.cxx
    experimental/profiler/statistical_analyzer.cxx
    experimental/profiler/stats_calculator.cxx
    experimental/profiler/traceme_recorder.cxx
)

# Add platform-specific sources
if(WIN32)
    list(APPEND CORE_SOURCES
        experimental/profiler/env_time_win.cxx
    )
endif()

# Add SMP backend-specific sources from selection.cmake
if(DEFINED xsigma_sources)
    # Prepend smp/ to relative paths that don't already have it
    foreach(source ${xsigma_sources})
        if(NOT source MATCHES "^smp/")
            if(source STREQUAL "tools.cxx")
                list(APPEND CORE_SOURCES "smp/${source}")
            else()
                list(APPEND CORE_SOURCES "${source}")
            endif()
        else()
            list(APPEND CORE_SOURCES "${source}")
        endif()
    endforeach()
endif()

# Collect header files
set(CORE_HEADERS
    # Common headers
    common/configure.h
    common/constants.h
    common/macros.h
    common/pointer.h
    common/wrapping_hints.h
    
    # C++17 compatibility
    c17/span.h
    
    # Memory management headers
    memory/allocator.h
    memory/data_ptr.h
    memory/device.h
    memory/numa.h
    memory/unified_memory_stats.h
    
    # SMP headers
    smp/multi_threader.h
    smp/threaded_callback_queue.h
    smp/threaded_callback_queue.hxx
    smp/threaded_task_queue.h
    smp/threaded_task_queue.hxx
    smp/tools.h
    smp/xsigma_thread_local.h
    
    # Utility headers
    util/back_trace.h
    util/container_util.h
    util/cpu_info.h
    util/exception.h
    util/flat_hash.h
    util/hash_util.h
    util/lazy.h
    util/logger.h
    util/logger_verbosity_enum.h
    util/logging.h
    util/merkel_hash.h
    util/multi_process_stream.h
    util/registry.h
    util/strcat.h
    util/string_util.h
    util/timer_log.h
    util/version.h
    util/xtea.h
)

# Add SMP backend-specific headers from selection.cmake
if(DEFINED xsigma_headers)
    # Prepend smp/ to relative paths that don't already have it
    foreach(header ${xsigma_headers})
        if(NOT header MATCHES "^smp/")
            if(header MATCHES "^(tools\\.h|xsigma_thread_local\\.h)$")
                list(APPEND CORE_HEADERS "smp/${header}")
            else()
                list(APPEND CORE_HEADERS "${header}")
            endif()
        else()
            list(APPEND CORE_HEADERS "${header}")
        endif()
    endforeach()
endif()

if(DEFINED xsigma_nowrap_headers)
    list(APPEND CORE_HEADERS ${xsigma_nowrap_headers})
endif()

# Add configured headers
list(APPEND CORE_HEADERS
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_features.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_build.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_smp.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_version_macros.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_threads.h"
)

# Create the Core library
if(XSIGMA_BUILD_SHARED_LIBS)
    add_library(Core SHARED ${CORE_SOURCES} ${CORE_HEADERS})
    set_target_properties(Core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        VERSION ${XSIGMA_VERSION}
        SOVERSION ${XSIGMA_MAJOR_VERSION}
    )
else()
    add_library(Core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
    target_compile_definitions(Core PUBLIC XSIGMA_CORE_STATIC_DEFINE)
endif()

# Create alias for consistent naming
add_library(XSigma::Core ALIAS Core)

# Set target properties
set_target_properties(Core PROPERTIES
    CXX_STANDARD ${XSIGMA_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
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
    target_compile_definitions(Core PUBLIC XSIGMA_HAS_FMT)
    message(STATUS "Core: Linked with fmt library (mandatory)")
endif()

if(TARGET XSigma::cpuinfo)
    target_link_libraries(Core PRIVATE XSigma::cpuinfo)
    target_compile_definitions(Core PRIVATE XSIGMA_HAS_CPUINFO)
    message(STATUS "Core: Linked with cpuinfo library (mandatory)")
endif()

# Optional dependencies (based on XSIGMA_ENABLE_XXX options)

if(XSIGMA_ENABLE_MAGIC_ENUM AND TARGET XSigma::magic_enum)
    target_link_libraries(Core PUBLIC XSigma::magic_enum)
    target_compile_definitions(Core PUBLIC XSIGMA_HAS_MAGIC_ENUM)
    message(STATUS "Core: Linked with magic_enum library")
endif()

if(XSIGMA_ENABLE_LOGURU AND TARGET XSigma::loguru)
    target_link_libraries(Core PRIVATE XSigma::loguru)
    target_compile_definitions(Core PRIVATE
        XSIGMA_HAS_LOGURU
        XSIGMA_MODULE_ENABLE_XSIGMA_loguru=1
    )
    message(STATUS "Core: Linked with loguru library")
endif()

# Optional dependencies
if(XSIGMA_ENABLE_TBB AND TARGET XSigma::tbb)
    target_link_libraries(Core PRIVATE XSigma::tbb)
    target_compile_definitions(Core PRIVATE XSIGMA_HAS_TBB)
    message(STATUS "Core: Linked with TBB library")
endif()

if(XSIGMA_ENABLE_MIMALLOC AND TARGET XSigma::mimalloc)
    target_link_libraries(Core PRIVATE XSigma::mimalloc)
    target_compile_definitions(Core PRIVATE XSIGMA_HAS_MIMALLOC)
    message(STATUS "Core: Linked with mimalloc library")
endif()

# Link with SMP backend libraries from selection.cmake
if(DEFINED xsigma_libraries)
    target_link_libraries(Core PRIVATE ${xsigma_libraries})
endif()

# Add compiler-specific options
if(MSVC)
    target_compile_options(Core PRIVATE /W4)
else()
    target_compile_options(Core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Apply vectorization flags if available
if(DEFINED VECTORIZATION_COMPILER_FLAGS AND VECTORIZATION)
    # Convert string to list for proper flag handling
    string(REPLACE " " ";" VECTORIZATION_COMPILER_FLAGS_LIST "${VECTORIZATION_COMPILER_FLAGS}")
    target_compile_options(Core PRIVATE ${VECTORIZATION_COMPILER_FLAGS_LIST})
    message(STATUS "Core: Applied vectorization flags: ${VECTORIZATION_COMPILER_FLAGS}")
endif()

# Configure testing
if(XSIGMA_BUILD_TESTING STREQUAL "ON" OR XSIGMA_BUILD_TESTING STREQUAL "WANT")
    add_subdirectory(Testing)
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
install(FILES ${CORE_HEADERS}
    DESTINATION include/XSigma/Core
    COMPONENT Development
)

# Install configured headers
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_features.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_build.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_smp.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_version_macros.h"
    "${CMAKE_CURRENT_BINARY_DIR}/xsigma_threads.h"
    "${CMAKE_CURRENT_BINARY_DIR}/CoreModule.h"
    DESTINATION include/XSigma/Core
    COMPONENT Development
)
