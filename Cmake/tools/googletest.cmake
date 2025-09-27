# Google Test Integration for XSigma
# This module configures the third-party Google Test included in ThirdParty/googletest

# Include guard to prevent multiple inclusions
if(XSIGMA_GOOGLETEST_INCLUDED)
    return()
endif()
set(XSIGMA_GOOGLETEST_INCLUDED TRUE)

# Only proceed if Google Test is enabled
if(NOT XSIGMA_GOOGLE_TEST AND NOT XSIGMA_ENABLE_BENCHMARK)
    return()
endif()

message(STATUS "Configuring third-party Google Test integration...")

# Verify that the third-party Google Test exists
set(GOOGLETEST_ROOT "${CMAKE_SOURCE_DIR}/ThirdParty/googletest")
if(NOT EXISTS "${GOOGLETEST_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "Third-party Google Test not found at ${GOOGLETEST_ROOT}")
endif()

# Configure Google Test options before adding subdirectory
set(BUILD_GMOCK OFF CACHE BOOL "Build Google Mock with Google Test" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable Google Test installation" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT for Google Test" FORCE)

# Disable Google Test's own testing and samples to speed up build
set(gtest_build_tests OFF CACHE BOOL "Disable Google Test's own tests" FORCE)
set(gtest_build_samples OFF CACHE BOOL "Disable Google Test samples" FORCE)

# Force static linking for Google Test and Google Mock to avoid DLL issues
set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)

# Additional Google Test/Mock static build configuration
set(gtest_disable_pthreads ON CACHE BOOL "Disable pthreads for Google Test" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "Force static libraries" FORCE)

# Add the third-party Google Test subdirectory
add_subdirectory(${GOOGLETEST_ROOT} ${CMAKE_BINARY_DIR}/ThirdParty/googletest EXCLUDE_FROM_ALL)

# Restore the original BUILD_SHARED_LIBS setting
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED})

# Force Google Test and Google Mock targets to be static libraries
if(TARGET gtest)
    set_target_properties(gtest PROPERTIES
        COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=0"
    )
endif()

if(TARGET gtest_main)
    set_target_properties(gtest_main PROPERTIES
        COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=0"
    )
endif()

if(TARGET gmock)
    set_target_properties(gmock PROPERTIES
        COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=0"
    )
endif()

if(TARGET gmock_main)
    set_target_properties(gmock_main PROPERTIES
        COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=0"
    )
endif()

# Create XSigma aliases for Google Test targets
if(TARGET gtest)
    add_library(XSigma::gtest ALIAS gtest)
    message(STATUS "Created XSigma::gtest alias for third-party Google Test")
endif()

if(TARGET gtest_main)
    add_library(XSigma::gtest_main ALIAS gtest_main)
    message(STATUS "Created XSigma::gtest_main alias for third-party Google Test")
endif()

if(TARGET gmock)
    add_library(XSigma::gmock ALIAS gmock)
    message(STATUS "Created XSigma::gmock alias for third-party Google Mock")
endif()

if(TARGET gmock_main)
    add_library(XSigma::gmock_main ALIAS gmock_main)
    message(STATUS "Created XSigma::gmock_main alias for third-party Google Mock")
endif()

message(STATUS "Third-party Google Test integration completed successfully")
