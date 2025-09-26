# XSigma GoogleTest Configuration
set(BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "Build shared libs" FORCE)

if(NOT DEFINED GOOGLETEST_SOURCE_DIR)
  set(GOOGLETEST_SOURCE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/googletest"
      CACHE STRING "googletest source directory from submodules")
endif()

if(MSVC AND BUILD_SHARED_LIBS)
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
endif()

add_subdirectory("${GOOGLETEST_SOURCE_DIR}"
                  "${CMAKE_BINARY_DIR}/ThirdParty/googletest")

if(XSIGMA_ENABLE_BENCHMARK)
  if(NOT TARGET benchmark)
    set(BENCHMARK_ENABLE_TESTING
        OFF
        CACHE BOOL "Disable benchmark testing as we don't need it.")
    set(BENCHMARK_ENABLE_INSTALL
        OFF
        CACHE BOOL
              "Disable benchmark install to avoid overwriting vendor install."
    )
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/benchmark)
  endif()
endif()

set(BUILD_SHARED_LIBS
    ${XSIGMA_BUILD_SHARED_LIBS}
    CACHE BOOL "Build shared libs" FORCE)

