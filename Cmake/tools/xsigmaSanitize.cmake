if(NOT XSIGMA_ENABLE_SANITIZER)
    return()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL
                                             "AppleClang")
  set(CMAKE_COMPILER_IS_CLANGXX 1)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
  if(XSIGMA_ENABLE_SANITIZER)
    if(UNIX AND NOT APPLE)
      # Tests using external binaries need additional help to load the ASan
      # runtime when in use.
      if(XSIGMA_SANITIZER_TYPE STREQUAL "address" OR XSIGMA_SANITIZER_TYPE
                                                     STREQUAL "undefined")
        find_library(
          XSIGMA_ASAN_LIBRARY
          NAMES libasan.so.6 libasan.so.5
          DOC "ASan library")
        mark_as_advanced(XSIGMA_ASAN_LIBRARY)

        set(_xsigma_testing_ld_preload "${XSIGMA_ASAN_LIBRARY}")
      endif()
    endif()

    set(xsigma_sanitize_args "-fsanitize=${XSIGMA_SANITIZER_TYPE}")

    if(CMAKE_COMPILER_IS_CLANGXX)
      configure_file(
        "${XSIGMA_SOURCE_DIR}/Utilities/DynamicAnalysis/sanitizer_ignore.txt.in"
        "${XSIGMA_BINARY_DIR}/sanitizer_ignore.txt" @ONLY)
      list(
        APPEND xsigma_sanitize_args
        "SHELL:-fsanitize-blacklist=${XSIGMA_BINARY_DIR}/sanitizer_ignore.txt")
    endif()

    target_compile_options(
      xsigmabuild INTERFACE "$<BUILD_INTERFACE:${xsigma_sanitize_args}>")
    target_link_options(xsigmabuild INTERFACE
                        "$<BUILD_INTERFACE:${xsigma_sanitize_args}>")
  endif()
endif()
