if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL
                                              "AppleClang")
  set(CMAKE_COMPILER_IS_CLANGXX 1)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
  set(xsigma_coverage_compile_args)
  set(xsigma_coverage_link_args)

  if(CMAKE_COMPILER_IS_CLANGXX)
    string(APPEND xsigma_coverage_compile_args
            "-g -O0 -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND xsigma_coverage_link_args
            "-fprofile-instr-generate -fcoverage-mapping")
  elseif(UNIX)
    string(APPEND xsigma_coverage_compile_args "-g -O0  -fprofile-arcs -ftest-coverage")
    string(APPEND xsigma_coverage_link_args "-lgcov --coverage")
  endif()

  # We're setting the CXX flags and C flags beacuse they're propagated down
  # independent of build type.
  string(APPEND CMAKE_CXX_FLAGS " ${xsigma_coverage_compile_args}")
  string(APPEND CMAKE_C_FLAGS " ${xsigma_coverage_compile_args}")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " ${xsigma_sanitize_args}")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " ${xsigma_coverage_link_args}")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS " ${xsigma_coverage_link_args}")
endif()
