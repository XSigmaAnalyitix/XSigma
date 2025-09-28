if(NOT XSIGMA_ENABLE_CLANGTIDY)
    return()
endif()

# XSigma ClangTidy Configuration
find_program(
  CLANG_TIDY_PATH
  NAMES clang-tidy
  DOC "Path to clang-tidy.")

if(NOT CLANG_TIDY_PATH)
  message(FATAL_ERROR "Could not find clang-tidy.")
endif()
set(CLANG_TIDY_FOUND
    ON
    CACHE BOOL "Found clang-tidy.")
mark_as_advanced(CLANG_TIDY_FOUND)

function(xsigma_module_enable_clang_tidy target_name)
  set_target_properties(${target_name} PROPERTIES
    C_CLANG_TIDY "${CLANG_TIDY_PATH};-warnings-as-errors=*"
    CXX_CLANG_TIDY "${CLANG_TIDY_PATH};-warnings-as-errors=*"
  )
endfunction()

function(disable_clang_tidy_for_target target_name)
  set_target_properties(${target_name} PROPERTIES
    C_CLANG_TIDY ""
    CXX_CLANG_TIDY ""
  )
endfunction()