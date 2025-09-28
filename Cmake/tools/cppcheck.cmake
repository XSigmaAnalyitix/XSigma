if(NOT XSIGMA_ENABLE_CPPCHECK)
    return()
endif()

# Skip cppcheck for third-party libraries
get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
if(CURRENT_DIR_NAME STREQUAL "ThirdParty" OR
   CMAKE_CURRENT_SOURCE_DIR MATCHES ".*/ThirdParty/.*" OR
   CMAKE_CURRENT_SOURCE_DIR MATCHES ".*/third_party/.*" OR
   CMAKE_CURRENT_SOURCE_DIR MATCHES ".*/3rdparty/.*")
    return()
endif()

find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck
    PATHS "C:/Program Files/Cppcheck"
    PATH_SUFFIXES bin)
if(NOT CMAKE_CXX_CPPCHECK)
    message(FATAL_ERROR "Cppcheck requested but not found!

Please install cppcheck:

  - Ubuntu/Debian: sudo apt-get install cppcheck
  - CentOS/RHEL/Fedora: sudo dnf install cppcheck
  - macOS: brew install cppcheck
  - Windows: choco install cppcheck or winget install cppcheck
  - Manual: Download from https://cppcheck.sourceforge.io/

Or set XSIGMA_ENABLE_CPPCHECK=OFF to disable static analysis")
else()
    # Check if suppressions file exists
    set(SUPPRESSIONS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Scripts/cppcheck_suppressions.txt")
    set(CPPCHECK_ARGS)
    
    if(EXISTS ${SUPPRESSIONS_FILE})
        list(APPEND CPPCHECK_ARGS "--suppressions-list=${SUPPRESSIONS_FILE}")
    endif()
    
    # Platform-specific setup
    if(WIN32)
        list(APPEND CPPCHECK_ARGS "--platform=win64" "--library=windows")
    else()
        list(APPEND CPPCHECK_ARGS "--platform=unix64" "--library=posix")
    endif()
    
    list(APPEND CPPCHECK_ARGS
        "--enable=warning,style,performance,portability"
        "--force"
        "--quiet"
        "--inline-suppr"
        "--template={id}:{file}:{line},{severity},{message}"
        "--relative-paths=${CMAKE_CURRENT_SOURCE_DIR}"
        "--output-file=${CMAKE_CURRENT_BINARY_DIR}/cppcheckoutput.log"
    )

    # Add automatic fix flag if enabled
    # WARNING: --fix modifies source files directly. Use with caution!
    if(XSIGMA_ENABLE_AUTOFIX)
        message(STATUS "Cppcheck automatic fixes enabled - source files will be modified!")
        message(WARNING "XSIGMA_ENABLE_AUTOFIX is ON: cppcheck will modify source files directly. "
                        "Ensure you have committed your changes before building.")
        list(APPEND CPPCHECK_ARGS "--fix")
    endif()
    
    set(CMAKE_CXX_CPPCHECK ${CMAKE_CXX_CPPCHECK} ${CPPCHECK_ARGS})
endif()