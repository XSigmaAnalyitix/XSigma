set(XSIGMA_REQUIRED_C_FLAGS)
set(XSIGMA_REQUIRED_CXX_FLAGS)

if(NOT XSIGMA_ENABLE_COVERAGE AND NOT XSIGMA_ENABLE_SANITIZER)
  message("--avx compiler flags: ${VECTORIZATION_COMPILER_FLAGS}")
  set(XSIGMA_REQUIRED_C_FLAGS ${VECTORIZATION_COMPILER_FLAGS})
  set(XSIGMA_REQUIRED_CXX_FLAGS ${VECTORIZATION_COMPILER_FLAGS})
endif()

if(XSIGMA_ENABLE_SANITIZER)
  set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -O1 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls")
  set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -O1 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls")
endif()

# make sure Crun is linked in with the native compiler, it is
# not used by default for shared libraries and is required for
# things like java to work.
if(CMAKE_SYSTEM MATCHES "SunOS.*")
  if(NOT CMAKE_COMPILER_IS_GNUCXX)
    find_library(XSIGMA_SUNCC_CRUN_LIBRARY Crun /opt/SUNWspro/lib)
    if(XSIGMA_SUNCC_CRUN_LIBRARY)
      link_libraries(${XSIGMA_SUNCC_CRUN_LIBRARY})
    endif()
    find_library(XSIGMA_SUNCC_CSTD_LIBRARY Cstd /opt/SUNWspro/lib)
    if(XSIGMA_SUNCC_CSTD_LIBRARY)
      link_libraries(${XSIGMA_SUNCC_CSTD_LIBRARY})
    endif()
  endif()
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # Enable exceptions because XSIGMA and third party code rely on C++ exceptions.
  # Allow C++ to catch exceptions. Emscripten disables it by default due to high overhead.
  # Generate helper functions to get stack traces for uncaught exceptions
  set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -fwasm-exceptions")
  set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -fwasm-exceptions")
  set(XSIGMA_REQUIRED_EXE_LINKER_FLAGS "${XSIGMA_REQUIRED_EXE_LINKER_FLAGS} -fwasm-exceptions -sEXCEPTION_STACK_TRACES=1")
  set(XSIGMA_REQUIRED_SHARED_LINKER_FLAGS "${XSIGMA_REQUIRED_SHARED_LINKER_FLAGS} -fwasm-exceptions -sEXCEPTION_STACK_TRACES=1")
  set(XSIGMA_REQUIRED_MODULE_LINKER_FLAGS "${XSIGMA_REQUIRED_MODULE_LINKER_FLAGS} -fwasm-exceptions -sEXCEPTION_STACK_TRACES=1")
  # Consumers linking to XSIGMA also need to add the exception flag.
  if (TARGET XSIGMAplatform)
    target_link_options(XSIGMAplatform
      INTERFACE
        "-fwasm-exceptions"
        "-sEXCEPTION_STACK_TRACES=1")
  endif ()
  if (XSIGMA_WEBASSEMBLY_THREADS)
    # Remove after https://github.com/WebAssembly/design/issues/1271 is closed
    # Set Wno flag globally because even though the flag is added in XSIGMACompilerWarningFlags.cmake,
    # wrapping tools do not link with `XSIGMAplatform`
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -pthread -Wno-pthreads-mem-growth")
    set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -pthread -Wno-pthreads-mem-growth")
    set(XSIGMA_REQUIRED_EXE_LINKER_FLAGS "${XSIGMA_REQUIRED_EXE_LINKER_FLAGS} -pthread")
    set(XSIGMA_REQUIRED_SHARED_LINKER_FLAGS "${XSIGMA_REQUIRED_SHARED_LINKER_FLAGS} -pthread")
    set(XSIGMA_REQUIRED_MODULE_LINKER_FLAGS "${XSIGMA_REQUIRED_MODULE_LINKER_FLAGS} -pthread")
    # Consumers linking to XSIGMA also need to add the pthread flag.
    if (TARGET XSIGMAplatform)
      target_compile_options(XSIGMAplatform
        INTERFACE
          "-pthread"
          "-Wno-pthreads-mem-growth")
      target_link_options(XSIGMAplatform
        INTERFACE
          "-pthread")
    endif ()
  endif ()
  if (XSIGMA_WEBASSEMBLY_64_BIT)
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -sMEMORY64=1")
    set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -sMEMORY64=1")
    set(XSIGMA_REQUIRED_EXE_LINKER_FLAGS "${XSIGMA_REQUIRED_EXE_LINKER_FLAGS} -sMEMORY64=1")
    set(XSIGMA_REQUIRED_SHARED_LINKER_FLAGS "${XSIGMA_REQUIRED_SHARED_LINKER_FLAGS} -sMEMORY64=1")
    set(XSIGMA_REQUIRED_MODULE_LINKER_FLAGS "${XSIGMA_REQUIRED_MODULE_LINKER_FLAGS} -sMEMORY64=1")
    # Consumers linking to XSIGMA also need to add the memory64 flag.
    if (TARGET XSIGMAplatform)
      target_compile_options(XSIGMAplatform
        INTERFACE
          "-sMEMORY64=1")
      target_link_options(XSIGMAplatform
        INTERFACE
          "-sMEMORY64=1")
    endif ()
  endif ()
endif ()

# A GCC compiler.
if(CMAKE_COMPILER_IS_GNUCXX)
  if(XSIGMA_USE_X)
    unset(WIN32)
  endif()
  if(WIN32)
# The platform is gcc on cygwin.
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -mwin32")
    set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -mwin32")
    link_libraries(-lgdi32)
  endif()
  if(MINGW)
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -mthreads")
    set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -mthreads")
    set(XSIGMA_REQUIRED_EXE_LINKER_FLAGS "${XSIGMA_REQUIRED_EXE_LINKER_FLAGS} -mthreads")
    set(XSIGMA_REQUIRED_SHARED_LINKER_FLAGS "${XSIGMA_REQUIRED_SHARED_LINKER_FLAGS} -mthreads")
    set(XSIGMA_REQUIRED_MODULE_LINKER_FLAGS "${XSIGMA_REQUIRED_MODULE_LINKER_FLAGS} -mthreads")
  endif()
  if(CMAKE_SYSTEM MATCHES "SunOS.*")
    # Disable warnings that occur in X11 headers.
    if(DART_ROOT AND BUILD_TESTING)
      set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -Wno-unknown-pragmas")
      set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} -Wno-unknown-pragmas")
    endif()
  endif()
else()
  if(CMAKE_ANSI_CFLAGS)
    set(XSIGMA_REQUIRED_C_FLAGS "${XSIGMA_REQUIRED_C_FLAGS} ${CMAKE_ANSI_CFLAGS}")
  endif()
  if(CMAKE_SYSTEM MATCHES "OSF1-V.*")
     set(XSIGMA_REQUIRED_CXX_FLAGS
         "${XSIGMA_REQUIRED_CXX_FLAGS} -timplicit_local -no_implicit_include")
  endif()
  if(CMAKE_SYSTEM MATCHES "AIX.*")
    # allow t-ypeid and d-ynamic_cast usage (normally off by default on xlC)
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -qrtti=all")
    # silence duplicate symbol warnings on AIX
    set(XSIGMA_REQUIRED_EXE_LINKER_FLAGS "${XSIGMA_REQUIRED_EXE_LINKER_FLAGS} -bhalt:5")
    set(XSIGMA_REQUIRED_SHARED_LINKER_FLAGS "${XSIGMA_REQUIRED_SHARED_LINKER_FLAGS} -bhalt:5")
    set(XSIGMA_REQUIRED_MODULE_LINKER_FLAGS "${XSIGMA_REQUIRED_MODULE_LINKER_FLAGS} -bhalt:5")
  endif()
  if(CMAKE_SYSTEM MATCHES "HP-UX.*")
     set(XSIGMA_REQUIRED_C_FLAGS
         "${XSIGMA_REQUIRED_C_FLAGS} +W2111 +W2236 +W4276")
     set(XSIGMA_REQUIRED_CXX_FLAGS
         "${XSIGMA_REQUIRED_CXX_FLAGS} +W2111 +W2236 +W4276")
  endif()
endif()

# figure out whether the compiler might be the Intel compiler
set(_MAY_BE_INTEL_COMPILER FALSE)
if(UNIX)
  if(CMAKE_CXX_COMPILER_ID)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      set(_MAY_BE_INTEL_COMPILER TRUE)
    endif()
  else()
    if(NOT CMAKE_COMPILER_IS_GNUCXX)
      set(_MAY_BE_INTEL_COMPILER TRUE)
    endif()
  endif()
endif()

#if so, test whether -i_dynamic is needed
if(_MAY_BE_INTEL_COMPILER)
  include(${CMAKE_CURRENT_LIST_DIR}/TestNO_ICC_IDYNAMIC_NEEDED.cmake)
  testno_icc_idynamic_needed(NO_ICC_IDYNAMIC_NEEDED ${CMAKE_CURRENT_LIST_DIR})
  if(NO_ICC_IDYNAMIC_NEEDED)
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS}")
  else()
    set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} -i_dynamic")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
  # --diag_suppress=236 is for constant value asserts used for error handling
  # This can be restricted to the implementation and doesn't need to propagate
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --diag_suppress=236")

  # --diag_suppress=381 is for redundant semi-colons used in macros
  # This needs to propagate to anything that includes XSIGMA headers
  set(XSIGMA_REQUIRED_CXX_FLAGS "${XSIGMA_REQUIRED_CXX_FLAGS} --diag_suppress=381")
endif()

if(MSVC)
# Use the highest warning level for visual c++ compiler.
  set(CMAKE_CXX_WARNING_LEVEL 4)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    STRING(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
  # Enable C++20 support: /Zc:__cplusplus
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
  # Treat warnings as errors
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
  # Disable C4244: conversion warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4715 /wd4018")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4244 /wd4267 /wd4715 /wd4018")

  # Disable deprecation warnings for standard C and STL functions in VS2015+
  # and later
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)

  # Enable /MP flag for Visual Studio
  set(CMAKE_CXX_MP_FLAG ON CACHE BOOL "Build with /MP flag enabled")
  set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
  set(CMAKE_CXX_MP_NUM_PROCESSORS ${PROCESSOR_COUNT} CACHE STRING "The maximum number of processes for the /MP flag")
  if (CMAKE_CXX_MP_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP${CMAKE_CXX_MP_NUM_PROCESSORS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP${CMAKE_CXX_MP_NUM_PROCESSORS}")
  endif ()

  # Enable /bigobj for MSVC to allow larger symbol tables
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")

  # Enable faster PDB generation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Zi")
  
  # Use parallel linking (VS 2019+)
  if(XSIGMA_ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEBUG:FULL /INCREMENTAL:NO")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DEBUG:FULL /INCREMENTAL:NO")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEBUG:FASTLINK")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DEBUG:FASTLINK")
  endif()
  # Use /utf-8 so that MSVC uses utf-8 in source files and object files
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /utf-8")

  #use /EHsc for exception handling
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHsc")
endif()

if(APPLE)
    message(STATUS "Applying macOS LLVM linker options")

    set(LLVM_LINK_FLAGS
        -L/opt/homebrew/opt/llvm/lib/c++
        -L/opt/homebrew/opt/llvm/lib
        -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++
        -Wl,-rpath,/opt/homebrew/opt/llvm/lib
    )

    # Add each flag only if not already included
    foreach(flag IN LISTS LLVM_LINK_FLAGS)
        string(FIND "${CMAKE_EXE_LINKER_FLAGS}" "${flag}" flag_found)
        if(flag_found EQUAL -1)
            add_link_options(${flag})
        endif()
    endforeach()
endif()



#-----------------------------------------------------------------------------
# Add compiler flags XSIGMA needs to work on this platform.  This must be
# done after the call to CMAKE_EXPORT_BUILD_SETTINGS, but before any
# try-compiles are done.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${XSIGMA_REQUIRED_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${XSIGMA_REQUIRED_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${XSIGMA_REQUIRED_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${XSIGMA_REQUIRED_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${XSIGMA_REQUIRED_MODULE_LINKER_FLAGS}")
