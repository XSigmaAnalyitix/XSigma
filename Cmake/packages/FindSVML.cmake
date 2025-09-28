set(SOURCE_DIR ${xsigma_cmake_dir}/svml)

set(SVML_INCLUDE_DIRS)

set(SVML_LIBRARY_SVML)
set(SVML_LIBRARY_IRC)
set(SVML_LIBRARY_INTLC)

if(WIN32)
  list(APPEND SVML_INCLUDE_DIRS ${SOURCE_DIR}/windows/lib)
  # list(APPEND SVML_INCLUDE_DIRS ${SOURCE_DIR}/windows/svml)

  find_library(
    SVML_LIBRARY_SVML
    NAMES svml_dispmd
    PATHS ${SVML_INCLUDE_DIRS})
  set(SVML_LIBRARIES ${SVML_LIBRARY_SVML})

endif()

if(UNIX)
  list(APPEND SVML_INCLUDE_DIRS ${SOURCE_DIR}/unix/lib)

  find_library(
    SVML_LIBRARY_SVML
    NAMES svml
    PATHS ${SVML_INCLUDE_DIRS})

  find_library(
    SVML_LIBRARY_IRC
    NAMES irc
    PATHS ${SVML_INCLUDE_DIRS})

  find_library(
    SVML_LIBRARY_INTLC
    NAMES libintlc.so.5
    PATHS ${SVML_INCLUDE_DIRS})

  set(SVML_LIBRARIES
      "${SVML_LIBRARY_SVML} ${SVML_LIBRARY_IRC} ${SVML_LIBRARY_INTLC}")

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SVML DEFAULT_MSG SVML_LIBRARIES
                                  SVML_INCLUDE_DIRS)

if(NOT TARGET SVML::SVML)
  add_library(SVML::SVML SHARED IMPORTED)
  set_target_properties(
    SVML::SVML
    PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C"
               IMPORTED_LOCATION "${SVML_LIBRARY_SVML}"
               IMPORTED_IMPLIB "${SVML_LIBRARY_SVML}"
               INTERFACE_INCLUDE_DIRECTORIES "${SVML_INCLUDE_DIRS}")
endif()

if(UNIX)
  if(NOT TARGET SVML::INTLC)
    add_library(SVML::INTLC UNKNOWN IMPORTED)
    set_target_properties(
      SVML::INTLC
      PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                 IMPORTED_LOCATION "${SVML_LIBRARY_INTLC}"
                 IMPORTED_IMPLIB "${SVML_LIBRARY_INTLC}"
                 INTERFACE_INCLUDE_DIRECTORIES "${SVML_INCLUDE_DIRS}")
  endif()
endif()

mark_as_advanced(SVML_LIBRARIES SVML_INCLUDE_DIRS)
# SVML_LIBRARY_SVML SVML_LIBRARY_IRC SVML_LIBRARY_INTLC)
