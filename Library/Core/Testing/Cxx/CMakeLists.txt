file(
  GLOB_RECURSE test_sources
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cu")

set(TestFiles)
foreach(_test_sources IN ITEMS ${test_sources})
  get_filename_component(file_name "${_test_sources}" NAME)
  if(NOT file_name MATCHES "TestLoggerDisableSignalHandler")
	list(APPEND TestFiles "${file_name}")
  endif()
endforeach()


include(GoogleTest)
add_executable(CoreCxxTests)
target_sources(CoreCxxTests PRIVATE ${TestFiles})
target_compile_definitions(CoreCxxTests PRIVATE XSIGMA_GOOGLE_TEST)
target_link_libraries(CoreCxxTests PRIVATE gtest gtest_main XSigma::Core ${XSIGMA_CUDA_LIBRARIES})
target_compile_definitions(CoreCxxTests PRIVATE -DUSE_GTEST)
target_compile_definitions(CoreCxxTests PRIVATE -D_VARIADIC_MAX=10)

# Apply XSigma output directory configuration
if(COMMAND xsigma_set_target_output_directories)
    xsigma_set_target_output_directories(CoreCxxTests)
endif()

# Add TBB support for tests
if(XSIGMA_ENABLE_TBB)
    # Disable TBB's pragma-based auto-linking on Windows to avoid conflicts
    if(WIN32)
        target_compile_definitions(CoreCxxTests PRIVATE __TBB_NO_IMPLICIT_LINKAGE)
    endif()

    # Ensure TBB include directories are available for tests
    if(TBB_INCLUDE_DIR AND EXISTS "${TBB_INCLUDE_DIR}")
        target_include_directories(CoreCxxTests PRIVATE "${TBB_INCLUDE_DIR}")
    endif()

    # Link with TBB libraries using a cross-platform approach
    # Use imported targets if available, otherwise fall back to manual linking
    if(TARGET TBB::tbb)
        target_link_libraries(CoreCxxTests PRIVATE TBB::tbb)
    elseif(TBB_LIBRARY_RELEASE)
        target_link_libraries(CoreCxxTests PRIVATE ${TBB_LIBRARY_RELEASE})
    endif()

    if(TARGET TBB::tbbmalloc)
        target_link_libraries(CoreCxxTests PRIVATE TBB::tbbmalloc)
    elseif(TBB_MALLOC_LIBRARY_RELEASE)
        target_link_libraries(CoreCxxTests PRIVATE ${TBB_MALLOC_LIBRARY_RELEASE})
    endif()
endif()
