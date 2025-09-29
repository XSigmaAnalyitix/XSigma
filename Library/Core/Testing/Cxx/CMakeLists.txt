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
if(COMMAND xsigma_apply_iwyu)
    xsigma_apply_iwyu(CoreCxxTests)
endif()
target_compile_definitions(CoreCxxTests PRIVATE XSIGMA_GOOGLE_TEST)
target_link_libraries(CoreCxxTests PRIVATE gtest gtest_main XSigma::Core ${XSIGMA_CUDA_LIBRARIES})
target_compile_definitions(CoreCxxTests PRIVATE -DUSE_GTEST)
target_compile_definitions(CoreCxxTests PRIVATE -D_VARIADIC_MAX=10)
set_target_properties(CoreCxxTests PROPERTIES FOLDER "Tests")

# Skip sanitizer flags for CoreCxxTests when using MSVC due to third-party library conflicts
# Third-party libraries (gtest, fmt) are not built with sanitizer support
if(DEFINED XSIGMA_SANITIZER_COMPILE_FLAGS AND XSIGMA_SANITIZER_COMPILE_FLAGS)
    target_compile_options(CoreCxxTests PRIVATE ${XSIGMA_SANITIZER_COMPILE_FLAGS})
endif()

if(DEFINED XSIGMA_SANITIZER_LINK_FLAGS AND XSIGMA_SANITIZER_LINK_FLAGS)
    target_link_options(CoreCxxTests PRIVATE ${XSIGMA_SANITIZER_LINK_FLAGS})
endif()

# Apply XSigma output directory configuration
if(COMMAND xsigma_set_target_output_directories)
    xsigma_set_target_output_directories(CoreCxxTests)
endif()


# Add TBB support for tests
if(XSIGMA_ENABLE_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(CoreCxxTests PRIVATE TBB::tbb TBB::tbbmalloc)
endif()