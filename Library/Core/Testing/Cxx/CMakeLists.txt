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

if(XSIGMA_GOOGLE_TEST AND TARGET XSigma::gtest)
    include(GoogleTest)
    xsigma_module_gtest_executable(CoreCxxTests ${TestFiles})

    # Link with Google Test only if enabled and available
    target_link_libraries(CoreCxxTests PRIVATE XSigma::gtest)
    if(TARGET XSigma::gtest_main)
        target_link_libraries(CoreCxxTests PRIVATE XSigma::gtest_main)
    endif()

    # Add compile definition to indicate Google Test is available
    target_compile_definitions(CoreCxxTests PRIVATE XSIGMA_HAS_GTEST)
    message(STATUS "CoreCxxTests: Using Google Test framework")
else()
    set(TestSystemInformation_ARGS ${CMAKE_BINARY_DIR})
    set(TestSMP_ARGS
        --Sequential=$<BOOL:${XSIGMA_SMP_ENABLE_SEQUENTIAL}>
        --STDThread=$<BOOL:${XSIGMA_SMP_ENABLE_STDTHREAD}>
        --TBB=$<OR:$<BOOL:${XSIGMA_SMP_ENABLE_TBB}>,$<STREQUAL:"${XSIGMA_SMP_IMPLEMENTATION_TYPE}","TBB">>
        --OpenMP=$<OR:$<BOOL:${XSIGMA_SMP_ENABLE_OPENMP}>,$<STREQUAL:"${XSIGMA_SMP_IMPLEMENTATION_TYPE}","OpenMP">>)
  
    xsigma_add_test_cxx(CoreCxxTests tests 
    NO_DATA NO_VALID NO_OUTPUT
    ${TestFiles})
    xsigma_test_cxx_executable(CoreCxxTests tests)

	# Special test to check whether disabling loguru signal handling, i.e.,
	# stack trace printing when it catches a signal, works. We add an
	# executable where the stack trace is disabled. The actual test is a
	# CMake script that runs the test executable and analyzes the output to
	# determine whether a stack trace was printed or not.
	add_executable(TestLoggerDisableSignalHandlerCxx
	  TestLoggerDisableSignalHandler.cxx)
	target_link_libraries(TestLoggerDisableSignalHandlerCxx
	  PRIVATE XSigma::Core)
	add_test(NAME TestLoggerDisableSignalHandler
	  COMMAND ${CMAKE_COMMAND}
	  -DEXECUTABLE_PATH:FILEPATH=$<TARGET_FILE:TestLoggerDisableSignalHandlerCxx>
	  -P ${CMAKE_CURRENT_SOURCE_DIR}/TestLoggerDisableSignalHandler.cmake)
	set_target_properties(TestLoggerDisableSignalHandlerCxx PROPERTIES FOLDER Tests)
endif()

# Add common test dependencies
xsigma_add_test_dependencies(CoreCxxTests)

# Add FMT dependency if available
if(TARGET fmt)
    target_include_directories(CoreCxxTests BEFORE
                               PRIVATE "${FMT_SOURCE_DIR}/include")
    target_link_libraries(CoreCxxTests PRIVATE fmt)
endif()