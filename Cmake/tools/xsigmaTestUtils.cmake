# XSigma Testing Utilities
# Function to create a test executable using standard CTest
function(xsigma_add_test_cxx test_name test_group)
    cmake_parse_arguments(XSIGMA_TEST "NO_DATA;NO_VALID;NO_OUTPUT" "" "" ${ARGN})
    
    # Create test executable
    add_executable(${test_name} ${XSIGMA_TEST_UNPARSED_ARGUMENTS})
    
    # Link with Core library
    target_link_libraries(${test_name} PRIVATE XSigma::Core)
    
    # Set target properties
    set_target_properties(${test_name} PROPERTIES
        CXX_STANDARD ${XSIGMA_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "Tests"
    )
    
    # Add include directories
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

# Function to register tests with CTest
function(xsigma_test_cxx_executable test_name test_group)
    # Add individual tests for each test file
    get_target_property(test_sources ${test_name} SOURCES)
    
    foreach(test_source ${test_sources})
        get_filename_component(test_file_name ${test_source} NAME_WE)
        if(test_file_name MATCHES "^Test")
            # Create individual test
            add_test(NAME ${test_file_name} COMMAND ${test_name})
            
            # Set test properties
            set_tests_properties(${test_file_name} PROPERTIES
                LABELS "${test_group}"
                TIMEOUT 300
            )
            
            # Add test arguments if defined
            if(DEFINED ${test_file_name}_ARGS)
                set_tests_properties(${test_file_name} PROPERTIES
                    COMMAND "${test_name} ${${test_file_name}_ARGS}"
                )
            endif()
        endif()
    endforeach()
endfunction()

# Function to create Google Test executable
function(xsigma_module_gtest_executable test_name)
    cmake_parse_arguments(XSIGMA_GTEST "" "" "" ${ARGN})
    
    # Create test executable
    add_executable(${test_name} ${XSIGMA_GTEST_UNPARSED_ARGUMENTS})
    
    # Link with Google Test and Core library
    target_link_libraries(${test_name} PRIVATE
        XSigma::Core
    )

    # Link with Google Test using XSigma aliases if available
    if(TARGET XSigma::gtest)
        target_link_libraries(${test_name} PRIVATE XSigma::gtest)
    elseif(TARGET gtest)
        target_link_libraries(${test_name} PRIVATE gtest)
    endif()

    if(TARGET XSigma::gtest_main)
        target_link_libraries(${test_name} PRIVATE XSigma::gtest_main)
    elseif(TARGET gtest_main)
        target_link_libraries(${test_name} PRIVATE gtest_main)
    endif()
    
    # Set target properties
    set_target_properties(${test_name} PROPERTIES
        CXX_STANDARD ${XSIGMA_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "Tests"
    )
    
    # Add include directories
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    # Discover and register Google Test cases
    if(COMMAND gtest_discover_tests)
        gtest_discover_tests(${test_name})
    else()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endfunction()

# Function to add compiler flags for vectorization
function(xsigma_add_vectorization_flags target)
    if(DEFINED VECTORIZATION_COMPILER_FLAGS AND VECTORIZATION)
        target_compile_options(${target} PRIVATE ${VECTORIZATION_COMPILER_FLAGS})
    endif()
endfunction()

# Function to configure test data directories
function(xsigma_configure_test_data test_name)
    # Set up test data directory if it exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Data")
        target_compile_definitions(${test_name} PRIVATE
            XSIGMA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Data"
        )
    endif()
endfunction()

# Function to add common test dependencies
function(xsigma_add_test_dependencies test_name)
    # Add common dependencies that tests might need
    if(TARGET fmt)
        target_link_libraries(${test_name} PRIVATE fmt)
    endif()
    
    # Add threading library if needed
    find_package(Threads QUIET)
    if(Threads_FOUND)
        target_link_libraries(${test_name} PRIVATE Threads::Threads)
    endif()
endfunction()

# Macro to set up test environment
macro(xsigma_setup_test_environment)
    # Set up common test environment variables
    if(XSIGMA_ENABLE_VALGRIND AND CMAKE_MEMORYCHECK_COMMAND)
        set(CTEST_MEMORYCHECK_COMMAND ${CMAKE_MEMORYCHECK_COMMAND})
    endif()
    
    # Set up coverage if enabled
    if(XSIGMA_ENABLE_COVERAGE)
        if(CMAKE_COMPILER_IS_GNUCXX)
            set(CTEST_COVERAGE_COMMAND gcov)
        elseif(CMAKE_COMPILER_IS_CLANGXX)
            set(CTEST_COVERAGE_COMMAND llvm-cov)
        endif()
    endif()
endmacro()
