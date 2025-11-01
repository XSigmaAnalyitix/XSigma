# Unit tests for smp_new module

set(SMP_NEW_TEST_SOURCES
    test_thread_pool.cxx
    test_parallel_for.cxx
    test_parallel_reduce.cxx
    test_task_execution.cxx
    test_backends.cxx
)

# Create test executable
add_executable(test_smp_new ${SMP_NEW_TEST_SOURCES})

# Link libraries
target_link_libraries(test_smp_new
    PRIVATE
        xsigma_smp_new
        xsigma_testing
        Threads::Threads
)

# Set C++ standard
set_target_properties(test_smp_new PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add tests
add_test(NAME smp_new_tests COMMAND test_smp_new)

# Set test properties
set_tests_properties(smp_new_tests PROPERTIES
    TIMEOUT 300
    LABELS "smp_new"
)

