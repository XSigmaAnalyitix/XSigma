# Benchmarks for smp_new module

set(SMP_NEW_BENCHMARK_SOURCES
    benchmark_parallel_for.cxx
    benchmark_parallel_reduce.cxx
)

# Create benchmark executable
add_executable(benchmark_smp_new ${SMP_NEW_BENCHMARK_SOURCES})

# Link libraries
target_link_libraries(benchmark_smp_new
    PRIVATE
        xsigma_smp_new
        benchmark::benchmark
        Threads::Threads
)

# Set C++ standard
set_target_properties(benchmark_smp_new PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add to benchmark target
add_custom_target(benchmark_smp_new_run
    COMMAND benchmark_smp_new
    DEPENDS benchmark_smp_new
)

