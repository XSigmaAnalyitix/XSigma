# XSigma SMP_NEW Module - PyTorch-Compatible Threading Backend
# This module provides a new threading backend that replicates PyTorch's CPU threading logic

set(SMP_NEW_SOURCES
    # Core thread pool implementation
    core/thread_pool.cxx

    # Parallel APIs
    parallel/parallel_api.cxx

    # Native backend
    native/parallel_native.cxx

    # OpenMP backend
    openmp/parallel_openmp.cxx
)

set(SMP_NEW_HEADERS
    # Core thread pool
    core/thread_pool.h

    # Parallel APIs
    parallel/parallel_api.h
    parallel/parallel_api.hxx

    # Native backend
    native/parallel_native.h

    # OpenMP backend
    openmp/parallel_openmp.h
)

# Create the smp_new library
add_library(xsigma_smp_new STATIC ${SMP_NEW_SOURCES})

# Set library properties
set_target_properties(xsigma_smp_new PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(xsigma_smp_new
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(xsigma_smp_new
    PUBLIC
        xsigma_common
        xsigma_memory
        xsigma_util
    PRIVATE
        Threads::Threads
)

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(xsigma_smp_new PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(xsigma_smp_new PRIVATE _OPENMP)
    message(STATUS "OpenMP support enabled for smp_new")
else()
    message(STATUS "OpenMP not found - smp_new will use native backend only")
endif()

# Add MKL support if available
find_package(MKL)
if(MKL_FOUND)
    target_link_libraries(xsigma_smp_new PRIVATE MKL::MKL)
    target_compile_definitions(xsigma_smp_new PRIVATE AT_MKL_ENABLED)
    message(STATUS "MKL support enabled for smp_new")
endif()

# Add NUMA support if available
if(XSIGMA_ENABLE_NUMA)
    target_compile_definitions(xsigma_smp_new PRIVATE XSIGMA_NUMA_ENABLED)
endif()

# Export headers
install(FILES ${SMP_NEW_HEADERS}
    DESTINATION include/xsigma/smp_new
)

# Install library
install(TARGETS xsigma_smp_new
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Add tests if enabled
if(XSIGMA_BUILD_TESTS)
    add_subdirectory(test)
endif()

# Add benchmarks if enabled
if(XSIGMA_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

