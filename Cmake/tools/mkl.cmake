if(NOT XSIGMA_ENABLE_MKL)
    return()
endif()

# MKL for Matrix Operations - Must be set BEFORE find_package(Ceres)
find_package(MKL REQUIRED)
include_directories("${MKL_ROOT}/include")
# Set BLAS/LAPACK variables FIRST - before Ceres configuration
#set(BLA_VENDOR Intel10_64lp CACHE STRING "BLAS vendor")
#set(BLAS_FOUND TRUE CACHE BOOL "BLAS found")
#set(LAPACK_FOUND TRUE CACHE BOOL "LAPACK found") 
#set(BLAS_LIBRARIES MKL::MKL CACHE STRING "BLAS libraries")
#set(LAPACK_LIBRARIES MKL::MKL CACHE STRING "LAPACK libraries")

# Core MKL setup
list(APPEND XSIGMA_MKL_INCLUDE_DIRS "${MKL_INCLUDE}")
list(APPEND XSIGMA_MKL_INCLUDE_LIBS MKL::MKL)
add_definitions(-DMKL_ILP64)
add_definitions(-DEIGEN_USE_MKL_ALL)

# Threading
if(MKL_THREADING STREQUAL "TBB")
    xsigma_module_find_package(PACKAGE XSIGMA::tbb REQUIRED)
    list(APPEND MATH_INCLUDE_LIBS TBB::tbb)
endif()

message(STATUS "MKL configured for matrix operations")
mark_as_advanced(XSIGMA_MKL_INCLUDE_DIRS)
mark_as_advanced(XSIGMA_MKL_INCLUDE_LIBS)