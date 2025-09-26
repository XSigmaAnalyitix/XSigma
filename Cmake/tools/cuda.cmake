if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message(STATUS "CUDA: Using Clang as host compiler ${CMAKE_CXX_COMPILER}")
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

find_package(CUDAToolkit REQUIRED)

# Enable CUDA language support
enable_language(CUDA)

# Version checks using consistent CUDAToolkit variables
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "XSigma requires CUDA 12.0 or above. Found: ${CUDAToolkit_VERSION}")
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.2")
    message(FATAL_ERROR "XSIGMA CUDA support requires compiler version 9.2+")
endif()

# Check for version conflicts
if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL CUDAToolkit_VERSION)
    message(FATAL_ERROR "Found two conflicting CUDA versions:\n"
                        "Compiler V${CMAKE_CUDA_COMPILER_VERSION} and\n"
                        "Toolkit V${CUDAToolkit_VERSION}")
endif()

message(STATUS "XSigma: CUDA detected: ${CUDAToolkit_VERSION}")
message(STATUS "XSigma: CUDA nvcc is: ${CMAKE_CUDA_COMPILER}")
message(STATUS "XSigma: CUDA toolkit directory: ${CUDAToolkit_ROOT}")

# Set C++ standard based on CUDA version
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
    message(STATUS "CUDA supports C++17 standard")
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 14)
endif()

set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Use modern CMake CUDA architecture handling
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# GPU Architecture options
set(XSIGMA_CUDA_ARCH_OPTIONS
    "native"
    CACHE STRING "Which GPU Architecture(s) to compile for")
set_property(
    CACHE XSIGMA_CUDA_ARCH_OPTIONS
    PROPERTY STRINGS
             native
             fermi
             kepler
             maxwell
             pascal
             volta
             turing
             ampere
             ada
             hopper
             all
             none)

# Set architectures based on user selection
if(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "native")
    # Let CMake handle native detection
    set(CMAKE_CUDA_ARCHITECTURES "native")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "fermi")
    set(CMAKE_CUDA_ARCHITECTURES "20")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "kepler")
    set(CMAKE_CUDA_ARCHITECTURES "30;35")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "maxwell")
    set(CMAKE_CUDA_ARCHITECTURES "50")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "pascal")
    set(CMAKE_CUDA_ARCHITECTURES "60;61")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "volta")
    set(CMAKE_CUDA_ARCHITECTURES "70")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "turing")
    set(CMAKE_CUDA_ARCHITECTURES "75")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "ampere")
    set(CMAKE_CUDA_ARCHITECTURES "80;86")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "ada")
    set(CMAKE_CUDA_ARCHITECTURES "89;90")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "hopper")
    set(CMAKE_CUDA_ARCHITECTURES "90")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "all")
    set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86;89;90")
elseif(XSIGMA_CUDA_ARCH_OPTIONS STREQUAL "none")
    # Don't set any architectures, let parent project handle it
endif()

# Set up CUDA libraries using modern imported targets
set(XSIGMA_CUDA_LIBRARIES
    CUDA::cudart
    CUDA::cusparse
    CUDA::curand
    CUDA::cublas
)

# Add CUDA libraries to the dependency list
list(APPEND XSIGMA_DEPENDENCY_LIBS ${XSIGMA_CUDA_LIBRARIES})

# Add include directories (using modern CUDAToolkit variables)
include_directories(SYSTEM "${CUDAToolkit_INCLUDE_DIRS}")
include_directories(SYSTEM "${CUDAToolkit_INCLUDE_DIRS}/thrust/system/cuda/detail")

# Add common CUDA flags
string(APPEND CMAKE_CUDA_FLAGS " -Xnvlink=--suppress-stack-size-warning")
string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets --expt-extended-lambda")

if(NOT MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        string(APPEND CMAKE_CUDA_FLAGS " -g -G")
    else()
        string(APPEND CMAKE_CUDA_FLAGS " -O3")
    endif()
endif()

# For backward compatibility, set legacy variables (if needed elsewhere)
set(XSIGMA_CUDA_FOUND TRUE)
set(XSIGMA_ENABLE_CUDA ON)