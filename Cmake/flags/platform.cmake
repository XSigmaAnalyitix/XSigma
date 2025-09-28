# =============================================================================
# XSigma Platform-Specific Compiler Optimization Module (Improved)
# =============================================================================

# Guard against multiple inclusions
if(XSIGMA_PLATFORM_CONFIGURED)
    return()
endif()
set(XSIGMA_PLATFORM_CONFIGURED TRUE CACHE INTERNAL "Platform module loaded")

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

# User-configurable optimization levels
option(XSIGMA_AGGRESSIVE_OPTS "Enable aggressive optimizations (may break IEEE compliance)" OFF)
option(XSIGMA_NATIVE_ARCH "Enable native architecture optimizations" ON)
option(XSIGMA_PARALLEL_COMPILE "Enable parallel compilation" ON)

# Initialize platform-specific flag lists (using lists for better handling)

message("--avx compiler flags: ${VECTORIZATION_COMPILER_FLAGS}")
set(XSIGMA_REQUIRED_C_FLAGS ${VECTORIZATION_COMPILER_FLAGS})
set(XSIGMA_REQUIRED_CXX_FLAGS ${VECTORIZATION_COMPILER_FLAGS})
set(XSIGMA_LINKER_FLAGS_COMMON)

# Helper function to safely add compiler flags
function(xsigma_add_cxx_flag_if_supported flag)
    string(MAKE_C_IDENTIFIER "HAVE_FLAG_${flag}" flag_var)
    check_cxx_compiler_flag("${flag}" ${flag_var})
    if(${flag_var})
        list(APPEND XSIGMA_REQUIRED_CXX_FLAGS "${flag}")
        set(XSIGMA_REQUIRED_CXX_FLAGS ${XSIGMA_REQUIRED_CXX_FLAGS} PARENT_SCOPE)
    else()
        message(WARNING "XSigma: Compiler flag '${flag}' not supported, skipping")
    endif()
endfunction()

function(xsigma_add_c_flag_if_supported flag)
    string(MAKE_C_IDENTIFIER "HAVE_C_FLAG_${flag}" flag_var)
    check_c_compiler_flag("${flag}" ${flag_var})
    if(${flag_var})
        list(APPEND XSIGMA_REQUIRED_C_FLAGS "${flag}")
        set(XSIGMA_REQUIRED_C_FLAGS ${XSIGMA_REQUIRED_C_FLAGS} PARENT_SCOPE)
    else()
        message(WARNING "XSigma: C compiler flag '${flag}' not supported, skipping")
    endif()
endfunction()

# =============================================================================
# Windows Platform Optimizations
# =============================================================================

if(WIN32)
    message(STATUS "XSigma: Configuring Windows optimizations...")
    
    if(MSVC)
        message(STATUS "XSigma: Using MSVC on Windows...")
        
        # Essential MSVC flags
        list(APPEND XSIGMA_REQUIRED_CXX_FLAGS
            "/Zc:__cplusplus"
            "/permissive-"
            "/Zc:inline"
            "/Zc:throwingNew"
            "/volatile:iso"
            "/bigobj"
            "/utf-8"
        )

        # Performance flags
        xsigma_add_cxx_flag_if_supported("/favor:INTEL64")
        xsigma_add_cxx_flag_if_supported("/Gy")  # Function-level linking
        xsigma_add_cxx_flag_if_supported("/Gw")  # Global data optimization

        # Warning configuration
        list(APPEND XSIGMA_REQUIRED_CXX_FLAGS
            "/W3"
            "/wd4244" "/wd4267" "/wd4996" "/wd4251" "/wd4018"
        )

        # Copy CXX flags to C flags (minus C++-specific ones)
        set(XSIGMA_REQUIRED_C_FLAGS ${XSIGMA_REQUIRED_CXX_FLAGS})
        list(REMOVE_ITEM XSIGMA_REQUIRED_C_FLAGS "/Zc:__cplusplus" "/permissive-")

        # Parallel compilation
        if(XSIGMA_PARALLEL_COMPILE)
            include(ProcessorCount)
            ProcessorCount(PROCESSOR_COUNT)
            if(PROCESSOR_COUNT EQUAL 0)
                set(PROCESSOR_COUNT 4)
            endif()
            list(APPEND XSIGMA_REQUIRED_CXX_FLAGS "/MP${PROCESSOR_COUNT}")
            list(APPEND XSIGMA_REQUIRED_C_FLAGS "/MP${PROCESSOR_COUNT}")
            message(STATUS "XSigma: MSVC parallel compilation enabled (${PROCESSOR_COUNT} processes)")
        endif()

        # MSVC linker optimizations
        list(APPEND XSIGMA_LINKER_FLAGS_COMMON "/OPT:REF" "/OPT:ICF")
        
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "XSigma: Using Clang on Windows...")
        
        # Clang on Windows can use either GCC-style or MSVC-style flags
        # Check if using clang-cl (MSVC-compatible interface)
        if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" OR 
           CMAKE_CXX_COMPILER MATCHES "clang-cl")
            message(STATUS "XSigma: Using clang-cl (MSVC-compatible interface)")
            
            # Use MSVC-style flags for clang-cl
            list(APPEND XSIGMA_REQUIRED_CXX_FLAGS
                "/clang:-march=native"  # Clang-specific native optimization
                "/W3"
                "/bigobj"
                "/utf-8"
            )
            
            # Clang-cl parallel compilation
            if(XSIGMA_PARALLEL_COMPILE)
                include(ProcessorCount)
                ProcessorCount(PROCESSOR_COUNT)
                if(PROCESSOR_COUNT EQUAL 0)
                    set(PROCESSOR_COUNT 4)
                endif()
                list(APPEND XSIGMA_REQUIRED_CXX_FLAGS "/MP${PROCESSOR_COUNT}")
                list(APPEND XSIGMA_REQUIRED_C_FLAGS "/MP${PROCESSOR_COUNT}")
                message(STATUS "XSigma: clang-cl parallel compilation enabled (${PROCESSOR_COUNT} processes)")
            endif()
            
            # MSVC-style linker flags work with clang-cl
            list(APPEND XSIGMA_LINKER_FLAGS_COMMON "/OPT:REF" "/OPT:ICF")
            
        else()
            message(STATUS "XSigma: Using Clang with GCC-style interface on Windows")
            
            # Use GCC-style flags for regular Clang on Windows
            xsigma_add_cxx_flag_if_supported("-fvisibility=hidden")
            xsigma_add_cxx_flag_if_supported("-fvisibility-inlines-hidden")
            
            # Windows-specific Clang flags
            if(MINGW)
                message(STATUS "XSigma: MinGW environment detected")
                xsigma_add_cxx_flag_if_supported("-mthreads")
                xsigma_add_c_flag_if_supported("-mthreads")
            endif()
            
            # Performance flags
            xsigma_add_cxx_flag_if_supported("-ftree-vectorize")
            xsigma_add_cxx_flag_if_supported("-finline-functions")
            
            # Native architecture optimization
            if(XSIGMA_NATIVE_ARCH)
                xsigma_add_cxx_flag_if_supported("-march=native")
            endif()
            
            # Warning configuration
            xsigma_add_cxx_flag_if_supported("-Wall")
            xsigma_add_cxx_flag_if_supported("-Wextra")
            xsigma_add_cxx_flag_if_supported("-Wno-unused-parameter")
            
            # MinGW linker optimizations
            if(MINGW)
                list(APPEND XSIGMA_LINKER_FLAGS_COMMON "-Wl,--gc-sections")
            endif()
        endif()
        
        # Copy flags to C
        set(XSIGMA_REQUIRED_C_FLAGS ${XSIGMA_REQUIRED_CXX_FLAGS})
        
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "XSigma: Using GCC on Windows (MinGW/MSYS2)...")
        
        # GCC on Windows (MinGW)
        xsigma_add_cxx_flag_if_supported("-fvisibility=hidden")
        xsigma_add_cxx_flag_if_supported("-fvisibility-inlines-hidden")
        xsigma_add_cxx_flag_if_supported("-mthreads")
        
        # Performance flags
        xsigma_add_cxx_flag_if_supported("-ftree-vectorize")
        xsigma_add_cxx_flag_if_supported("-finline-functions")
        
        if(XSIGMA_NATIVE_ARCH)
            xsigma_add_cxx_flag_if_supported("-march=native")
        endif()
        
        # Warning configuration
        xsigma_add_cxx_flag_if_supported("-Wall")
        xsigma_add_cxx_flag_if_supported("-Wextra")
        
        # Copy flags to C
        set(XSIGMA_REQUIRED_C_FLAGS ${XSIGMA_REQUIRED_CXX_FLAGS})
        
        # MinGW linker optimizations
        list(APPEND XSIGMA_LINKER_FLAGS_COMMON "-Wl,--gc-sections")
    endif()

    # Common Windows definitions
    add_definitions(
        -D_CRT_SECURE_NO_WARNINGS
        -D_SCL_SECURE_NO_WARNINGS
        -DWIN32_LEAN_AND_MEAN
        -DNOMINMAX
    )

# =============================================================================
# Unix-like Platform Optimizations (Linux, macOS, BSD, etc.)
# =============================================================================

else()
    message(STATUS "XSigma: Configuring Unix-like platform optimizations...")

    # Essential flags for Unix-like systems
    xsigma_add_cxx_flag_if_supported("-fPIC")
    xsigma_add_cxx_flag_if_supported("-fvisibility=hidden")
    xsigma_add_cxx_flag_if_supported("-fvisibility-inlines-hidden")

    # Threading support
    xsigma_add_cxx_flag_if_supported("-pthread")
    xsigma_add_c_flag_if_supported("-pthread")

    # Safe performance flags
    xsigma_add_cxx_flag_if_supported("-ftree-vectorize")
    xsigma_add_cxx_flag_if_supported("-finline-functions")
    
    # Aggressive optimizations (optional)
    if(XSIGMA_AGGRESSIVE_OPTS)
        message(WARNING "XSigma: Enabling aggressive optimizations - may break IEEE compliance")
        xsigma_add_cxx_flag_if_supported("-ffast-math")
        xsigma_add_cxx_flag_if_supported("-funroll-loops")
    endif()

    # Native architecture optimization
    if(XSIGMA_NATIVE_ARCH)
        xsigma_add_cxx_flag_if_supported("-march=native")
    endif()

    # Warning configuration
    xsigma_add_cxx_flag_if_supported("-Wall")
    xsigma_add_cxx_flag_if_supported("-Wextra")
    xsigma_add_cxx_flag_if_supported("-Wno-ignored-attributes")
    xsigma_add_cxx_flag_if_supported("-Wno-unused-parameter")

    # Copy flags to C (they should be compatible)
    set(XSIGMA_REQUIRED_C_FLAGS ${XSIGMA_REQUIRED_CXX_FLAGS})

    # Unix linker optimizations
    list(APPEND XSIGMA_LINKER_FLAGS_COMMON "-Wl,--gc-sections" "-Wl,--as-needed")

endif()

# =============================================================================
# Compiler-Specific Optimizations (Cross-Platform)
# =============================================================================

# GCC specific (works on Windows via MinGW and Unix systems)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "XSigma: Applying GCC-specific optimizations...")
    
    # Version-specific optimizations
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0")
        xsigma_add_cxx_flag_if_supported("-fconcepts")
    endif()
    
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
        xsigma_add_cxx_flag_if_supported("-fmodules-ts")
    endif()
    
    # GCC-specific performance optimizations
    xsigma_add_cxx_flag_if_supported("-fdevirtualize-speculatively")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "10.0")
        xsigma_add_cxx_flag_if_supported("-fipa-pta")
    endif()
endif()

# Clang specific (works on Windows, macOS, and Linux)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "XSigma: Applying Clang-specific optimizations...")
    
    # Cross-platform Clang optimizations
    xsigma_add_cxx_flag_if_supported("-fstrict-aliasing")
    xsigma_add_cxx_flag_if_supported("-fvectorize")
    xsigma_add_cxx_flag_if_supported("-fslp-vectorize")
    
    # Clang version-specific features
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
        xsigma_add_cxx_flag_if_supported("-fexperimental-new-pass-manager")
    endif()
    
    # Platform-specific Clang configurations
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        message(STATUS "XSigma: Apple Clang detected")
        xsigma_add_cxx_flag_if_supported("-stdlib=libc++")
        # Apple Clang specific optimizations
        xsigma_add_cxx_flag_if_supported("-fcolor-diagnostics")
    elseif(WIN32 AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        message(STATUS "XSigma: Clang on Windows with GCC interface")
        # Additional Windows-specific Clang flags can go here
    endif()
endif()

# Intel Compiler
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(STATUS "XSigma: Applying Intel compiler optimizations...")
    xsigma_add_cxx_flag_if_supported("-xHost")
    xsigma_add_cxx_flag_if_supported("-ipo")
    xsigma_add_cxx_flag_if_supported("-vec-report=2")
endif()

# =============================================================================
# Final Flag Application
# =============================================================================

# Convert lists to space-separated strings for CMake compatibility
if(XSIGMA_REQUIRED_CXX_FLAGS)
    string(REPLACE ";" " " XSIGMA_CXX_FLAGS_STR "${XSIGMA_REQUIRED_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${XSIGMA_CXX_FLAGS_STR}")
endif()

if(XSIGMA_REQUIRED_C_FLAGS)
    string(REPLACE ";" " " XSIGMA_C_FLAGS_STR "${XSIGMA_REQUIRED_C_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${XSIGMA_C_FLAGS_STR}")
endif()

if(XSIGMA_LINKER_FLAGS_COMMON)
    string(REPLACE ";" " " XSIGMA_LINKER_FLAGS_STR "${XSIGMA_LINKER_FLAGS_COMMON}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${XSIGMA_LINKER_FLAGS_STR}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${XSIGMA_LINKER_FLAGS_STR}")
endif()

# Configuration summary
message(STATUS "XSigma: Platform optimization completed")
message(STATUS "XSigma: Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "XSigma: C++ Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "XSigma: Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}")

# Cache important settings
set(XSIGMA_PLATFORM_CONFIGURED TRUE CACHE INTERNAL "Platform optimization completed")