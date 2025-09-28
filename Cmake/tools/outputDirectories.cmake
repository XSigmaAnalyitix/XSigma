# =============================================================================
# XSigma Output Directory Configuration Module
# =============================================================================
# This module configures output directories for executables and libraries
# to ensure consistent placement across different CMake generators.
#
# Features:
# - Automatic detection of multi-config vs single-config generators
# - Proper handling of Visual Studio, Ninja, and Makefiles generators
# - Windows-specific DLL and executable placement
# - Cross-platform compatibility
# =============================================================================

# Guard against multiple inclusions
if(XSIGMA_OUTPUT_DIRECTORIES_CONFIGURED)
    return()
endif()
set(XSIGMA_OUTPUT_DIRECTORIES_CONFIGURED TRUE CACHE INTERNAL "Output directories module loaded")

# =============================================================================
# Generator Detection
# =============================================================================

# Detect if we're using a multi-config generator (like Visual Studio, Xcode)
# vs single-config generator (like Ninja, Unix Makefiles)
get_property(XSIGMA_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

if(NOT DEFINED XSIGMA_IS_MULTI_CONFIG)
    # Fallback for older CMake versions
    if(CMAKE_GENERATOR MATCHES "Visual Studio" OR 
       CMAKE_GENERATOR MATCHES "Xcode" OR
       CMAKE_GENERATOR MATCHES "Multi-Config")
        set(XSIGMA_IS_MULTI_CONFIG TRUE)
    else()
        set(XSIGMA_IS_MULTI_CONFIG FALSE)
    endif()
endif()

# =============================================================================
# Output Directory Configuration
# =============================================================================

# Set base output directories
set(XSIGMA_BASE_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
set(XSIGMA_BASE_LIB_DIR "${CMAKE_BINARY_DIR}/lib")

if(XSIGMA_IS_MULTI_CONFIG)
    # Multi-config generators (Visual Studio, Xcode)
    # These generators automatically append the configuration name
    message(STATUS "XSigma: Configuring output directories for multi-config generator")
    
    # Set global output directories
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${XSIGMA_BASE_BIN_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${XSIGMA_BASE_LIB_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${XSIGMA_BASE_LIB_DIR}")
    
    # For Windows, also set PDB output directory
    if(WIN32)
        set(CMAKE_PDB_OUTPUT_DIRECTORY "${XSIGMA_BASE_BIN_DIR}")
    endif()
    
    # Set per-configuration directories explicitly
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${config} config_upper)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BASE_BIN_DIR}/${config}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BASE_LIB_DIR}/${config}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BASE_LIB_DIR}/${config}")
        
        if(WIN32)
            set(CMAKE_PDB_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BASE_BIN_DIR}/${config}")
        endif()
    endforeach()
    
else()
    # Single-config generators (Ninja, Unix Makefiles)
    message(STATUS "XSigma: Configuring output directories for single-config generator")
    
    # Set global output directories without configuration subdirectories
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${XSIGMA_BASE_BIN_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${XSIGMA_BASE_LIB_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${XSIGMA_BASE_LIB_DIR}")
    
    # For Windows, also set PDB output directory
    if(WIN32)
        set(CMAKE_PDB_OUTPUT_DIRECTORY "${XSIGMA_BASE_BIN_DIR}")
    endif()
endif()

# =============================================================================
# Utility Functions
# =============================================================================

# Function to apply output directory settings to a specific target
function(xsigma_set_target_output_directories target_name)
    if(XSIGMA_IS_MULTI_CONFIG)
        # Multi-config: Set base directories and let generator handle config subdirs
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${XSIGMA_BASE_BIN_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${XSIGMA_BASE_LIB_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY "${XSIGMA_BASE_LIB_DIR}"
        )
        
        if(WIN32)
            set_target_properties(${target_name} PROPERTIES
                PDB_OUTPUT_DIRECTORY "${XSIGMA_BASE_BIN_DIR}"
            )
        endif()
        
        # Set per-configuration directories
        foreach(config ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${config} config_upper)
            set_target_properties(${target_name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BASE_BIN_DIR}/${config}"
                LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BASE_LIB_DIR}/${config}"
                ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BASE_LIB_DIR}/${config}"
            )
            
            if(WIN32)
                set_target_properties(${target_name} PROPERTIES
                    PDB_OUTPUT_DIRECTORY_${config_upper} "${XSIGMA_BASE_BIN_DIR}/${config}"
                )
            endif()
        endforeach()
    else()
        # Single-config: Set directories directly
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${XSIGMA_BASE_BIN_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${XSIGMA_BASE_LIB_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY "${XSIGMA_BASE_LIB_DIR}"
        )
        
        if(WIN32)
            set_target_properties(${target_name} PROPERTIES
                PDB_OUTPUT_DIRECTORY "${XSIGMA_BASE_BIN_DIR}"
            )
        endif()
    endif()
endfunction()

# Function to get the appropriate output directory for a given target type and config
function(xsigma_get_output_directory output_var target_type config)
    if(target_type STREQUAL "EXECUTABLE" OR target_type STREQUAL "SHARED_LIBRARY")
        set(base_dir "${XSIGMA_BASE_BIN_DIR}")
    else()
        set(base_dir "${XSIGMA_BASE_LIB_DIR}")
    endif()
    
    if(XSIGMA_IS_MULTI_CONFIG AND config)
        set(${output_var} "${base_dir}/${config}" PARENT_SCOPE)
    else()
        set(${output_var} "${base_dir}" PARENT_SCOPE)
    endif()
endfunction()

# Function to create output directories if they don't exist
function(xsigma_create_output_directories)
    file(MAKE_DIRECTORY "${XSIGMA_BASE_BIN_DIR}")
    file(MAKE_DIRECTORY "${XSIGMA_BASE_LIB_DIR}")
    
    if(XSIGMA_IS_MULTI_CONFIG)
        foreach(config ${CMAKE_CONFIGURATION_TYPES})
            file(MAKE_DIRECTORY "${XSIGMA_BASE_BIN_DIR}/${config}")
            file(MAKE_DIRECTORY "${XSIGMA_BASE_LIB_DIR}/${config}")
        endforeach()
    endif()
endfunction()

# =============================================================================
# Information Display
# =============================================================================

message(STATUS "XSigma Output Directory Configuration:")
message(STATUS "  Generator: ${CMAKE_GENERATOR}")
message(STATUS "  Multi-config: ${XSIGMA_IS_MULTI_CONFIG}")
message(STATUS "  Base bin directory: ${XSIGMA_BASE_BIN_DIR}")
message(STATUS "  Base lib directory: ${XSIGMA_BASE_LIB_DIR}")

if(XSIGMA_IS_MULTI_CONFIG)
    message(STATUS "  Configuration types: ${CMAKE_CONFIGURATION_TYPES}")
    message(STATUS "  Output structure: bin/<config>/ and lib/<config>/")
else()
    message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Output structure: bin/ and lib/")
endif()

# Create the output directories
xsigma_create_output_directories()

# Cache important variables for other modules
set(XSIGMA_OUTPUT_BIN_DIR "${XSIGMA_BASE_BIN_DIR}" CACHE INTERNAL "XSigma binary output directory")
set(XSIGMA_OUTPUT_LIB_DIR "${XSIGMA_BASE_LIB_DIR}" CACHE INTERNAL "XSigma library output directory")
set(XSIGMA_IS_MULTI_CONFIG_CACHED "${XSIGMA_IS_MULTI_CONFIG}" CACHE INTERNAL "Whether using multi-config generator")
