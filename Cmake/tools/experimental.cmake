# ============================================================================= XSigma Experimental
# Features Configuration Module
# =============================================================================
# This module configures experimental features for the XSigma project. Experimental features
# are under development and may not be stable or fully tested. They are disabled by default
# and should only be enabled for development and testing purposes.
# =============================================================================

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# Experimental Features Support Flag Controls whether experimental features are enabled.
# When enabled, provides access to features that are under active development and may
# change or be removed in future releases. Use with caution in production environments.
option(XSIGMA_ENABLE_EXPERIMENTAL "Enable experimental features (use with caution)" OFF)
mark_as_advanced(XSIGMA_ENABLE_EXPERIMENTAL)

# Only proceed if experimental features are enabled
if(NOT XSIGMA_ENABLE_EXPERIMENTAL)
  message(STATUS "Experimental features are disabled (XSIGMA_ENABLE_EXPERIMENTAL=OFF)")
  return()
endif()

message(STATUS "Configuring experimental features...")

# ============================================================================= Experimental
# Features Configuration
# =============================================================================

# Set flag to indicate experimental features are available
set(XSIGMA_EXPERIMENTAL_FOUND TRUE CACHE BOOL "Experimental features are enabled" FORCE)

message(STATUS "✅ Experimental features enabled")
message(STATUS "   WARNING: Experimental features may be unstable or incomplete")
message(STATUS "   WARNING: API and behavior may change without notice")
message(STATUS "   WARNING: Not recommended for production use")

message(STATUS "Experimental features configuration complete")

