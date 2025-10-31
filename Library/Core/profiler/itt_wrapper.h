/*
 * XSigma ITT API Wrapper
 *
 * This header provides C++ wrapper functions for Intel Instrumentation and
 * Tracing Technology (ITT) API, aligned with PyTorch's implementation.
 *
 * Features:
 * - Global ITT domain for XSigma
 * - Task range annotations (push/pop)
 * - Event markers
 * - Thread-safe operations
 *
 * Usage:
 *   #ifdef XSIGMA_HAS_ITTAPI
 *   itt_range_push("my_operation");
 *   // ... code to profile ...
 *   itt_range_pop();
 *   itt_mark("checkpoint");
 *   #endif
 */

#pragma once

#include <string>

#include "common/export.h"

#ifdef XSIGMA_HAS_ITTAPI
#include <ittnotify.h>
#endif

namespace xsigma
{
namespace profiler
{

#ifdef XSIGMA_HAS_ITTAPI
constexpr bool kITTAvailable{true};
#else
constexpr bool kITTAvailable{false};
#endif

// ============================================================================
// ITT API Wrapper Functions
// ============================================================================

#ifdef XSIGMA_HAS_ITTAPI

/**
 * @brief Initialize ITT API
 *
 * Creates the global XSigma ITT domain. Should be called once before
 * any ITT operations.
 */
XSIGMA_API void itt_init();

/**
 * @brief Push a named range onto the ITT stack
 *
 * Marks the beginning of a named task/operation for VTune profiling.
 *
 * @param name Name of the operation (must be a string literal or persistent)
 */
XSIGMA_API void itt_range_push(const char* name);

/**
 * @brief Pop the current range from the ITT stack
 *
 * Marks the end of the current task/operation.
 */
XSIGMA_API void itt_range_pop();

/**
 * @brief Mark an event at the current time
 *
 * Records an instantaneous event for VTune profiling.
 *
 * @param name Name of the event (must be a string literal or persistent)
 */
XSIGMA_API void itt_mark(const char* name);

/**
 * @brief Get the global ITT domain
 *
 * @return Pointer to the global XSigma ITT domain, or nullptr if ITT not available
 */
XSIGMA_API __itt_domain* itt_get_domain();

#else

// Stub implementations when XSIGMA_HAS_ITTAPI is not defined
inline void  itt_init() {}
inline void  itt_range_push(const char*) {}
inline void  itt_range_pop() {}
inline void  itt_mark(const char*) {}
inline void* itt_get_domain()
{
    return nullptr;
}

#endif  // XSIGMA_HAS_ITTAPI

}  // namespace profiler
}  // namespace xsigma
