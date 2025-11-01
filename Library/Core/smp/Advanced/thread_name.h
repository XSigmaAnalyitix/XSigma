#pragma once

#include <string>

#include "common/macros.h"

namespace xsigma::detail::smp::Advanced
{

/**
 * @brief Sets the name of the current thread.
 *
 * This function sets the name of the current thread. The name is used for
 * debugging and profiling purposes. On systems that support thread naming
 * (e.g., Linux with glibc 2.12+), the name is set in the kernel. On other
 * systems, this function is a no-op.
 *
 * The maximum thread name length is 15 characters (including null terminator).
 * If the provided name is longer, it will be truncated.
 *
 * @param name The name to set for the current thread.
 *
 * @note Thread-safe. Each thread can set its own name independently.
 * @note Platform-dependent. Only works on Linux with glibc 2.12+.
 */
XSIGMA_API void set_thread_name(std::string name);

/**
 * @brief Gets the name of the current thread.
 *
 * This function retrieves the name of the current thread. On systems that
 * support thread naming (e.g., Linux with glibc 2.12+), the name is retrieved
 * from the kernel. On other systems, this function returns an empty string.
 *
 * @return The name of the current thread, or an empty string if thread naming
 *         is not supported on this platform.
 *
 * @note Thread-safe. Each thread can retrieve its own name independently.
 * @note Platform-dependent. Only works on Linux with glibc 2.12+.
 */
XSIGMA_API std::string get_thread_name();

}  // namespace xsigma::detail::smp::Advanced
