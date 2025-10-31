/*
 * XSigma Kineto Shim Implementation
 *
 * Direct libkineto integration matching PyTorch's implementation.
 *
 * Note: This file is only compiled when XSIGMA_HAS_KINETO=ON.
 * Build-time exclusion is handled in CMakeLists.txt.
 */

#include "kineto_shim.h"

#include <libkineto.h>

#include <mutex>

namespace xsigma
{
namespace profiler
{

namespace
{
// Thread-local state for Kineto
thread_local bool g_kineto_initialized = false;
std::mutex        g_kineto_init_mutex;
}  // namespace

void kineto_init(bool cpu_only, bool log_on_error)
{
    std::lock_guard<std::mutex> lock(g_kineto_init_mutex);

    if (g_kineto_initialized)
    {
        return;
    }

    // Initialize libkineto
    libkineto_init(cpu_only, log_on_error);

    // Suppress log messages if requested
    if (log_on_error)
    {
        libkineto::api().suppressLogMessages();
    }

    g_kineto_initialized = true;
}

bool kineto_is_profiler_registered()
{
    return libkineto::api().isProfilerRegistered();
}

bool kineto_is_profiler_initialized()
{
    return libkineto::api().isProfilerInitialized();
}

void kineto_prepare_trace(
    const std::set<libkineto::ActivityType>& activities, const std::string& config_str)
{
    // Reset thread-local state
    libkineto::api().resetKinetoTLS();

    // Initialize if not already done
    if (!kineto_is_profiler_registered())
    {
        kineto_init(false, true);
    }

    // Initialize profiler if needed
    if (!kineto_is_profiler_initialized())
    {
        libkineto::api().initProfilerIfRegistered();
    }

    // Prepare trace with specified activities
    libkineto::api().activityProfiler().prepareTrace(activities, config_str);
}

void kineto_start_trace()
{
    libkineto::api().activityProfiler().startTrace();
}

std::unique_ptr<libkineto::ActivityTraceInterface> kineto_stop_trace()
{
    return libkineto::api().activityProfiler().stopTrace();
}

void kineto_reset_tls()
{
    libkineto::api().resetKinetoTLS();
}

}  // namespace profiler
}  // namespace xsigma
