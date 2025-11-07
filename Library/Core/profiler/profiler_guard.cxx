/*
 * XSigma Profiler RAII Guards Implementation
 */

#include "profiler_guard.h"

#include <chrono>

#include "itt_wrapper.h"
#include "profiler_api.h"

namespace xsigma
{
namespace profiler
{

// ============================================================================
// == profiler_guard Implementation ============================================
// ============================================================================

profiler_guard::profiler_guard(const profiler_config& config) : config_(config)
{
    auto& profiler = profiler_session::instance();
    was_profiling_ = profiler.start(config);
}

profiler_guard::~profiler_guard()
{
    if (was_profiling_)
    {
        auto& profiler = profiler_session::instance();
        profiler.stop();
    }
}

bool profiler_guard::export_trace(const std::string& path)
{
    auto& profiler = profiler_session::instance();
    return profiler.export_trace(path);
}

// ============================================================================
// == record_function Implementation ===========================================
// ============================================================================

record_function::record_function(const char* name) : name_(name)
{
    // Record start time
    start_time_ns_ = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Emit ITT range push if available
#if XSIGMA_HAS_ITT
    itt_range_push(name_);
#endif

    // Check if profiler is active
    if (profiler_enabled())
    {
        //fixme:
        // In a full implementation, this would register the function
        // with the active profiler for event collection
    }
}

record_function::~record_function()
{
    // Record end time
    end_time_ns_ = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Emit ITT range pop if available
#if XSIGMA_HAS_ITT
    itt_range_pop();
#endif

    // Check if profiler is active
    if (profiler_enabled())
    {
         //fixme:
        // In a full implementation, this would finalize the function
        // recording and add it to the event collection
    }
}

// ============================================================================
// == scoped_activity Implementation ===========================================
// ============================================================================

scoped_activity::scoped_activity(const char* name) : name_(name)
{
    // Emit ITT range push if available
#if XSIGMA_HAS_ITT
    itt_range_push(name_);
#endif
}

scoped_activity::~scoped_activity()
{
    // Emit ITT range pop if available
#if XSIGMA_HAS_ITT
    itt_range_pop();
#endif
}

}  // namespace profiler
}  // namespace xsigma
