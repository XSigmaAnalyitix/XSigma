/*
 * XSigma Profiler API Implementation
 */

#include "profiler_api.h"
#if XSIGMA_HAS_KINETO
#include "kineto_shim.h"
#endif
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "profiler/pytroch_profiler/itt_wrapper.h"

namespace xsigma
{
namespace profiler
{

// ============================================================================
// == profiler_session Implementation ==========================================
// ============================================================================

profiler_session& profiler_session::instance()
{
    static profiler_session instance_;
    return instance_;
}

bool profiler_session::start(const profiler_config& config)
{
    std::scoped_lock const lock(state_mutex_);

    if (state_ != profiler_state_enum::Disabled)
    {
        if (config.verbose)
        {
            std::cerr << "Profiler already running" << std::endl;
        }
        return false;
    }

    config_ = config;
    state_  = profiler_state_enum::Ready;

    // Initialize Kineto if needed
    initialize_kineto();

    // Initialize ITT if needed
    if (config_.verbose)
    {
        initialize_itt();
    }

    // Record start time
    start_time_ns_ = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    state_ = profiler_state_enum::Recording;

    if (config_.verbose)
    {
        std::cout << "Profiler started with " << config_.activities.size() << " activities"
                  << std::endl;
    }

    return true;
}

bool profiler_session::stop()
{
    std::scoped_lock const lock(state_mutex_);

    if (state_ != profiler_state_enum::Recording)
    {
        if (config_.verbose)
        {
            std::cerr << "Profiler not recording" << std::endl;
        }
        return false;
    }

    // Record end time
    end_time_ns_ = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Collect events
    collect_events();

    // Finalize Kineto
    finalize_kineto();

    // Finalize ITT
    finalize_itt();

    state_ = profiler_state_enum::Ready;

    if (config_.verbose)
    {
        std::cout << "Profiler stopped. Collected " << events_.size() << " events" << std::endl;
    }

    return true;
}

bool profiler_session::is_profiling() const
{
    std::scoped_lock const lock(state_mutex_);
    return state_ == profiler_state_enum::Recording;
}

profiler_state_enum profiler_session::get_state() const
{
    std::scoped_lock const lock(state_mutex_);
    return state_;
}

const profiler_config& profiler_session::get_config() const
{
    std::scoped_lock const lock(state_mutex_);
    return config_;
}

bool profiler_session::export_trace(const std::string& path)
{
    std::scoped_lock const lock(state_mutex_);

    if (state_ == profiler_state_enum::Recording)
    {
        if (config_.verbose)
        {
            std::cerr << "Cannot export while profiling" << std::endl;
        }
        return false;
    }

    try
    {
        std::ofstream file(path);
        if (!file.is_open())
        {
            if (config_.verbose)
            {
                std::cerr << "Failed to open file: " << path << std::endl;
            }
            return false;
        }

        // Write JSON header
        file << "{\n";
        file << "  \"traceEvents\": [\n";

        // Write events
        for (size_t i = 0; i < events_.size(); ++i)
        {
            file << "    " << events_[i];
            if (i < events_.size() - 1)
            {
                file << ",";
            }
            file << "\n";
        }

        file << "  ],\n";
        file << "  \"displayTimeUnit\": \"ns\",\n";
        file << R"(  "traceID": ")" << config_.trace_id << "\"\n";
        file << "}\n";

        file.close();

        if (config_.verbose)
        {
            std::cout << "Trace exported to: " << path << std::endl;
        }

        return true;
    }
    catch (const std::exception& e)
    {
        if (config_.verbose)
        {
            std::cerr << "Export failed: " << e.what() << std::endl;
        }
        return false;
    }
}

void profiler_session::clear()
{
    std::scoped_lock const lock(state_mutex_);
    events_.clear();
    start_time_ns_ = 0;
    end_time_ns_   = 0;
}

size_t profiler_session::event_count() const
{
    std::scoped_lock const lock(state_mutex_);
    return events_.size();
}

void profiler_session::reset()
{
    std::scoped_lock const lock(state_mutex_);
    state_ = profiler_state_enum::Disabled;
    events_.clear();
    start_time_ns_ = 0;
    end_time_ns_   = 0;
}

void profiler_session::initialize_kineto() const
{
#if XSIGMA_HAS_KINETO
    kineto_init(false, config_.verbose);
#endif
}

void profiler_session::initialize_itt()
{
#if XSIGMA_HAS_ITT
    itt_init();
#endif
}

void profiler_session::finalize_kineto()
{
#if XSIGMA_HAS_KINETO
    kineto_reset_tls();
#endif
}

void profiler_session::finalize_itt()
{
    // ITT cleanup is automatic
}

void profiler_session::collect_events() const
{
    //fixme:
    // Simplified event collection
    // In a full implementation, this would:
    // 1. Collect CPU events from record_function callbacks
    // 2. Collect GPU events from Kineto
    // 3. Merge and correlate events
    // 4. Build event tree
    // 5. Export to JSON

    if (config_.verbose)
    {
        std::cout << "Collecting events from " << start_time_ns_ << " to " << end_time_ns_
                  << std::endl;
    }
}

// ============================================================================
// == Global API Functions ====================================================
// ============================================================================

bool profiler_enabled()
{
    return profiler_session::instance().get_state() != profiler_state_enum::Disabled;
}

profiler_state_enum get_profiler_state()
{
    return profiler_session::instance().get_state();
}

const profiler_config& get_profiler_config()
{
    return profiler_session::instance().get_config();
}

}  // namespace profiler
}  // namespace xsigma
