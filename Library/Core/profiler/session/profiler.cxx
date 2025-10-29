/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

/**
 * @file profiler.cxx
 * @brief Implementation of the enhanced profiler system for XSigma applications
 *
 * Provides high-performance, thread-safe profiling capabilities with comprehensive
 * timing, memory tracking, and statistical analysis features.
 *
 * @author XSigma Development Team
 * @version 1.0
 * @date 2024
 */

#include "profiler.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

#include "common/macros.h"
#include "logging/logger.h"
#include "profiler/analysis/statistical_analyzer.h"
#include "profiler/core/profiler_collection.h"
#include "profiler/core/profiler_factory.h"
#include "profiler/exporters/xplane/xplane_schema.h"
#include "profiler/memory/memory_tracker.h"
#include "profiler/session/profiler_report.h"

// Prevent Windows min/max macros from interfering with std::numeric_limits
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <algorithm>
#include <limits>

namespace xsigma
{
namespace
{
std::string JsonQuote(std::string_view value)
{
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char c : value)
    {
        switch (c)
        {
        case '"':
            escaped.append("\\\"");
            break;
        case '\\':
            escaped.append("\\\\");
            break;
        case '\b':
            escaped.append("\\b");
            break;
        case '\f':
            escaped.append("\\f");
            break;
        case '\n':
            escaped.append("\\n");
            break;
        case '\r':
            escaped.append("\\r");
            break;
        case '\t':
            escaped.append("\\t");
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char buffer[7];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned char>(c));
                escaped.append(buffer);
            }
            else
            {
                escaped.push_back(c);
            }
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string ConvertXSpaceToChromeTrace(const x_space& space)
{
    constexpr uint32_t kPid = 0;

    std::ostringstream out;
    out << R"({"displayTimeUnit":"ns","metadata":{"highres-ticks":true},"traceEvents":[)";
    bool first        = true;
    auto append_event = [&](const std::string& event_json)
    {
        if (!first)
        {
            out << ',';
        }
        first = false;
        out << event_json;
    };

    append_event(
        std::string("{\"ph\":\"M\",\"pid\":0,\"name\":\"process_name\",\"args\":{\"name\":") +
        JsonQuote("XSigma CPU Profiler") + "}}");
    append_event(
        "{\"ph\":\"M\",\"pid\":0,\"name\":\"process_sort_index\",\"args\":{\"sort_index\":0}}");

    for (const xplane& plane : space.planes())
    {
        if (plane.name() != kHostThreadsPlaneName)
        {
            continue;
        }

        const auto& metadata     = plane.event_metadata();
        int         thread_index = 0;
        for (const xline& line : plane.lines())
        {
            ++thread_index;
            int64_t tid = line.display_id() != 0 ? line.display_id() : line.id();
            if (tid == 0)
            {
                tid = thread_index;
            }

            std::string thread_name = !line.display_name().empty()
                                          ? std::string(line.display_name())
                                          : std::string(line.name());
            if (thread_name.empty())
            {
                thread_name = "Thread " + std::to_string(thread_index);
            }

            append_event(
                std::string("{\"ph\":\"M\",\"pid\":") + std::to_string(kPid) +
                ",\"tid\":" + std::to_string(tid) +
                ",\"name\":\"thread_name\",\"args\":{\"name\":" + JsonQuote(thread_name) + "}}");
            append_event(
                std::string("{\"ph\":\"M\",\"pid\":") + std::to_string(kPid) + ",\"tid\":" +
                std::to_string(tid) + ",\"name\":\"thread_sort_index\",\"args\":{\"sort_index\":" +
                std::to_string(thread_index) + "}}");

            int64_t line_timestamp_ns = line.timestamp_ns();
            for (const xevent& event : line.events())
            {
                if (event.data_case() == xevent::data_case_type::kNumOccurrences)
                {
                    continue;
                }

                std::string event_name;
                if (const auto it = metadata.find(event.metadata_id()); it != metadata.end())
                {
                    const auto& md = it->second;
                    event_name     = !md.display_name().empty() ? std::string(md.display_name())
                                                                : std::string(md.name());
                }
                else
                {
                    event_name = "TraceEvent";
                }

                uint64_t offset_ps = static_cast<uint64_t>(std::max<int64_t>(0, event.offset_ps()));
                uint64_t start_ps =
                    static_cast<uint64_t>(std::max<int64_t>(0, line_timestamp_ns)) * 1000ULL +
                    offset_ps;
                uint64_t ts_us       = start_ps / 1000ULL;
                uint64_t duration_ps = static_cast<uint64_t>(event.duration_ps());
                if (duration_ps == 0)
                {
                    duration_ps = 1000ULL;  // 1 ns to ensure visibility
                }
                uint64_t dur_us = std::max<uint64_t>(1, duration_ps / 1000ULL);

                append_event(
                    std::string("{\"ph\":\"X\",\"pid\":") + std::to_string(kPid) +
                    ",\"tid\":" + std::to_string(tid) + ",\"ts\":" + std::to_string(ts_us) +
                    ",\"dur\":" + std::to_string(dur_us) + ",\"name\":" + JsonQuote(event_name) +
                    "}");
            }
        }
    }

    out << "]}";
    return out.str();
}

}  // namespace

// Static thread-local storage for current scope (DLL-compatible implementation)
thread_local xsigma::profiler_scope_data* profiler_session::thread_current_scope_ = nullptr;

// Static current session management with atomic operations for thread safety
static std::atomic<xsigma::profiler_session*> g_current_session{nullptr};

//=============================================================================
// timing_stats Implementation
//=============================================================================

void timing_stats::add_sample(double time_ms)
{
    min_time_ = std::min(time_ms, min_time_);
    max_time_ = std::max(time_ms, max_time_);
    total_time_ += time_ms;
    ++sample_count_;
    samples_.push_back(time_ms);
}

void timing_stats::calculate_statistics()
{
    if (sample_count_ == 0)
    {
        return;
    }

    mean_time_ = total_time_ / sample_count_;

    // Compute variance/standard deviation using collected samples
    double variance_sum = 0.0;
    for (double sample : samples_)
    {
        double const diff = sample - mean_time_;
        variance_sum += diff * diff;
    }
    std_deviation_ = sample_count_ > 0 ? std::sqrt(variance_sum / sample_count_) : 0.0;

    // Initialize percentiles vector if needed (25th, 50th, 75th, 90th, 95th, 99th)
    std::array<double, 6> percentile_targets = {25.0, 50.0, 75.0, 90.0, 95.0, 99.0};
    std::vector<double>   sorted_samples     = samples_;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    percentiles_.assign(percentile_targets.begin(), percentile_targets.end());
    for (size_t i = 0; i < percentile_targets.size(); ++i)
    {
        double const percentile = percentile_targets[i];
        if (sorted_samples.empty())
        {
            percentiles_[i] = 0.0;
            continue;
        }
        double const index = (percentile / 100.0) * (sorted_samples.size() - 1);
        auto const   lower = static_cast<size_t>(std::floor(index));
        auto const   upper = static_cast<size_t>(std::ceil(index));
        if (lower == upper)
        {
            percentiles_[i] = sorted_samples[lower];
        }
        else
        {
            double const weight = index - lower;
            percentiles_[i] =
                (sorted_samples[lower] * (1.0 - weight)) + (sorted_samples[upper] * weight);
        }
    }
}

void timing_stats::reset()
{
    min_time_      = (std::numeric_limits<double>::max)();
    max_time_      = 0.0;
    total_time_    = 0.0;
    mean_time_     = 0.0;
    std_deviation_ = 0.0;
    sample_count_  = 0;
    percentiles_.clear();
    samples_.clear();
}

//=============================================================================
// profiler_scope_data Implementation
//=============================================================================

double profiler_scope_data::get_duration_ms() const
{
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
    return duration.count() / 1000.0;
}

double profiler_scope_data::get_duration_us() const
{
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
    return static_cast<double>(duration.count());
}

double profiler_scope_data::get_duration_ns() const
{
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
    return static_cast<double>(duration.count());
}

//=============================================================================
// profiler_session Implementation
//=============================================================================

profiler_session::profiler_session(xsigma::profiler_options options) : options_(std::move(options))
{
    initialize_components();
}

profiler_session::~profiler_session()
{
    if (active_.load())
    {
        stop();
    }
    cleanup_components();
}

bool profiler_session::start()
{
    if (active_.exchange(true))
    {
        return false;  // Already active
    }

    auto maybe_lock = ProfilerLock::Acquire();
    if (!maybe_lock.has_value())
    {
        active_.store(false);
        return false;
    }
    profiler_lock_ = std::move(*maybe_lock);

    backend_profile_options_ = build_backend_profile_options();
    uint64_t const start_ns  = static_cast<uint64_t>(get_current_time_nanos());
    backend_profile_options_.set_start_timestamp_ns(start_ns);
    start_time_ns_ = start_ns;
    xspace_ready_  = false;

    auto profilers = xsigma::create_profilers(backend_profile_options_);
    if (!profilers.empty())
    {
        backend_profilers_ = std::make_unique<profiler_collection>(std::move(profilers));
        profiler_status backend_status = backend_profilers_->start();
        if (!backend_status.ok())
        {
            XSIGMA_LOG_ERROR(
                "Failed to start one or more profiler backends: {}", backend_status.message());
            backend_profilers_.reset();
            profiler_lock_.ReleaseIfActive();
            active_.store(false);
            return false;
        }
    }
    else
    {
        backend_profilers_.reset();
    }

    start_time_ = std::chrono::high_resolution_clock::now();

    // Initialize root scope for hierarchical profiling
    if (options_.enable_hierarchical_profiling_)
    {
        std::scoped_lock const lock(scope_mutex_);
        root_scope_               = std::make_unique<xsigma::profiler_scope_data>();
        root_scope_->name_        = "ROOT";
        root_scope_->start_time_  = start_time_;
        root_scope_->thread_id_   = std::this_thread::get_id();
        root_scope_->depth_level_ = 0;
        current_scope_            = root_scope_.get();
        thread_current_scope_     = current_scope_;
    }

    // Start memory tracking
    if (options_.enable_memory_tracking_ && memory_tracker_)
    {
        memory_tracker_->start_tracking();
    }

    // Start statistical analysis
    if (options_.enable_statistical_analysis_ && statistical_analyzer_)
    {
        statistical_analyzer_->start_analysis();
    }

    set_current_session(this);

    return true;
}

bool profiler_session::stop()
{
    if (!active_.exchange(false))
    {
        return false;  // Not active
    }

    end_time_    = std::chrono::high_resolution_clock::now();
    end_time_ns_ = static_cast<uint64_t>(get_current_time_nanos());

    // Finalize root scope
    if (options_.enable_hierarchical_profiling_ && root_scope_)
    {
        std::scoped_lock const lock(scope_mutex_);
        root_scope_->end_time_ = end_time_;
        thread_current_scope_  = nullptr;
    }

    // Stop memory tracking
    if (options_.enable_memory_tracking_ && memory_tracker_)
    {
        memory_tracker_->stop_tracking();
    }

    // Stop statistical analysis
    if (options_.enable_statistical_analysis_ && statistical_analyzer_)
    {
        statistical_analyzer_->stop_analysis();
    }

    if (backend_profilers_)
    {
        std::string     backend_errors;
        profiler_status stop_status = backend_profilers_->stop();
        if (!stop_status.ok() && !stop_status.message().empty())
        {
            backend_errors = stop_status.message();
        }

        x_space collected_space;
        xspace_                        = x_space();
        profiler_status collect_status = backend_profilers_->collect_data(&collected_space);
        if (collect_status.ok())
        {
            xspace_ = std::move(collected_space);
            normalize_xspace(&xspace_);
        }
        else
        {
            if (!collect_status.message().empty())
            {
                if (!backend_errors.empty())
                {
                    backend_errors.append("\n");
                }
                backend_errors.append(collect_status.message());
            }
        }
        xspace_ready_ = collect_status.ok();

        if (!backend_errors.empty())
        {
            XSIGMA_LOG_ERROR("Profiler backend errors: {}", backend_errors);
        }

        backend_profilers_.reset();
    }
    else
    {
        xspace_ready_ = false;
    }

    if (current_session() == this)
    {
        set_current_session(nullptr);
    }

    profiler_lock_.ReleaseIfActive();

    return true;
}

std::unique_ptr<xsigma::profiler_scope> profiler_session::create_scope(const std::string& name)
{
    return std::make_unique<xsigma::profiler_scope>(name, this);
}

std::unique_ptr<xsigma::profiler_report> profiler_session::generate_report() const
{
    return std::make_unique<xsigma::profiler_report>(*this);
}

void profiler_session::export_report(const std::string& filename) const
{
    auto report = generate_report();
    report->export_to_file(filename, options_.output_format_);
}

void profiler_session::print_report() const
{
    auto report = generate_report();
    report->print_detailed_report();
}

profiler_session* profiler_session::current_session()
{
    return g_current_session.load();
}

void profiler_session::set_current_session(xsigma::profiler_session* session)
{
    g_current_session.store(session);
}

void profiler_session::initialize_components()
{
    if (options_.enable_memory_tracking_)
    {
        memory_tracker_ = std::make_unique<xsigma::memory_tracker>();
    }

    if (options_.enable_statistical_analysis_)
    {
        statistical_analyzer_ = std::make_unique<xsigma::statistical_analyzer>();
    }

    backend_profile_options_ = build_backend_profile_options();
}

void profiler_session::cleanup_components()
{
    memory_tracker_.reset();  //NOLINT
    memory_tracker_ = nullptr;

    statistical_analyzer_.reset();  //NOLINT

    backend_profilers_.reset();
    profiler_lock_.ReleaseIfActive();
    xspace_        = x_space();
    xspace_ready_  = false;
    start_time_ns_ = 0;
    end_time_ns_   = 0;

    std::scoped_lock const lock(scope_mutex_);
    root_scope_.reset();  //NOLINT
    current_scope_ = nullptr;
}

profile_options profiler_session::build_backend_profile_options() const
{
    profile_options opts;
    opts.set_version(5);
    opts.set_device_type(profile_options::device_type_enum::CPU);
    opts.set_include_dataset_ops(false);
    opts.set_host_tracer_level(options_.enable_timing_ ? 2U : 0U);
    opts.set_device_tracer_level(0);
    opts.set_python_tracer_level(0);
    opts.set_enable_hlo_proto(false);
    opts.set_duration_ms(0);
    return opts;
}

void profiler_session::normalize_xspace(x_space* space) const
{
    if (space == nullptr)
    {
        return;
    }
    int64_t const base_time = static_cast<int64_t>(start_time_ns_);
    for (auto& plane : *space->mutable_planes())
    {
        for (auto& line : *plane.mutable_lines())
        {
            int64_t ts = line.timestamp_ns() - base_time;
            if (ts < 0)
            {
                ts = 0;
            }
            line.set_timestamp_ns(ts);
        }
    }
    if (space->hostnames().empty())
    {
        space->add_hostname("localhost");
    }
}
void profiler_session::register_scope_start(xsigma::profiler_scope* scope)
{
    if (!options_.enable_hierarchical_profiling_ || !active_.load())
    {
        return;
    }

    std::scoped_lock const lock(scope_mutex_);

    // Find the current scope for this thread
    xsigma::profiler_scope_data* parent_scope = thread_current_scope_;
    if (parent_scope == nullptr)
    {
        parent_scope = root_scope_.get();
    }

    if (parent_scope != nullptr)
    {
        // Add as child to current scope
        auto child_scope          = std::make_unique<xsigma::profiler_scope_data>();
        child_scope->name_        = scope->data().name_;
        child_scope->start_time_  = scope->data().start_time_;
        child_scope->thread_id_   = std::this_thread::get_id();
        child_scope->depth_level_ = parent_scope->depth_level_ + 1;
        child_scope->parent_      = parent_scope;

        xsigma::profiler_scope_data* child_ptr = child_scope.get();
        parent_scope->children_.push_back(std::move(child_scope));

        // Update thread-local current scope
        thread_current_scope_ = child_ptr;
    }
}

void profiler_session::register_scope_end(xsigma::profiler_scope* scope)
{
    if (!options_.enable_hierarchical_profiling_ || !active_.load())
    {
        return;
    }

    std::scoped_lock const lock(scope_mutex_);

    if (thread_current_scope_ != nullptr)
    {
        thread_current_scope_->end_time_     = scope->data().end_time_;
        thread_current_scope_->memory_stats_ = scope->data().memory_stats_;
        thread_current_scope_->timing_stats_ = scope->data().timing_stats_;

        // Move back to parent scope
        thread_current_scope_ = thread_current_scope_->parent_;
    }
}

//=============================================================================
// profiler_scope Implementation
//=============================================================================

profiler_scope::profiler_scope(const std::string& name, xsigma::profiler_session* session)
    : session_((session != nullptr) ? session : xsigma::profiler_session::current_session())
{
    data_             = std::make_unique<xsigma::profiler_scope_data>();
    data_->name_      = name;
    data_->thread_id_ = std::this_thread::get_id();

    // Auto-start if session is active
    if ((session_ != nullptr) && session_->is_active())
    {
        start();
    }
}

profiler_scope::~profiler_scope()
{
    if (started_ && !stopped_)
    {
        stop();
    }
}

void profiler_scope::start()
{
    if (started_)
    {
        return;
    }

    started_           = true;
    data_->start_time_ = std::chrono::high_resolution_clock::now();

    // Register with session for hierarchical tracking
    if (session_ != nullptr)
    {
        session_->register_scope_start(this);

        memory_annotation_ = std::make_unique<scoped_memory_debug_annotation>(data_->name_.c_str());

        // Start memory tracking for this scope
        if (session_->options_.enable_memory_tracking_ && session_->memory_tracker_)
        {
            start_memory_stats_     = session_->memory_tracker_->get_current_stats();
            data_->memory_stats_    = start_memory_stats_;
            has_start_memory_stats_ = true;
        }
    }
}

void profiler_scope::stop()
{
    if (!started_ || stopped_)
    {
        return;
    }

    stopped_         = true;
    data_->end_time_ = std::chrono::high_resolution_clock::now();

    // Calculate timing statistics
    double const duration_ms = data_->get_duration_ms();
    data_->timing_stats_.add_sample(duration_ms);
    data_->timing_stats_.calculate_statistics();

    // Update memory statistics
    if (session_ != nullptr)
    {
        if (session_->options_.enable_memory_tracking_ && session_->memory_tracker_)
        {
            auto current_stats   = session_->memory_tracker_->get_current_stats();
            data_->memory_stats_ = current_stats;
            data_->memory_stats_.delta_since_start_ =
                has_start_memory_stats_
                    ? static_cast<int64_t>(current_stats.current_usage_) -
                          static_cast<int64_t>(start_memory_stats_.current_usage_)
                    : static_cast<int64_t>(current_stats.current_usage_);
            data_->memory_stats_.peak_usage_ =
                (std::max)(current_stats.peak_usage_, start_memory_stats_.peak_usage_);
        }

        // Add timing sample to statistical analyzer
        if (session_->options_.enable_statistical_analysis_ && session_->statistical_analyzer_)
        {
            session_->statistical_analyzer_->add_timing_sample(data_->name_, duration_ms);
        }

        // Register scope end with session
        session_->register_scope_end(this);
    }

    memory_annotation_.reset();
}

std::string profiler_session::generate_chrome_trace_json() const
{
    if (!xspace_ready_)
    {
        return "{}";
    }
    return ConvertXSpaceToChromeTrace(xspace_);
}

bool profiler_session::write_chrome_trace(const std::string& filename) const
{
    if (!xspace_ready_)
    {
        return false;
    }
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out)
    {
        return false;
    }
    out << ConvertXSpaceToChromeTrace(xspace_);
    return out.good();
}

}  // namespace xsigma
