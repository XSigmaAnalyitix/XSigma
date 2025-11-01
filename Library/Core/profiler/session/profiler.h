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

#pragma once

/**
 * @file profiler.h
 * @brief Enhanced profiler system for comprehensive performance analysis in XSigma applications
 *
 * This header provides a high-performance, thread-safe profiling system with:
 * - High-precision timing measurements (nanosecond accuracy)
 * - Memory usage tracking with allocation/deallocation monitoring
 * - Hierarchical profiling for nested function call analysis
 * - Statistical analysis with comprehensive metrics
 * - Multiple output formats (console, JSON, CSV, XML)
 * - Minimal performance overhead designed for production use
 *
 * @author XSigma Development Team
 * @version 1.0
 * @date 2024
 */

// Prevent Windows min/max macros from interfering with std::numeric_limits
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "logging/tracing/traceme.h"
#include "profiler/core/profiler_interface.h"
#include "profiler/core/profiler_lock.h"
#include "profiler/core/profiler_options.h"
#include "profiler/exporters/xplane/xplane.h"
#include "profiler/memory/scoped_memory_debug_annotation.h"

namespace xsigma
{

// Forward declarations for enhanced profiler components
class profiler_session;
class profiler_scope;
class memory_tracker;
class statistical_analyzer;
class profiler_report;
class profiler_collection;

class profiler_session_builder;

/**
 * @brief Configuration options for the enhanced profiler system
 *
 * This structure contains all configuration parameters that control the behavior
 * of the enhanced profiler, including timing, memory tracking, statistical analysis,
 * and output formatting options.
 */
struct profiler_options
{
    /// Enable high-precision timing measurements
    bool enable_timing_ = true;

    /// Enable memory allocation/deallocation tracking
    bool enable_memory_tracking_ = true;

    /// Enable hierarchical profiling for nested scopes
    bool enable_hierarchical_profiling_ = true;

    /// Enable statistical analysis of profiling data
    bool enable_statistical_analysis_ = true;

    /// Enable thread-safe profiling for multi-threaded applications
    bool enable_thread_safety_ = true;

    /**
     * @brief Output format options for profiling reports
     */
    enum class output_format_enum : int
    {
        CONSOLE,    ///< Human-readable console output
        FILE,       ///< Plain text file output
        JSON,       ///< JSON format for programmatic processing
        CSV,        ///< CSV format for spreadsheet analysis
        STRUCTURED  ///< Structured format with detailed hierarchy
    };

    /// Selected output format for reports
    output_format_enum output_format_ = output_format_enum::CONSOLE;

    /// File path for report output (when using FILE format)
    std::string output_file_path_;

    /// Maximum number of samples to collect for statistical analysis
    size_t max_samples_ = 1000;

    /// Whether to calculate percentile statistics (25th, 50th, 75th, 90th, 95th, 99th)
    bool calculate_percentiles_ = true;

    /// Whether to track peak memory usage
    bool track_peak_memory_ = true;

    /// Whether to track memory usage deltas between measurements
    bool track_memory_deltas_ = true;

    /// Size of thread pool for concurrent profiling operations
    size_t thread_pool_size_ = std::thread::hardware_concurrency();
};

/**
 * @brief Memory usage statistics container
 *
 * Tracks various memory usage metrics including current usage, peak usage,
 * total allocations/deallocations, and deltas since profiling started.
 */
struct memory_stats
{
    /// Current memory usage in bytes
    size_t current_usage_ = 0;

    /// Peak memory usage observed in bytes
    size_t peak_usage_ = 0;

    /// Total memory allocated in bytes
    size_t total_allocated_ = 0;

    /// Total memory deallocated in bytes
    size_t total_deallocated_ = 0;

    /// Memory usage change since profiling started (can be negative)
    int64_t delta_since_start_ = 0;

    /**
     * @brief Reset all memory statistics to zero
     */
    void reset()
    {
        current_usage_     = 0;
        peak_usage_        = 0;
        total_allocated_   = 0;
        total_deallocated_ = 0;
        delta_since_start_ = 0;
    }
};

/**
 * @brief Timing statistics for repeated measurements
 *
 * Collects and calculates statistical metrics for timing measurements,
 * including min/max/mean times, standard deviation, and percentiles.
 */
struct timing_stats
{
    /// Minimum time observed in milliseconds
    double min_time_ = (std::numeric_limits<double>::max)();

    /// Maximum time observed in milliseconds
    double max_time_ = 0.0;

    /// Total accumulated time in milliseconds
    double total_time_ = 0.0;

    /// Mean (average) time in milliseconds
    double mean_time_ = 0.0;

    /// Standard deviation of timing measurements
    double std_deviation_ = 0.0;

    /// Number of timing samples collected
    size_t sample_count_ = 0;

    /// Percentile values: 25th, 50th, 75th, 90th, 95th, 99th
    std::vector<double> percentiles_;

    /// Raw timing samples collected for this scope
    std::vector<double> samples_;

    /**
     * @brief Add a new timing sample to the statistics
     * @param time_ms Time measurement in milliseconds
     */
    void add_sample(double time_ms);

    /**
     * @brief Calculate statistical metrics from collected samples
     */
    void calculate_statistics(bool include_percentiles = true);

    /**
     * @brief Reset all timing statistics to initial state
     */
    void reset();
};

/**
 * @brief Profiler scope data for hierarchical profiling
 *
 * Contains all data associated with a profiling scope, including timing information,
 * memory statistics, hierarchical relationships, and thread context.
 */
struct profiler_scope_data
{
    /// Human-readable name of the profiling scope
    std::string name_;

    /// High-resolution timestamp when scope started
    std::chrono::high_resolution_clock::time_point start_time_;

    /// High-resolution timestamp when scope ended
    std::chrono::high_resolution_clock::time_point end_time_;

    /// Memory usage statistics for this scope
    xsigma::memory_stats memory_stats_;

    /// Timing statistics for this scope
    xsigma::timing_stats timing_stats_;

    /// ID of the thread that executed this scope
    std::thread::id thread_id_;

    /// Nesting depth level in the profiling hierarchy (0 = root)
    size_t depth_level_ = 0;

    /// Child scopes nested within this scope
    std::vector<std::unique_ptr<xsigma::profiler_scope_data>> children_;

    /// Pointer to parent scope (nullptr for root scope)
    xsigma::profiler_scope_data* parent_ = nullptr;

    /**
     * @brief Get the duration of this scope in milliseconds
     * @return Duration in milliseconds as a double
     */
    double get_duration_ms() const;

    /**
     * @brief Get the duration of this scope in microseconds
     * @return Duration in microseconds as a double
     */
    double get_duration_us() const;

    /**
     * @brief Get the duration of this scope in nanoseconds
     * @return Duration in nanoseconds as a double
     */
    double get_duration_ns() const;
};

/**
 * @brief Enhanced profiler session with profiler_session_builder pattern support
 *
 * The main profiler session class that manages the entire profiling lifecycle.
 * Provides high-precision timing, memory tracking, hierarchical profiling,
 * and statistical analysis capabilities. Uses the profiler_session_builder pattern for
 * flexible configuration.
 *
 * Thread-safe for concurrent profiling operations when enabled.
 */
class XSIGMA_VISIBILITY profiler_session
{
public:
    /**
     * @brief Construct a new enhanced profiler session
     * @param options Configuration options for the profiler
     */
    XSIGMA_API explicit profiler_session(xsigma::profiler_options options);

    /**
     * @brief Destructor - automatically stops profiling if active
     */
    XSIGMA_API ~profiler_session();

    /**
     * @brief Start the profiling session
     * @return true if successfully started, false if already active or failed
     */
    XSIGMA_API bool start();

    /**
     * @brief Stop the profiling session
     * @return true if successfully stopped, false if not active or failed
     */
    XSIGMA_API bool stop();

    /**
     * @brief Check if the profiling session is currently active
     * @return true if profiling is active, false otherwise
     */
    bool is_active() const { return active_.load(); }

    /**
     * @brief Create a new profiling scope
     * @param name Human-readable name for the scope
     * @return Unique pointer to the created profiler scope
     */
    XSIGMA_API std::unique_ptr<xsigma::profiler_scope> create_scope(const std::string& name);

    /**
     * @brief Get reference to the memory tracker component
     * @return Reference to the memory tracker
     */
    xsigma::memory_tracker& memory_tracker() { return *memory_tracker_; }

    /**
     * @brief Get reference to the statistical analyzer component
     * @return Reference to the statistical analyzer
     */
    xsigma::statistical_analyzer& statistical_analyzer() { return *statistical_analyzer_; }

    /**
     * @brief Generate a comprehensive profiling report
     * @return Unique pointer to the generated report
     */
    XSIGMA_API std::unique_ptr<xsigma::profiler_report> generate_report() const;

    /**
     * @brief Access the raw XSpace captured during profiling.
     * @return Const reference to the collected XSpace timeline.
     */
    const xsigma::x_space& collected_xspace() const { return xspace_; }

    /**
     * @brief Check whether the session captured any XSpace data.
     */
    bool has_collected_xspace() const { return xspace_ready_; }

    /**
     * @brief Generate a Chrome trace (JSON) representation of collected profiler data.
     */
    XSIGMA_API std::string generate_chrome_trace_json() const;

    /**
     * @brief Write the Chrome trace JSON to the specified file.
     */
    XSIGMA_API bool write_chrome_trace(const std::string& filename) const;

    /**
     * @brief Export profiling report to a file
     * @param filename Path to the output file
     */
    XSIGMA_API void export_report(const std::string& filename) const;

    /**
     * @brief Print profiling report to console
     */
    XSIGMA_API void print_report() const;

    /**
     * @brief Retrieve the session start timestamp (nanoseconds)
     * @return Start timestamp
     */
    std::chrono::high_resolution_clock::time_point session_start_time() const
    {
        return start_time_;
    }

    /**
     * @brief Retrieve the session end timestamp (nanoseconds)
     * @return End timestamp
     */
    std::chrono::high_resolution_clock::time_point session_end_time() const { return end_time_; }

    /**
     * @brief Access the memory tracker in read-only form
     */
    const xsigma::memory_tracker* memory_tracker_ptr() const { return memory_tracker_.get(); }

    /**
     * @brief Access the statistical analyzer in read-only form
     */
    const xsigma::statistical_analyzer* statistical_analyzer_ptr() const
    {
        return statistical_analyzer_.get();
    }

    /**
     * @brief Get the current active profiling session (thread-safe)
     * @return Pointer to current session or nullptr if none active
     */
    XSIGMA_API static profiler_session* current_session();

    /**
     * @brief Set the current active profiling session (thread-safe)
     * @param session Pointer to the session to set as current
     */
    XSIGMA_API static void set_current_session(profiler_session* session);

    /**
     * @brief Get read-only access to the root profiling scope
     * @return Const pointer to root scope data
     */
    const xsigma::profiler_scope_data* get_root_scope() const { return root_scope_.get(); }

private:
    friend class profiler_session_builder;

    /// Configuration options for this profiler session
    xsigma::profiler_options options_;

    /// Atomic flag indicating if profiling is currently active
    std::atomic<bool> active_{false};

    /// High-resolution timestamp when profiling session started
    std::chrono::high_resolution_clock::time_point start_time_;

    /// High-resolution timestamp when profiling session ended
    std::chrono::high_resolution_clock::time_point end_time_;

    /// Nanosecond timestamps compatible with XSpace exporters
    uint64_t start_time_ns_ = 0;
    uint64_t end_time_ns_   = 0;

    /// Memory tracking component
    std::unique_ptr<xsigma::memory_tracker> memory_tracker_;

    /// Statistical analysis component
    std::unique_ptr<xsigma::statistical_analyzer> statistical_analyzer_;

    /// Mutex for thread-safe access to hierarchical profiling data
    mutable std::mutex scope_mutex_;

    /// Root scope of the profiling hierarchy
    std::unique_ptr<xsigma::profiler_scope_data> root_scope_;

    /// Pointer to the currently active scope
    xsigma::profiler_scope_data* current_scope_ = nullptr;

    /// Thread-local storage for current scope (DLL-compatible implementation)
    static thread_local xsigma::profiler_scope_data* thread_current_scope_;

    /**
     * @brief Initialize all profiler components
     */
    void initialize_components();

    /**
     * @brief Clean up all profiler components
     */
    void cleanup_components();

    /// Build backend profiling options for registered profilers
    xsigma::profile_options build_backend_profile_options() const;

    /// Normalize timestamps of captured XSpace relative to session start
    void normalize_xspace(x_space* space) const;

    /// Profiler lock to avoid concurrent sessions
    xsigma::ProfilerLock profiler_lock_;

    /// Backend profiler configuration and collection
    xsigma::profile_options                      backend_profile_options_;
    std::unique_ptr<xsigma::profiler_collection> backend_profilers_;

    /// Captured XSpace timeline from backend profilers
    xsigma::x_space xspace_;
    bool            xspace_ready_ = false;

    /// Allow profiler_scope to access private registration methods
    friend class xsigma::profiler_scope;

    /**
     * @brief Register the start of a profiling scope
     * @param scope Pointer to the scope being started
     */
    void register_scope_start(xsigma::profiler_scope* scope);

    /**
     * @brief Register the end of a profiling scope
     * @param scope Pointer to the scope being ended
     */
    void register_scope_end(xsigma::profiler_scope* scope);
};

/**
 * @brief Builder pattern implementation for profiler_session
 *
 * Provides a fluent interface for configuring profiler options before
 * creating a profiler session. All methods return a reference to the
 * profiler_session_builder for method chaining.
 */
class XSIGMA_VISIBILITY profiler_session_builder
{
public:
    /**
     * @brief Default constructor
     */
    profiler_session_builder() = default;

    /**
     * @brief Enable or disable timing measurements
     * @param enable true to enable timing, false to disable
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_timing(bool enable = true)
    {
        options_.enable_timing_ = enable;
        return *this;
    }

    /**
     * @brief Enable or disable memory tracking
     * @param enable true to enable memory tracking, false to disable
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_memory_tracking(bool enable = true)
    {
        options_.enable_memory_tracking_ = enable;
        return *this;
    }

    /**
     * @brief Enable or disable hierarchical profiling
     * @param enable true to enable hierarchical profiling, false to disable
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_hierarchical_profiling(bool enable = true)
    {
        options_.enable_hierarchical_profiling_ = enable;
        return *this;
    }

    /**
     * @brief Enable or disable statistical analysis
     * @param enable true to enable statistical analysis, false to disable
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_statistical_analysis(bool enable = true)
    {
        options_.enable_statistical_analysis_ = enable;
        return *this;
    }

    /**
     * @brief Enable or disable thread safety
     * @param enable true to enable thread safety, false to disable
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_thread_safety(bool enable = true)
    {
        options_.enable_thread_safety_ = enable;
        return *this;
    }

    /**
     * @brief Set the output format for reports
     * @param format Output format to use
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_output_format(
        xsigma::profiler_options::output_format_enum format)
    {
        options_.output_format_ = format;
        return *this;
    }

    /**
     * @brief Set the output file path for reports
     * @param path File path for report output
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_output_file(const std::string& path)
    {
        options_.output_file_path_ = path;
        return *this;
    }

    /**
     * @brief Set maximum number of samples for statistical analysis
     * @param max_samples Maximum number of samples to collect
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_max_samples(size_t max_samples)
    {
        options_.max_samples_ = max_samples;
        return *this;
    }

    /**
     * @brief Enable or disable percentile calculations
     * @param enable true to calculate percentiles, false to disable
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_percentiles(bool enable = true)
    {
        options_.calculate_percentiles_ = enable;
        return *this;
    }

    /**
     * @brief Enable or disable peak memory tracking
     * @param enable true to track peak memory, false to disable
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_peak_memory_tracking(bool enable = true)
    {
        options_.track_peak_memory_ = enable;
        return *this;
    }

    /**
     * @brief Enable or disable memory delta tracking
     * @param enable true to track memory deltas, false to disable
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_memory_deltas(bool enable = true)
    {
        options_.track_memory_deltas_ = enable;
        return *this;
    }

    /**
     * @brief Set the thread pool size for concurrent operations
     * @param size Number of threads in the thread pool
     * @return Reference to this profiler_session_builder for method chaining
     */
    profiler_session_builder& with_thread_pool_size(size_t size)
    {
        options_.thread_pool_size_ = size;
        return *this;
    }

    /**
     * @brief Build the configured profiler session
     * @return Unique pointer to the created profiler session
     */
    std::unique_ptr<xsigma::profiler_session> build()
    {
        return std::make_unique<xsigma::profiler_session>(options_);
    }

private:
    /// Configuration options being built
    xsigma::profiler_options options_;
};

/**
 * @brief RAII profiler scope for automatic timing and memory tracking
 *
 * Provides automatic profiling scope management using RAII principles.
 * When created, starts timing and memory tracking. When destroyed,
 * automatically stops and records the measurements.
 *
 * Thread-safe when used with a thread-safe profiler session.
 */
class XSIGMA_VISIBILITY profiler_scope
{
public:
    /**
     * @brief Construct a new profiler scope
     * @param name Human-readable name for this profiling scope
     * @param session Pointer to the profiler session (uses current session if nullptr)
     */
    XSIGMA_API explicit profiler_scope(
        const std::string& name, xsigma::profiler_session* session = nullptr);

    /**
     * @brief Destructor - automatically stops profiling
     */
    XSIGMA_API ~profiler_scope();

    /**
     * @brief Manually start profiling (called automatically in constructor)
     */
    XSIGMA_API void start();

    /**
     * @brief Manually stop profiling (called automatically in destructor)
     */
    XSIGMA_API void stop();

    /**
     * @brief Get read-only access to the scope data
     * @return Const reference to the profiler scope data
     */
    const xsigma::profiler_scope_data& data() const { return *data_; }

    /// Disable copy constructor to ensure RAII semantics
    profiler_scope(const profiler_scope&) = delete;

    /// Disable copy assignment to ensure RAII semantics
    profiler_scope& operator=(const profiler_scope&) = delete;

    /// Disable move constructor to ensure RAII semantics
    profiler_scope(profiler_scope&&) = delete;

    /// Disable move assignment to ensure RAII semantics
    profiler_scope& operator=(profiler_scope&&) = delete;

private:
    /// Scope data containing timing and memory measurements
    std::unique_ptr<xsigma::profiler_scope_data> data_;

    /// Pointer to the profiler session managing this scope
    xsigma::profiler_session* session_;

    /// Memory stats snapshot captured at scope start (for delta computation)
    xsigma::memory_stats start_memory_stats_;

    /// Whether start_memory_stats_ contains valid data
    bool has_start_memory_stats_ = false;

    /// Flag indicating if profiling has been started
    bool started_ = false;

    /// Flag indicating if profiling has been stopped
    bool stopped_ = false;

    std::unique_ptr<scoped_memory_debug_annotation> memory_annotation_;
};

/**
 * @brief Convenience macro for creating a named profiling scope
 * @param name String literal name for the profiling scope
 *
 * Creates a profiler_scope object that will automatically profile
 * the current code block until the scope ends.
 */
#define XSIGMA_PROFILE_SCOPE(name) \
    XSIGMA_UNUSED xsigma::profiler_scope XSIGMA_ANONYMOUS_VARIABLE(_xsigma_profile_scope_)(name)

/**
 * @brief Convenience macro for profiling the current function
 *
 * Creates a profiler_scope object using the current function name
 * as the scope name. Profiles the entire function execution.
 */
#define XSIGMA_PROFILE_FUNCTION() \
    XSIGMA_UNUSED                 \
    xsigma::profiler_scope XSIGMA_ANONYMOUS_VARIABLE(_xsigma_profile_scope_)(__FUNCTION__)

/**
 * @brief Convenience macro for profiling a specific code block
 * @param name String literal name for the profiling scope
 *
 * Creates a profiler_scope that profiles only the code within
 * the immediately following block or statement.
 */
#define XSIGMA_PROFILE_BLOCK(name)                                                              \
    if (XSIGMA_UNUSED xsigma::profiler_scope XSIGMA_ANONYMOUS_VARIABLE(_xsigma_profile_scope_)( \
            name);                                                                              \
        true)

}  // namespace xsigma
