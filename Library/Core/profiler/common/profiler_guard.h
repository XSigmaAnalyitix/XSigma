/*
 * XSigma Profiler RAII Guards
 *
 * RAII wrappers for automatic profiler lifecycle management.
 * Ensures profiler is properly started and stopped.
 *
 * Features:
 * - Automatic profiler start on construction
 * - Automatic profiler stop on destruction
 * - Exception-safe operation
 * - Scope-based function recording
 *
 * Usage:
 *   {
 *       xsigma::profiler::profiler_config config;
 *       config.activities = {xsigma::profiler::activity_type_enum::CPU};
 *       xsigma::profiler::profiler_guard guard(config);
 *       // Profiling active here
 *       // ... code to profile ...
 *   } // Profiling stops automatically
 */

#pragma once

#include <string>

#include "common/export.h"
#include "profiler_api.h"

namespace xsigma
{
namespace profiler
{

// ============================================================================
// == Profiler Guard (RAII) ===================================================
// ============================================================================

/**
 * @brief RAII guard for automatic profiler lifecycle management
 *
 * Starts profiler on construction, stops on destruction.
 * Ensures proper cleanup even if exceptions occur.
 */
class XSIGMA_VISIBILITY profiler_guard
{
public:
    /**
   * @brief Construct and start profiler
   *
   * @param config Profiler configuration
   */
    explicit profiler_guard(const profiler_config& config);

    /**
   * @brief Destruct and stop profiler
   */
    ~profiler_guard();

    /**
   * @brief Check if profiler is active
   *
   * @return true if profiler started successfully
   */
    bool is_active() const { return was_profiling_; }

    /**
   * @brief Export trace to file
   *
   * @param path Output file path
   * @return true if export successful
   */
    static bool export_trace(const std::string& path);

private:
    bool            was_profiling_ = false;
    profiler_config config_;
};

// ============================================================================
// == Record Function (Scope-based Recording) ================================
// ============================================================================

/**
 * @brief RAII guard for recording a function/scope
 *
 * Records entry and exit of a function or code block.
 * Automatically annotates with ITT if available.
 *
 * Usage:
 *   void my_function() {
 *       xsigma::profiler::record_function record("my_function");
 *       // ... function code ...
 *   } // Automatically recorded
 */
class XSIGMA_VISIBILITY record_function
{
public:
    /**
   * @brief Construct and record function entry
   *
   * @param name Function name (must be string literal or persistent)
   */
    explicit record_function(const char* name);

    /**
   * @brief Destruct and record function exit
   */
    ~record_function();

    /**
   * @brief Get function name
   *
   * @return Function name
   */
    const char* name() const { return name_; }

    /**
   * @brief Get start time in nanoseconds
   *
   * @return Start time
   */
    uint64_t start_time_ns() const { return start_time_ns_; }

    /**
   * @brief Get end time in nanoseconds
   *
   * @return End time (0 if still recording)
   */
    uint64_t end_time_ns() const { return end_time_ns_; }

private:
    const char* name_;
    uint64_t    start_time_ns_;
    uint64_t    end_time_ns_ = 0;
};

// ============================================================================
// == Scoped Activity Recording ===============================================
// ============================================================================

/**
 * @brief RAII guard for recording a named activity
 *
 * Records a named activity with automatic start/stop.
 * Integrates with both Kineto and ITT.
 *
 * Usage:
 *   {
 *       xsigma::profiler::scoped_activity activity("matrix_multiply");
 *       // ... code to profile ...
 *   } // Activity automatically recorded
 */
class XSIGMA_VISIBILITY scoped_activity
{
public:
    /**
   * @brief Construct and start activity
   *
   * @param name Activity name
   */
    explicit scoped_activity(const char* name);

    /**
   * @brief Destruct and stop activity
   */
    ~scoped_activity();

    /**
   * @brief Get activity name
   *
   * @return Activity name
   */
    const char* name() const { return name_; }

private:
    const char* name_;
};

}  // namespace profiler
}  // namespace xsigma
