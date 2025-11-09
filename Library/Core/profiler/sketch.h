/*
 * XSigma Profiler Forward Declarations Sketch
 *
 * This header provides forward declarations for all classes used in the profiler
 * module. It serves as a quick reference for class hierarchies and dependencies.
 *
 * COMPONENT CLASSIFICATION: UTILITY
 * This is a utility header for forward declarations and documentation purposes.
 */

#pragma once

namespace xsigma
{

// ============================================================================
// == Core Profiler Classes ==================================================
// ============================================================================

/**
 * @brief Main profiler session manager
 *
 * Singleton class managing profiler lifecycle and event collection.
 * Thread-safe for concurrent profiling from multiple threads.
 *
 * Location: profiler/profiler_api.h
 * Namespace: xsigma::profiler
 */
class profiler_session;

/**
 * @brief RAII guard for automatic profiler lifecycle management
 *
 * Starts profiler on construction, stops on destruction.
 * Ensures proper cleanup even if exceptions occur.
 *
 * Location: profiler/profiler_guard.h
 * Namespace: xsigma::profiler
 */
class profiler_guard;

/**
 * @brief RAII guard for recording a function/scope
 *
 * Records entry and exit of a function or code block.
 * Automatically annotates with ITT if available.
 *
 * Location: profiler/profiler_guard.h
 * Namespace: xsigma::profiler
 */
class record_function;

/**
 * @brief RAII guard for recording a named activity
 *
 * Records a named activity with automatic start/stop.
 * Integrates with both Kineto and ITT.
 *
 * Location: profiler/profiler_guard.h
 * Namespace: xsigma::profiler
 */
class scoped_activity;

// ============================================================================
// == Profiler Interface Classes =============================================
// ============================================================================

/**
 * @brief Interface for XSigma profiler plugins
 *
 * Base interface that all profiler implementations must inherit from.
 * Defines the contract for profiler lifecycle management.
 *
 * Location: profiler/native/core/profiler_interface.h
 * Namespace: xsigma
 */
class profiler_interface;

/**
 * @brief Decorator for XSigma profiler plugins
 *
 * Tracks that calls to the underlying profiler interface functions are made
 * in the expected order: start, stop and collect_data.
 *
 * Location: profiler/native/core/profiler_controller.h
 * Namespace: xsigma
 */
class profiler_controller;

/**
 * @brief Multiplexes profiler_interface calls into a collection of profilers
 *
 * Allows multiple profilers to run concurrently, forwarding calls to all.
 *
 * Location: profiler/native/core/profiler_collection.h
 * Namespace: xsigma
 */
class profiler_collection;

// ============================================================================
// == Configuration Classes ==================================================
// ============================================================================

/**
 * @brief Configuration options for profiling sessions
 *
 * Contains all configuration parameters needed to set up and control
 * profiling behavior across different device types and profiling modes.
 *
 * Location: profiler/native/core/profiler_options.h
 * Namespace: xsigma
 */
class profile_options;

/**
 * @brief Options for remote profiler session management
 *
 * Contains configuration for managing remote profiling sessions across
 * multiple service addresses with timing and duration controls.
 *
 * Location: profiler/native/core/profiler_options.h
 * Namespace: xsigma
 */
class remote_profiler_session_manager_options;

/**
 * @brief Profiler configuration structure
 *
 * Configuration for profiler session including activities, output format,
 * and other profiling parameters.
 *
 * Location: profiler/profiler_api.h
 * Namespace: xsigma::profiler
 */
struct profiler_config;

// ============================================================================
// == Status and Result Classes ==============================================
// ============================================================================

/**
 * @brief Profiler status result type
 *
 * Represents the result of profiler operations with success/failure status
 * and optional error messages.
 *
 * Location: profiler/native/core/profiler_status.h
 * Namespace: xsigma
 */
class profiler_status;

// ============================================================================
// == Utility Classes ========================================================
// ============================================================================

/**
 * @brief Timespan representation for profiling events
 *
 * Represents a time extent of an event: a pair of (begin, duration).
 * Events may have duration 0 ("instant events") but duration can't be negative.
 *
 * Location: profiler/native/core/timespan.h
 * Namespace: xsigma
 */
class timespan;

/**
 * @brief Handle for the profiler lock
 *
 * At most one instance of this class, the "active" instance, owns the
 * profiler lock. Ensures only one profiling session is active at a time.
 *
 * Location: profiler/native/core/profiler_lock.h
 * Namespace: xsigma
 */
class ProfilerLock;

// ============================================================================
// == Enumerations ===========================================================
// ============================================================================

/**
 * @brief Types of activities that can be profiled
 *
 * Enumeration of supported profiling activity types:
 * - CPU: CPU operations
 * - CUDA: NVIDIA CUDA operations
 * - ROCM: AMD ROCm operations
 * - XPU: Intel XPU operations
 * - Memory: Memory allocations/deallocations
 *
 * Location: profiler/profiler_api.h
 * Namespace: xsigma::profiler
 */
enum class activity_type_enum;

/**
 * @brief Profiler operational state
 *
 * Enumeration of profiler states:
 * - Disabled: Profiler not running
 * - Ready: Profiler configured, ready to start
 * - Recording: Profiler actively recording
 *
 * Location: profiler/profiler_api.h
 * Namespace: xsigma::profiler
 */
enum class profiler_state_enum;

// ============================================================================
// == ITT API Wrapper Classes ================================================
// ============================================================================

/**
 * @brief ITT API wrapper functions
 *
 * Provides C++ wrapper functions for Intel Instrumentation and Tracing
 * Technology (ITT) API, aligned with XSigma's implementation.
 *
 * Location: profiler/itt_wrapper.h
 * Namespace: xsigma::profiler
 *
 * Functions:
 * - itt_init(): Initialize ITT API
 * - itt_range_push(name): Push a named range onto the ITT stack
 * - itt_range_pop(): Pop the current range from the ITT stack
 * - itt_mark(name): Mark an event at the current time
 * - itt_get_domain(): Get the global ITT domain
 */

// ============================================================================
// == Kineto Shim Classes ====================================================
// ============================================================================

/**
 * @brief Kineto profiling library wrapper functions
 *
 * Provides direct access to XSigma Kineto profiling library,
 * aligned with XSigma's implementation for feature parity.
 *
 * Location: profiler/kineto_shim.h
 * Namespace: xsigma::profiler
 *
 * Functions:
 * - kineto_init(cpu_only, log_on_error): Initialize Kineto profiling library
 * - kineto_is_profiler_registered(): Check if Kineto profiler is registered
 * - kineto_is_profiler_initialized(): Check if Kineto profiler is initialized
 * - kineto_prepare_trace(activities, config_str): Prepare trace with specified activity types
 * - kineto_start_trace(): Start profiling trace
 * - kineto_stop_trace(): Stop profiling trace and return trace interface
 * - kineto_reset_tls(): Reset Kineto thread-local state
 */

// ============================================================================
// == XSpace and Export Classes ==============================================
// ============================================================================

/**
 * @brief XSpace timeline data structure
 *
 * Represents the collected profiling data in XSpace format.
 * Used for storing and exporting profiling results.
 *
 * Location: profiler/native/exporters/xplane/xplane.h
 * Namespace: xsigma
 */
class x_space;

}  // namespace xsigma
