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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "experimental/profiler/no_init.h"
#include "experimental/profiler/traceme_encode.h"
#include "experimental/profiler/traceme_encode.h"  // IWYU pragma: export
#include "logging/logger.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "experimental/profiler/traceme_recorder.h"
//#include "experimental/profiler/time_utils.h"
#endif

namespace xsigma
{

XSIGMA_FORCE_INLINE int64_t get_current_time_nanos()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return nanos;
}

// Predefined levels:
// - Level 1 (CRITICAL) is the default and used only for user instrumentation.
// - Level 2 (INFO) is used by profiler for instrumenting high level program
//   execution details (expensive TF ops, XLA ops, etc).
// - Level 3 (VERBOSE) is also used by profiler to instrument more verbose
//   (low-level) program execution details (cheap TF ops, etc).
enum class trace_me_level_enum
{
    CRITICAL = 1,
    INFO     = 2,
    VERBOSE  = 3,
};

// This is specifically used for instrumenting Tensorflow ops.
// Takes input as whether a TF op is expensive or not and returns the trace_me
// level to be assigned to trace that particular op. Assigns level 2 for
// expensive ops (these are high-level details and shown by default in profiler
// UI). Assigns level 3 for cheap ops (low-level details not shown by default).
inline int get_tf_trace_me_level(bool is_expensive)
{
    return is_expensive ? static_cast<int>(trace_me_level_enum::INFO)
                        : static_cast<int>(trace_me_level_enum::VERBOSE);
}

// This class permits user-specified (CPU) tracing activities. A trace activity
// is started when an object of this class is created and stopped when the
// object is destroyed.
//
// CPU tracing can be useful when trying to understand what parts of GPU
// computation (e.g., kernels and memcpy) correspond to higher level activities
// in the overall program. For instance, a collection of kernels maybe
// performing one "step" of a program that is better visualized together than
// interspersed with kernels from other "steps". Therefore, a trace_me object
// can be created at each "step".
//
// Two APIs are provided:
//   (1) Scoped object: a trace_me object starts tracing on construction, and
//       stops tracing when it goes out of scope.
//          {
//            trace_me trace("step");
//            ... do some work ...
//          }
//       trace_me objects can be members of a class, or allocated on the heap.
//   (2) Static methods: activity_start and activity_end may be called in pairs.
//          auto id = activity_start("step");
//          ... do some work ...
//          activity_end(id);
//       The two static methods should be called within the same thread.
class trace_me
{
public:
    // Constructor that traces a user-defined activity labeled with name
    // in the UI. Level defines the trace priority, used for filtering trace_me
    // events. By default, traces with trace_me level <= 2 are recorded. Levels:
    // - Must be a positive integer.
    // - Can be a value in enum trace_me_level_enum.
    // Users are welcome to use level > 3 in their code, if they wish to filter
    // out their host traces based on verbosity.
    explicit trace_me(std::string_view name, int level = 1)
    {
        XSIGMA_CHECK_DEBUG(level >= 1, "level is less than 1");
#if !defined(IS_MOBILE_PLATFORM)
        if XSIGMA_UNLIKELY (trace_me_recorder::active(level))
        {
            name_.Emplace(std::string(name));
            start_time_ = get_current_time_nanos();
        }
#endif
    }

    // Do not allow passing a temporary string as the overhead of generating that
    // string should only be incurred when tracing is enabled. Wrap the temporary
    // string generation (e.g., StrCat) in a lambda and use the name_generator
    // template instead.
    explicit trace_me(std::string&& name, int level = 1) = delete;

    // Do not allow passing strings by reference or value since the caller
    // may unintentionally maintain ownership of the name.
    // Explicitly wrap the name in a string_view if you really wish to maintain
    // ownership of a string already generated for other purposes. For temporary
    // strings (e.g., result of StrCat) use the name_generator template.
    explicit trace_me(const std::string& name, int level = 1) = delete;

    // This overload is necessary to make trace_me's with string literals work.
    // Otherwise, the name_generator template would be used.
    explicit trace_me(const char* raw, int level = 1) : trace_me(std::string_view(raw), level) {}

    // This overload only generates the name (and possibly metadata) if tracing is
    // enabled. Useful for avoiding expensive operations (e.g., string
    // concatenation) when tracing is disabled.
    // name_generator may be a lambda or functor that returns a type that the
    // string() constructor can take, e.g., the result of TraceMeEncode.
    // name_generator is templated, rather than a std::function to avoid
    // allocations std::function might make even if never called.
    // Example Usage:
    //   TraceMe trace_me([&]() {
    //     return StrCat("my_trace", id);
    //   }
    //   TraceMe op_trace_me([&]() {
    //     return TraceMeOp(op_name, op_type);
    //   }
    //   TraceMe trace_me_with_metadata([&value1]() {
    //     return TraceMeEncode("my_trace", {{"key1", value1}, {"key2", 42}});
    //   });
    template <
        typename NameGeneratorT,
        std::enable_if_t<std::is_invocable_v<NameGeneratorT>, bool> = true>
    explicit trace_me(NameGeneratorT&& name_generator, int level = 1)
    {
        XSIGMA_CHECK_DEBUG(level >= 1, "level is less than 1");
#if !defined(IS_MOBILE_PLATFORM)
        if XSIGMA_UNLIKELY (trace_me_recorder::active(level))
        {
            name_.Emplace(std::forward<NameGeneratorT>(name_generator)());
            start_time_ = get_current_time_nanos();
        }
#endif
    }

    // Movable.
    trace_me(trace_me&& other) noexcept { *this = std::move(other); }
    trace_me& operator=(trace_me&& other) noexcept
    {
#if !defined(IS_MOBILE_PLATFORM)
        if XSIGMA_UNLIKELY (other.start_time_ != kUntracedActivity)
        {
            name_.Emplace(std::move(other.name_).Consume());
            start_time_ = std::exchange(other.start_time_, kUntracedActivity);
        }
#endif
        return *this;
    }

    ~trace_me() { stop(); }

    // Stop tracing the activity. Called by the destructor, but exposed to allow
    // stopping tracing before the object goes out of scope. Only has an effect
    // the first time it is called.
    void stop()
    {
        // We do not need to check the trace level again here.
        // - If tracing wasn't active to start with, we have kUntracedActivity.
        // - If tracing was active and was stopped, we have
        //   trace_me_recorder::active().
        // - If tracing was active and was restarted at a lower level, we may
        //   spuriously record the event. This is extremely rare, and acceptable as
        //   event will be discarded when its start timestamp fall outside of the
        //   start/stop session timestamp.
#if !defined(IS_MOBILE_PLATFORM)
        if XSIGMA_UNLIKELY (start_time_ != kUntracedActivity)
        {
            if XSIGMA_LIKELY (trace_me_recorder::active())
            {
                trace_me_recorder::record(
                    {std::move(name_.value), start_time_, get_current_time_nanos()});
            }
            name_.Destroy();
            start_time_ = kUntracedActivity;
        }
#endif
    }

    // Appends new_metadata to the TraceMe name passed to the constructor.
    // metadata_generator may be a lambda or functor that returns a type that the
    // string() constructor can take, e.g., the result of TraceMeEncode.
    // metadata_generator is only evaluated when tracing is enabled.
    // metadata_generator is templated, rather than a std::function to avoid
    // allocations std::function might make even if never called.
    // Example Usage:
    //   trace_me.append_metadata([&value1]() {
    //     return trace_me_encode({{"key1", value1}, {"key2", 42}});
    //   });
    template <
        typename MetadataGeneratorT,
        std::enable_if_t<std::is_invocable_v<MetadataGeneratorT>, bool> = true>
    void append_metadata(MetadataGeneratorT&& metadata_generator)
    {
#if !defined(IS_MOBILE_PLATFORM)
        if XSIGMA_UNLIKELY (start_time_ != kUntracedActivity)
        {
            if XSIGMA_LIKELY (trace_me_recorder::active())
            {
                traceme_internal::append_metadata(
                    &name_.value, std::forward<MetadataGeneratorT>(metadata_generator)());
            }
        }
#endif
    }

    // Static API, for use when scoped objects are inconvenient.

    // Record the start time of an activity.
    // Returns the activity ID, which is used to stop the activity.
    // Calls `name_generator` to get the name for activity.
    template <
        typename NameGeneratorT,
        std::enable_if_t<std::is_invocable_v<NameGeneratorT>, bool> = true>
    static int64_t activity_start(NameGeneratorT&& name_generator, int level = 1)
    {
#if !defined(IS_MOBILE_PLATFORM)
        if XSIGMA_UNLIKELY (trace_me_recorder::active(level))
        {
            int64_t activity_id = trace_me_recorder::new_activity_id();
            trace_me_recorder::record(
                {std::forward<NameGeneratorT>(name_generator)(),
                 get_current_time_nanos(),
                 -activity_id});
            return activity_id;
        }
#endif
        return kUntracedActivity;
    }

    // Record the start time of an activity.
    // Returns the activity ID, which is used to stop the activity.
    static int64_t activity_start(std::string_view name, int level = 1)
    {
#if !defined(IS_MOBILE_PLATFORM)
        if XSIGMA_UNLIKELY (trace_me_recorder::active(level))
        {
            int64_t activity_id = trace_me_recorder::new_activity_id();
            trace_me_recorder::record({std::string(name), get_current_time_nanos(), -activity_id});
            return activity_id;
        }
#endif
        return kUntracedActivity;
    }

    // Same as activity_start above, an overload for "const std::string&"
    static int64_t activity_start(const std::string& name, int level = 1)
    {
        return activity_start(std::string_view(name), level);
    }

    // Same as activity_start above, an overload for "const char*"
    static int64_t activity_start(const char* name, int level = 1)
    {
        return activity_start(std::string_view(name), level);
    }

    // Record the end time of an activity started by activity_start().
    static void activity_end(int64_t activity_id)
    {
#if !defined(IS_MOBILE_PLATFORM)
        // We don't check the level again (see trace_me::stop()).
        if XSIGMA_UNLIKELY (activity_id != kUntracedActivity)
        {
            if XSIGMA_LIKELY (trace_me_recorder::active())
            {
                trace_me_recorder::record({std::string(), -activity_id, get_current_time_nanos()});
            }
        }
#endif
    }

    // Records the time of an instant activity.
    template <
        typename NameGeneratorT,
        std::enable_if_t<std::is_invocable_v<NameGeneratorT>, bool> = true>
    static void instant_activity(NameGeneratorT&& name_generator, int level = 1)
    {
#if !defined(IS_MOBILE_PLATFORM)
        if XSIGMA_UNLIKELY (trace_me_recorder::active(level))
        {
            int64_t now = get_current_time_nanos();
            trace_me_recorder::record(
                {std::forward<NameGeneratorT>(name_generator)(),
                 /*start_time=*/now,
                 /*end_time=*/now});
        }
#endif
    }

    static bool active(int level = 1)
    {
#if !defined(IS_MOBILE_PLATFORM)
        return trace_me_recorder::active(level);
#else
        return false;
#endif
    }

    static int64_t new_activity_id()
    {
#if !defined(IS_MOBILE_PLATFORM)
        return trace_me_recorder::new_activity_id();
#else
        return 0;
#endif
    }

private:
    // Start time used when tracing is disabled.
    constexpr static int64_t kUntracedActivity = 0;

    trace_me(const trace_me&)       = delete;
    void operator=(const trace_me&) = delete;

    no_init<std::string> name_;

    int64_t start_time_ = kUntracedActivity;
};

// Whether OpKernel::TraceString will populate additional information for
// profiler, such as tensor shapes.
inline bool tf_op_details_enabled()
{
    return trace_me::active(static_cast<int>(trace_me_level_enum::VERBOSE));
}

}  // namespace xsigma
