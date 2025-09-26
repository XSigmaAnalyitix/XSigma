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

#include <atomic>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common/macros.h"

namespace xsigma
{
namespace internal
{

// Current trace level.
// Static atomic so trace_me_recorder::active can be fast and non-blocking.
// Modified by trace_me_recorder singleton when tracing starts/stops.
XSIGMA_API extern std::atomic<int> g_trace_level;

}  // namespace internal

// trace_me_recorder is a singleton repository of trace_me events.
// It can be safely and cheaply appended to by multiple threads.
//
// start() and stop() must be called in pairs, stop() returns the events added
// since the previous start().
//
// This is the backend for trace_me instrumentation.
// The profiler starts the recorder, the trace_me destructor records complete
// events. trace_me::activity_start records start events, and trace_me::activity_end
// records end events. The profiler then stops the recorder and finds start/end
// pairs. (Unpaired start/end events are discarded at that point).
class trace_me_recorder
{
public:
    // An Event is either the start of a trace_me, the end of a trace_me, or both.
    // Times are in ns since the Unix epoch.
    // A negative time encodes the activity_id used to pair up the start of an
    // event with its end.
    struct Event
    {
        bool is_complete() const { return start_time > 0 && end_time > 0; }
        bool is_start() const { return end_time < 0; }
        bool is_end() const { return start_time < 0; }

        int64_t activity_id() const
        {
            if (is_start())
                return -end_time;
            if (is_end())
                return -start_time;
            return 1;  // complete
        }

        std::string name;
        int64_t     start_time;
        int64_t     end_time;
    };
    struct ThreadInfo
    {
        uint32_t    tid;
        std::string name;
    };
    struct ThreadEvents
    {
        ThreadInfo        thread;
        std::deque<Event> events;
    };
    using Events = std::vector<ThreadEvents>;

    // Starts recording of trace_me().
    // Only traces <= level will be recorded.
    // Level must be >= 0. If level is 0, no traces will be recorded.
    static bool start(int level);

    // Stops recording and returns events recorded since start().
    // Events passed to record after stop has started will be dropped.
    static Events stop();

    // Returns whether we're currently recording. Racy, but cheap!
    static inline bool active(int level = 1)
    {
        return internal::g_trace_level.load(std::memory_order_acquire) >= level;
    }

    // Default value for trace_level_ when tracing is disabled
    static constexpr int kTracingDisabled = -1;

    // Records an event. Non-blocking.
    static void record(Event&& event);

    // Returns an activity_id for trace_me::activity_start.
    static int64_t new_activity_id();

private:
    trace_me_recorder()  = delete;
    ~trace_me_recorder() = delete;

    // Clears events from all active threads that were added due to record
    // racing with stop.
    static void clear();

    // Gathers events from all active threads, and clears their buffers.
    static Events consume();
};

}  // namespace xsigma