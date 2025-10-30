/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "profiler/exporters/chrome_trace_exporter.h"
#include "profiler/exporters/xplane/xplane_builder.h"
#include "xsigmaTest.h"

using namespace xsigma;
using namespace xsigma::profiler;

// ============================================================================
// Chrome Trace Exporter Tests
// ============================================================================

XSIGMATEST(Profiler, chrome_trace_export_empty_space)
{
    x_space     space;
    std::string json = export_to_chrome_trace_json(space);

    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("traceEvents"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_with_single_plane)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_NE(json.find("CPU"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_with_events)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(100);
    line->set_name("Thread-1");
    line->set_timestamp_ns(1000000);

    auto* event = line->add_events();
    event->set_offset_ps(0);
    event->set_duration_ps(500000);
    event->set_metadata_id(1);

    auto* metadata = plane->mutable_event_metadata();
    (*metadata)[1].set_name("compute");

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_NE(json.find("compute"), std::string::npos);
    EXPECT_NE(json.find("\"ph\":\"X\""), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_pretty_print)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    std::string json = export_to_chrome_trace_json(space, true);

    EXPECT_FALSE(json.empty());
    // Pretty printed JSON should have newlines
    EXPECT_NE(json.find("\n"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_json_escaping)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(100);
    line->set_name("Thread-1");

    auto* event = line->add_events();
    event->set_offset_ps(0);
    event->set_duration_ps(500000);
    event->set_metadata_id(1);

    auto* metadata = plane->mutable_event_metadata();
    (*metadata)[1].set_name("test\"quote");

    std::string json = export_to_chrome_trace_json(space);

    // Should escape quotes
    EXPECT_NE(json.find("\\\""), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_multiple_threads)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    for (int i = 0; i < 3; ++i)
    {
        auto* line = plane->add_lines();
        line->set_id(100 + i);
        line->set_name("Thread-" + std::to_string(i));
    }

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_NE(json.find("Thread-0"), std::string::npos);
    EXPECT_NE(json.find("Thread-1"), std::string::npos);
    EXPECT_NE(json.find("Thread-2"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_with_stats)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(100);
    line->set_name("Thread-1");

    auto* event = line->add_events();
    event->set_offset_ps(0);
    event->set_duration_ps(500000);
    event->set_metadata_id(1);

    auto* stat = event->add_stats();
    stat->set_value(int64_t(42));

    auto* metadata = plane->mutable_event_metadata();
    (*metadata)[1].set_name("compute");

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_NE(json.find("compute"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_multiple_planes)
{
    x_space space;

    for (int i = 0; i < 2; ++i)
    {
        auto* plane = space.add_planes();
        plane->set_id(i + 1);
        plane->set_name("Plane-" + std::to_string(i));
    }

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_NE(json.find("Plane-0"), std::string::npos);
    EXPECT_NE(json.find("Plane-1"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_large_timestamps)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(100);
    line->set_name("Thread-1");
    line->set_timestamp_ns(9223372036854775000LL);  // Large timestamp

    auto* event = line->add_events();
    event->set_offset_ps(1000000000000LL);
    event->set_duration_ps(500000000000LL);
    event->set_metadata_id(1);

    auto* metadata = plane->mutable_event_metadata();
    (*metadata)[1].set_name("compute");

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("compute"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_special_characters_in_names)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(100);
    line->set_name("Thread\\with\\backslash");

    auto* event = line->add_events();
    event->set_offset_ps(0);
    event->set_duration_ps(500000);
    event->set_metadata_id(1);

    auto* metadata = plane->mutable_event_metadata();
    (*metadata)[1].set_name("func\nwith\nnewlines");

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_FALSE(json.empty());
    // Should properly escape special characters
    EXPECT_NE(json.find("\\\\"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_zero_duration_events)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(100);
    line->set_name("Thread-1");

    auto* event = line->add_events();
    event->set_offset_ps(0);
    event->set_duration_ps(0);  // Zero duration
    event->set_metadata_id(1);

    auto* metadata = plane->mutable_event_metadata();
    (*metadata)[1].set_name("instant_event");

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_NE(json.find("instant_event"), std::string::npos);
    EXPECT_NE(json.find("\"dur\":0"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_many_events)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(100);
    line->set_name("Thread-1");

    // Add many events
    for (int i = 0; i < 100; ++i)
    {
        auto* event = line->add_events();
        event->set_offset_ps(i * 1000000);
        event->set_duration_ps(500000);
        event->set_metadata_id(1);
    }

    auto* metadata = plane->mutable_event_metadata();
    (*metadata)[1].set_name("event");

    std::string json = export_to_chrome_trace_json(space);

    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("event"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_file_write)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(100);
    line->set_name("Thread-1");

    auto* event = line->add_events();
    event->set_offset_ps(0);
    event->set_duration_ps(500000);
    event->set_metadata_id(1);

    auto* metadata = plane->mutable_event_metadata();
    (*metadata)[1].set_name("compute");

    // Try to write to invalid path (should fail gracefully)
    bool result = export_to_chrome_trace_json_file(space, "/invalid/path/trace.json");
    EXPECT_FALSE(result);
}

XSIGMATEST(Profiler, chrome_trace_export_display_time_unit)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    std::string json = export_to_chrome_trace_json(space);

    // Should contain displayTimeUnit
    EXPECT_NE(json.find("displayTimeUnit"), std::string::npos);
    EXPECT_NE(json.find("\"ns\""), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_process_metadata)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(42);
    plane->set_name("CustomProcess");

    std::string json = export_to_chrome_trace_json(space);

    // Should contain process metadata
    EXPECT_NE(json.find("process_name"), std::string::npos);
    EXPECT_NE(json.find("CustomProcess"), std::string::npos);
}

XSIGMATEST(Profiler, chrome_trace_export_thread_metadata)
{
    x_space space;
    auto*   plane = space.add_planes();
    plane->set_id(1);
    plane->set_name("CPU");

    auto* line = plane->add_lines();
    line->set_id(123);
    line->set_name("CustomThread");

    std::string json = export_to_chrome_trace_json(space);

    // Should contain thread metadata
    EXPECT_NE(json.find("thread_name"), std::string::npos);
    EXPECT_NE(json.find("CustomThread"), std::string::npos);
}
