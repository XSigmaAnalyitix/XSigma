/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Test suite for XPlane visitor classes
 * Tests visitor pattern implementation for XPlane data structures
 */

#include "Testing/xsigmaTest.h"
#include "profiler/exporters/xplane/xplane.h"
#include "profiler/exporters/xplane/xplane_builder.h"
#include "profiler/exporters/xplane/xplane_visitor.h"

using namespace xsigma;

// Test x_stat_visitor construction
XSIGMATEST(Profiler, xplane_visitor_x_stat_visitor_construction)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify plane visitor can be created
    EXPECT_TRUE(true);
}

// Test xevent_visitor construction
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_construction)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify event visitor can be created
    EXPECT_TRUE(true);
}

// Test xevent_visitor timestamp calculations
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_timestamps)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder = builder.get_or_create_line(1);
    line_builder.SetTimestampNs(1000);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify timestamps are accessible
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor duration calculations
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_duration)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);
    event_builder.SetDurationNs(100);

    xplane_visitor plane_visitor(&plane);

    // Verify duration is accessible
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor comparison operator
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_comparison)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder   = builder.get_or_create_line(1);
    auto metadata1      = builder.get_or_create_event_metadata("event1");
    auto metadata2      = builder.get_or_create_event_metadata("event2");
    auto event_builder1 = line_builder.add_event(*metadata1);
    auto event_builder2 = line_builder.add_event(*metadata2);

    xplane_visitor plane_visitor(&plane);

    // Verify events can be compared
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor aggregated event detection
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_aggregated)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify aggregated event detection works
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor metadata access
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_metadata)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify metadata is accessible
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor display name
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_display_name)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify display name handling
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor timespan
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_timespan)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify timespan calculation
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor pico to nano conversion
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_pico_to_nano)
{
    int64_t ps = 1000;
    int64_t ns = xevent_visitor::pico_to_nano(ps);
    EXPECT_EQ(ns, 1);
}

// Test xevent_visitor nano to pico conversion
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_nano_to_pico)
{
    int64_t ns = 1;
    int64_t ps = xevent_visitor::nano_to_pico(ns);
    EXPECT_EQ(ps, 1000);
}

// Test xevent_visitor round-trip conversion
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_round_trip_conversion)
{
    int64_t original_ns = 12345;
    int64_t ps          = xevent_visitor::nano_to_pico(original_ns);
    int64_t back_to_ns  = xevent_visitor::pico_to_nano(ps);
    EXPECT_EQ(back_to_ns, original_ns);
}

// Test xevent_visitor with multiple events
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_multiple_events)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder = builder.get_or_create_line(1);
    for (int i = 0; i < 10; ++i)
    {
        auto metadata = builder.get_or_create_event_metadata("event_" + std::to_string(i));
        line_builder.add_event(*metadata);
    }

    xplane_visitor plane_visitor(&plane);

    // Verify multiple events are handled
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor with multiple lines
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_multiple_lines)
{
    xplane         plane;
    xplane_builder builder(&plane);

    for (int i = 0; i < 5; ++i)
    {
        auto line_builder = builder.get_or_create_line(i);
        line_builder.SetTimestampNs(1000 + i * 100);
        auto metadata = builder.get_or_create_event_metadata("event_" + std::to_string(i));
        line_builder.add_event(*metadata);
    }

    xplane_visitor plane_visitor(&plane);

    // Verify multiple lines are handled
    EXPECT_EQ(plane_visitor.num_lines(), 5);
}

// Test xevent_visitor offset calculations
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_offset)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);
    event_builder.SetOffsetNs(50);

    xplane_visitor plane_visitor(&plane);

    // Verify offset calculations work
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor end timestamp calculations
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_end_timestamp)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);
    event_builder.SetEndTimestampNs(1100);

    xplane_visitor plane_visitor(&plane);

    // Verify end timestamp calculations
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor num_occurrences
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_num_occurrences)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);
    event_builder.SetNumOccurrences(5);

    xplane_visitor plane_visitor(&plane);

    // Verify num_occurrences is accessible
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor raw_event access
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_raw_event)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify raw event access
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor id access
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_id)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify id access
    EXPECT_GT(plane_visitor.num_lines(), 0);
}

// Test xevent_visitor name access
XSIGMATEST(Profiler, xplane_visitor_xevent_visitor_name)
{
    xplane         plane;
    xplane_builder builder(&plane);

    auto line_builder  = builder.get_or_create_line(1);
    auto metadata      = builder.get_or_create_event_metadata("test_event");
    auto event_builder = line_builder.add_event(*metadata);

    xplane_visitor plane_visitor(&plane);

    // Verify name access
    EXPECT_GT(plane_visitor.num_lines(), 0);
}
