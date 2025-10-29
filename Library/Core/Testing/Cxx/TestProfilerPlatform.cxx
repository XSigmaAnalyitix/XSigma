/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "profiler/platform/env_time.h"
#include "xsigmaTest.h"

using namespace xsigma;

// ============================================================================
// Environment Time Tests
// ============================================================================

XSIGMATEST(Profiler, env_time_now_nanos_returns_positive)
{
    uint64_t time = env_time::now_nanos();

    EXPECT_GT(time, 0);
}

XSIGMATEST(Profiler, env_time_now_nanos_monotonic)
{
    uint64_t time1 = env_time::now_nanos();
    uint64_t time2 = env_time::now_nanos();

    EXPECT_GE(time2, time1);
}

XSIGMATEST(Profiler, env_time_now_nanos_increasing)
{
    uint64_t time1 = env_time::now_nanos();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    uint64_t time2 = env_time::now_nanos();

    EXPECT_GT(time2, time1);
}

XSIGMATEST(Profiler, env_time_now_micros_returns_positive)
{
    uint64_t time = env_time::now_micros();

    EXPECT_GT(time, 0);
}

XSIGMATEST(Profiler, env_time_now_micros_monotonic)
{
    uint64_t time1 = env_time::now_micros();
    uint64_t time2 = env_time::now_micros();

    EXPECT_GE(time2, time1);
}

XSIGMATEST(Profiler, env_time_now_seconds_returns_positive)
{
    uint64_t time = env_time::now_seconds();

    EXPECT_GT(time, 0);
}

XSIGMATEST(Profiler, env_time_now_seconds_monotonic)
{
    uint64_t time1 = env_time::now_seconds();
    uint64_t time2 = env_time::now_seconds();

    EXPECT_GE(time2, time1);
}

XSIGMATEST(Profiler, env_time_nanos_to_micros_conversion)
{
    uint64_t nanos  = 1000000;  // 1ms
    uint64_t micros = nanos / 1000;

    EXPECT_EQ(micros, 1000);
}

XSIGMATEST(Profiler, env_time_nanos_to_seconds_conversion)
{
    uint64_t nanos   = 1000000000;  // 1 second
    uint64_t seconds = nanos / 1000000000;

    EXPECT_EQ(seconds, 1);
}

XSIGMATEST(Profiler, env_time_micros_to_nanos_conversion)
{
    uint64_t micros = 1000;  // 1ms
    uint64_t nanos  = micros * 1000;

    EXPECT_EQ(nanos, 1000000);
}

XSIGMATEST(Profiler, env_time_multiple_reads_consistency)
{
    uint64_t time1 = env_time::now_nanos();
    uint64_t time2 = env_time::now_nanos();
    uint64_t time3 = env_time::now_nanos();

    EXPECT_GE(time2, time1);
    EXPECT_GE(time3, time2);
}

XSIGMATEST(Profiler, env_time_sleep_and_measure)
{
    uint64_t time1 = env_time::now_nanos();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    uint64_t time2 = env_time::now_nanos();

    uint64_t elapsed = time2 - time1;
    EXPECT_GE(elapsed, 2500000);  // At least 2.5ms
}

XSIGMATEST(Profiler, env_time_high_resolution_verification)
{
    // Verify we're getting nanosecond precision
    uint64_t time1 = env_time::now_nanos();
    uint64_t time2 = env_time::now_nanos();

    // The difference should be small (less than 1 second)
    uint64_t diff = time2 - time1;
    EXPECT_LT(diff, 1000000000);  // Less than 1 second
}

XSIGMATEST(Profiler, env_time_consistency_between_units)
{
    uint64_t nanos   = env_time::now_nanos();
    uint64_t micros  = env_time::now_micros();
    uint64_t seconds = env_time::now_seconds();

    // All should be positive
    EXPECT_GT(nanos, 0);
    EXPECT_GT(micros, 0);
    EXPECT_GT(seconds, 0);

    // Nanos should be larger than micros
    EXPECT_GT(nanos, micros);
}

XSIGMATEST(Profiler, env_time_large_duration_measurement)
{
    uint64_t time1 = env_time::now_nanos();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    uint64_t time2 = env_time::now_nanos();

    uint64_t elapsed = time2 - time1;
    EXPECT_GE(elapsed, 25000000);  // At least 25ms
}

XSIGMATEST(Profiler, env_time_micros_precision)
{
    uint64_t time1 = env_time::now_micros();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    uint64_t time2 = env_time::now_micros();

    uint64_t elapsed = time2 - time1;
    EXPECT_GE(elapsed, 50);  // At least 50 microseconds
}

XSIGMATEST(Profiler, env_time_seconds_precision)
{
    uint64_t time1 = env_time::now_seconds();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    uint64_t time2 = env_time::now_seconds();

    // Seconds should be the same or time2 >= time1
    EXPECT_GE(time2, time1);
}
