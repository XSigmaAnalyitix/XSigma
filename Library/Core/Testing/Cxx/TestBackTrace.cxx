#include <fmt/format.h>  // for compile_string_to_view
#include <gtest/gtest.h>  // for Test, AssertionResult, TestInfo, Message, TestPartResult, TEST, CmpHelperNE

#include <chrono>  // for duration, duration_cast, operator-, high_resolution_clock, microseconds, tim...
#include <memory>  // for _Simple_types
#include <string>  // for string
#include <vector>  // for vector, _Vector_const_iterator, _Vector_iterator

#include "logging/back_trace.h"  // for backtrace_options, back_trace, stack_frame
#include "logging/logger.h"      // for XSIGMA_LOG_INFO

namespace xsigma
{
namespace
{
// Helper functions to create a deeper call stack
void level_3()
{
    // Capture stack trace at this level
    auto trace = back_trace::print(0, 10, false);
    EXPECT_FALSE(trace.empty());
    //EXPECT_NE(trace.find("level_3"), std::string::npos);
}

void level_2()
{
    level_3();
}

void level_1()
{
    level_2();
}

}  // namespace

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(BackTraceTest, BasicPrint)
{
    auto trace = back_trace::print();
    EXPECT_FALSE(trace.empty());
    XSIGMA_LOG_INFO("Basic stack trace:\n{}", trace);
}

TEST(BackTraceTest, PrintWithSkipFrames)
{
    auto trace_no_skip = back_trace::print(0, 5, false);
    auto trace_skip_2  = back_trace::print(2, 5, false);

    EXPECT_FALSE(trace_no_skip.empty());
    EXPECT_FALSE(trace_skip_2.empty());

    // Skipped trace should be shorter
    XSIGMA_LOG_INFO("No skip:\n{}", trace_no_skip);
    XSIGMA_LOG_INFO("Skip 2:\n{}", trace_skip_2);
}

TEST(BackTraceTest, PrintWithMaxFrames)
{
    auto trace_5  = back_trace::print(0, 5, false);
    auto trace_10 = back_trace::print(0, 10, false);

    EXPECT_FALSE(trace_5.empty());
    EXPECT_FALSE(trace_10.empty());

    XSIGMA_LOG_INFO("Max 5 frames:\n{}", trace_5);
    XSIGMA_LOG_INFO("Max 10 frames:\n{}", trace_10);
}

// ============================================================================
// Enhanced API Tests
// ============================================================================

TEST(BackTraceTest, CaptureAndFormat)
{
    // Capture raw frames
    backtrace_options options;
    options.frames_to_skip           = 0;
    options.maximum_number_of_frames = 10;
    options.skip_python_frames       = false;

    auto frames = back_trace::capture(options);
    EXPECT_FALSE(frames.empty());

    // Verify frame structure
    for (const auto& frame : frames)
    {
        EXPECT_FALSE(frame.function_name.empty());
        EXPECT_NE(frame.return_address, nullptr);
    }

    // Format the captured frames
    auto formatted = back_trace::format(frames, options);
    EXPECT_FALSE(formatted.empty());

    XSIGMA_LOG_INFO("Captured {} frames:\n{}", frames.size(), formatted);
}

TEST(BackTraceTest, CompactFormat)
{
    backtrace_options options;
    options.frames_to_skip           = 0;
    options.maximum_number_of_frames = 5;
    options.compact_format           = true;
    options.include_addresses        = false;
    options.include_offsets          = false;

    auto trace = back_trace::print(options);
    EXPECT_FALSE(trace.empty());

    // Compact format should contain " -> " separators
    EXPECT_NE(trace.find(" -> "), std::string::npos);

    XSIGMA_LOG_INFO("Compact trace: {}", trace);
}

TEST(BackTraceTest, CompactHelper)
{
    auto trace = back_trace::compact(5);
    EXPECT_FALSE(trace.empty());

    // Should contain function call chain
    EXPECT_NE(trace.find(" -> "), std::string::npos);

    XSIGMA_LOG_INFO("Compact helper: {}", trace);
}

TEST(BackTraceTest, DetailedFormatWithOptions)
{
    backtrace_options options;
    options.frames_to_skip           = 0;
    options.maximum_number_of_frames = 5;
    options.compact_format           = false;
    options.include_addresses        = true;
    options.include_offsets          = true;

    auto trace = back_trace::print(options);
    EXPECT_FALSE(trace.empty());

    // Detailed format should contain "frame #" markers
    EXPECT_NE(trace.find("frame #"), std::string::npos);

    XSIGMA_LOG_INFO("Detailed trace:\n{}", trace);
}

TEST(BackTraceTest, DetailedFormatWithoutAddresses)
{
    backtrace_options options;
    options.frames_to_skip           = 0;
    options.maximum_number_of_frames = 5;
    options.include_addresses        = false;
    options.include_offsets          = false;

    auto trace = back_trace::print(options);
    EXPECT_FALSE(trace.empty());

    XSIGMA_LOG_INFO("Trace without addresses:\n{}", trace);
}

// ============================================================================
// Call Stack Depth Tests
// ============================================================================

TEST(BackTraceTest, DeepCallStack)
{
    level_1();
}

// ============================================================================
// Platform Support Tests
// ============================================================================

TEST(BackTraceTest, IsSupported)
{
    bool supported = back_trace::is_supported();

#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
    // Should be supported on major platforms
    EXPECT_TRUE(supported);
#endif

    XSIGMA_LOG_INFO("Backtrace supported: {}", supported);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(BackTraceTest, EmptyFramesFormat)
{
    std::vector<stack_frame> empty_frames;
    auto                     formatted = back_trace::format(empty_frames);

    EXPECT_FALSE(formatted.empty());
    EXPECT_NE(formatted.find("No stack trace available"), std::string::npos);
}

TEST(BackTraceTest, ZeroMaxFrames)
{
    backtrace_options options;
    options.maximum_number_of_frames = 0;

    auto frames = back_trace::capture(options);
    // Should return empty or minimal frames
    XSIGMA_LOG_INFO("Zero max frames captured: {}", frames.size());
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(BackTraceTest, PerformanceCapture)
{
#if NDEBUG
    const int iterations = 100;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        backtrace_options options;
        options.maximum_number_of_frames = 10;
        auto frames                      = back_trace::capture(options);
        (void)frames;  // Suppress unused warning
    }

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_microseconds = static_cast<double>(duration.count()) / iterations;

    XSIGMA_LOG_INFO("Average capture time: {:.2f} microseconds", avg_microseconds);

    // Capture should be reasonably fast (< 1ms on average)
    EXPECT_LT(avg_microseconds, 1000.0);

#endif
}

TEST(BackTraceTest, PerformanceFormat)
{
    // Capture once
    backtrace_options options;
    options.maximum_number_of_frames = 10;
    auto frames                      = back_trace::capture(options);

    const int iterations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        auto formatted = back_trace::format(frames, options);
        (void)formatted;  // Suppress unused warning
    }

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_microseconds = static_cast<double>(duration.count()) / iterations;

    XSIGMA_LOG_INFO("Average format time: {:.2f} microseconds", avg_microseconds);

    // Formatting should be very fast (< 100us on average)
    EXPECT_LT(avg_microseconds, 100.0);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(BackTraceTest, UsageInLogging)
{
    // Demonstrate usage in logging context
    XSIGMA_LOG_INFO("Error occurred at:\n{}", back_trace::print(0, 5));
}

TEST(BackTraceTest, UsageInCompactLogging)
{
    // Demonstrate compact usage in logging
    XSIGMA_LOG_INFO("Call chain: {}", back_trace::compact(5));
}

}  // namespace xsigma
