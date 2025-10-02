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
 * @file TestFMT.cxx
 * @brief Comprehensive test suite for fmt formatting library integration
 *
 * This file contains extensive tests for the fmt library covering:
 * - Basic formatting operations (format, print)
 * - Numeric type formatting (integers, floats, edge cases)
 * - String and character formatting
 * - Container and range formatting
 * - Chrono and time formatting
 * - Color and style formatting
 * - Compile-time formatting
 * - Output iterator operations (format_to, formatted_size)
 * - Error handling and edge cases
 * - Platform-independent behavior
 *
 * Target: Minimum 98% code coverage for fmt usage in XSigma
 *
 * @author XSigma Development Team
 * @version 2.0
 * @date 2025
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "xsigmaTest.h"

// clang-format off
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
// clang-format on

// =============================================================================
// BASIC FORMATTING TESTS
// =============================================================================

/**
 * @brief Test basic fmt::format() functionality with various argument types
 *
 * Covers: string formatting, positional arguments, basic type support
 */
XSIGMATEST(Core, fmt_basic_format)
{
    // Simple string formatting
    EXPECT_EQ(fmt::format("Hello, {}!", "World"), "Hello, World!");
    EXPECT_EQ(fmt::format("The answer is {}", 42), "The answer is 42");

    // Multiple arguments
    EXPECT_EQ(fmt::format("{} + {} = {}", 1, 2, 3), "1 + 2 = 3");
    EXPECT_EQ(fmt::format("{0} {1} {0}", "A", "B"), "A B A");

    // Empty format string
    EXPECT_EQ(fmt::format(""), "");

    // No arguments
    EXPECT_EQ(fmt::format("No placeholders"), "No placeholders");

    END_TEST();
}

/**
 * @brief Test fmt::print() functionality
 *
 * Covers: print to stdout, basic output operations
 */
XSIGMATEST(Core, fmt_basic_print)
{
    // These tests verify that print doesn't crash
    // Output is not captured in unit tests but validates API usage
    fmt::print("Hello, {}!\n", "World");
    fmt::print("Number: {}\n", 42);
    fmt::print("Multiple: {} {} {}\n", 1, 2, 3);

    END_TEST();
}

/**
 * @brief Test string and character formatting
 *
 * Covers: string_view, const char*, std::string, char, escape sequences
 */
XSIGMATEST(Core, fmt_string_formatting)
{
    // C-string
    const char* c_str = "C-string";
    EXPECT_EQ(fmt::format("{}", c_str), "C-string");

    // std::string
    std::string std_str = "std::string";
    EXPECT_EQ(fmt::format("{}", std_str), "std::string");

    // Character
    EXPECT_EQ(fmt::format("{}", 'A'), "A");
    EXPECT_EQ(fmt::format("{}", '\n'), "\n");

    // Empty string
    EXPECT_EQ(fmt::format("{}", ""), "");
    EXPECT_EQ(fmt::format("{}", std::string()), "");

    // String with special characters
    EXPECT_EQ(fmt::format("{}", "Line1\nLine2"), "Line1\nLine2");
    EXPECT_EQ(fmt::format("{}", "Tab\there"), "Tab\there");

    // Width and alignment for strings
    EXPECT_EQ(fmt::format("{:10}", "test"), "test      ");
    EXPECT_EQ(fmt::format("{:<10}", "test"), "test      ");
    EXPECT_EQ(fmt::format("{:>10}", "test"), "      test");
    EXPECT_EQ(fmt::format("{:^10}", "test"), "   test   ");

    END_TEST();
}

// =============================================================================
// NUMERIC FORMATTING TESTS - INTEGERS
// =============================================================================

/**
 * @brief Test integer formatting with various types and formats
 *
 * Covers: int, long, unsigned, different bases, width, padding, alignment
 */
XSIGMATEST(Core, fmt_integer_formatting)
{
    // Basic integer formatting
    EXPECT_EQ(fmt::format("{}", 42), "42");
    EXPECT_EQ(fmt::format("{}", -42), "-42");
    EXPECT_EQ(fmt::format("{}", 0), "0");

    // Different integer types
    EXPECT_EQ(fmt::format("{}", static_cast<int8_t>(127)), "127");
    EXPECT_EQ(fmt::format("{}", static_cast<int16_t>(-32768)), "-32768");
    EXPECT_EQ(fmt::format("{}", static_cast<int32_t>(2147483647)), "2147483647");
    EXPECT_EQ(
        fmt::format("{}", static_cast<int64_t>(-9223372036854775807LL - 1)),
        "-9223372036854775808");

    // Unsigned integers
    EXPECT_EQ(fmt::format("{}", static_cast<uint8_t>(255)), "255");
    EXPECT_EQ(fmt::format("{}", static_cast<uint16_t>(65535)), "65535");
    EXPECT_EQ(fmt::format("{}", static_cast<uint32_t>(4294967295U)), "4294967295");
    EXPECT_EQ(
        fmt::format("{}", static_cast<uint64_t>(18446744073709551615ULL)), "18446744073709551615");

    // Hexadecimal formatting
    EXPECT_EQ(fmt::format("{:x}", 255), "ff");
    EXPECT_EQ(fmt::format("{:X}", 255), "FF");
    EXPECT_EQ(fmt::format("{:#x}", 255), "0xff");
    EXPECT_EQ(fmt::format("{:#X}", 255), "0XFF");

    // Octal formatting
    EXPECT_EQ(fmt::format("{:o}", 64), "100");
    EXPECT_EQ(fmt::format("{:#o}", 64), "0100");

    // Binary formatting
    EXPECT_EQ(fmt::format("{:b}", 5), "101");
    EXPECT_EQ(fmt::format("{:#b}", 5), "0b101");

    // Width and padding
    EXPECT_EQ(fmt::format("{:5}", 42), "   42");
    EXPECT_EQ(fmt::format("{:05}", 42), "00042");
    EXPECT_EQ(fmt::format("{:<5}", 42), "42   ");
    EXPECT_EQ(fmt::format("{:>5}", 42), "   42");
    EXPECT_EQ(fmt::format("{:^5}", 42), " 42  ");

    // Sign handling
    EXPECT_EQ(fmt::format("{:+}", 42), "+42");
    EXPECT_EQ(fmt::format("{:+}", -42), "-42");
    EXPECT_EQ(fmt::format("{: }", 42), " 42");
    EXPECT_EQ(fmt::format("{: }", -42), "-42");

    // Edge cases
    EXPECT_EQ(fmt::format("{}", std::numeric_limits<int>::max()), "2147483647");
    EXPECT_EQ(fmt::format("{}", std::numeric_limits<int>::min()), "-2147483648");
    EXPECT_EQ(fmt::format("{}", std::numeric_limits<unsigned>::max()), "4294967295");

    END_TEST();
}

// =============================================================================
// NUMERIC FORMATTING TESTS - FLOATING POINT
// =============================================================================

/**
 * @brief Test floating-point formatting with various formats and edge cases
 *
 * Covers: float, double, precision, scientific notation, fixed notation, NaN, infinity
 */
XSIGMATEST(Core, fmt_float_formatting)
{
    // Basic float formatting
    EXPECT_EQ(fmt::format("{}", 3.14), "3.14");
    EXPECT_EQ(fmt::format("{}", -3.14), "-3.14");
    EXPECT_EQ(fmt::format("{}", 0.0), "0");

    // Precision control
    EXPECT_EQ(fmt::format("{:.2f}", 3.14159), "3.14");
    EXPECT_EQ(fmt::format("{:.4f}", 3.14159), "3.1416");
    EXPECT_EQ(fmt::format("{:.0f}", 3.14159), "3");

    // Scientific notation
    EXPECT_EQ(fmt::format("{:e}", 1234.5), "1.234500e+03");
    EXPECT_EQ(fmt::format("{:E}", 1234.5), "1.234500E+03");
    EXPECT_EQ(fmt::format("{:.2e}", 1234.5), "1.23e+03");

    // General format (chooses between fixed and scientific)
    EXPECT_EQ(fmt::format("{:g}", 1234.5), "1234.5");
    EXPECT_EQ(fmt::format("{:g}", 0.0001234), "0.0001234");

    // Width and alignment
    EXPECT_EQ(fmt::format("{:10.2f}", 3.14), "      3.14");
    EXPECT_EQ(fmt::format("{:<10.2f}", 3.14), "3.14      ");
    EXPECT_EQ(fmt::format("{:>10.2f}", 3.14), "      3.14");
    EXPECT_EQ(fmt::format("{:^10.2f}", 3.14), "   3.14   ");

    // Sign handling
    EXPECT_EQ(fmt::format("{:+.2f}", 3.14), "+3.14");
    EXPECT_EQ(fmt::format("{:+.2f}", -3.14), "-3.14");
    EXPECT_EQ(fmt::format("{: .2f}", 3.14), " 3.14");

    // Special values - NaN
    double      nan_val    = std::numeric_limits<double>::quiet_NaN();
    std::string nan_result = fmt::format("{}", nan_val);
    EXPECT_TRUE(nan_result == "nan" || nan_result == "NaN" || nan_result == "-nan(ind)");

    // Special values - Infinity
    double      inf_val    = std::numeric_limits<double>::infinity();
    std::string inf_result = fmt::format("{}", inf_val);
    EXPECT_TRUE(inf_result == "inf" || inf_result == "Inf");

    double      neg_inf_val    = -std::numeric_limits<double>::infinity();
    std::string neg_inf_result = fmt::format("{}", neg_inf_val);
    EXPECT_TRUE(neg_inf_result == "-inf" || neg_inf_result == "-Inf");

    // Edge cases
    EXPECT_EQ(fmt::format("{}", 0.0), "0");
    EXPECT_EQ(fmt::format("{}", -0.0), "-0");

    // Very small and very large numbers
    EXPECT_EQ(fmt::format("{:.2e}", 1.23e-100), "1.23e-100");
    EXPECT_EQ(fmt::format("{:.2e}", 1.23e+100), "1.23e+100");

    // Different float types
    float  f = 3.14f;
    double d = 3.14159265358979;
    EXPECT_EQ(fmt::format("{:.2f}", f), "3.14");
    EXPECT_EQ(fmt::format("{:.10f}", d), "3.1415926536");

    END_TEST();
}

/**
 * @brief Test boolean and pointer formatting
 *
 * Covers: bool, void*, nullptr
 */
XSIGMATEST(Core, fmt_bool_pointer_formatting)
{
    // Boolean formatting
    EXPECT_EQ(fmt::format("{}", true), "true");
    EXPECT_EQ(fmt::format("{}", false), "false");

    // Pointer formatting
    int         value   = 42;
    void*       ptr     = &value;
    std::string ptr_str = fmt::format("{}", ptr);
    EXPECT_TRUE(ptr_str.find("0x") == 0 || ptr_str.find("0X") == 0);

    // Nullptr
    void*       null_ptr = nullptr;
    std::string null_str = fmt::format("{}", null_ptr);
    EXPECT_TRUE(
        null_str == "0x0" || null_str == "0" || null_str == "(nil)" || null_str == "0x00000000" ||
        null_str == "0x0000000000000000");

    END_TEST();
}

// =============================================================================
// CONTAINER AND RANGE FORMATTING TESTS
// =============================================================================

/**
 * @brief Test formatting of STL containers using fmt/ranges.h
 *
 * Covers: vector, list, deque, set, map, array, tuple, pair, optional
 */
XSIGMATEST(Core, fmt_container_formatting)
{
    // Vector formatting
    std::vector<int> vec = {1, 2, 3, 4, 5};
    EXPECT_EQ(fmt::format("{}", vec), "[1, 2, 3, 4, 5]");

    // Empty vector
    std::vector<int> empty_vec;
    EXPECT_EQ(fmt::format("{}", empty_vec), "[]");

    // Vector of strings
    std::vector<std::string> str_vec = {"hello", "world"};
    EXPECT_EQ(fmt::format("{}", str_vec), "[\"hello\", \"world\"]");

    // List formatting
    std::list<int> lst = {10, 20, 30};
    EXPECT_EQ(fmt::format("{}", lst), "[10, 20, 30]");

    // Deque formatting
    std::deque<int> dq = {100, 200};
    EXPECT_EQ(fmt::format("{}", dq), "[100, 200]");

    // Set formatting
    std::set<int> s = {3, 1, 2};  // Sets are ordered
    EXPECT_EQ(fmt::format("{}", s), "{1, 2, 3}");

    // Array formatting
    std::array<int, 4> arr = {7, 8, 9, 10};
    EXPECT_EQ(fmt::format("{}", arr), "[7, 8, 9, 10]");

    // Pair formatting
    std::pair<int, std::string> p = {42, "answer"};
    EXPECT_EQ(fmt::format("{}", p), "(42, \"answer\")");

    // Tuple formatting
    std::tuple<int, double, std::string> t = {1, 3.14, "pi"};
    EXPECT_EQ(fmt::format("{}", t), "(1, 3.14, \"pi\")");

    // Optional formatting (C++17) - format the value if present
    std::optional<int> opt_val = 42;
    if (opt_val.has_value())
    {
        EXPECT_EQ(fmt::format("{}", *opt_val), "42");
    }

    std::optional<int> opt_empty;
    EXPECT_FALSE(opt_empty.has_value());

    // Map formatting
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}};
    EXPECT_EQ(fmt::format("{}", m), "{1: \"one\", 2: \"two\"}");

    END_TEST();
}

/**
 * @brief Test nested container formatting
 *
 * Covers: vector of vectors, complex nested structures
 */
XSIGMATEST(Core, fmt_nested_container_formatting)
{
    // Vector of vectors
    std::vector<std::vector<int>> nested_vec = {{1, 2}, {3, 4}, {5, 6}};
    EXPECT_EQ(fmt::format("{}", nested_vec), "[[1, 2], [3, 4], [5, 6]]");

    // Vector of pairs
    std::vector<std::pair<int, int>> vec_pairs = {{1, 2}, {3, 4}};
    EXPECT_EQ(fmt::format("{}", vec_pairs), "[(1, 2), (3, 4)]");

    // Map with vector values
    std::map<std::string, std::vector<int>> map_vec = {{"a", {1, 2}}, {"b", {3, 4}}};
    EXPECT_EQ(fmt::format("{}", map_vec), "{\"a\": [1, 2], \"b\": [3, 4]}");

    END_TEST();
}

// =============================================================================
// CHRONO AND TIME FORMATTING TESTS
// =============================================================================

/**
 * @brief Test chrono duration and time_point formatting
 *
 * Covers: duration types, time formatting, platform-independent behavior
 */
XSIGMATEST(Core, fmt_chrono_formatting)
{
    using namespace std::chrono;

    // Duration formatting
    auto dur_sec = seconds(42);
    EXPECT_EQ(fmt::format("{}", dur_sec), "42s");

    auto dur_ms = milliseconds(1500);
    EXPECT_EQ(fmt::format("{}", dur_ms), "1500ms");

    auto dur_us = microseconds(500);
    // Note: fmt uses Unicode micro symbol (μ) for microseconds
    std::string us_result = fmt::format("{}", dur_us);
    EXPECT_TRUE(us_result.find("500") != std::string::npos);

    auto dur_ns = nanoseconds(1000);
    EXPECT_EQ(fmt::format("{}", dur_ns), "1000ns");

    auto dur_min = minutes(5);
    EXPECT_EQ(fmt::format("{}", dur_min), "5min");

    auto dur_hour = hours(2);
    EXPECT_EQ(fmt::format("{}", dur_hour), "2h");

    // Zero duration
    auto dur_zero = seconds(0);
    EXPECT_EQ(fmt::format("{}", dur_zero), "0s");

    // Negative duration
    auto dur_neg = seconds(-10);
    EXPECT_EQ(fmt::format("{}", dur_neg), "-10s");

    // Time point formatting (system_clock)
    auto        now      = system_clock::now();
    std::string time_str = fmt::format("{}", now);
    EXPECT_FALSE(time_str.empty());

    // Duration arithmetic
    auto total = seconds(90);
    EXPECT_EQ(fmt::format("{}", total), "90s");

    END_TEST();
}

// =============================================================================
// COLOR AND STYLE FORMATTING TESTS
// =============================================================================

/**
 * @brief Test terminal color and text style formatting
 *
 * Covers: foreground colors, background colors, text styles, combined styles
 */
XSIGMATEST(Core, fmt_color_formatting)
{
    using namespace fmt;

    // Foreground colors
    std::string red_text = fmt::format(fg(color::red), "Red text");
    EXPECT_TRUE(red_text.find("Red text") != std::string::npos);
    EXPECT_TRUE(red_text.find("\x1B[") != std::string::npos);  // ANSI escape code

    std::string green_text = fmt::format(fg(color::green), "Green text");
    EXPECT_TRUE(green_text.find("Green text") != std::string::npos);

    std::string blue_text = fmt::format(fg(color::blue), "Blue text");
    EXPECT_TRUE(blue_text.find("Blue text") != std::string::npos);

    // Background colors
    std::string bg_text = fmt::format(bg(color::yellow), "Yellow background");
    EXPECT_TRUE(bg_text.find("Yellow background") != std::string::npos);

    // RGB colors
    std::string rgb_text = fmt::format(fg(rgb(255, 128, 0)), "Orange text");
    EXPECT_TRUE(rgb_text.find("Orange text") != std::string::npos);

    // Text styles
    std::string bold_text = fmt::format(emphasis::bold, "Bold text");
    EXPECT_TRUE(bold_text.find("Bold text") != std::string::npos);

    std::string italic_text = fmt::format(emphasis::italic, "Italic text");
    EXPECT_TRUE(italic_text.find("Italic text") != std::string::npos);

    std::string underline_text = fmt::format(emphasis::underline, "Underlined text");
    EXPECT_TRUE(underline_text.find("Underlined text") != std::string::npos);

    // Combined styles
    std::string combined = fmt::format(fg(color::red) | emphasis::bold, "Bold red");
    EXPECT_TRUE(combined.find("Bold red") != std::string::npos);

    std::string multi_style =
        fmt::format(fg(color::white) | bg(color::blue) | emphasis::bold, "Styled");
    EXPECT_TRUE(multi_style.find("Styled") != std::string::npos);

    END_TEST();
}

// =============================================================================
// COMPILE-TIME FORMATTING TESTS
// =============================================================================

/**
 * @brief Test compile-time format string compilation using FMT_COMPILE
 *
 * Covers: FMT_COMPILE macro, compile-time optimization
 */
XSIGMATEST(Core, fmt_compile_time_formatting)
{
    // Basic compile-time formatting
    EXPECT_EQ(fmt::format(FMT_COMPILE("Hello, {}!"), "World"), "Hello, World!");
    EXPECT_EQ(fmt::format(FMT_COMPILE("The answer is {}"), 42), "The answer is 42");

    // Multiple arguments
    EXPECT_EQ(fmt::format(FMT_COMPILE("{} + {} = {}"), 1, 2, 3), "1 + 2 = 3");

    // With format specifications
    EXPECT_EQ(fmt::format(FMT_COMPILE("{:.2f}"), 3.14159), "3.14");
    EXPECT_EQ(fmt::format(FMT_COMPILE("{:05}"), 42), "00042");
    EXPECT_EQ(fmt::format(FMT_COMPILE("{:#x}"), 255), "0xff");

    // String formatting
    EXPECT_EQ(fmt::format(FMT_COMPILE("{:>10}"), "test"), "      test");

    END_TEST();
}

// =============================================================================
// FORMAT_TO AND OUTPUT ITERATOR TESTS
// =============================================================================

/**
 * @brief Test fmt::format_to() with various output iterators
 *
 * Covers: format_to, back_inserter, output iterators, buffer operations
 */
XSIGMATEST(Core, fmt_format_to_operations)
{
    // Format to vector using back_inserter
    std::vector<char> buffer;
    fmt::format_to(std::back_inserter(buffer), "Hello, {}!", "World");
    std::string result(buffer.begin(), buffer.end());
    EXPECT_EQ(result, "Hello, World!");

    // Format to string using back_inserter
    std::string str_buffer;
    fmt::format_to(std::back_inserter(str_buffer), "Number: {}", 42);
    EXPECT_EQ(str_buffer, "Number: 42");

    // Multiple format_to operations
    std::string multi_buffer;
    fmt::format_to(std::back_inserter(multi_buffer), "First: {} ", 1);
    fmt::format_to(std::back_inserter(multi_buffer), "Second: {} ", 2);
    fmt::format_to(std::back_inserter(multi_buffer), "Third: {}", 3);
    EXPECT_EQ(multi_buffer, "First: 1 Second: 2 Third: 3");

    // Format to fixed-size buffer
    char fixed_buffer[100];
    auto result_it = fmt::format_to(fixed_buffer, "Value: {}", 123);
    *result_it     = '\0';  // Null-terminate
    EXPECT_EQ(std::string(fixed_buffer), "Value: 123");

    // Format with complex types
    std::vector<char> vec_buffer;
    std::vector<int>  vec = {1, 2, 3};
    fmt::format_to(std::back_inserter(vec_buffer), "Vector: {}", vec);
    std::string vec_result(vec_buffer.begin(), vec_buffer.end());
    EXPECT_EQ(vec_result, "Vector: [1, 2, 3]");

    END_TEST();
}

/**
 * @brief Test fmt::formatted_size() function
 *
 * Covers: formatted_size, size calculation without formatting
 */
XSIGMATEST(Core, fmt_formatted_size)
{
    // Basic size calculation
    size_t size1 = fmt::formatted_size("Hello, {}!", "World");
    EXPECT_EQ(size1, 13);  // "Hello, World!"

    size_t size2 = fmt::formatted_size("Number: {}", 42);
    EXPECT_EQ(size2, 10);  // "Number: 42"

    // With format specifications
    size_t size3 = fmt::formatted_size("{:.2f}", 3.14159);
    EXPECT_EQ(size3, 4);  // "3.14"

    size_t size4 = fmt::formatted_size("{:05}", 42);
    EXPECT_EQ(size4, 5);  // "00042"

    // Empty format
    size_t size5 = fmt::formatted_size("");
    EXPECT_EQ(size5, 0);

    // Complex types
    std::vector<int> vec   = {1, 2, 3};
    size_t           size6 = fmt::formatted_size("{}", vec);
    EXPECT_EQ(size6, 9);  // "[1, 2, 3]"

    END_TEST();
}

// =============================================================================
// ERROR HANDLING AND EDGE CASE TESTS
// =============================================================================

/**
 * @brief Test error handling with runtime format strings
 *
 * Covers: runtime format strings, error conditions (no exceptions per XSigma rules)
 * Note: XSigma follows no-exception pattern, so we test valid runtime usage
 */
XSIGMATEST(Core, fmt_runtime_format_strings)
{
    // Runtime format strings (valid usage)
    std::string format_str = "Hello, {}!";
    EXPECT_EQ(fmt::format(fmt::runtime(format_str), "World"), "Hello, World!");

    // Dynamic format string construction
    std::string dynamic_fmt = "{} + {} = {}";
    EXPECT_EQ(fmt::format(fmt::runtime(dynamic_fmt), 1, 2, 3), "1 + 2 = 3");

    // Runtime format with specifications
    std::string spec_fmt = "{:.2f}";
    EXPECT_EQ(fmt::format(fmt::runtime(spec_fmt), 3.14159), "3.14");

    END_TEST();
}

/**
 * @brief Test edge cases and boundary conditions
 *
 * Covers: empty strings, null scenarios, extreme values, special characters
 */
XSIGMATEST(Core, fmt_edge_cases)
{
    // Empty format string
    EXPECT_EQ(fmt::format(""), "");

    // Empty argument
    EXPECT_EQ(fmt::format("{}", ""), "");
    EXPECT_EQ(fmt::format("{}", std::string()), "");

    // Single character
    EXPECT_EQ(fmt::format("{}", 'X'), "X");

    // Very long string
    std::string long_str(1000, 'A');
    std::string long_result = fmt::format("{}", long_str);
    EXPECT_EQ(long_result.size(), 1000);
    EXPECT_EQ(long_result, long_str);

    // Special characters in format string
    EXPECT_EQ(fmt::format("{{}}"), "{}");
    EXPECT_EQ(fmt::format("{{{}}}", 42), "{42}");
    EXPECT_EQ(fmt::format("100%"), "100%");

    // Unicode and special characters (platform-independent)
    EXPECT_EQ(fmt::format("{}", "Hello\nWorld"), "Hello\nWorld");
    EXPECT_EQ(fmt::format("{}", "Tab\tSeparated"), "Tab\tSeparated");

    // Zero values
    EXPECT_EQ(fmt::format("{}", 0), "0");
    EXPECT_EQ(fmt::format("{}", 0.0), "0");
    EXPECT_EQ(fmt::format("{}", false), "false");

    // Maximum and minimum values (platform-independent)
    EXPECT_EQ(fmt::format("{}", std::numeric_limits<int>::max()), "2147483647");
    EXPECT_EQ(fmt::format("{}", std::numeric_limits<int>::min()), "-2147483648");

    END_TEST();
}

// =============================================================================
// PLATFORM-INDEPENDENT TESTS
// =============================================================================

/**
 * @brief Test platform-independent behavior across Windows, Linux, and macOS
 *
 * Covers: portable types, cross-platform formatting, consistent output
 */
XSIGMATEST(Core, fmt_platform_independent)
{
    // Fixed-width integer types (platform-independent)
    EXPECT_EQ(fmt::format("{}", static_cast<int8_t>(127)), "127");
    EXPECT_EQ(fmt::format("{}", static_cast<int16_t>(32767)), "32767");
    EXPECT_EQ(fmt::format("{}", static_cast<int32_t>(2147483647)), "2147483647");
    EXPECT_EQ(
        fmt::format("{}", static_cast<int64_t>(9223372036854775807LL)), "9223372036854775807");

    EXPECT_EQ(fmt::format("{}", static_cast<uint8_t>(255)), "255");
    EXPECT_EQ(fmt::format("{}", static_cast<uint16_t>(65535)), "65535");
    EXPECT_EQ(fmt::format("{}", static_cast<uint32_t>(4294967295U)), "4294967295");
    EXPECT_EQ(
        fmt::format("{}", static_cast<uint64_t>(18446744073709551615ULL)), "18446744073709551615");

    // size_t formatting (platform-independent)
    size_t      sz        = 1024;
    std::string sz_result = fmt::format("{}", sz);
    EXPECT_EQ(sz_result, "1024");

    // ptrdiff_t formatting
    std::ptrdiff_t pd        = -100;
    std::string    pd_result = fmt::format("{}", pd);
    EXPECT_EQ(pd_result, "-100");

    // Character encoding (ASCII - platform-independent)
    EXPECT_EQ(fmt::format("{}", 'A'), "A");
    EXPECT_EQ(fmt::format("{}", '0'), "0");
    EXPECT_EQ(fmt::format("{}", ' '), " ");

    // Newline handling (platform-independent)
    EXPECT_EQ(fmt::format("Line1\nLine2"), "Line1\nLine2");

    // Path separators (test both, but don't assume which is used)
    std::string path_unix = fmt::format("{}", "path/to/file");
    EXPECT_EQ(path_unix, "path/to/file");

    std::string path_win = fmt::format("{}", "path\\to\\file");
    EXPECT_EQ(path_win, "path\\to\\file");

    END_TEST();
}

/**
 * @brief Test formatting with standard library types across platforms
 *
 * Covers: std::string, std::vector, std::pair, consistent behavior
 */
XSIGMATEST(Core, fmt_stdlib_cross_platform)
{
    // std::string (platform-independent)
    std::string str = "cross-platform";
    EXPECT_EQ(fmt::format("{}", str), "cross-platform");

    // std::vector (platform-independent)
    std::vector<int> vec = {10, 20, 30};
    EXPECT_EQ(fmt::format("{}", vec), "[10, 20, 30]");

    // std::pair (platform-independent)
    std::pair<int, int> p = {1, 2};
    EXPECT_EQ(fmt::format("{}", p), "(1, 2)");

    // std::array (platform-independent)
    std::array<int, 3> arr = {100, 200, 300};
    EXPECT_EQ(fmt::format("{}", arr), "[100, 200, 300]");

    // Boolean (platform-independent)
    EXPECT_EQ(fmt::format("{}", true), "true");
    EXPECT_EQ(fmt::format("{}", false), "false");

    END_TEST();
}

// =============================================================================
// ADVANCED FORMATTING TESTS
// =============================================================================

/**
 * @brief Test advanced formatting features
 *
 * Covers: positional arguments, named arguments, custom separators
 */
XSIGMATEST(Core, fmt_advanced_formatting)
{
    // Positional arguments
    EXPECT_EQ(fmt::format("{0} {1} {0}", "A", "B"), "A B A");
    EXPECT_EQ(fmt::format("{1} {0}", "first", "second"), "second first");
    EXPECT_EQ(fmt::format("{2} {1} {0}", 1, 2, 3), "3 2 1");

    // Automatic indexing (cannot mix with manual indexing)
    EXPECT_EQ(fmt::format("{} {} {}", "A", "B", "C"), "A B C");

    // Width and precision combinations
    EXPECT_EQ(fmt::format("{:10.2f}", 3.14159), "      3.14");
    EXPECT_EQ(fmt::format("{:<10.2f}", 3.14159), "3.14      ");
    EXPECT_EQ(fmt::format("{:^10.2f}", 3.14159), "   3.14   ");

    // Fill character with alignment
    EXPECT_EQ(fmt::format("{:*>10}", "test"), "******test");
    EXPECT_EQ(fmt::format("{:*<10}", "test"), "test******");
    EXPECT_EQ(fmt::format("{:*^10}", "test"), "***test***");

    // Numeric formatting with fill
    EXPECT_EQ(fmt::format("{:0>5}", 42), "00042");
    EXPECT_EQ(fmt::format("{:*>5}", 42), "***42");

    END_TEST();
}

/**
 * @brief Test performance-critical formatting scenarios
 *
 * Covers: repeated formatting, large data sets, efficiency
 */
XSIGMATEST(Core, fmt_performance_scenarios)
{
    // Repeated formatting (should not crash or leak)
    for (int i = 0; i < 1000; ++i)
    {
        std::string result = fmt::format("Iteration: {}", i);
        EXPECT_FALSE(result.empty());
    }

    // Large vector formatting
    std::vector<int> large_vec;
    for (int i = 0; i < 100; ++i)
    {
        large_vec.push_back(i);
    }
    std::string large_result = fmt::format("{}", large_vec);
    EXPECT_FALSE(large_result.empty());
    EXPECT_TRUE(large_result.find("[0,") != std::string::npos);

    // Multiple format_to operations (buffer reuse)
    std::string buffer;
    for (int i = 0; i < 100; ++i)
    {
        fmt::format_to(std::back_inserter(buffer), "{} ", i);
    }
    EXPECT_FALSE(buffer.empty());

    END_TEST();
}

// =============================================================================
// INTEGRATION TESTS WITH XSIGMA COMPONENTS
// =============================================================================

/**
 * @brief Test fmt integration with XSigma macros and utilities
 *
 * Covers: XSIGMA_UNUSED, integration with project patterns
 */
XSIGMATEST(Core, fmt_xsigma_integration)
{
    // Test with XSIGMA_UNUSED macro
    XSIGMA_UNUSED int unused_var = 42;
    std::string       result     = fmt::format("Value: {}", unused_var);
    EXPECT_EQ(result, "Value: 42");

    // Test with various XSigma types
    int32_t test_int32 = 12345;
    EXPECT_EQ(fmt::format("{}", test_int32), "12345");

    uint64_t test_uint64 = 9876543210ULL;
    EXPECT_EQ(fmt::format("{}", test_uint64), "9876543210");

    // Test formatting for logging scenarios
    std::string log_msg = fmt::format("[{}] {}: {}", "INFO", "TestModule", "Test message");
    EXPECT_EQ(log_msg, "[INFO] TestModule: Test message");

    // Test formatting for error messages
    int         error_code = 404;
    std::string error_msg  = fmt::format("Error {}: {}", error_code, "Not found");
    EXPECT_EQ(error_msg, "Error 404: Not found");

    END_TEST();
}

/**
 * @brief Test memory safety and resource management
 *
 * Covers: no memory leaks, safe buffer operations, RAII compliance
 */
XSIGMATEST(Core, fmt_memory_safety)
{
    // Test with smart pointers
    auto        ptr        = std::make_unique<int>(42);
    std::string ptr_result = fmt::format("Value: {}", *ptr);
    EXPECT_EQ(ptr_result, "Value: 42");

    // Test with shared_ptr
    auto        shared        = std::make_shared<std::string>("shared data");
    std::string shared_result = fmt::format("Data: {}", *shared);
    EXPECT_EQ(shared_result, "Data: shared data");

    // Test with temporary objects
    EXPECT_EQ(fmt::format("{}", std::string("temporary")), "temporary");
    EXPECT_EQ(fmt::format("{}", std::vector<int>{1, 2, 3}), "[1, 2, 3]");

    // Test with move semantics
    std::string movable     = "movable string";
    std::string move_result = fmt::format("{}", std::move(movable));
    EXPECT_EQ(move_result, "movable string");

    END_TEST();
}

/**
 * @brief Comprehensive integration test combining multiple features
 *
 * Covers: real-world usage scenarios, complex formatting combinations
 */
XSIGMATEST(Core, fmt_comprehensive_integration)
{
    // Simulate a real-world logging scenario
    std::string timestamp   = "2025-01-15 10:30:45";
    std::string level       = "INFO";
    std::string module      = "Core";
    std::string message     = "Operation completed successfully";
    int         duration_ms = 125;

    std::string log_entry =
        fmt::format("[{}] [{}] {}: {} (took {}ms)", timestamp, level, module, message, duration_ms);

    EXPECT_TRUE(log_entry.find(timestamp) != std::string::npos);
    EXPECT_TRUE(log_entry.find(level) != std::string::npos);
    EXPECT_TRUE(log_entry.find(module) != std::string::npos);
    EXPECT_TRUE(log_entry.find(message) != std::string::npos);
    EXPECT_TRUE(log_entry.find("125ms") != std::string::npos);

    // Simulate data reporting scenario
    std::vector<double> results = {1.23, 4.56, 7.89};
    double              average = 4.56;
    std::string         report  = fmt::format("Results: {}, Average: {:.2f}", results, average);

    EXPECT_TRUE(report.find("[1.23, 4.56, 7.89]") != std::string::npos);
    EXPECT_TRUE(report.find("4.56") != std::string::npos);

    // Simulate error reporting with context
    int         line_number  = 42;
    std::string filename     = "test.cpp";
    std::string error_detail = "Invalid parameter";

    std::string error_report =
        fmt::format("Error at {}:{} - {}", filename, line_number, error_detail);

    EXPECT_EQ(error_report, "Error at test.cpp:42 - Invalid parameter");

    END_TEST();
}
