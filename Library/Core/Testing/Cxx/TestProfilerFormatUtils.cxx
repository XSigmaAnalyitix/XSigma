/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <string>

#include "Testing/xsigmaTest.h"
#include "profiler/utils/format_utils.h"

using namespace xsigma::profiler;

// ============================================================================
// format_utils tests
// ============================================================================

XSIGMATEST(Profiler, format_one_digit_rounds_to_single_decimal)
{
    EXPECT_EQ(one_digit(3.14), "3.1");
    EXPECT_EQ(one_digit(-7.95), "-8.0");
}

XSIGMATEST(Profiler, format_two_digits_rounds_half_up)
{
    EXPECT_EQ(two_digits(2.345), "2.35");
    EXPECT_EQ(two_digits(-2.344), "-2.34");
}

XSIGMATEST(Profiler, format_three_digits_preserves_precision)
{
    EXPECT_EQ(three_digits(1.23456), "1.235");
    EXPECT_EQ(three_digits(-9.87654), "-9.877");
}

XSIGMATEST(Profiler, format_max_precision_round_trips_double_value)
{
    const double value      = 12345.678901234567;
    const auto   text       = max_precision(value);
    const double round_trip = std::stod(text);

    EXPECT_NEAR(round_trip, value, 1e-12);
}

XSIGMATEST(Profiler, format_handles_small_and_large_magnitudes)
{
    EXPECT_EQ(two_digits(0.000123), "0.00");
    EXPECT_EQ(one_digit(9876.543), "9876.5");
}

XSIGMATEST(Profiler, format_handles_zero)
{
    EXPECT_EQ(one_digit(0.0), "0.0");
    EXPECT_EQ(two_digits(0.0), "0.00");
    EXPECT_EQ(three_digits(0.0), "0.000");
    EXPECT_EQ(max_precision(0.0), "0");
}
