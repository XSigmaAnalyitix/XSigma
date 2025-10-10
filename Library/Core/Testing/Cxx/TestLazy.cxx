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

#include <memory>
#include <string>
#include <vector>

#include "util/lazy.h"
#include "xsigmaTest.h"

using namespace xsigma;

// ============================================================================
// Basic optimistic_lazy Tests
// ============================================================================

XSIGMATEST(Lazy, optimistic_lazy_basic)
{
    optimistic_lazy<int> lazy_int;

    // Test that factory is called on first access
    int  call_count = 0;
    auto factory    = [&call_count]() -> int
    {
        call_count++;
        return 42;
    };

    int& value1 = lazy_int.ensure(factory);
    EXPECT_EQ(value1, 42);
    EXPECT_EQ(call_count, 1);

    // Test that factory is not called on subsequent access
    int& value2 = lazy_int.ensure(factory);
    EXPECT_EQ(value2, 42);
    EXPECT_EQ(call_count, 1);  // Should still be 1

    // Test that both references point to same object
    EXPECT_EQ(&value1, &value2);

    END_TEST();
}

XSIGMATEST(Lazy, optimistic_lazy_complex_types)
{
    // Test with std::string
    optimistic_lazy<std::string> lazy_str;
    std::string& str = lazy_str.ensure([]() { return std::string("Hello, World!"); });
    EXPECT_EQ(str, "Hello, World!");

    // Test with std::vector
    optimistic_lazy<std::vector<int>> lazy_vec;
    std::vector<int>& vec = lazy_vec.ensure([]() { return std::vector<int>{1, 2, 3}; });
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[2], 3);

    END_TEST();
}

// ============================================================================
// lazy_value Tests
// ============================================================================

XSIGMATEST(Lazy, optimistic_lazy_reset)
{
    optimistic_lazy<int> lazy_int;

    // Initialize with first value
    int& value1 = lazy_int.ensure([]() { return 42; });
    EXPECT_EQ(value1, 42);

    // Reset
    lazy_int.reset();

    // Re-initialize with different value
    int& value2 = lazy_int.ensure([]() { return 99; });
    EXPECT_EQ(value2, 99);

    END_TEST();
}

XSIGMATEST(Lazy, lazy_value_interface)
{
    // Test with optimistic_lazy_value
    class test_lazy_value : public optimistic_lazy_value<int>
    {
    private:
        int compute() const override { return 42; }
    };

    test_lazy_value lazy_val;
    const int&      value1 = lazy_val.get();
    EXPECT_EQ(value1, 42);

    // Test that value is cached
    const int& value2 = lazy_val.get();
    EXPECT_EQ(value2, 42);
    EXPECT_EQ(&value1, &value2);

    END_TEST();
}

XSIGMATEST(Lazy, precomputed_lazy_value)
{
    precomputed_lazy_value<int> lazy_val(42);

    const int& value1 = lazy_val.get();
    EXPECT_EQ(value1, 42);

    const int& value2 = lazy_val.get();
    EXPECT_EQ(value2, 42);
    EXPECT_EQ(&value1, &value2);

    // Test with complex type
    precomputed_lazy_value<std::string> lazy_str(std::string("Hello"));
    const std::string&                  str = lazy_str.get();
    EXPECT_EQ(str, "Hello");

    END_TEST();
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

XSIGMATEST(Lazy, edge_cases)
{
    // Test with empty string
    optimistic_lazy<std::string> lazy_empty;
    std::string&                 empty = lazy_empty.ensure([]() { return std::string(); });
    EXPECT_TRUE(empty.empty());

    // Test with zero value
    optimistic_lazy<int> lazy_zero;
    int&                 zero = lazy_zero.ensure([]() { return 0; });
    EXPECT_EQ(zero, 0);

    END_TEST();
}
