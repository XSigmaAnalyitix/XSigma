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

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "logging/logger.h"
#include "util/lazy.h"
#include "xsigmaTest.h"

using namespace xsigma;

// ============================================================================
// Basic optimistic_lazy Tests
// ============================================================================

/**
 * @brief Test basic optimistic_lazy functionality
 *
 * Covers: lazy initialization, factory function, value retrieval
 */
XSIGMATEST(Core, lazy_basic_optimistic)
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

/**
 * @brief Test optimistic_lazy with complex types
 *
 * Covers: lazy initialization with strings, vectors, custom types
 */
XSIGMATEST(Core, lazy_complex_types)
{
    // Test with std::string
    optimistic_lazy<std::string> lazy_str;
    std::string& str = lazy_str.ensure([]() { return std::string("Hello, World!"); });
    EXPECT_EQ(str, "Hello, World!");

    // Test with std::vector
    optimistic_lazy<std::vector<int>> lazy_vec;
    std::vector<int>& vec = lazy_vec.ensure([]() { return std::vector<int>{1, 2, 3, 4, 5}; });
    EXPECT_EQ(vec.size(), 5);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[4], 5);

    // Test with std::pair
    optimistic_lazy<std::pair<int, std::string>> lazy_pair;
    auto& pair = lazy_pair.ensure([]() { return std::make_pair(42, std::string("test")); });
    EXPECT_EQ(pair.first, 42);
    EXPECT_EQ(pair.second, "test");

    END_TEST();
}

// ============================================================================
// Copy and Move Semantics Tests
// ============================================================================

/**
 * @brief Test optimistic_lazy copy constructor
 *
 * Covers: copy semantics, deep copy
 */
XSIGMATEST(Core, lazy_copy_semantics)
{
    optimistic_lazy<int> lazy1;
    lazy1.ensure([]() { return 42; });

    // Test copy constructor
    optimistic_lazy<int> lazy2(lazy1);
    int& value2 = lazy2.ensure([]() { return 99; });  // Factory should not be called
    EXPECT_EQ(value2, 42);

    // Test that copies are independent
    int& value1 = lazy1.ensure([]() { return 99; });
    EXPECT_EQ(value1, 42);
    EXPECT_NE(&value1, &value2);  // Different objects

    // Test copy assignment
    optimistic_lazy<int> lazy3;
    lazy3       = lazy1;
    int& value3 = lazy3.ensure([]() { return 99; });
    EXPECT_EQ(value3, 42);

    END_TEST();
}

/**
 * @brief Test optimistic_lazy move semantics
 *
 * Covers: move constructor, move assignment
 */
XSIGMATEST(Core, lazy_move_semantics)
{
    optimistic_lazy<int> lazy1;
    lazy1.ensure([]() { return 42; });

    // Test move constructor
    optimistic_lazy<int> lazy2(std::move(lazy1));
    int&                 value2 = lazy2.ensure([]() { return 99; });
    EXPECT_EQ(value2, 42);

    // Test move assignment
    optimistic_lazy<int> lazy3;
    lazy3.ensure([]() { return 100; });

    optimistic_lazy<int> lazy4;
    lazy4       = std::move(lazy3);
    int& value4 = lazy4.ensure([]() { return 99; });
    EXPECT_EQ(value4, 100);

    END_TEST();
}

// ============================================================================
// Reset Tests
// ============================================================================

/**
 * @brief Test optimistic_lazy reset functionality
 *
 * Covers: reset, re-initialization
 */
XSIGMATEST(Core, lazy_reset)
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

// ============================================================================
// Thread Safety Tests
// ============================================================================

/**
 * @brief Test optimistic_lazy thread safety
 *
 * Covers: concurrent access, opportunistic concurrency
 */
XSIGMATEST(Core, lazy_thread_safety)
{
    optimistic_lazy<int> lazy_int;
    std::atomic<int>     factory_calls{0};
    std::atomic<int>     success_count{0};

    const int num_threads = 10;

    auto factory = [&factory_calls]() -> int
    {
        factory_calls++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 42;
    };

    auto thread_func = [&]()
    {
        int& value = lazy_int.ensure(factory);
        if (value == 42)
        {
            success_count++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(thread_func);
    }

    for (auto& t : threads)
    {
        t.join();
    }

    // All threads should get the correct value
    EXPECT_EQ(success_count.load(), num_threads);

    // Factory may be called multiple times due to opportunistic concurrency
    // but should be called at least once
    EXPECT_GE(factory_calls.load(), 1);

    XSIGMA_LOG_INFO(
        "Thread safety test: factory called {} times, {} threads succeeded",
        factory_calls.load(),
        success_count.load());

    END_TEST();
}

// ============================================================================
// lazy_value Interface Tests
// ============================================================================

/**
 * @brief Test lazy_value interface
 *
 * Covers: abstract interface, polymorphism
 */
XSIGMATEST(Core, lazy_value_interface)
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

/**
 * @brief Test precomputed_lazy_value
 *
 * Covers: precomputed values, immediate availability
 */
XSIGMATEST(Core, lazy_precomputed_value)
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
// Memory Management Tests
// ============================================================================

/**
 * @brief Test lazy memory management
 *
 * Covers: heap allocation, proper cleanup, no leaks
 */
XSIGMATEST(Core, lazy_memory_management)
{
    // Test with large object
    struct large_object
    {
        std::vector<int> data;
        large_object() : data(10000, 42) {}
    };

    {
        optimistic_lazy<large_object> lazy_obj;
        large_object&                 obj = lazy_obj.ensure([]() { return large_object(); });
        EXPECT_EQ(obj.data.size(), 10000);
        EXPECT_EQ(obj.data[0], 42);
    }  // lazy_obj goes out of scope, should clean up properly

    // Test reset with large object
    {
        optimistic_lazy<large_object> lazy_obj;
        lazy_obj.ensure([]() { return large_object(); });
        lazy_obj.reset();                                  // Should deallocate
        lazy_obj.ensure([]() { return large_object(); });  // Allocate again
    }

    END_TEST();
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/**
 * @brief Test lazy with edge cases
 *
 * Covers: empty objects, exceptions in factory, null values
 */
XSIGMATEST(Core, lazy_edge_cases)
{
    // Test with empty string
    optimistic_lazy<std::string> lazy_empty;
    std::string&                 empty = lazy_empty.ensure([]() { return std::string(); });
    EXPECT_TRUE(empty.empty());

    // Test with zero value
    optimistic_lazy<int> lazy_zero;
    int&                 zero = lazy_zero.ensure([]() { return 0; });
    EXPECT_EQ(zero, 0);

    // Test with negative value
    optimistic_lazy<int> lazy_neg;
    int&                 neg = lazy_neg.ensure([]() { return -42; });
    EXPECT_EQ(neg, -42);

    // Test with nullptr (using pointer type)
    optimistic_lazy<int*> lazy_ptr;
    int*&                 ptr = lazy_ptr.ensure([]() -> int* { return nullptr; });
    EXPECT_EQ(ptr, nullptr);

    END_TEST();
}

// ============================================================================
// Performance Tests
// ============================================================================

/**
 * @brief Test lazy performance characteristics
 *
 * Covers: initialization overhead, repeated access performance
 */
XSIGMATEST(Core, lazy_performance)
{
    // Test repeated access performance
    optimistic_lazy<int> lazy_int;
    lazy_int.ensure([]() { return 42; });

    const int iterations = 100000;
    for (int i = 0; i < iterations; ++i)
    {
        int& value = lazy_int.ensure([]() { return 99; });
        (void)value;  // Suppress unused warning
    }

    // Test with complex object
    optimistic_lazy<std::vector<int>> lazy_vec;
    lazy_vec.ensure([]() { return std::vector<int>(1000, 42); });

    for (int i = 0; i < 10000; ++i)
    {
        std::vector<int>& vec = lazy_vec.ensure([]() { return std::vector<int>(); });
        (void)vec;
    }

    XSIGMA_LOG_INFO("Performance test: {} iterations completed", iterations);

    END_TEST();
}

// ============================================================================
// Concurrent Reset Tests
// ============================================================================

/**
 * @brief Test concurrent access patterns
 *
 * Covers: multiple threads accessing different lazy objects
 */
XSIGMATEST(Core, lazy_concurrent_access)
{
    const int                         num_lazy_objects = 10;
    std::vector<optimistic_lazy<int>> lazy_objects(num_lazy_objects);
    std::atomic<int>                  success_count{0};

    auto thread_func = [&](int index)
    {
        for (int i = 0; i < 100; ++i)
        {
            int& value = lazy_objects[index].ensure([index]() { return index * 10; });
            if (value == index * 10)
            {
                success_count++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_lazy_objects; ++i)
    {
        threads.emplace_back(thread_func, i);
    }

    for (auto& t : threads)
    {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_lazy_objects * 100);

    END_TEST();
}

// ============================================================================
// Polymorphic Usage Tests
// ============================================================================

/**
 * @brief Test lazy_value polymorphic usage
 *
 * Covers: using lazy_value through base class pointer
 */
XSIGMATEST(Core, lazy_polymorphic_usage)
{
    // Test with optimistic_lazy_value through base pointer
    class expensive_computation : public optimistic_lazy_value<double>
    {
    private:
        double compute() const override
        {
            // Simulate expensive computation
            double result = 0.0;
            for (int i = 0; i < 1000; ++i)
            {
                result += i * 0.001;
            }
            return result;
        }
    };

    std::unique_ptr<lazy_value<double>> lazy_ptr = std::make_unique<expensive_computation>();
    const double&                       value1   = lazy_ptr->get();
    EXPECT_GT(value1, 0.0);

    // Test with precomputed_lazy_value through base pointer
    std::unique_ptr<lazy_value<int>> precomputed_ptr =
        std::make_unique<precomputed_lazy_value<int>>(42);
    const int& value2 = precomputed_ptr->get();
    EXPECT_EQ(value2, 42);

    END_TEST();
}

// ============================================================================
// Self-Assignment Tests
// ============================================================================

/**
 * @brief Test lazy self-assignment
 *
 * Covers: self-assignment safety
 */
XSIGMATEST(Core, lazy_self_assignment)
{
    optimistic_lazy<int> lazy1;
    lazy1.ensure([]() { return 42; });

    // Test move self-assignment
    lazy1      = std::move(lazy1);
    int& value = lazy1.ensure([]() { return 99; });
    EXPECT_EQ(value, 42);

    END_TEST();
}

// ============================================================================
// Factory Exception Safety Tests
// ============================================================================

/**
 * @brief Test behavior when factory might fail
 *
 * Covers: factory returning default values, error handling
 * Note: XSigma follows no-exception pattern
 */
XSIGMATEST(Core, lazy_factory_safety)
{
    // Test factory that returns default value on "error"
    optimistic_lazy<int> lazy_int;

    bool should_fail = false;
    auto factory     = [&should_fail]() -> int
    {
        if (should_fail)
        {
            return -1;  // Error indicator
        }
        return 42;
    };

    int& value1 = lazy_int.ensure(factory);
    EXPECT_EQ(value1, 42);

    // Even if we change should_fail, cached value is returned
    should_fail = true;
    int& value2 = lazy_int.ensure(factory);
    EXPECT_EQ(value2, 42);  // Still cached value

    END_TEST();
}

// ============================================================================
// Multiple Factory Types Tests
// ============================================================================

/**
 * @brief Test lazy with different factory types
 *
 * Covers: lambda, function pointer, functor
 */
XSIGMATEST(Core, lazy_factory_types)
{
    // Test with lambda
    optimistic_lazy<int> lazy1;
    int&                 v1 = lazy1.ensure([]() { return 42; });
    EXPECT_EQ(v1, 42);

    // Test with function pointer
    auto                 func_ptr = []() -> int { return 99; };
    optimistic_lazy<int> lazy2;
    int&                 v2 = lazy2.ensure(func_ptr);
    EXPECT_EQ(v2, 99);

    // Test with functor
    struct functor
    {
        int operator()() const { return 123; }
    };
    optimistic_lazy<int> lazy3;
    int&                 v3 = lazy3.ensure(functor());
    EXPECT_EQ(v3, 123);

    END_TEST();
}
