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

#include "common/configure.h"
#include "common/macros.h"
#include "logging/logger.h"
#include "util/exception.h"
#include "util/string_util.h"
#include "xsigmaTest.h"

using namespace xsigma;

// ============================================================================
// Basic Exception Tests
// ============================================================================

XSIGMATEST(Core, ExceptionBasic)
{
#ifndef NDEBUG
    ASSERT_ANY_THROW({ XSIGMA_CHECK_DEBUG(false, "XSIGMA_CHECK_DEBUG: Should throw"); });
#endif

    ASSERT_ANY_THROW({ XSIGMA_CHECK(false, "XSIGMA_CHECK: Should throw"); });

    ASSERT_ANY_THROW({ XSIGMA_THROW("XSIGMA_THROW: should throw"); });

    END_TEST();
}

// ============================================================================
// Exception Mode Tests
// ============================================================================

XSIGMATEST(Core, ExceptionMode)
{
    // Test default mode
    auto default_mode = xsigma::get_exception_mode();
    XSIGMA_LOG_INFO(
        "Default exception mode: {}",
        default_mode == xsigma::exception_mode::THROW ? "THROW" : "LOG_FATAL");

    // Test mode switching
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);
    ASSERT_EQ(xsigma::get_exception_mode(), xsigma::exception_mode::THROW);

    xsigma::set_exception_mode(xsigma::exception_mode::LOG_FATAL);
    ASSERT_EQ(xsigma::get_exception_mode(), xsigma::exception_mode::LOG_FATAL);

    // Restore default mode for other tests
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    END_TEST();
}

// ============================================================================
// Exception Chaining Tests
// ============================================================================

XSIGMATEST(Core, ExceptionChaining)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Create a nested exception using new/shared_ptr to avoid MSVC ICE
    xsigma::source_location            inner_loc{__func__, __FILE__, __LINE__};
    std::shared_ptr<xsigma::exception> inner(new xsigma::exception(
        inner_loc,
        "Inner error: database connection failed",
        xsigma::exception_category::RUNTIME_ERROR));

    // Create outer exception with nested
    xsigma::source_location outer_loc{__func__, __FILE__, __LINE__};
    xsigma::exception       outer(outer_loc, "Outer error: failed to process request", inner);

    // Check nested exception is accessible
    ASSERT_TRUE(outer.nested() != nullptr);
    ASSERT_EQ(outer.nested(), inner);

    // Check what() includes nested information
    std::string full_msg(outer.what());
    ASSERT_TRUE(full_msg.find("Outer error") != std::string::npos);
    ASSERT_TRUE(full_msg.find("Caused by") != std::string::npos);
    ASSERT_TRUE(full_msg.find("Inner error") != std::string::npos);

    XSIGMA_LOG_INFO("Exception chain:\n{}", outer.what());

    END_TEST();
}

// ============================================================================
// Context Accumulation Tests
// ============================================================================

XSIGMATEST(Core, ExceptionContext)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    try
    {
        auto e = xsigma::exception(
            xsigma::source_location{__func__, __FILE__, __LINE__},
            "Base error",
            xsigma::exception_category::GENERIC);

        e.add_context("Context 1: processing file");
        e.add_context("Context 2: parsing line 42");
        e.add_context("Context 3: invalid token");

        // Check context is accumulated
        const auto& contexts = e.context();
        ASSERT_EQ(contexts.size(), 3);
        ASSERT_EQ(contexts[0], "Context 1: processing file");
        ASSERT_EQ(contexts[1], "Context 2: parsing line 42");
        ASSERT_EQ(contexts[2], "Context 3: invalid token");

        // Check what() includes all context
        std::string full_msg(e.what());
        ASSERT_TRUE(full_msg.find("Context 1") != std::string::npos);
        ASSERT_TRUE(full_msg.find("Context 2") != std::string::npos);
        ASSERT_TRUE(full_msg.find("Context 3") != std::string::npos);

        XSIGMA_LOG_INFO("Exception with context:\n{}", e.what());
    }
    catch (...)
    {
        FAIL() << "Should not throw in this test";
    }

    END_TEST();
}

#if 0
// ============================================================================
// Error Category Tests
// ============================================================================

XSIGMATEST(Core, ExceptionErrorCategory)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test exception category to string conversion
    ASSERT_STREQ(xsigma::enum_to_string(xsigma::exception_category::GENERIC).data(), "GENERIC");
    ASSERT_STREQ(
        xsigma::enum_to_string(xsigma::exception_category::VALUE_ERROR).data(), "VALUE_ERROR");
    ASSERT_STREQ(
        xsigma::enum_to_string(xsigma::exception_category::TYPE_ERROR).data(), "TYPE_ERROR");
    ASSERT_STREQ(
        xsigma::enum_to_string(xsigma::exception_category::INDEX_ERROR).data(), "INDEX_ERROR");
    ASSERT_STREQ(
        xsigma::enum_to_string(xsigma::exception_category::NOT_IMPLEMENTED).data(),
        "NOT_IMPLEMENTED");
    ASSERT_STREQ(
        xsigma::enum_to_string(xsigma::exception_category::ENFORCE_FINITE).data(),
        "ENFORCE_FINITE");

    // Test exception with different categories
    try
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, "Value error test", xsigma::exception_category::VALUE_ERROR);
    }
    catch (const xsigma::exception& e)
    {
        //ASSERT_EQ(e.category(), xsigma::exception_category::VALUE_ERROR);
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Value error test") != std::string::npos);
    }

    // Test exception with TYPE_ERROR category
    try
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, "Type error test", xsigma::exception_category::TYPE_ERROR);
    }
    catch (const xsigma::exception& e)
    {
        ASSERT_EQ(e.category(), xsigma::exception_category::TYPE_ERROR);
    }

    // Test exception with GENERIC category
    try
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, "Generic error test", xsigma::exception_category::GENERIC);
    }
    catch (const xsigma::exception& e)
    {
        ASSERT_EQ(e.category(), xsigma::exception_category::GENERIC);
    }

    END_TEST();
}

// ============================================================================
// Stack Trace Tests
// ============================================================================

XSIGMATEST(Core, ExceptionStackTrace)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    try
    {
        XSIGMA_THROW("Test exception with stack trace");
        FAIL() << "Should have thrown";
    }
    catch (const xsigma::exception& e)
    {
        // Check that backtrace is captured
        const std::string& backtrace = e.backtrace();
        ASSERT_FALSE(backtrace.empty());

        // Backtrace should contain "Exception raised from"
        ASSERT_TRUE(backtrace.find("Exception raised from") != std::string::npos);

        XSIGMA_LOG_INFO("Exception with backtrace:\n{}", e.what());
    }

    END_TEST();
}

// ============================================================================
// Specialized Exception Types Tests
// ============================================================================

XSIGMATEST(Core, ExceptionTypes)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test exception with VALUE_ERROR category
    try
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, "Invalid value", xsigma::exception_category::VALUE_ERROR);
    }
    catch (const xsigma::exception& e)
    {
        XSIGMA_LOG_INFO("Caught exception with VALUE_ERROR: {}", e.msg());
        ASSERT_EQ(e.category(), xsigma::exception_category::VALUE_ERROR);
    }
    catch (...)
    {
        FAIL() << "Should have caught exception";
    }

    // Test exception with TYPE_ERROR category
    try
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, "Invalid type", xsigma::exception_category::TYPE_ERROR);
    }
    catch (const xsigma::exception& e)
    {
        XSIGMA_LOG_INFO("Caught exception with TYPE_ERROR: {}", e.msg());
        ASSERT_EQ(e.category(), xsigma::exception_category::TYPE_ERROR);
    }
    catch (...)
    {
        FAIL() << "Should have caught exception";
    }

    // Test exception with INDEX_ERROR category
    try
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(
            loc, "Index out of bounds", xsigma::exception_category::INDEX_ERROR);
    }
    catch (const xsigma::exception& e)
    {
        XSIGMA_LOG_INFO("Caught exception with INDEX_ERROR: {}", e.msg());
        ASSERT_EQ(e.category(), xsigma::exception_category::INDEX_ERROR);
    }
    catch (...)
    {
        FAIL() << "Should have caught exception";
    }

    // Test NOT_IMPLEMENTED category using macro
    try
    {
        XSIGMA_NOT_IMPLEMENTED("Feature not yet implemented");
        FAIL() << "Should have thrown exception";
    }
    catch (const xsigma::exception& e)
    {
        XSIGMA_LOG_INFO("Caught exception with NOT_IMPLEMENTED: {}", e.msg());
        ASSERT_EQ(e.category(), xsigma::exception_category::NOT_IMPLEMENTED);
    }
    catch (...)
    {
        FAIL() << "Should have caught exception";
    }

    END_TEST();
}

// ============================================================================
// Cross-Platform Tests
// ============================================================================

XSIGMATEST(Core, ExceptionCrossPlatform)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test that exceptions work consistently across platforms
    try
    {
        XSIGMA_THROW("Platform test: {}", "cross-platform");
        FAIL() << "Should have thrown";
    }
    catch (const xsigma::exception& e)
    {
        // Should work on Windows, Linux, and macOS
        ASSERT_TRUE(e.what() != nullptr);
        ASSERT_FALSE(std::string(e.what()).empty());

#ifdef _WIN32
        XSIGMA_LOG_INFO("Running on Windows");
#elif defined(__linux__)
        XSIGMA_LOG_INFO("Running on Linux");
#elif defined(__APPLE__)
        XSIGMA_LOG_INFO("Running on macOS");
#else
        XSIGMA_LOG_INFO("Running on unknown platform");
#endif

        XSIGMA_LOG_INFO("Exception message: {}", e.what());
    }

    END_TEST();
}

// ============================================================================
// Additional Exception Category Tests
// ============================================================================

XSIGMATEST(Core, exception_all_categories)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test all exception categories
    const std::vector<xsigma::exception_category> categories = {
        xsigma::exception_category::GENERIC,
        xsigma::exception_category::VALUE_ERROR,
        xsigma::exception_category::TYPE_ERROR,
        xsigma::exception_category::INDEX_ERROR,
        xsigma::exception_category::NOT_IMPLEMENTED,
        xsigma::exception_category::ENFORCE_FINITE,
        xsigma::exception_category::RUNTIME_ERROR,
        xsigma::exception_category::LOGIC_ERROR,
        xsigma::exception_category::SYSTEM_ERROR,
        xsigma::exception_category::MEMORY_ERROR};

    for (const auto& cat : categories)
    {
        try
        {
            xsigma::source_location loc{__func__, __FILE__, __LINE__};
            throw xsigma::exception(loc, "Test exception", cat);
        }
        catch (const xsigma::exception& e)
        {
            ASSERT_EQ(e.category(), cat);
            ASSERT_FALSE(std::string(e.what()).empty());
        }
    }

    END_TEST();
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

XSIGMATEST(Core, exception_edge_cases)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test empty message
    try
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, "", xsigma::exception_category::GENERIC);
    }
    catch (const xsigma::exception& e)
    {
        // Should not crash with empty message
        ASSERT_TRUE(e.what() != nullptr);
    }

    // Test very long message
    try
    {
        std::string             long_msg(10000, 'x');
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, long_msg, xsigma::exception_category::GENERIC);
    }
    catch (const xsigma::exception& e)
    {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("xxx") != std::string::npos);
    }

    // Test message with special characters
    try
    {
        XSIGMA_THROW("Special chars: \n\t\r\\\"\'");
    }
    catch (const xsigma::exception& e)
    {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Special chars") != std::string::npos);
    }

    // Test message with Unicode (if supported)
    try
    {
        XSIGMA_THROW("Unicode: \u03C0 \u2248 3.14");
    }
    catch (const xsigma::exception& e)
    {
        // Should not crash with Unicode
        ASSERT_TRUE(e.what() != nullptr);
    }

    END_TEST();
}

// ============================================================================
// Source Location Tests
// ============================================================================

XSIGMATEST(Core, exception_source_location)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    const int test_line = __LINE__ + 3;
    try
    {
        XSIGMA_THROW("Test source location");
    }
    catch (const xsigma::exception& e)
    {
        // Check that backtrace contains file and line information
        const std::string& backtrace = e.backtrace();
        ASSERT_FALSE(backtrace.empty());
        ASSERT_TRUE(backtrace.find("TestException.cxx") != std::string::npos);

        // Verify line number is captured (approximate check)
        std::string line_str = std::to_string(test_line);
        ASSERT_TRUE(backtrace.find(line_str) != std::string::npos);
    }

    END_TEST();
}

#endif

// ============================================================================
// Memory Safety Tests
// ============================================================================

XSIGMATEST(Core, exception_memory_safety)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test exception with shared_ptr (no memory leaks)
    {
        std::shared_ptr<xsigma::exception> ex_ptr;
        try
        {
            xsigma::source_location loc{__func__, __FILE__, __LINE__};
            ex_ptr = std::make_shared<xsigma::exception>(
                loc, "Shared ptr exception", xsigma::exception_category::GENERIC);
        }
        catch (...)
        {
            FAIL() << "Should not throw during construction";
        }

        ASSERT_TRUE(ex_ptr != nullptr);
        ASSERT_FALSE(std::string(ex_ptr->what()).empty());
    }

    // Test exception copy
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        xsigma::exception original(loc, "Original exception", xsigma::exception_category::GENERIC);

        // Copy constructor
        xsigma::exception copy(original);
        ASSERT_EQ(std::string(copy.msg()), std::string(original.msg()));
        ASSERT_EQ(copy.category(), original.category());
    }

    END_TEST();
}

// ============================================================================
// Thread Safety Tests
// ============================================================================
#if 0
XSIGMATEST(Core, exception_thread_safety)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    std::atomic<int> exception_count{0};
    std::atomic<int> success_count{0};
    const int        num_threads           = 10;
    const int        iterations_per_thread = 100;

    auto thread_func = [&]()
    {
        for (int i = 0; i < iterations_per_thread; ++i)
        {
            try
            {
                XSIGMA_THROW("Thread exception {}", i);
            }
            catch (const xsigma::exception& e)
            {
                exception_count++;
                // Verify exception is valid
                if (e.what() != nullptr &&
                    std::string(e.what()).find("Thread exception") != std::string::npos)
                {
                    success_count++;
                }
            }
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

    ASSERT_EQ(exception_count.load(), num_threads * iterations_per_thread);
    ASSERT_EQ(success_count.load(), num_threads * iterations_per_thread);

    XSIGMA_LOG_INFO(
        "Thread safety test: {} exceptions thrown successfully", exception_count.load());

    END_TEST();
}

// ============================================================================
// Exception Mode Thread Safety Tests
// ============================================================================

XSIGMATEST(Core, exception_mode_thread_safety)
{
    // Test that exception mode can be safely read from multiple threads
    std::atomic<int> read_count{0};
    const int        num_threads      = 10;
    const int        reads_per_thread = 24;

    auto read_func = [&]()
    {
        for (int i = 0; i < reads_per_thread; ++i)
        {
            xsigma::exception_mode mode = xsigma::get_exception_mode();
            (void)mode;  // Suppress unused variable warning
            read_count++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(read_func);
    }

    for (auto& t : threads)
    {
        t.join();
    }

    ASSERT_EQ(read_count.load(), num_threads * reads_per_thread);

    END_TEST();
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

XSIGMATEST(Core, exception_performance)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test repeated exception creation and destruction
    const int iterations = 2;
    for (int i = 0; i < iterations; ++i)
    {
        try
        {
            XSIGMA_THROW("Performance test iteration {}", i);
        }
        catch (const xsigma::exception& e)
        {
            // Verify exception is valid
            ASSERT_TRUE(e.what() != nullptr);
        }
    }

    XSIGMA_LOG_INFO("Performance test: {} exceptions created successfully", iterations);

    END_TEST();
}

// ============================================================================
// Nested Exception Depth Tests
// ============================================================================

XSIGMATEST(Core, exception_nested_depth)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Create a chain of nested exceptcdxions
    std::shared_ptr<xsigma::exception> current;
    const int                          depth = 5;

    for (int i = 0; i < depth; ++i)
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        std::string             msg = fmt::format("Exception level {}", i);
        current                     = std::make_shared<xsigma::exception>(loc, msg, current);
    }

    // Verify the chain
    int  count = 0;
    auto ptr   = current;
    while (ptr != nullptr)
    {
        count++;
        ptr = ptr->nested();
    }

    ASSERT_EQ(count, depth);

    // Verify what() includes all levels
    std::string full_msg(current->what());
    for (int i = 0; i < depth; ++i)
    {
        std::string level_msg = fmt::format("Exception level {}", i);
        ASSERT_TRUE(full_msg.find(level_msg) != std::string::npos);
    }

    END_TEST();
}

// ============================================================================
// Context Accumulation Stress Tests
// ============================================================================

XSIGMATEST(Core, exception_context_stress)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    xsigma::source_location loc{__func__, __FILE__, __LINE__};
    xsigma::exception       ex(loc, "Base exception", xsigma::exception_category::GENERIC);

    // Add many context entries
    const int context_count = 2;
    for (int i = 0; i < context_count; ++i)
    {
        ex.add_context(fmt::format("Context entry {}", i));
    }

    // Verify all contexts are present
    const auto& contexts = ex.context();
    ASSERT_EQ(contexts.size(), static_cast<size_t>(context_count));

    for (int i = 0; i < context_count; ++i)
    {
        std::string expected = fmt::format("Context entry {}", i);
        ASSERT_EQ(contexts[i], expected);
    }

    END_TEST();
}
#endif