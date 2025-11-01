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

#include "util/exception.h"
#include "xsigmaTest.h"

using namespace xsigma;

// ============================================================================
// Basic Exception Tests
// ============================================================================

XSIGMATEST(Exception, basic_macros)
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

XSIGMATEST(Exception, mode_configuration)
{
    // Test default mode
    XSIGMA_UNUSED auto default_mode = xsigma::get_exception_mode();

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

XSIGMATEST(Exception, chaining)
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

    END_TEST();
}

// ============================================================================
// Context Accumulation Tests
// ============================================================================

XSIGMATEST(Exception, context_accumulation)
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
    }
    catch (...)
    {
        FAIL() << "Should not throw in this test";
    }

    END_TEST();
}

// ============================================================================
// Error Category Tests
// ============================================================================

XSIGMATEST(Exception, error_categories)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test exception with different categories
    try
    {
        xsigma::source_location loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, "Value error test", xsigma::exception_category::VALUE_ERROR);
    }
    catch (const xsigma::exception& e)
    {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Value error test") != std::string::npos);
    }
#if 0
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
#endif

    END_TEST();
}

// ============================================================================
// Stack Trace Tests
// ============================================================================

XSIGMATEST(Exception, stack_trace)
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
    }

    END_TEST();
}

// ============================================================================
// NOT_IMPLEMENTED Macro Test
// ============================================================================

XSIGMATEST(Exception, not_implemented_macro)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test NOT_IMPLEMENTED category using macro
    try
    {
        XSIGMA_NOT_IMPLEMENTED("Feature not yet implemented");
        FAIL() << "Should have thrown exception";
    }
    catch (const xsigma::exception& e)
    {
        ASSERT_EQ(e.category(), xsigma::exception_category::NOT_IMPLEMENTED);
        ASSERT_TRUE(std::string(e.msg()).find("Feature not yet implemented") != std::string::npos);
    }

    END_TEST();
}

// ============================================================================
// Cross-Platform Tests
// ============================================================================

XSIGMATEST(Exception, cross_platform)
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
    }

    END_TEST();
}

// ============================================================================
// All Exception Categories Test
// ============================================================================

XSIGMATEST(Exception, all_categories)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test all exception categories
    const std::vector<xsigma::exception_category> categories = {
        xsigma::exception_category::GENERIC  // ,
        // xsigma::exception_category::VALUE_ERROR,
        // xsigma::exception_category::TYPE_ERROR,
        // xsigma::exception_category::INDEX_ERROR,
        // xsigma::exception_category::NOT_IMPLEMENTED,
        // xsigma::exception_category::ENFORCE_FINITE,
        // xsigma::exception_category::RUNTIME_ERROR,
        // xsigma::exception_category::LOGIC_ERROR,
        // xsigma::exception_category::SYSTEM_ERROR,
        // xsigma::exception_category::MEMORY_ERROR
    };

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

XSIGMATEST(Exception, edge_cases)
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

    END_TEST();
}

// ============================================================================
// Source Location Tests
// ============================================================================

XSIGMATEST(Exception, source_location)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

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
    }

    END_TEST();
}

// ============================================================================
// Memory Safety Tests
// ============================================================================

XSIGMATEST(Exception, memory_safety)
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
// Exception Base Constructor Tests
// ============================================================================

XSIGMATEST(Exception, base_constructor)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test base constructor with all parameters
    xsigma::exception ex(
        "Test message", "Test backtrace", nullptr, xsigma::exception_category::VALUE_ERROR);

    ASSERT_EQ(std::string(ex.msg()), "Test message");
    ASSERT_EQ(std::string(ex.backtrace()), "Test backtrace");
    ASSERT_EQ(ex.caller(), nullptr);
    ASSERT_EQ(ex.category(), xsigma::exception_category::VALUE_ERROR);

    END_TEST();
}

XSIGMATEST(Exception, base_constructor_with_caller)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test base constructor with caller pointer
    int               dummy_obj = 42;
    xsigma::exception ex(
        "Test message", "Test backtrace", &dummy_obj, xsigma::exception_category::RUNTIME_ERROR);

    ASSERT_EQ(ex.caller(), &dummy_obj);
    ASSERT_EQ(ex.category(), xsigma::exception_category::RUNTIME_ERROR);

    END_TEST();
}

// ============================================================================
// Exception Accessor Tests
// ============================================================================

XSIGMATEST(Exception, accessors)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    xsigma::source_location loc{__func__, __FILE__, __LINE__};
    xsigma::exception       ex(loc, "Test message", xsigma::exception_category::LOGIC_ERROR);

    // Test msg() accessor
    ASSERT_EQ(std::string(ex.msg()), "Test message");

    // Test category() accessor
    ASSERT_EQ(ex.category(), xsigma::exception_category::LOGIC_ERROR);

    // Test backtrace() accessor
    ASSERT_FALSE(ex.backtrace().empty());
    ASSERT_TRUE(ex.backtrace().find("Exception raised from") != std::string::npos);

    // Test context() accessor
    ASSERT_TRUE(ex.context().empty());  // No context added yet

    // Test caller() accessor
    ASSERT_EQ(ex.caller(), nullptr);

    // Test nested() accessor
    ASSERT_EQ(ex.nested(), nullptr);

    END_TEST();
}

// ============================================================================
// Exception Compute What Tests
// ============================================================================

XSIGMATEST(Exception, compute_what_with_context)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    xsigma::source_location loc{__func__, __FILE__, __LINE__};
    xsigma::exception       ex(loc, "Base message", xsigma::exception_category::GENERIC);

    ex.add_context("Context line 1");
    ex.add_context("Context line 2");

    std::string what_str(ex.what());
    ASSERT_TRUE(what_str.find("Base message") != std::string::npos);
    ASSERT_TRUE(what_str.find("Context line 1") != std::string::npos);
    ASSERT_TRUE(what_str.find("Context line 2") != std::string::npos);

    END_TEST();
}

// ============================================================================
// Exception Category Enum Tests
// ============================================================================

XSIGMATEST(Exception, all_exception_categories)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test all exception categories
    const std::vector<xsigma::exception_category> all_categories = {
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

    for (const auto& cat : all_categories)
    {
        try
        {
            xsigma::source_location loc{__func__, __FILE__, __LINE__};
            throw xsigma::exception(loc, "Test", cat);
        }
        catch (const xsigma::exception& e)
        {
            ASSERT_EQ(e.category(), cat);
        }
    }

    END_TEST();
}