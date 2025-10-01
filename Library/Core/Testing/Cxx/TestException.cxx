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

#include "common/configure.h"
#include "common/macros.h"
#include "logging/logger.h"
#include "util/exception.h"
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
#if 0
// ============================================================================
// Error Category Tests
// ============================================================================

XSIGMATEST(Core, ExceptionErrorCategory)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test error category to string conversion
    ASSERT_STREQ(xsigma::error_category_to_string(xsigma::error_category_enum::GENERIC), "GENERIC");
    ASSERT_STREQ(xsigma::error_category_to_string(xsigma::error_category_enum::VALUE_ERROR), "VALUE_ERROR");
    ASSERT_STREQ(xsigma::error_category_to_string(xsigma::error_category_enum::TYPE_ERROR), "TYPE_ERROR");
    ASSERT_STREQ(xsigma::error_category_to_string(xsigma::error_category_enum::INDEX_ERROR), "INDEX_ERROR");
    ASSERT_STREQ(xsigma::error_category_to_string(xsigma::error_category_enum::NOT_IMPLEMENTED), "NOT_IMPLEMENTED");
    ASSERT_STREQ(xsigma::error_category_to_string(xsigma::error_category_enum::ENFORCE_FINITE), "ENFORCE_FINITE");

    // Test exception with different categories
    try
    {
        xsigma::SourceLocation loc{__func__, __FILE__, __LINE__};
        throw xsigma::exception(loc, "Value error test", xsigma::exception_category::VALUE_ERROR);
    }
    catch (const xsigma::exception& e)
    {
        ASSERT_EQ(e.category(), xsigma::error_category_enum::VALUE_ERROR);
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Value error test") != std::string::npos);
    }

    // Test backward compatibility aliases
    try
    {
        xsigma::SourceLocation loc{__func__, __FILE__, __LINE__};
        throw xsigma::ValueError(loc, "Using ValueError alias", xsigma::error_category_enum::VALUE_ERROR);
    }
    catch (const xsigma::exception& e)
    {
        ASSERT_EQ(e.category(), xsigma::error_category_enum::VALUE_ERROR);
    }

    // Test that Error alias works
    try
    {
        xsigma::SourceLocation loc{__func__, __FILE__, __LINE__};
        throw xsigma::Error(loc, "Using Error alias");
    }
    catch (const xsigma::exception& e)
    {
        ASSERT_EQ(e.category(), xsigma::error_category_enum::GENERIC);
    }

    END_TEST();
}

// ============================================================================
// fmt-Style Formatting Tests
// ============================================================================

XSIGMATEST(Core, ExceptionFmtFormatting)
{
    // Ensure we're in THROW mode for these tests
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test XSIGMA_THROW with fmt formatting
    try
    {
        int value = 42;
        XSIGMA_THROW("Test error with value: {}", value);
        FAIL() << "Should have thrown";
    }
    catch (const xsigma::Error& e)
    {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("42") != std::string::npos);
        XSIGMA_LOG_INFO("Caught expected exception: {}", e.what());
    }

    // Test XSIGMA_CHECK with fmt formatting
    try
    {
        int x = -5;
        XSIGMA_CHECK(x > 0, "x must be positive, got {}", x);
        FAIL() << "Should have thrown";
    }
    catch (const xsigma::Error& e)
    {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("-5") != std::string::npos);
        XSIGMA_LOG_INFO("Caught expected exception: {}", e.what());
    }

    // Test XSIGMA_CHECK with formatted double
    try
    {
        double value = 3.14159;
        XSIGMA_CHECK(value < 3.0, "Value too large: {:.2f}", value);
        FAIL() << "Should have thrown";
    }
    catch (const xsigma::Error& e)
    {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("3.14") != std::string::npos);
        XSIGMA_LOG_INFO("Caught expected exception: {}", e.what());
    }

    // Test multiple format arguments
    try
    {
        XSIGMA_THROW("Multiple values: {}, {}, {}", 1, 2.5, "test");
        FAIL() << "Should have thrown";
    }
    catch (const xsigma::Error& e)
    {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("1") != std::string::npos);
        ASSERT_TRUE(msg.find("2.5") != std::string::npos);
        ASSERT_TRUE(msg.find("test") != std::string::npos);
        XSIGMA_LOG_INFO("Caught expected exception: {}", e.what());
    }

    END_TEST();
}


// ============================================================================
// Exception Chaining Tests
// ============================================================================

XSIGMATEST(Core, ExceptionChaining)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Create a nested exception using new/shared_ptr to avoid MSVC ICE
    xsigma::SourceLocation inner_loc{__func__, __FILE__, __LINE__};
    std::shared_ptr<xsigma::exception> inner(
        new xsigma::exception(inner_loc, "Inner error: database connection failed"));

    // Create outer exception with nested
    xsigma::SourceLocation outer_loc{__func__, __FILE__, __LINE__};
    xsigma::exception outer(outer_loc, "Outer error: failed to process request", inner);

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
    catch (const xsigma::Error& e)
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
// Context Accumulation Tests
// ============================================================================

XSIGMATEST(Core, ExceptionContext)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    try
    {
        auto e = xsigma::Error(
            xsigma::SourceLocation{__func__, __FILE__, __LINE__},
            "Base error");

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

// ============================================================================
// Specialized Exception Types Tests
// ============================================================================

XSIGMATEST(Core, ExceptionTypes)
{
    xsigma::set_exception_mode(xsigma::exception_mode::THROW);

    // Test ValueError (using error category)
    try
    {
        xsigma::SourceLocation loc{__func__, __FILE__, __LINE__};
        throw xsigma::ValueError(loc, "Invalid value", xsigma::error_category::VALUE_ERROR);
    }
    catch (const xsigma::ValueError& e)
    {
        XSIGMA_LOG_INFO("Caught ValueError: {}", e.msg());
        ASSERT_EQ(e.category(), xsigma::error_category::VALUE_ERROR);
    }
    catch (...)
    {
        FAIL() << "Should have caught ValueError";
    }

    // Test TypeError (using error category)
    try
    {
        xsigma::SourceLocation loc{__func__, __FILE__, __LINE__};
        throw xsigma::TypeError(loc, "Invalid type", xsigma::error_category::TYPE_ERROR);
    }
    catch (const xsigma::TypeError& e)
    {
        XSIGMA_LOG_INFO("Caught TypeError: {}", e.msg());
        ASSERT_EQ(e.category(), xsigma::error_category::TYPE_ERROR);
    }
    catch (...)
    {
        FAIL() << "Should have caught TypeError";
    }

    // Test IndexError (using error category)
    try
    {
        xsigma::SourceLocation loc{__func__, __FILE__, __LINE__};
        throw xsigma::IndexError(loc, "Index out of bounds", xsigma::error_category::INDEX_ERROR);
    }
    catch (const xsigma::IndexError& e)
    {
        XSIGMA_LOG_INFO("Caught IndexError: {}", e.msg());
        ASSERT_EQ(e.category(), xsigma::error_category::INDEX_ERROR);
    }
    catch (...)
    {
        FAIL() << "Should have caught IndexError";
    }

    // Test NotImplementedError using macro
    try
    {
        XSIGMA_NOT_IMPLEMENTED("Feature not yet implemented");
        FAIL() << "Should have thrown NotImplementedError";
    }
    catch (const xsigma::NotImplementedError& e)
    {
        XSIGMA_LOG_INFO("Caught NotImplementedError: {}", e.msg());
        ASSERT_EQ(e.category(), xsigma::error_category::NOT_IMPLEMENTED);
    }
    catch (...)
    {
        FAIL() << "Should have caught NotImplementedError";
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
    catch (const xsigma::Error& e)
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

#endif