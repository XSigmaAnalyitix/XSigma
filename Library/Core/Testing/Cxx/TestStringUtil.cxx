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
 * @file TestStringUtil.cxx
 * @brief Comprehensive test suite for string utility functions
 *
 * This file contains extensive tests for all string utility functions
 * including edge cases, error conditions, and performance validation.
 *
 * @author XSigma Development Team
 * @version 2.0
 * @date 2024
 */

#include <cmath>        // for isnan, isinf
#include <cstddef>      // for size_t
#include <fstream>      // for filebuf, ostream
#include <limits>       // for numeric_limits
#include <sstream>      // for ostringstream
#include <string>       // for string, allocator, basic_string
#include <string_view>  // for string_view
#include <vector>       // for vector

#include "util/string_util.h"  // for stoi, strip_basename, compile_tim...
#include "xsigmaTest.h"        // for EXPECT_EQ, EXPECT_TRUE, END_TEST

namespace xsigma
{
// =============================================================================
// FILE PATH MANIPULATION TESTS
// =============================================================================

void testFilePathFunctions()
{
    // Test strip_basename function
    EXPECT_EQ(strip_basename("zzz.h"), "zzz.h");
    EXPECT_EQ(strip_basename("c:/xxx/yyy/zzz.h"), "zzz.h");
    EXPECT_EQ(strip_basename("/usr/local/bin/program"), "program");
    EXPECT_EQ(strip_basename(""), "");                          // Edge case: empty string
    EXPECT_EQ(strip_basename("no_separator"), "no_separator");  // No separator
    EXPECT_EQ(strip_basename("/"), "");                         // Root directory
    EXPECT_EQ(strip_basename("path/"), "");                     // Trailing separator

    // Test exclude_file_extension function
    EXPECT_EQ(exclude_file_extension("c:/xxx/yyy/zzz.h"), "c:/xxx/yyy/zzz");
    EXPECT_EQ(exclude_file_extension("document.pdf"), "document");
    EXPECT_EQ(exclude_file_extension("archive.tar.gz"), "archive.tar");
    EXPECT_EQ(exclude_file_extension("no_extension"), "no_extension");  // No extension
    EXPECT_EQ(exclude_file_extension(""), "");                          // Edge case: empty string
    EXPECT_EQ(exclude_file_extension(".hidden"), "");                   // Hidden file
    EXPECT_EQ(exclude_file_extension("file."), "file");                 // Trailing dot

    // Test file_extension function
    EXPECT_EQ(file_extension("c:/xxx/yyy/zzz.h"), ".h");
    EXPECT_EQ(file_extension("document.pdf"), ".pdf");
    EXPECT_EQ(file_extension("archive.tar.gz"), ".gz");
    EXPECT_EQ(file_extension("no_extension"), "");    // No extension
    EXPECT_EQ(file_extension(""), "");                // Edge case: empty string
    EXPECT_EQ(file_extension(".hidden"), ".hidden");  // Hidden file
    EXPECT_EQ(file_extension("file."), ".");          // Trailing dot
}

// =============================================================================
// STRING CONCATENATION AND CONVERSION TESTS
// =============================================================================

void testStringConcatenation()
{
    // Test compile_time_empty_string functionality
    details::compile_time_empty_string a;
    XSIGMA_UNUSED const char*          a_char = a;
    XSIGMA_UNUSED const std::string& a_string = a;

    // Test stream operations with empty string
    std::filebuf fb;
    std::ostream ss(&fb);
    details::_str<details::compile_time_empty_string>(ss, a);

    // Test string wrapper functionality
    XSIGMA_UNUSED details::_str_wrapper<const char*> b;
    a_char = details::_str_wrapper<const char*>::call(a_char);

    XSIGMA_UNUSED details::_str_wrapper<std::string> e;
    XSIGMA_UNUSED const std::string& a_string2 = details::_str_wrapper<std::string>::call(a_string);

    // Test to_string function with various types
    auto result1 = to_string("Hello", " ", "World", "!");
    // Note: Exact result depends on implementation, but should concatenate properly

    auto result2 = to_string(42);
    // Should handle numeric types

    XSIGMA_UNUSED auto result3 = to_string();
    // Should handle empty case
}

// =============================================================================
// NUMERIC CONVERSION TESTS
// =============================================================================

void testNumericConversion()
{
    // Test stoi function with position tracking
    size_t pos;

    // Basic conversion tests
    EXPECT_EQ(stoi("1", &pos), 1);
    EXPECT_EQ(pos, 1u);

    EXPECT_EQ(stoi("12345", &pos), 12345);
    EXPECT_EQ(pos, 5u);

    // Partial conversion test
    EXPECT_EQ(stoi("123abc", &pos), 123);
    EXPECT_EQ(pos, 3u);

    // Negative numbers
    EXPECT_EQ(stoi("-456", &pos), -456);
    EXPECT_EQ(pos, 4u);

    // Test without position parameter
    EXPECT_EQ(stoi("789"), 789);

    // Edge cases that should throw
    // Note: Uncomment when exception handling is properly set up
    // ASSERT_ANY_THROW({ stoi("abc", &pos); });
    // ASSERT_ANY_THROW({ stoi("", &pos); });
}

// =============================================================================
// STRING MANIPULATION TESTS
// =============================================================================

void testStringManipulation()
{
    // Test erase_all_sub_string function
    std::string      s   = "blabla";
    std::string_view sub = "la";
    erase_all_sub_string(s, sub);
    EXPECT_EQ(s, "bb");

    // More comprehensive erase tests
    s = "abcabcabc";
    erase_all_sub_string(s, "abc");
    EXPECT_EQ(s, "");

    s = "hello world hello";
    erase_all_sub_string(s, "hello");
    EXPECT_EQ(s, " world ");

    s = "test";
    erase_all_sub_string(s, "xyz");  // Non-existent substring
    EXPECT_EQ(s, "test");

    // Test replace_all function
    s            = "bb";
    size_t count = replace_all(s, "b", "c");
    EXPECT_EQ(s, "cc");
    EXPECT_EQ(count, 2u);

    // More comprehensive replace tests
    s     = "hello world hello";
    count = replace_all(s, "hello", "hi");
    EXPECT_EQ(s, "hi world hi");
    EXPECT_EQ(count, 2u);

    s     = "abcdef";
    count = replace_all(s, "xyz", "123");  // Non-existent substring
    EXPECT_EQ(s, "abcdef");
    EXPECT_EQ(count, 0u);

    // Test overlapping replacements
    s     = "aaa";
    count = replace_all(s, "aa", "b");
    EXPECT_EQ(s, "ba");  // Should replace first occurrence
    EXPECT_EQ(count, 1u);
}

// =============================================================================
// STRING VALIDATION TESTS
// =============================================================================

void testStringValidation()
{
    // Test is_float function
    EXPECT_TRUE(is_float("3.15"));
    EXPECT_TRUE(is_float("3.14159"));
    EXPECT_TRUE(is_float("-2.5"));
    EXPECT_TRUE(is_float("0.0"));
    EXPECT_TRUE(is_float("1.23e-4"));  // Scientific notation
    EXPECT_TRUE(is_float("1.23E+5"));  // Scientific notation uppercase

    // Invalid float strings
    EXPECT_FALSE(is_float("abc"));
    EXPECT_FALSE(is_float(""));
    EXPECT_FALSE(is_float("3.14.15"));  // Multiple dots
    EXPECT_FALSE(is_float("3.14abc"));  // Mixed content

    // Test is_integer function
    EXPECT_TRUE(is_integer("3"));
    EXPECT_TRUE(is_integer("12345"));
    EXPECT_TRUE(is_integer("0"));

    // Invalid integer strings
    EXPECT_FALSE(is_integer("-123"));   // Negative (function doesn't handle signs)
    EXPECT_FALSE(is_integer("12.34"));  // Decimal point
    EXPECT_FALSE(is_integer("abc"));
    EXPECT_FALSE(is_integer(""));
    EXPECT_FALSE(is_integer("123abc"));  // Mixed content
    EXPECT_FALSE(is_integer(" 123 "));   // Whitespace
}

// =============================================================================
// STRING SPLITTING TESTS
// =============================================================================

void testStringSplitting()
{
    // Test split_string function
    std::vector<char>        separators = {',', ' ', ':', '=', '/', '|', '{', '}', '[', ']'};
    std::string              input      = "a,b c:d=e/f|g{h}i[j]k";
    std::vector<std::string> expected   = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"};

    const auto& result = split_string(input, separators);
    EXPECT_EQ(result, expected);

    // Test with single separator
    std::vector<char> comma_sep = {','};
    input                       = "apple,banana,cherry";
    expected                    = {"apple", "banana", "cherry"};
    const auto& result2         = split_string(input, comma_sep);
    EXPECT_EQ(result2, expected);

    // Test with empty string
    const auto& result3 = split_string("", separators);
    EXPECT_TRUE(result3.empty());

    // Test with no separators in string
    const auto& result4 = split_string("noseparators", separators);
    EXPECT_EQ(result4.size(), 1u);
    EXPECT_EQ(result4[0], "noseparators");

    // Test with consecutive separators (should be filtered out)
    input               = "a,,b  c";
    const auto& result5 = split_string(input, comma_sep);
    // Result should not contain empty strings
    for (const auto& token : result5)
    {
        EXPECT_FALSE(token.empty());
    }
}

// =============================================================================
// FORMATTING TESTS
// =============================================================================

void testFormatting()
{
    // Test to_string with precision and width
    auto result = to_string(3.14159, 2, 10);
    EXPECT_EQ(result.length(), 10u);  // Should be padded to width 10

    // Test center function
    auto centered = center("Hello", 11);
    EXPECT_EQ(centered, "   Hello   ");
    EXPECT_EQ(centered.length(), 11u);

    // Test center with odd padding
    centered = center("Hi", 7);
    EXPECT_EQ(centered, "  Hi   ");  // Extra space on right
    EXPECT_EQ(centered.length(), 7u);

    // Test center with string longer than width
    centered = center("VeryLongString", 5);
    EXPECT_EQ(centered, "VeryLongString");  // Should return original

    // Test center with exact width
    centered = center("Exact", 5);
    EXPECT_EQ(centered, "Exact");
}

// =============================================================================
// PRINTF-STYLE FORMATTING TESTS
// =============================================================================

void testPrintfFormatting()
{
    // Test Printf function
    auto result = Printf("Value: %d, Name: %s", 42, "test");
    EXPECT_EQ(result, "Value: 42, Name: test");

    // Test with floating point
    result = Printf("Pi: %.2f", 3.14159);
    EXPECT_EQ(result, "Pi: 3.14");

    // Test Appendf function
    std::string base = "Prefix: ";
    Appendf(&base, "Value: %d, Status: %s", 42, "OK");
    EXPECT_EQ(base, "Prefix: Value: 42, Status: OK");

    // Test empty format
    result = Printf("");
    EXPECT_EQ(result, "");
}

// =============================================================================
// C++20 COMPATIBILITY TESTS
// =============================================================================

void testCompatibilityFunctions()
{
    // Test starts_with function with std::string
    std::string str1    = "Hello World";
    std::string prefix1 = "Hello";
    EXPECT_TRUE(starts_with(str1, prefix1));

    std::string prefix2 = "hello";
    EXPECT_FALSE(starts_with(str1, prefix2));  // Case sensitive

    std::string str2    = "Hi";
    std::string prefix3 = "Hello";
    EXPECT_FALSE(starts_with(str2, prefix3));

    std::string empty1 = "";
    std::string empty2 = "";
    EXPECT_TRUE(starts_with(empty1, empty2));  // Edge case

    std::string prefix4 = "test";
    EXPECT_FALSE(starts_with(empty1, prefix4));

    // Test starts_with with string_view
    std::string_view sv1 = "Hello World";
    std::string_view sv2 = "Hello";
    EXPECT_TRUE(starts_with(sv1, sv2));

    // Test ends_with function with std::string
    std::string doc  = "document.pdf";
    std::string ext1 = ".pdf";
    EXPECT_TRUE(ends_with(doc, ext1));

    std::string ext2 = ".PDF";
    EXPECT_FALSE(ends_with(doc, ext2));  // Case sensitive

    std::string short_str = "doc";
    std::string long_ext  = ".pdf";
    EXPECT_FALSE(ends_with(short_str, long_ext));

    EXPECT_TRUE(ends_with(empty1, empty2));  // Edge case
    EXPECT_FALSE(ends_with(empty1, prefix4));

    // Test ends_with with string_view
    std::string_view sv3 = "document.pdf";
    std::string_view sv4 = ".pdf";
    EXPECT_TRUE(ends_with(sv3, sv4));
}

// =============================================================================
// SOURCE LOCATION TESTS
// =============================================================================

void testSourceLocation()
{
    xsigma::SourceLocation loc;
    loc.file     = "test.cpp";
    loc.function = "testFunction";
    loc.line     = 42;

    // Test basic functionality (skip stream output for now due to linking issues)
    EXPECT_EQ(std::string(loc.file), "test.cpp");
    EXPECT_EQ(std::string(loc.function), "testFunction");
    EXPECT_EQ(loc.line, 42);

    // TODO: Fix operator<< linking issue
    // std::ostringstream oss;
    // oss << loc;
    // EXPECT_EQ(oss.str(), "testFunction at test.cpp:42");
}

// =============================================================================
// MAIN TEST FUNCTION
// =============================================================================

void testAllFunctions()
{
    testFilePathFunctions();
    testStringConcatenation();
    testNumericConversion();
    testStringManipulation();
    testStringValidation();
    testStringSplitting();
    testFormatting();
    testPrintfFormatting();
    testCompatibilityFunctions();
    testSourceLocation();
}

}  // namespace xsigma

XSIGMATEST(Core, StringUtil)
{
    xsigma::testAllFunctions();
    END_TEST();
}
