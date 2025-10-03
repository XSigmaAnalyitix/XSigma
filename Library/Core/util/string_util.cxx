/**
 * @file string_util.cxx
 * @brief Implementation of high-performance string utility functions
 *
 * This file contains optimized implementations of string manipulation,
 * conversion, and utility functions for the XSigma Core library.
 *
 * Key design principles:
 * - Performance-first approach for financial computing applications
 * - Thread-safe implementations where applicable
 * - Locale-independent behavior for consistent results
 * - Robust error handling with clear failure modes
 *
 * @author XSigma Development Team
 * @version 2.0
 * @date 2024
 */

#include "util/string_util.h"

#include <cmath>           // for signbit, HUGE_VAL
#include <array>          // for array
#include <cctype>         // for isdigit, isspace, tolower
#include <cfloat>         // for DBL_DIG, FLT_DIG, DBL_MAX
#include <cmath>          // for isnan, signbit, abs
#include <cstdarg>        // for va_end, va_list, va_copy, va_start
#include <cstdio>         // for snprintf, vsnprintf
#include <cstdlib>        // for strtod, strtof, abs, strtol
#include <cstring>        // for strlen, memcpy
#include <limits>         // for numeric_limits
#include <locale>         // for ctype_base::space, locale, ctype, tolower, ctype_base
#include <memory>         // for _Simple_types
#include <string>         // for char_traits, string, operator<<, allocator, operator==, oper...
#include <unordered_map>  // for unordered_map
#include <utility>        // for reverse, find_if

#include "common/macros.h"   // for XSIGMA_UNUSED, XSIGMA_HAS_CXA_DEMANGLE
#include "util/exception.h"  // for XSIGMA_CHECK_DEBUG, XSIGMA_CHECK, XSIGMA_CHECK_VALUE

// =============================================================================
// PLATFORM-SPECIFIC CONFIGURATION
// =============================================================================

#if XSIGMA_HAS_CXA_DEMANGLE
#include <cxxabi.h>
#endif

// =============================================================================
// MAIN XSIGMA STRING UTILITIES IMPLEMENTATION
// =============================================================================

namespace xsigma
{
// Constants for cleaning up demangled names
constexpr std::string_view CLASS_NAME = "class ";  ///< C++ class prefix to remove
constexpr std::string_view SPACE_STR  = " ";       ///< Spaces to remove for cleaner output
constexpr std::string_view SPACE_LIB1 = "__1::";   ///< libstdc++ namespace to remove

/**
 * @brief Implementation of C++ symbol demangling
 * @note This function converts mangled C++ symbols to human-readable names
 * @note Falls back gracefully on platforms without demangling support
 */
std::string demangle(const char* name)
{
    // Handle null or empty input
    if (!name || *name == '\0')
    {
        return "<unknown>";
    }

    std::string ret = name;

#if HAS_DEMANGLE
    int status = -1;

    // Use GCC/Clang ABI demangling function
    // This converts mangled names like "_Z1gv" to readable names like "g()"
    // Reference: https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/libsupc%2B%2B/cxxabi.h
    // NOTE: __cxa_demangle returns malloc'd memory that must be freed
    std::unique_ptr<char, std::function<void(char*)>> demangled(
        abi::__cxa_demangle(name, nullptr, 0, &status),  // NOLINT - C API
        /*deleter=*/std::free);                          // NOLINT - C API

    // Demangling may fail for symbols that don't follow the standard C++
    // (Itanium ABI) mangling scheme. Examples include 'main', 'clone', etc.
    // In such cases, the original mangled name is a reasonable fallback.
    if (status == 0 && demangled != nullptr)
    {
        ret = demangled.get();
    }
#endif  // HAS_DEMANGLE

    // Clean up common unwanted prefixes and suffixes for better readability
    erase_all_sub_string(ret, CLASS_NAME);  // Remove "class " prefix
    erase_all_sub_string(ret, SPACE_STR);   // Remove extra spaces
    erase_all_sub_string(ret, SPACE_LIB1);  // Remove libstdc++ internal namespace

    return ret;
}
// =============================================================================
// STRING MANIPULATION AND REPLACEMENT FUNCTIONS
// =============================================================================

/**
 * @brief Replace all occurrences of substring with another string
 * @note Performs in-place modification for memory efficiency
 * @note Uses iterative approach to handle overlapping replacements correctly
 */
size_t replace_all(std::string& s, const char* from, const char* to)
{
    // Validate input parameters
    XSIGMA_CHECK(from && *from, "Source string cannot be null or empty");
    XSIGMA_CHECK(to, "Replacement string cannot be null");

    size_t     numReplaced = 0;
    const auto lenFrom     = std::strlen(from);
    const auto lenTo       = std::strlen(to);

    // Find and replace all occurrences
    for (auto pos = s.find(from); pos != std::string::npos; pos = s.find(from, pos + lenTo))
    {
        s.replace(pos, lenFrom, to);
        numReplaced++;
    }
    return numReplaced;
}

/**
 * @brief Remove all occurrences of a substring from a string
 * @note More efficient than replace_all when replacing with empty string
 * @note Uses iterative approach to handle multiple occurrences
 * @note Marked noexcept for performance in exception-sensitive contexts
 */
void erase_all_sub_string(std::string& mainStr, std::string_view const& toErase) noexcept
{
    size_t pos;

    // Search for the substring in a loop until nothing is found
    while ((pos = mainStr.find(toErase)) != std::string::npos)
    {
        // Erase the found substring
        mainStr.erase(pos, toErase.length());
    }
}
}  // namespace xsigma

namespace xsigma
{

// C++20 compatibility utility for std::string_view::starts_with()
// Note: These functions provide C++20 functionality with C++17 fallback
bool starts_with(std::string_view str, std::string_view prefix)
{
#if __cplusplus >= 202002L
    return str.starts_with(prefix);
#else
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
#endif
}

// C++20 compatibility utility for std::string_view::ends_with()
bool ends_with(std::string_view str, std::string_view suffix)
{
#if __cplusplus >= 202002L
    return str.ends_with(suffix);
#else
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
#endif
}

}  // namespace xsigma
