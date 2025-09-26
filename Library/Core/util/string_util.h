/**
 * @file string_util.h
 * @brief Comprehensive string utility functions for the XSigma Core library
 *
 * This header provides a collection of high-performance string manipulation,
 * conversion, and utility functions designed for financial computing applications.
 * All functions are optimized for performance and thread-safety where applicable.
 *
 * @author XSigma Development Team
 * @version 2.0
 * @date 2024
 */

#pragma once

#ifndef __XSIGMA_WRAP__

#include <cstddef>      // for size_t
#include <cstdint>      // for int64_t
#include <sstream>      // for ostream, ostringstream, stringstream
#include <string>       // for string, allocator, basic_string
#include <string_view>  // for string_view
#include <typeinfo>     // for type_info
#include <vector>       // for vector

#include "common/macros.h"

// =============================================================================
// ENUM CONVERSION UTILITIES
// =============================================================================

#if XSIGMA_MODULE_ENABLE_XSIGMA_magicenum
#include "xsigma_magic_enum.h"
#include XSIGMA_MAGIC_ENUM(magic_enum.hpp)

namespace xsigma
{
/**
 * @brief Convert an enum value to its string representation using magic_enum
 * @tparam E The enum type
 * @param x The enum value to convert
 * @return String view containing the enum name
 * @note This function uses magic_enum for reflection-based enum conversion
 * @example
 * @code
 * enum class Color { Red, Green, Blue };
 * auto name = enum_to_string(Color::Red); // Returns "Red"
 * @endcode
 */
template <typename E>
inline std::string_view enum_to_string(E x)
{
    return magic_enum::enum_name(x);
}

/**
 * @brief Convert a string to an enum value using magic_enum (case-insensitive)
 * @tparam E The enum type
 * @param str The string to convert
 * @return The corresponding enum value
 * @throws std::bad_optional_access if the string doesn't match any enum value
 * @note This function performs case-insensitive matching
 * @example
 * @code
 * enum class Color { Red, Green, Blue };
 * auto color = string_to_enum<Color>("red"); // Returns Color::Red
 * @endcode
 */
template <typename E>
E string_to_enum(std::string_view str)
{
    return magic_enum::enum_cast<E>(str, magic_enum::case_insensitive).value();
}
}  // namespace xsigma
#else
namespace xsigma
{
/**
 * @brief Convert an enum value to its numeric string representation (fallback)
 * @tparam E The enum type
 * @param x The enum value to convert
 * @return String view containing the numeric representation
 * @note This is a fallback implementation when magic_enum is not available
 * @warning Uses xsigma_thread_local storage for performance
 */
template <typename E>
inline std::string_view enum_to_string(E x)
{
    static thread_local std::string buffer;
    buffer = std::to_string(static_cast<int>(x));
    return buffer;
}

/**
 * @brief Convert a string to an enum value using numeric conversion (fallback)
 * @tparam E The enum type
 * @param str The string containing numeric value
 * @return The corresponding enum value
 * @throws std::invalid_argument if the string is not a valid integer
 * @note This is a fallback implementation when magic_enum is not available
 */
template <typename E>
E string_to_enum(std::string_view str)
{
    return static_cast<E>(std::stoi(std::string(str)));
}
}  // namespace xsigma
#endif

// =============================================================================
// TYPE INTROSPECTION AND DEMANGLING UTILITIES
// =============================================================================

namespace xsigma
{
/**
 * @brief Demangle a C++ symbol name to human-readable format
 * @param name The mangled symbol name (typically from typeid().name())
 * @return Demangled string representation of the symbol
 * @note On platforms without demangling support, returns the original name
 * @note The function handles null/empty input gracefully
 * @example
 * @code
 * const char* mangled = typeid(std::vector<int>).name();
 * std::string readable = demangle(mangled); // Returns "std::vector<int>"
 * @endcode
 */
XSIGMA_API std::string demangle(const char* name);

/**
 * @brief Get the printable name of a type using RTTI
 * @tparam T The type to get the name for
 * @return C-string containing the demangled type name
 * @note Uses static storage for performance - the returned pointer remains valid
 * @note Returns a descriptive message when RTTI is disabled
 * @warning The returned pointer should not be freed
 * @example
 * @code
 * const char* name = demangle_type<std::vector<double>>();
 * // Returns "std::vector<double>" or similar
 * @endcode
 */
template <typename T>
inline const char* demangle_type()
{
#ifdef __GXX_RTTI
    static const auto& name = *(new std::string(demangle(typeid(T).name())));
    return name.c_str();
#else   // __GXX_RTTI
    return "(RTTI disabled, cannot show name)";
#endif  // __GXX_RTTI
}

// =============================================================================
// STRING CONCATENATION IMPLEMENTATION DETAILS
// =============================================================================

namespace details
{
/**
 * @brief Compile-time empty string optimization for performance
 * @note This struct provides implicit conversions to both std::string and const char*
 * @note Used to avoid unnecessary string construction in assert macros and similar contexts
 */
struct compile_time_empty_string
{
    /// @brief Implicit conversion to const std::string& (returns static empty string)
    operator const std::string&() const
    {
        static const std::string empty_string_literal;
        return empty_string_literal;
    }

    /// @brief Implicit conversion to const char* (returns empty C-string)
    operator const char*() const { return ""; }
};

/**
 * @brief Type trait to canonicalize string-like types for optimal passing
 * @tparam T The type to canonicalize
 * @note Converts array types to pointer types for efficient parameter passing
 */
template <typename T>
struct CanonicalizeStrTypes
{
    using type = const T&;  ///< Default: pass by const reference
};

/// @brief Specialization for character arrays - convert to const char*
template <size_t N>
struct CanonicalizeStrTypes<char[N]>  // NOLINT
{
    using type = const char*;
};

/**
 * @brief Internal function to stream a single argument
 * @tparam T The type of the argument
 * @param ss The output stream
 * @param t The argument to stream
 * @return Reference to the output stream for chaining
 */
template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t)
{
    ss << t;
    return ss;
}

/**
 * @brief Specialization for compile_time_empty_string - no-op for performance
 * @param ss The output stream (unchanged)
 * @param unused The empty string object (ignored)
 * @return Reference to the unchanged output stream
 */
template <>
inline std::ostream& _str<compile_time_empty_string>(
    std::ostream& ss, const compile_time_empty_string& /*unused*/)
{
    return ss;
}

/**
 * @brief Variadic template function to stream multiple arguments recursively
 * @tparam T The type of the first argument
 * @tparam Args The types of the remaining arguments
 * @param ss The output stream
 * @param t The first argument
 * @param args The remaining arguments
 * @return Reference to the output stream after streaming all arguments
 */
template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args)
{
    return _str(_str(ss, t), args...);
}

/**
 * @brief Primary template for string concatenation wrapper
 * @tparam Args The types of arguments to concatenate
 * @note This is the general case that uses stringstream for concatenation
 */
template <typename... Args>
struct _str_wrapper final
{
    /**
     * @brief Concatenate all arguments into a single string
     * @param args The arguments to concatenate
     * @return String containing all arguments concatenated
     */
    static std::string call(const Args&... args)
    {
        std::ostringstream ss;
        _str(ss, args...);
        return ss.str();
    }
};

/**
 * @brief Specialization for single std::string argument - avoid unnecessary copy
 * @note Returns by reference to avoid binary size overhead of string copying
 */
template <>
struct _str_wrapper<std::string> final
{
    /// @brief Return the string by reference (no copy needed)
    static const std::string& call(const std::string& str) { return str; }
};

/**
 * @brief Specialization for single const char* argument - pass through directly
 * @note Avoids string construction when not needed
 */
template <>
struct _str_wrapper<const char*> final
{
    /// @brief Return the C-string directly (no conversion needed)
    static const char* call(const char* str) { return str; }
};

/**
 * @brief Specialization for empty argument list - return compile-time empty string
 * @note Common in assert macros - avoids stringstream construction/destruction overhead
 */
template <>
struct _str_wrapper<> final
{
    /// @brief Return compile-time empty string for maximum performance
    static compile_time_empty_string call() { return compile_time_empty_string(); }
};
}  // namespace details

/**
 * @brief Convert a list of string-like arguments into a single string
 * @tparam Args The types of arguments to concatenate
 * @param args The arguments to concatenate
 * @return String or string-like object containing all arguments concatenated
 * @note This function is optimized for performance with specializations for common cases
 * @note Returns by reference when possible to avoid unnecessary copies
 * @example
 * @code
 * auto result = to_string("Value: ", 42, ", Status: ", true);
 * // Returns "Value: 42, Status: 1"
 * @endcode
 */
template <typename... Args>
inline decltype(auto) to_string(const Args&... args)
{
    return details::_str_wrapper<typename details::CanonicalizeStrTypes<Args>::type...>::call(
        args...);
}

// =============================================================================
// SOURCE LOCATION AND DEBUGGING UTILITIES
// =============================================================================

/**
 * @brief Represents a location in source code for debugging purposes
 * @note Used primarily in assertion macros and error reporting
 * @note All members are const char* for minimal memory overhead
 */
struct XSIGMA_VISIBILITY SourceLocation
{
    const char* function;  ///< Function name where the location was captured
    const char* file;      ///< Source file name
    int         line;      ///< Line number in the source file
};

// =============================================================================
// FILE PATH MANIPULATION UTILITIES
// =============================================================================

/**
 * @brief Extract the base name (filename) from a full file path
 * @param full_path The complete file path
 * @return The filename portion of the path (everything after the last '/')
 * @note Uses '/' as the path separator (Unix-style)
 * @note If no separator is found, returns the entire input string
 * @example
 * @code
 * auto name = strip_basename("/usr/local/bin/program"); // Returns "program"
 * auto name2 = strip_basename("file.txt"); // Returns "file.txt"
 * @endcode
 */
XSIGMA_API std::string strip_basename(const std::string& full_path);

/**
 * @brief Remove the file extension from a filename
 * @param file_name The filename with extension
 * @return The filename without extension (everything before the last '.')
 * @note If no extension is found, returns the entire input string
 * @example
 * @code
 * auto name = exclude_file_extension("document.pdf"); // Returns "document"
 * auto name2 = exclude_file_extension("archive.tar.gz"); // Returns "archive.tar"
 * @endcode
 */
XSIGMA_API std::string exclude_file_extension(const std::string& file_name);

/**
 * @brief Extract the file extension from a filename
 * @param file_name The filename
 * @return The file extension including the dot (everything from the last '.')
 * @note If no extension is found, returns an empty string
 * @example
 * @code
 * auto ext = file_extension("document.pdf"); // Returns ".pdf"
 * auto ext2 = file_extension("archive.tar.gz"); // Returns ".gz"
 * @endcode
 */
XSIGMA_API std::string file_extension(const std::string& file_name);

// =============================================================================
// STRING MANIPULATION AND TRANSFORMATION UTILITIES
// =============================================================================

/**
 * @brief Join container elements into a single string with delimiter
 * @tparam Container The container type (must be iterable)
 * @param delimiter The string to insert between elements
 * @param v The container of elements to join
 * @return String containing all elements separated by the delimiter
 * @note Elements must be streamable to std::ostream
 * @note No delimiter is added after the last element
 * @example
 * @code
 * std::vector<int> nums = {1, 2, 3, 4};
 * auto result = join(", ", nums); // Returns "1, 2, 3, 4"
 * @endcode
 */
template <class Container>
inline std::string join(const std::string& delimiter, const Container& v)
{
    std::stringstream s;
    int               cnt = static_cast<int64_t>(v.size()) - 1;
    for (auto i = v.begin(); i != v.end(); ++i, --cnt)
    {
        s << (*i) << (cnt ? delimiter : "");
    }
    return s.str();
}

/**
 * @brief Replace all occurrences of a substring with another string
 * @param s The string to modify (modified in-place)
 * @param from The substring to find and replace
 * @param to The replacement string
 * @return Number of replacements performed
 * @pre from must not be null or empty
 * @pre to must not be null
 * @note Modifies the input string in-place for performance
 * @example
 * @code
 * std::string text = "Hello world, world!";
 * size_t count = replace_all(text, "world", "universe");
 * // text becomes "Hello universe, universe!", count is 2
 * @endcode
 */
size_t XSIGMA_API replace_all(std::string& s, const char* from, const char* to);

/**
 * @brief Convert string to integer with position tracking
 * @param str The string to convert
 * @param pos Optional pointer to store the position after conversion
 * @return The converted integer value
 * @throws xsigma exception if conversion fails
 * @note More robust than std::stoi with better error handling
 * @example
 * @code
 * size_t pos;
 * int value = stoi("123abc", &pos); // value=123, pos=3
 * @endcode
 */
XSIGMA_API int stoi(const std::string& str, std::size_t* pos = nullptr);

/**
 * @brief Stream output operator for SourceLocation
 * @param out The output stream
 * @param loc The source location to output
 * @return Reference to the output stream
 * @note Formats as "function at file:line"
 */
std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

/**
 * @brief Remove all occurrences of a substring from a string
 * @param mainStr The string to modify (modified in-place)
 * @param toErase The substring to remove
 * @note This function is noexcept and performs in-place modification
 * @note More efficient than replace_all when replacing with empty string
 * @example
 * @code
 * std::string text = "a-b-c-d";
 * erase_all_sub_string(text, "-"); // text becomes "abcd"
 * @endcode
 */
XSIGMA_API void erase_all_sub_string(std::string& mainStr, std::string_view const& toErase) noexcept;

// =============================================================================
// STRING VALIDATION AND TYPE CHECKING UTILITIES
// =============================================================================

/**
 * @brief Check if a string represents a valid floating-point number
 * @param str The string to validate
 * @return true if the string is a valid float, false otherwise
 * @note Handles scientific notation and special values (inf, nan)
 * @note Uses locale-independent parsing
 * @example
 * @code
 * bool valid1 = is_float("3.14159"); // Returns true
 * bool valid2 = is_float("1.23e-4"); // Returns true
 * bool valid3 = is_float("abc");      // Returns false
 * @endcode
 */
XSIGMA_API bool is_float(const std::string& str);

/**
 * @brief Check if a string represents a valid integer
 * @param str The string to validate
 * @return true if the string contains only digits, false otherwise
 * @note Does not handle negative numbers or leading/trailing whitespace
 * @note For more robust integer parsing, use safe_strto64 or similar
 * @example
 * @code
 * bool valid1 = is_integer("12345"); // Returns true
 * bool valid2 = is_integer("-123");  // Returns false (negative sign)
 * bool valid3 = is_integer("12.34"); // Returns false (decimal point)
 * @endcode
 */
XSIGMA_API bool is_integer(const std::string& str);

/**
 * @brief Split a string using multiple separator characters
 * @param s The string to split
 * @param separators Vector of characters to use as separators
 * @return Vector of strings containing the split parts
 * @note Uses locale-based tokenization for robust parsing
 * @note Empty tokens are automatically filtered out
 * @example
 * @code
 * std::vector<char> seps = {',', ';', ' '};
 * auto parts = split_string("a,b;c d", seps); // Returns {"a", "b", "c", "d"}
 * @endcode
 */
XSIGMA_API std::vector<std::string> split_string(
    std::string_view s, const std::vector<char>& separators);

// =============================================================================
// NUMERIC FORMATTING UTILITIES
// =============================================================================

/**
 * @brief Convert double to formatted string with specified precision and width
 * @param x The double value to convert
 * @param decDigits Number of decimal places
 * @param width Minimum field width (padded with spaces)
 * @return Formatted string representation
 * @note Uses fixed-point notation and right alignment
 * @example
 * @code
 * auto str = to_string(3.14159, 2, 10); // Returns "      3.14"
 * @endcode
 */
XSIGMA_API std::string to_string(const double x, const int decDigits, const int width);

/**
 * @brief Center a string within a specified width
 * @param s The string to center
 * @param width The total width of the result
 * @return String padded with spaces to center the input
 * @note If width is less than string length, returns the original string
 * @note Uses space characters for padding
 * @example
 * @code
 * auto centered = center("Hello", 11); // Returns "   Hello   "
 * @endcode
 */
XSIGMA_API std::string center(const std::string s, const int width);

// =============================================================================
// PLATFORM-SPECIFIC AND LEGACY UTILITIES
// =============================================================================

/**
 * @brief Convert wide character string to narrow (UTF-8) string
 * @param x Wide character string to convert
 * @return Narrow string representation
 * @note Platform-specific implementation using xsigmasys::Encoding
 * @note Primarily used for Windows Unicode interoperability
 */
XSIGMA_API std::string ToNarrow(wchar_t* x);

/**
 * @brief Create a dynamically allocated copy of a C-string
 * @param str The C-string to duplicate
 * @return Pointer to newly allocated string copy
 * @note Caller is responsible for freeing the returned pointer
 * @note Uses xsigmasys::SystemTools for platform-specific allocation
 * @warning Remember to free the returned pointer to avoid memory leaks
 */
XSIGMA_API char* DuplicateString(const char* str);

// =============================================================================
// PRINTF-STYLE FORMATTING UTILITIES
// =============================================================================

/**
 * @brief Create a formatted string using printf-style format specifiers
 * @param format Printf-style format string
 * @param ... Variable arguments matching the format specifiers
 * @return Formatted string
 * @note Provides type-safe printf formatting with automatic memory management
 * @note Compiler will perform format string validation when available
 * @example
 * @code
 * auto str = Printf("Value: %d, Name: %s", 42, "test");
 * // Returns "Value: 42, Name: test"
 * @endcode
 */
XSIGMA_API std::string Printf(const char* format, ...)
    // Tell the compiler to do printf format string checking.
    XSIGMA_PRINTF_ATTRIBUTE(1, 2);

/**
 * @brief Append formatted text to an existing string using printf-style format
 * @param dst Pointer to the string to append to (must not be null)
 * @param format Printf-style format string
 * @param ... Variable arguments matching the format specifiers
 * @note More efficient than string concatenation for multiple append operations
 * @note Compiler will perform format string validation when available
 * @example
 * @code
 * std::string result = "Prefix: ";
 * Appendf(&result, "Value: %d, Status: %s", 42, "OK");
 * // result becomes "Prefix: Value: 42, Status: OK"
 * @endcode
 */
XSIGMA_API void Appendf(std::string* dst, const char* format, ...)
    // Tell the compiler to do printf format string checking.
    XSIGMA_PRINTF_ATTRIBUTE(2, 3);

// =============================================================================
// C++20 COMPATIBILITY UTILITIES
// =============================================================================

/**
 * @brief Check if a string starts with a given prefix (C++20 compatibility)
 * @param str The string to check
 * @param prefix The prefix to look for
 * @return true if str starts with prefix, false otherwise
 * @note Uses C++20 std::string::starts_with() when available, fallback otherwise
 * @note Performs exact character matching (case-sensitive)
 * @example
 * @code
 * bool result = starts_with("Hello World", "Hello"); // Returns true
 * bool result2 = starts_with("Hello World", "hello"); // Returns false
 * @endcode
 */
XSIGMA_API bool starts_with(const std::string& str, const std::string& prefix);

/**
 * @brief Check if a string_view starts with a given prefix (C++20 compatibility)
 * @param str The string_view to check
 * @param prefix The prefix to look for
 * @return true if str starts with prefix, false otherwise
 * @note Uses C++20 std::string_view::starts_with() when available, fallback otherwise
 * @note More efficient than the std::string version for temporary strings
 */
XSIGMA_API bool starts_with(std::string_view str, std::string_view prefix);

/**
 * @brief Check if a string ends with a given suffix (C++20 compatibility)
 * @param str The string to check
 * @param suffix The suffix to look for
 * @return true if str ends with suffix, false otherwise
 * @note Uses C++20 std::string::ends_with() when available, fallback otherwise
 * @note Performs exact character matching (case-sensitive)
 * @example
 * @code
 * bool result = ends_with("document.pdf", ".pdf"); // Returns true
 * bool result2 = ends_with("document.pdf", ".PDF"); // Returns false
 * @endcode
 */
XSIGMA_API bool ends_with(const std::string& str, const std::string& suffix);

/**
 * @brief Check if a string_view ends with a given suffix (C++20 compatibility)
 * @param str The string_view to check
 * @param suffix The suffix to look for
 * @return true if str ends with suffix, false otherwise
 * @note Uses C++20 std::string_view::ends_with() when available, fallback otherwise
 * @note More efficient than the std::string version for temporary strings
 */
XSIGMA_API bool ends_with(std::string_view str, std::string_view suffix);

/**
 * @brief Low-level printf-style formatting with va_list
 * @param dst Pointer to the string to append to (must not be null)
 * @param format Printf-style format string
 * @param ap Variable argument list
 * @note This is the core implementation used by Printf() and Appendf()
 * @note Handles buffer management and platform-specific formatting differences
 * @note Most users should use Printf() or Appendf() instead of this function
 */
XSIGMA_API void Appendv(std::string* dst, const char* format, va_list ap);

}  // namespace xsigma

// =============================================================================
// HIGH-PERFORMANCE NUMERIC CONVERSION UTILITIES
// =============================================================================

namespace xsigma
{
namespace numbers
{
/**
 * @brief High-performance numeric to string conversion utilities
 *
 * This namespace provides optimized functions for converting numeric values
 * to string representations. These functions are designed for maximum performance
 * in financial computing applications where speed is critical.
 *
 * Key features:
 * - Zero-allocation conversions using caller-provided buffers
 * - Optimized algorithms for common numeric types
 * - Locale-independent formatting for consistent results
 * - Thread-safe implementations
 *
 * Buffer size requirements:
 * - Int32, UInt32: minimum 12 bytes
 * - Int64, UInt64: minimum 22 bytes
 * - Float, Double: minimum 32 bytes
 * - Time formatting: exactly 30 bytes
 *
 * @note All functions return the number of characters written
 * @note Buffers must be at least kFastToBufferSize bytes for safety
 * @note In 64-bit systems, time_t values may exceed 4-digit year representation
 */

/**
 * @brief Recommended buffer size for all fast conversion functions
 * @note This size accommodates the largest possible output from any conversion
 * @note Using this constant ensures forward compatibility with future changes
 */
static const int kFastToBufferSize = 32;

/**
 * @brief Convert 32-bit signed integer to left-aligned ASCII string
 * @param i The integer value to convert
 * @param buffer Output buffer (must be at least 12 bytes)
 * @return Number of characters written to the buffer
 * @note Buffer is not null-terminated automatically
 * @note Negative values include the minus sign in the character count
 * @note Optimized for speed over convenience - use for performance-critical code
 * @example
 * @code
 * char buf[kFastToBufferSize];
 * size_t len = FastInt32ToBufferLeft(-12345, buf);
 * std::string result(buf, len); // "-12345"
 * @endcode
 */
XSIGMA_API size_t FastInt32ToBufferLeft(int32_t i, char* buffer);

/**
 * @brief Convert 32-bit unsigned integer to left-aligned ASCII string
 * @param i The unsigned integer value to convert
 * @param buffer Output buffer (must be at least 12 bytes)
 * @return Number of characters written to the buffer
 * @note Buffer is not null-terminated automatically
 * @note Optimized for speed over convenience - use for performance-critical code
 */
XSIGMA_API size_t FastUInt32ToBufferLeft(uint32_t i, char* buffer);

/**
 * @brief Convert 64-bit signed integer to left-aligned ASCII string
 * @param i The 64-bit integer value to convert
 * @param buffer Output buffer (must be at least 22 bytes)
 * @return Number of characters written to the buffer
 * @note Buffer is not null-terminated automatically
 * @note Negative values include the minus sign in the character count
 * @note Handles the full range of int64_t values
 */
XSIGMA_API size_t FastInt64ToBufferLeft(int64_t i, char* buffer);

/**
 * @brief Convert 64-bit unsigned integer to left-aligned ASCII string
 * @param i The 64-bit unsigned integer value to convert
 * @param buffer Output buffer (must be at least 22 bytes)
 * @return Number of characters written to the buffer
 * @note Buffer is not null-terminated automatically
 * @note Handles the full range of uint64_t values
 */
XSIGMA_API size_t FastUInt64ToBufferLeft(uint64_t i, char* buffer);

/**
 * @brief Convert double-precision floating point to ASCII string
 * @param value The double value to convert
 * @param buffer Output buffer (must be at least kFastToBufferSize bytes)
 * @return Number of characters written to the buffer
 * @note Uses optimal precision to ensure round-trip accuracy
 * @note Handles special values (NaN, infinity) appropriately
 * @note Uses locale-independent formatting for consistent results
 * @note Buffer is null-terminated
 */
XSIGMA_API size_t DoubleToBuffer(double value, char* buffer);

/**
 * @brief Convert single-precision floating point to ASCII string
 * @param value The float value to convert
 * @param buffer Output buffer (must be at least kFastToBufferSize bytes)
 * @return Number of characters written to the buffer
 * @note Uses optimal precision to ensure round-trip accuracy
 * @note Handles special values (NaN, infinity) appropriately
 * @note Uses locale-independent formatting for consistent results
 * @note Buffer is null-terminated
 */
XSIGMA_API size_t FloatToBuffer(float value, char* buffer);

// =============================================================================
// HEXADECIMAL CONVERSION UTILITIES
// =============================================================================

/**
 * @brief Convert 64-bit unsigned integer to hexadecimal string
 * @param v The 64-bit value to convert
 * @param buf Output buffer (must be at least kFastToBufferSize bytes)
 * @return String view of the hexadecimal representation
 * @note Output is null-terminated and uses lowercase hex digits
 * @note Always produces exactly 16 hex characters (with leading zeros)
 * @note Primarily used for fingerprint and hash value formatting
 * @example
 * @code
 * char buffer[kFastToBufferSize];
 * auto hex = Uint64ToHexString(0xDEADBEEF12345678ULL, buffer);
 * // hex contains "deadbeef12345678"
 * @endcode
 */
XSIGMA_API std::string_view Uint64ToHexString(uint64_t v, char* buf);

/**
 * @brief Parse hexadecimal string to 64-bit unsigned integer
 * @param s The hexadecimal string to parse
 * @param result Pointer to store the parsed value
 * @return true if parsing succeeded, false otherwise
 * @note Accepts both uppercase and lowercase hex digits
 * @note Does not require "0x" prefix
 * @note Stops parsing at first non-hex character
 * @example
 * @code
 * uint64_t value;
 * bool success = HexStringToUint64("DEADBEEF", &value);
 * // success is true, value is 0xDEADBEEF
 * @endcode
 */
XSIGMA_API bool HexStringToUint64(const std::string_view& s, uint64_t* result);

// =============================================================================
// SAFE STRING TO NUMERIC CONVERSION UTILITIES
// =============================================================================

/**
 * @brief Safely convert string to 32-bit unsigned integer
 * @param str The string to parse
 * @param value Pointer to store the parsed value
 * @return true if conversion succeeded, false on overflow or invalid input
 * @note Allows leading and trailing whitespace
 * @note Detects overflow conditions and returns false
 * @note More robust than standard library functions
 * @example
 * @code
 * uint32_t value;
 * bool ok = safe_strtou32("  12345  ", &value); // ok=true, value=12345
 * bool bad = safe_strtou32("99999999999", &value); // bad=false (overflow)
 * @endcode
 */
XSIGMA_API bool safe_strtou32(std::string_view str, uint32_t* value);

/**
 * @brief Safely convert string to 64-bit signed integer
 * @param str The string to parse
 * @param value Pointer to store the parsed value
 * @return true if conversion succeeded, false on overflow or invalid input
 * @note Allows leading and trailing whitespace
 * @note Handles negative values correctly
 * @note Detects overflow conditions and returns false
 */
XSIGMA_API bool safe_strto64(std::string_view str, int64_t* value);

/**
 * @brief Safely convert string to 64-bit unsigned integer
 * @param str The string to parse
 * @param value Pointer to store the parsed value
 * @return true if conversion succeeded, false on overflow or invalid input
 * @note Allows leading and trailing whitespace
 * @note Detects overflow conditions and returns false
 */
XSIGMA_API bool safe_strtou64(std::string_view str, uint64_t* value);

/**
 * @brief Safely convert string to single-precision floating point
 * @param str The string to parse
 * @param value Pointer to store the parsed value
 * @return true if conversion succeeded, false on invalid input
 * @note Allows leading and trailing whitespace
 * @note Handles special values (inf, nan, scientific notation)
 * @note Values may be rounded on over/underflow
 * @note Returns false if string length >= kFastToBufferSize
 */
XSIGMA_API bool safe_strtof(std::string_view str, float* value);

/**
 * @brief Safely convert string to double-precision floating point
 * @param str The string to parse
 * @param value Pointer to store the parsed value
 * @return true if conversion succeeded, false on invalid input
 * @note Allows leading and trailing whitespace
 * @note Handles special values (inf, nan, scientific notation)
 * @note Values may be rounded on over/underflow
 * @note Returns false if string length >= kFastToBufferSize
 */
XSIGMA_API bool safe_strtod(std::string_view str, double* value);

// =============================================================================
// GENERIC NUMERIC PARSING UTILITIES
// =============================================================================

/**
 * @brief Generic numeric parser for uint32_t (Protocol Buffer compatibility)
 * @param s The string to parse
 * @param value Pointer to store the parsed value
 * @return true if parsing succeeded, false otherwise
 * @note Part of the ProtoParseNumeric family for template-based parsing
 */
inline bool ProtoParseNumeric(std::string_view s, uint32_t* value)
{
    return safe_strtou32(s, value);
}

/**
 * @brief Generic numeric parser for uint64_t (Protocol Buffer compatibility)
 * @param s The string to parse
 * @param value Pointer to store the parsed value
 * @return true if parsing succeeded, false otherwise
 * @note Part of the ProtoParseNumeric family for template-based parsing
 */
inline bool ProtoParseNumeric(std::string_view s, uint64_t* value)
{
    return safe_strtou64(s, value);
}

/**
 * @brief Generic numeric parser for double (Protocol Buffer compatibility)
 * @param s The string to parse
 * @param value Pointer to store the parsed value
 * @return true if parsing succeeded, false otherwise
 * @note Part of the ProtoParseNumeric family for template-based parsing
 */
inline bool ProtoParseNumeric(std::string_view s, double* value)
{
    return safe_strtod(s, value);
}

/**
 * @brief Template-based safe string to numeric conversion
 * @tparam T The numeric type to convert to
 * @param s The string to parse
 * @param value Pointer to store the parsed value
 * @return true if conversion succeeded, false otherwise
 * @note Uses ProtoParseNumeric internally for type-specific parsing
 * @note Supports uint32_t, uint64_t, and double types
 * @example
 * @code
 * double d;
 * bool ok = SafeStringToNumeric("3.14159", &d); // ok=true, d=3.14159
 * @endcode
 */
template <typename T>
bool SafeStringToNumeric(std::string_view s, T* value)
{
    return ProtoParseNumeric(s, value);
}

// =============================================================================
// HUMAN-READABLE FORMATTING UTILITIES
// =============================================================================

/**
 * @brief Convert integer to human-readable string with SI prefixes
 * @param value The integer value to format
 * @return String with appropriate SI prefix (k, M, B, T)
 * @note Uses decimal powers (1000-based) for formatting
 * @note Handles negative values appropriately
 * @note Uses scientific notation for very large numbers (>= 1E15)
 * @example
 * @code
 * auto str1 = HumanReadableNum(1200000); // Returns "1.20M"
 * auto str2 = HumanReadableNum(1500);    // Returns "1.50k"
 * auto str3 = HumanReadableNum(42);      // Returns "42"
 * @endcode
 */
std::string HumanReadableNum(int64_t value);

/**
 * @brief Convert byte count to human-readable string with binary prefixes
 * @param num_bytes The number of bytes to format
 * @return String with appropriate binary prefix (KiB, MiB, GiB, etc.)
 * @note Uses binary powers (1024-based) for byte formatting
 * @note Follows IEC 60027-2 standard for binary prefixes
 * @note Handles negative values and edge cases appropriately
 * @example
 * @code
 * auto str1 = HumanReadableNumBytes(12345678); // Returns "11.77MiB"
 * auto str2 = HumanReadableNumBytes(1536);     // Returns "1.5KiB"
 * auto str3 = HumanReadableNumBytes(512);      // Returns "512B"
 * @endcode
 */
std::string HumanReadableNumBytes(int64_t num_bytes);
}  // namespace numbers
}  // namespace xsigma

#endif  // __XSIGMA_WRAP__
