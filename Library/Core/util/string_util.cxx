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

// Standard library includes
#include <algorithm>   // For std::find_if, std::reverse, etc.
#include <array>       // For std::array in locale customization
#include <cctype>      // For character classification functions
#include <cerrno>      // For errno handling
#include <cfloat>      // For floating-point constants (DBL_DIG, FLT_DIG)
#include <cmath>       // For mathematical functions (isnan, signbit)
#include <cstdarg>     // For va_list and related operations
#include <cstdio>      // For snprintf and related functions
#include <cstdlib>     // For strtod, strtol, etc.
#include <cstring>     // For strlen, memcpy, etc.
#include <functional>  // For std::function
#include <iterator>    // For iterator utilities
#include <locale>      // For locale-specific operations
#include <string>      // For std::string

// XSigma system includes
#include <xsigmasys/Encoding.hxx>           // For character encoding utilities
#include <xsigmasys/SystemInformation.hxx>  // For system information
#include <xsigmasys/SystemTools.hxx>        // For system utility functions

// XSigma core includes
#include "common/macros.h"
#include "common/pointer.h"      // For pointer utilities
#include "util/exception.h"      // For XSigma exception handling
#include "xsigmasys/Encoding.h"  // For encoding functions

// =============================================================================
// PLATFORM-SPECIFIC CONFIGURATION
// =============================================================================

#if XSIGMA_HAS_CXA_DEMANGLE
#include <cxxabi.h>
#endif

// =============================================================================
// NUMERIC LIMITS CONSTANTS
// =============================================================================

/**
 * @brief Compile-time constants for numeric type limits
 * @note These constants are used throughout the safe conversion functions
 * @note Using explicit casts to ensure correct type representation
 */

// Unsigned integer maximum values
static const uint8_t  kuint8max  = static_cast<uint8_t>(0xFF);         ///< Max value for uint8_t
static const uint16_t kuint16max = static_cast<uint16_t>(0xFFFF);      ///< Max value for uint16_t
static const uint32_t kuint32max = static_cast<uint32_t>(0xFFFFFFFF);  ///< Max value for uint32_t
static const uint64_t kuint64max =
    static_cast<uint64_t>(0XFFFFFFFFFFFFFFFFULL);  ///< Max value for uint64_t

// Signed integer minimum and maximum values
static const int8_t  kint8min  = static_cast<int8_t>(~0x7F);         ///< Min value for int8_t
static const int8_t  kint8max  = static_cast<int8_t>(0x7F);          ///< Max value for int8_t
static const int16_t kint16min = static_cast<int16_t>(~0x7FFF);      ///< Min value for int16_t
static const int16_t kint16max = static_cast<int16_t>(0x7FFF);       ///< Max value for int16_t
static const int32_t kint32min = static_cast<int32_t>(~0x7FFFFFFF);  ///< Min value for int32_t
static const int32_t kint32max = static_cast<int32_t>(0x7FFFFFFF);   ///< Max value for int32_t
static const int64_t kint64min =
    static_cast<int64_t>(~0X7FFFFFFFFFFFFFFFLL);  ///< Min value for int64_t
static const int64_t kint64max =
    static_cast<int64_t>(0X7FFFFFFFFFFFFFFFLL);  ///< Max value for int64_t

// =============================================================================
// ADVANCED DOUBLE CONVERSION IMPLEMENTATION
// =============================================================================

namespace xsigma::double_conversion
{
/**
 * @brief Configuration flags for string-to-double conversion
 * @note These flags control parsing behavior and can be combined using bitwise OR
 */
struct StringToDoubleFlags
{
    static constexpr int NO_FLAGS                 = 0;       ///< No special parsing behavior
    static constexpr int ALLOW_LEADING_SPACES     = 1 << 0;  ///< Skip whitespace at start
    static constexpr int ALLOW_TRAILING_SPACES    = 1 << 1;  ///< Skip whitespace at end
    static constexpr int ALLOW_HEX                = 1 << 2;  ///< Parse hexadecimal numbers (0x...)
    static constexpr int ALLOW_CASE_INSENSIBILITY = 1 << 3;  ///< Case-insensitive parsing
};

/**
 * @brief High-performance string to double converter with configurable behavior
 * @note This class provides locale-independent, configurable string-to-double conversion
 * @note Optimized for financial applications where precision and performance matter
 */
class StringToDoubleConverter
{
public:
    /**
     * @brief Construct a converter with specified configuration
     * @param flags Bitwise OR of StringToDoubleFlags values
     * @param empty_string_value Value to return for empty strings
     * @param junk_string_value Value to return for invalid strings
     * @param infinity_symbol String representation of infinity (e.g., "inf")
     * @param nan_symbol String representation of NaN (e.g., "nan")
     */
    StringToDoubleConverter(
        int         flags,
        double      empty_string_value,
        double      junk_string_value,
        const char* infinity_symbol,
        const char* nan_symbol)
        : flags_(flags),
          empty_string_value_(empty_string_value),
          junk_string_value_(junk_string_value),
          infinity_symbol_(infinity_symbol),
          nan_symbol_(nan_symbol)
    {
    }

    /**
     * @brief Convert string to double with comprehensive parsing support
     * @param str Input string to parse
     * @param length Length of the input string
     * @param processed_chars_ptr Pointer to store number of characters processed
     * @return Parsed double value or configured fallback value
     * @note This is the main entry point for string-to-double conversion
     * @note Handles empty strings, whitespace, special values, hex, and decimal numbers
     */
    double StringToDouble(const char* str, int length, int* processed_chars_ptr) const
    {
        // Handle empty string case
        if (length == 0)
        {
            *processed_chars_ptr = 0;
            return empty_string_value_;
        }

        int current = 0;

        // Skip leading whitespace if configured to do so
        if (flags_ & StringToDoubleFlags::ALLOW_LEADING_SPACES)
        {
            while (current < length && std::isspace(str[current]))
            {
                ++current;
            }
        }

        // Check if we consumed the entire string with whitespace
        if (current == length)
        {
            *processed_chars_ptr = current;
            return empty_string_value_;
        }

        // Check for special values (infinity, NaN) first
        if (CheckSpecialValues(str + current, length - current, processed_chars_ptr, current))
        {
            *processed_chars_ptr = current + *processed_chars_ptr;
            return result_;
        }

        // Handle hexadecimal numbers (0x prefix)
        if ((flags_ & StringToDoubleFlags::ALLOW_HEX) && (current + 2 < length) &&
            (str[current] == '0') && (ToLower(str[current + 1]) == 'x'))
        {
            return HandleHex(str + current, length - current, processed_chars_ptr, current);
        }

        // Handle regular decimal numbers (most common case)
        return HandleDecimal(str + current, length - current, processed_chars_ptr, current);
    }

private:
    // Configuration members
    const int      flags_;               ///< Parsing behavior flags
    const double   empty_string_value_;  ///< Value returned for empty strings
    const double   junk_string_value_;   ///< Value returned for invalid strings
    const char*    infinity_symbol_;     ///< String representation of infinity
    const char*    nan_symbol_;          ///< String representation of NaN
    mutable double result_;              ///< Temporary storage for special values

    /**
     * @brief Convert character to lowercase if case-insensitive mode is enabled
     * @param c Character to potentially convert
     * @return Lowercase character if case-insensitive, original character otherwise
     * @note Only performs conversion when ALLOW_CASE_INSENSIBILITY flag is set
     */
    char ToLower(char c) const
    {
        return (flags_ & StringToDoubleFlags::ALLOW_CASE_INSENSIBILITY) ? std::tolower(c) : c;
    }

    /**
     * @brief Check if string starts with a special value (infinity or NaN)
     * @param str String to check
     * @param length Length of the string (unused but kept for interface consistency)
     * @param processed_chars Pointer to store number of characters consumed
     * @param start Starting position (unused but kept for interface consistency)
     * @return true if a special value was found and parsed, false otherwise
     * @note Sets result_ member variable when a special value is found
     */
    bool CheckSpecialValues(
        const char*       str,
        XSIGMA_UNUSED int length,
        int*              processed_chars,
        XSIGMA_UNUSED int start) const
    {
        // Check for infinity symbol (e.g., "inf", "infinity")
        if (infinity_symbol_ && CompareInsensitive(str, infinity_symbol_))
        {
            size_t len       = strlen(infinity_symbol_);
            *processed_chars = static_cast<int>(len);
            result_          = std::numeric_limits<double>::infinity();
            return true;
        }

        // Check for NaN symbol (e.g., "nan")
        if (nan_symbol_ && CompareInsensitive(str, nan_symbol_))
        {
            size_t len       = strlen(nan_symbol_);
            *processed_chars = static_cast<int>(len);
            result_          = std::numeric_limits<double>::quiet_NaN();
            return true;
        }

        return false;  // No special value found
    }

    /**
     * @brief Compare two strings with optional case-insensitive matching
     * @param str1 First string to compare
     * @param str2 Second string to compare
     * @return true if strings match (considering case sensitivity settings), false otherwise
     * @note Uses ToLower() method which respects the ALLOW_CASE_INSENSIBILITY flag
     * @note Performs character-by-character comparison for maximum control
     */
    bool CompareInsensitive(const char* str1, const char* str2) const
    {
        // Compare characters one by one
        while (*str1 && *str2)
        {
            if (ToLower(*str1) != ToLower(*str2))
            {
                return false;  // Mismatch found
            }
            str1++;
            str2++;
        }
        // Both strings must end at the same time for a match
        return *str1 == *str2;
    }

    /**
     * @brief Parse hexadecimal number (with 0x prefix) to double
     * @param str String starting with "0x"
     * @param length Length of the string
     * @param processed_chars_ptr Pointer to store number of characters processed
     * @param start Starting position in the original string
     * @return Parsed double value or junk_string_value_ on error
     * @note Handles both uppercase and lowercase hex digits
     * @note Includes overflow detection to prevent undefined behavior
     */
    double HandleHex(const char* str, int length, int* processed_chars_ptr, int start) const
    {
        // Skip the "0x" prefix
        str += 2;
        length -= 2;

        uint64_t value     = 0;
        int      processed = 0;

        // Parse hex digits one by one
        while (processed < length)
        {
            char c = str[processed];
            int  digit;

            // Convert character to hex digit value
            if (c >= '0' && c <= '9')
            {
                digit = c - '0';  // 0-9
            }
            else if (c >= 'a' && c <= 'f')
            {
                digit = 10 + (c - 'a');  // a-f (lowercase)
            }
            else if (c >= 'A' && c <= 'F')
            {
                digit = 10 + (c - 'A');  // A-F (uppercase)
            }
            else
            {
                break;  // Non-hex character encountered
            }

            // Check for overflow before shifting
            if (value > (std::numeric_limits<uint64_t>::max() >> 4))
            {
                break;  // Would overflow on next shift
            }

            // Accumulate the digit (shift left by 4 bits and add new digit)
            value = (value << 4) | digit;
            processed++;
        }

        // Check if we parsed any hex digits
        if (processed == 0)
        {
            *processed_chars_ptr = start;
            return junk_string_value_;  // No valid hex digits found
        }

        // Update processed character count (include "0x" prefix)
        *processed_chars_ptr = start + processed + 2;
        return static_cast<double>(value);
    }

    /**
     * @brief Parse decimal number to double using standard library
     * @param str String containing decimal number
     * @param length Length of the string
     * @param processed_chars_ptr Pointer to store number of characters processed
     * @param start Starting position in the original string
     * @return Parsed double value or junk_string_value_ on error
     * @note Uses std::strtod for the actual parsing (locale-independent)
     * @note Handles trailing whitespace if configured to do so
     */
    double HandleDecimal(const char* str, int length, int* processed_chars_ptr, int start) const
    {
        char*  end_ptr;
        double result = std::strtod(str, &end_ptr);

        // Check if any characters were consumed
        if (end_ptr == str)
        {
            *processed_chars_ptr = start;
            return junk_string_value_;  // No valid number found
        }

        int processed = static_cast<int>(end_ptr - str);

        // Skip trailing whitespace if configured to do so
        if (flags_ & StringToDoubleFlags::ALLOW_TRAILING_SPACES)
        {
            while (processed < length && std::isspace(str[processed]))
            {
                ++processed;
            }
        }

        *processed_chars_ptr = start + processed;
        return result;
    }
};
}  // namespace xsigma::double_conversion

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
// FILE PATH MANIPULATION FUNCTIONS
// =============================================================================

/**
 * @brief Extract basename from a file path (Unix-style separator)
 * @note Uses '/' as path separator for cross-platform consistency
 * @note Optimized for performance with single pass through string
 */
std::string strip_basename(const std::string& full_path)
{
    const char kSeparator = '/';
    const auto pos        = full_path.rfind(kSeparator);

    // If no separator found, the entire path is the basename
    return (pos != std::string::npos) ? full_path.substr(pos + 1, std::string::npos) : full_path;
}

/**
 * @brief Remove file extension from filename
 * @note Finds the last dot and removes everything from that point onward
 * @note If no extension exists, returns the original filename
 */
std::string exclude_file_extension(const std::string& file_name)
{
    const char sep     = '.';
    const auto dot_pos = file_name.find_last_of(sep);

    // If no dot found, return entire filename
    if (dot_pos == std::string::npos)
    {
        return file_name;
    }

    return file_name.substr(0, dot_pos);
}

/**
 * @brief Extract file extension from filename
 * @note Returns everything from the last dot onward (including the dot)
 * @note If no extension exists, returns empty string
 */
std::string file_extension(const std::string& file_name)
{
    const char sep     = '.';
    const auto dot_pos = file_name.find_last_of(sep);

    // If no dot found, return empty string
    if (dot_pos == std::string::npos)
    {
        return "";
    }

    return file_name.substr(dot_pos);
}

// =============================================================================
// SOURCE LOCATION AND DEBUGGING UTILITIES
// =============================================================================

/**
 * @brief Stream output operator for SourceLocation debugging info
 * @note Formats as "function at file:line" for easy parsing by IDEs
 */
std::ostream& operator<<(std::ostream& out, const SourceLocation& loc)
{
    out << loc.function << " at " << loc.file << ":" << loc.line;
    return out;
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

// =============================================================================
// STRING TO NUMERIC CONVERSION FUNCTIONS
// =============================================================================

/**
 * @brief Convert string to integer with position tracking
 * @note More robust than std::stoi with better error reporting
 * @note Uses stringstream for parsing with comprehensive error checking
 */
int stoi(const std::string& str, std::size_t* pos)
{
    std::stringstream ss;

    int n = 0;
    ss << str;
    ss >> n;

    // Check for conversion failure
    XSIGMA_CHECK_VALUE(!ss.fail(), "String is not a valid integer");

    // Update position if requested
    if (pos != nullptr)
    {
        if (ss.tellg() == std::streampos(-1))
        {
            // Entire string was consumed
            *pos = str.size();
        }
        else
        {
            // Partial consumption - return position where parsing stopped
            *pos = static_cast<std::size_t>(ss.tellg());
        }
    }
    return n;
}

// =============================================================================
// STRING VALIDATION FUNCTIONS
// =============================================================================

/**
 * @brief Check if string represents a valid floating-point number
 * @note Uses standard library strtod for robust parsing
 * @note Handles special cases like infinity and NaN
 * @note Rejects HUGE_VAL to avoid overflow issues
 */
bool is_float(const std::string& str)
{
    char*      end = nullptr;
    const auto val = strtod(str.c_str(), &end);

    // Valid if: parsing consumed characters, reached end of string, and no overflow
    return end != str.c_str() && *end == '\0' && val != HUGE_VAL;
}

/**
 * @brief Check if string contains only digit characters
 * @note Simple validation - does not handle negative numbers or whitespace
 * @note For more robust integer validation, use safe_strto64 functions
 * @note Uses std::find_if with lambda for efficient character checking
 */
bool is_integer(const std::string& str)
{
    return !str.empty() &&
           std::find_if(
               str.begin(), str.end(), [](unsigned char c) { return std::isdigit(c) == 0; }) ==
               str.end();
}

// =============================================================================
// STRING SPLITTING AND TOKENIZATION
// =============================================================================

/**
 * @brief Split string using multiple separator characters
 * @note Uses custom locale facet to treat separators as whitespace
 * @note This approach is more robust than simple character-by-character parsing
 * @note Automatically filters out empty tokens
 */
std::vector<std::string> split_string(std::string_view s, const std::vector<char>& separators)
{
    /**
     * @brief Custom ctype facet that treats specified characters as whitespace
     * @note This allows us to use standard stream extraction (>>) for tokenization
     * @note More efficient than manual parsing for complex separator sets
     */
    struct tokens : std::ctype<char>
    {
        /**
         * @brief Construct facet with custom separator characters
         * @param Separators Vector of characters to treat as separators
         */
        explicit tokens(const std::vector<char>& Separators)
            : std::ctype<char>(get_table(Separators))
        {
        }

        /**
         * @brief Create character classification table with custom separators
         * @param Separators Characters to mark as whitespace/separators
         * @return Pointer to character classification table
         * @note Uses static storage for performance - table persists across calls
         */
        static std::ctype_base::mask const* get_table(const std::vector<char>& Separators)
        {
            using cctype = std::ctype<char>;

            // Get the standard character classification table
            static const cctype::mask* const_rc = cctype::classic_table();

            // Create our custom table based on the standard one
            static std::array<cctype::mask, cctype::table_size> rc;
            std::memcpy(rc.data(), const_rc, cctype::table_size * sizeof(cctype::mask));

            // Mark each separator character as whitespace
            for (auto c : Separators)
            {
                rc[static_cast<unsigned char>(c)] = std::ctype_base::space;
            }
            return rc.data();
        }
    };

    // Convert string_view to string for stringstream compatibility
    const auto&       tmp = std::string(s);
    std::stringstream ss(tmp);

    // Apply our custom locale that treats separators as whitespace
    ss.imbue(std::locale(std::locale(), new tokens(separators)));

    // Extract tokens using standard stream extraction
    std::vector<std::string> ret;
    std::string              token;
    while (ss >> token)
    {
        ret.push_back(token);
    }

    return ret;
}

// =============================================================================
// NUMERIC FORMATTING FUNCTIONS
// =============================================================================

/**
 * @brief Format double with specified precision and field width
 * @note Uses fixed-point notation for consistent decimal places
 * @note Right-aligned with space padding for tabular output
 */
std::string to_string(const double x, const int decDigits, const int width)
{
    std::stringstream ss;
    ss << std::fixed << std::right;
    ss.fill(' ');             // Use spaces for padding
    ss.width(width);          // Set minimum field width
    ss.precision(decDigits);  // Set number of decimal places
    ss << x;
    return ss.str();
}

/**
 * @brief Center a string within a specified width using space padding
 * @note If the string is longer than the width, returns the original string
 * @note Handles odd padding by adding extra space to the right
 */
std::string center(const std::string s, const int width)
{
    std::stringstream ss;
    std::stringstream spaces;
    const auto        padding = width - static_cast<int>(s.size());

    // If no padding needed, return original string
    if (padding <= 0)
    {
        return s;
    }

    // Create left padding (half of total padding)
    for (int i = 0; i < padding / 2; ++i)
    {
        spaces << " ";
    }

    // Assemble: left_padding + string + right_padding
    ss << spaces.str() << s << spaces.str();

    // Add extra space on the right if padding is odd
    if (padding % 2 != 0)
    {
        ss << " ";
    }

    return ss.str();
}

// =============================================================================
// PLATFORM-SPECIFIC UTILITY FUNCTIONS
// =============================================================================

/**
 * @brief Convert wide character string to narrow (UTF-8) string
 * @note Delegates to xsigmasys::Encoding for platform-specific implementation
 * @note Primarily used for Windows Unicode interoperability
 */
std::string ToNarrow(wchar_t* x)
{
    return xsigmasys::Encoding::ToNarrow(x);
}

/**
 * @brief Create a heap-allocated copy of a C-string
 * @note Delegates to xsigmasys::SystemTools for platform-specific allocation
 * @note Caller must free the returned pointer to avoid memory leaks
 * @warning Always pair with appropriate deallocation (free() or delete[])
 */
char* DuplicateString(const char* str)
{
    return xsigmasys::SystemTools::DuplicateString(str);
}

// =============================================================================
// PRINTF-STYLE FORMATTING IMPLEMENTATION
// =============================================================================

/**
 * @brief Core printf-style formatting function using va_list
 * @note This is the foundation for all printf-style functions in this module
 * @note Handles buffer management and platform-specific formatting differences
 * @note Uses two-phase approach: try small buffer first, then allocate as needed
 */
void Appendv(std::string* dst, const char* format, va_list ap)
{
    // Phase 1: Try with a reasonably-sized stack buffer first
    static const int kSpaceLength = 1024;
    char             space[kSpaceLength];

    // Important: va_list may be invalidated after use, so make a copy
    // This is required by the C standard and critical for portability
    va_list backup_ap;
    va_copy(backup_ap, ap);
    int result = vsnprintf(space, kSpaceLength, format, backup_ap);
    va_end(backup_ap);

    // Check if the small buffer was sufficient
    if (result < kSpaceLength)
    {
        if (result >= 0)
        {
            // Success: everything fit in the small buffer
            dst->append(space, result);
            return;
        }

#ifdef _MSC_VER
        // Handle MSVC-specific behavior: older versions return -1 on overflow
        // MSVC 8.0+ can tell us the required buffer size with this idiom:
        va_copy(backup_ap, ap);
        result = vsnprintf(nullptr, 0, format, backup_ap);
        va_end(backup_ap);
#endif

        if (result < 0)
        {
            // Formatting error occurred - abort
            return;
        }
    }

    // Phase 2: Allocate a buffer large enough for the full result
    // Add 1 for null terminator
    int   length = result + 1;
    char* buf    = new char[length];

    // Format again with the properly-sized buffer
    va_copy(backup_ap, ap);
    result = vsnprintf(buf, length, format, backup_ap);
    va_end(backup_ap);

    // Verify the formatting succeeded and append the result
    if (result >= 0 && result < length)
    {
        dst->append(buf, result);
    }

    // Clean up the allocated buffer
    delete[] buf;
}

/**
 * @brief Create formatted string using printf-style format specifiers
 * @note Convenience wrapper around Appendv for creating new strings
 * @note Provides automatic memory management for formatted output
 */
std::string Printf(const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    std::string result;
    Appendv(&result, format, ap);
    va_end(ap);
    return result;
}

/**
 * @brief Append formatted text to existing string using printf-style format
 * @note Convenience wrapper around Appendv for appending to existing strings
 * @note More efficient than string concatenation for multiple operations
 */
void Appendf(std::string* dst, const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    Appendv(dst, format, ap);
    va_end(ap);
}
}  // namespace xsigma

namespace xsigma
{
namespace
{
template <typename T>
const std::unordered_map<std::string, T>* GetSpecialNumsSingleton()
{
    static const std::unordered_map<std::string, T>* special_nums =
        new const std::unordered_map<std::string, T>{
            {"inf", std::numeric_limits<T>::infinity()},
            {"+inf", std::numeric_limits<T>::infinity()},
            {"-inf", -std::numeric_limits<T>::infinity()},
            {"infinity", std::numeric_limits<T>::infinity()},
            {"+infinity", std::numeric_limits<T>::infinity()},
            {"-infinity", -std::numeric_limits<T>::infinity()},
            {"nan", std::numeric_limits<T>::quiet_NaN()},
            {"+nan", std::numeric_limits<T>::quiet_NaN()},
            {"-nan", -std::numeric_limits<T>::quiet_NaN()},
        };
    return special_nums;
}

template <typename T>
T locale_independent_strtonum(const char* str, const char** endptr)
{
    auto              special_nums = GetSpecialNumsSingleton<T>();
    std::stringstream s(str);

    // Check if str is one of the special numbers.
    std::string special_num_str;
    s >> special_num_str;

    for (char& tmp : special_num_str)
    {
        tmp = std::tolower(tmp, std::locale::classic());
    }

    auto entry = special_nums->find(special_num_str);
    if (entry != special_nums->end())
    {
        *endptr = str + (s.eof() ? static_cast<std::iostream::pos_type>(strlen(str)) : s.tellg());
        return entry->second;
    }
    else
    {
        // Perhaps it's a hex number
        if (special_num_str.compare(0, 2, "0x") == 0 || special_num_str.compare(0, 3, "-0x") == 0)
        {
            return strtol(str, const_cast<char**>(endptr), 16);
        }
    }
    // Reset the stream
    s.str(str);
    s.clear();
    // Use the "C" locale
    s.imbue(std::locale::classic());

    T result;
    s >> result;

    // Set to result to what strto{f,d} functions would have returned. If the
    // number was outside the range, the stringstream sets the fail flag, but
    // returns the +/-max() value, whereas strto{f,d} functions return +/-INF.
    if (s.fail())
    {
        if (result == std::numeric_limits<T>::max() || result == std::numeric_limits<T>::infinity())
        {
            result = std::numeric_limits<T>::infinity();
            s.clear(s.rdstate() & ~std::ios::failbit);
        }
        else if (
            result == -std::numeric_limits<T>::max() ||
            result == -std::numeric_limits<T>::infinity())
        {
            result = -std::numeric_limits<T>::infinity();
            s.clear(s.rdstate() & ~std::ios::failbit);
        }
    }

    if (endptr)
    {
        *endptr = str + (s.fail() ? static_cast<std::iostream::pos_type>(0)
                                  : (s.eof() ? static_cast<std::iostream::pos_type>(strlen(str))
                                             : s.tellg()));
    }
    return result;
}

inline const double_conversion::StringToDoubleConverter& StringToFloatConverter()
{
    static const double_conversion::StringToDoubleConverter converter(
        double_conversion::StringToDoubleFlags::ALLOW_LEADING_SPACES |
            double_conversion::StringToDoubleFlags::ALLOW_HEX |
            double_conversion::StringToDoubleFlags::ALLOW_TRAILING_SPACES |
            double_conversion::StringToDoubleFlags::ALLOW_CASE_INSENSIBILITY,
        0.,
        0.,
        "inf",
        "nan");
    return converter;
}

}  // namespace
namespace numbers
{
size_t FastInt32ToBufferLeft(int32_t i, char* buffer)
{
    uint32_t u      = i;
    size_t   length = 0;
    if (i < 0)
    {
        *buffer++ = '-';
        ++length;
        // We need to do the negation in modular (i.e., "unsigned")
        // arithmetic; MSVC++ apparently warns for plain "-u", so
        // we write the equivalent expression "0 - u" instead.
        u = 0 - u;
    }
    length += FastUInt32ToBufferLeft(u, buffer);
    return length;
}

size_t FastUInt32ToBufferLeft(uint32_t i, char* buffer)
{
    char* start = buffer;
    do
    {
        *buffer++ = ((i % 10) + '0');
        i /= 10;
    } while (i > 0);
    *buffer = 0;
    std::reverse(start, buffer);
    return buffer - start;
}

size_t FastInt64ToBufferLeft(int64_t i, char* buffer)
{
    uint64_t u      = i;
    size_t   length = 0;
    if (i < 0)
    {
        *buffer++ = '-';
        ++length;
        u = 0 - u;
    }
    length += FastUInt64ToBufferLeft(u, buffer);
    return length;
}

size_t FastUInt64ToBufferLeft(uint64_t i, char* buffer)
{
    char* start = buffer;
    do
    {
        *buffer++ = ((i % 10) + '0');
        i /= 10;
    } while (i > 0);
    *buffer = 0;
    std::reverse(start, buffer);
    return buffer - start;
}

static const double kDoublePrecisionCheckMax = DBL_MAX / 1.000000000000001;

size_t DoubleToBuffer(double value, char* buffer)
{
    // DBL_DIG is 15 for IEEE-754 doubles, which are used on almost all
    // platforms these days.  Just in case some system exists where DBL_DIG
    // is significantly larger -- and risks overflowing our buffer -- we have
    // this assert.
    static_assert(DBL_DIG < 20, "DBL_DIG is too big");

    if (std::isnan(value))
    {
        int snprintf_result =
            snprintf(buffer, kFastToBufferSize, "%snan", std::signbit(value) ? "-" : "");
        // Paranoid check to ensure we don't overflow the buffer.
        XSIGMA_CHECK_DEBUG(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
        return snprintf_result;
    }

    if (std::abs(value) <= kDoublePrecisionCheckMax)
    {
        int snprintf_result = snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG, value);

        // The snprintf should never overflow because the buffer is significantly
        // larger than the precision we asked for.
        XSIGMA_CHECK_DEBUG(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

        if (locale_independent_strtonum<double>(buffer, nullptr) == value)
        {
            // Round-tripping the string to double works; we're done.
            return snprintf_result;
        }
        // else: full precision formatting needed. Fall through.
    }

    int snprintf_result = snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG + 2, value);

    // Should never overflow; see above.
    XSIGMA_CHECK_DEBUG(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

    return snprintf_result;
}

namespace
{
char SafeFirstChar(std::string_view str)
{
    if (str.empty())
        return '\0';
    return str[0];
}
void SkipSpaces(std::string_view* str)
{
    while (isspace(SafeFirstChar(*str)))
        str->remove_prefix(1);
}
inline bool consume_prefix(std::string_view& str, std::string_view prefix)
{
    if (starts_with(str, prefix))
    {
        str.remove_prefix(prefix.length());
        return true;
    }
    return false;
}
}  // namespace

bool safe_strto64(std::string_view str, int64_t* value)
{
    SkipSpaces(&str);

    int64_t vlimit = kint64max;
    int     sign   = 1;
    if (consume_prefix(str, "-"))
    {
        sign = -1;
        // Different limit for positive and negative integers.
        vlimit = kint64min;
    }

    if (!isdigit(SafeFirstChar(str)))
        return false;

    int64_t result = 0;
    if (sign == 1)
    {
        do
        {
            int digit = SafeFirstChar(str) - '0';
            if ((vlimit - digit) / 10 < result)
            {
                return false;
            }
            result = result * 10 + digit;
            str.remove_prefix(1);
        } while (isdigit(SafeFirstChar(str)));
    }
    else
    {
        do
        {
            int digit = SafeFirstChar(str) - '0';
            if ((vlimit + digit) / 10 > result)
            {
                return false;
            }
            result = result * 10 - digit;
            str.remove_prefix(1);
        } while (isdigit(SafeFirstChar(str)));
    }

    SkipSpaces(&str);
    if (!str.empty())
        return false;

    *value = result;
    return true;
}

bool safe_strtou64(std::string_view str, uint64_t* value)
{
    SkipSpaces(&str);
    if (!isdigit(SafeFirstChar(str)))
        return false;

    uint64_t result = 0;
    do
    {
        int digit = SafeFirstChar(str) - '0';
        if ((kuint64max - digit) / 10 < result)
        {
            return false;
        }
        result = result * 10 + digit;
        str.remove_prefix(1);
    } while (isdigit(SafeFirstChar(str)));

    SkipSpaces(&str);
    if (!str.empty())
        return false;

    *value = result;
    return true;
}

//bool safe_strto32(std::string_view str, int32_t* value)
//{
//    SkipSpaces(&str);
//
//    int64_t vmax = kint32max;
//    int     sign = 1;
//    if (std::ConsumePrefix(&str, "-"))
//    {
//        sign = -1;
//        // Different max for positive and negative integers.
//        ++vmax;
//    }
//
//    if (!isdigit(SafeFirstChar(str)))
//        return false;
//
//    int64_t result = 0;
//    do
//    {
//        result = result * 10 + SafeFirstChar(str) - '0';
//        if (result > vmax)
//        {
//            return false;
//        }
//        str.remove_prefix(1);
//    } while (isdigit(SafeFirstChar(str)));
//
//    SkipSpaces(&str);
//
//    if (!str.empty())
//        return false;
//
//    *value = static_cast<int32_t>(result * sign);
//    return true;
//}

bool safe_strtou32(std::string_view str, uint32_t* value)
{
    SkipSpaces(&str);
    if (!isdigit(SafeFirstChar(str)))
        return false;

    int64_t result = 0;
    do
    {
        result = result * 10 + SafeFirstChar(str) - '0';
        if (result > kuint32max)
        {
            return false;
        }
        str.remove_prefix(1);
    } while (isdigit(SafeFirstChar(str)));

    SkipSpaces(&str);
    if (!str.empty())
    {
        return false;
    }

    *value = static_cast<uint32_t>(result);
    return true;
}

bool safe_strtof(std::string_view str, float* value)
{
    int  processed_characters_count = -1;
    auto len                        = str.size();

    // If string length exceeds buffer size or int max, fail.
    if (len >= kFastToBufferSize)
    {
        return false;
    }
    if (len > std::numeric_limits<int>::max())
    {
        return false;
    }

    *value = StringToFloatConverter().StringToDouble(
        str.data(), static_cast<int>(len), &processed_characters_count);
    return processed_characters_count > 0;
}

bool safe_strtod(std::string_view str, double* value)
{
    int  processed_characters_count = -1;
    auto len                        = str.size();

    // If string length exceeds buffer size or int max, fail.
    if (len >= kFastToBufferSize)
        return false;
    if (len > std::numeric_limits<int>::max())
        return false;

    *value = StringToFloatConverter().StringToDouble(
        str.data(), static_cast<int>(len), &processed_characters_count);
    return processed_characters_count > 0;
}

size_t FloatToBuffer(float value, char* buffer)
{
    // FLT_DIG is 6 for IEEE-754 floats, which are used on almost all
    // platforms these days.  Just in case some system exists where FLT_DIG
    // is significantly larger -- and risks overflowing our buffer -- we have
    // this assert.
    static_assert(FLT_DIG < 10, "FLT_DIG is too big");

    if (std::isnan(value))
    {
        int snprintf_result =
            snprintf(buffer, kFastToBufferSize, "%snan", std::signbit(value) ? "-" : "");
        // Paranoid check to ensure we don't overflow the buffer.
        XSIGMA_CHECK_DEBUG(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
        return snprintf_result;
    }

    int snprintf_result = snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG, value);

    // The snprintf should never overflow because the buffer is significantly
    // larger than the precision we asked for.
    XSIGMA_CHECK_DEBUG(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

    float parsed_value;
    if (parsed_value != value)
    {
        snprintf_result = snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG + 3, value);

        // Should never overflow; see above.
        XSIGMA_CHECK_DEBUG(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
    }
    return snprintf_result;
}

std::string_view Uint64ToHexString(uint64_t v, char* buf)
{
    static const char* hexdigits = "0123456789abcdef";
    const int          num_byte  = 16;
    buf[num_byte]                = '\0';
    for (int i = num_byte - 1; i >= 0; i--)
    {
        buf[i] = hexdigits[v & 0xf];
        v >>= 4;
    }
    return std::string_view(buf, num_byte);  //NOLINT
}

bool HexStringToUint64(const std::string_view& s, uint64_t* result)
{
    uint64_t v = 0;
    if (s.empty())
    {
        return false;
    }

    for (char c : s)
    {
        if (c >= '0' && c <= '9')
        {
            v = (v << 4) + (c - '0');
        }
        else if (c >= 'a' && c <= 'f')
        {
            v = (v << 4) + 10 + (c - 'a');
        }
        else if (c >= 'A' && c <= 'F')
        {
            v = (v << 4) + 10 + (c - 'A');
        }
        else
        {
            return false;
        }
    }
    *result = v;
    return true;
}

std::string HumanReadableNum(int64_t value)
{
    std::string s;
    if (value < 0)
    {
        s += "-";
        value = -value;
    }
    if (value < 1000)
    {
        Appendf(&s, "%lld", static_cast<long long>(value));
    }
    else if (value >= static_cast<int64_t>(1e15))
    {
        // Number bigger than 1E15; use that notation.
        Appendf(&s, "%0.3G", static_cast<double>(value));
    }
    else
    {
        static const char units[] = "kMBT";
        const char*       unit    = units;
        while (value >= static_cast<int64_t>(1000000))
        {
            value /= static_cast<int64_t>(1000);
            ++unit;
            // XSIGMA_CHECK(unit < units + TF_ARRAYSIZE(units));
        }
        Appendf(&s, "%.2f%c", value / 1000.0, *unit);
    }
    return s;
}

std::string HumanReadableNumBytes(int64_t num_bytes)
{
    if (num_bytes == kint64min)
    {
        // Special case for number with not representable negation.
        return "-8E";
    }

    const char* neg_str = (num_bytes < 0) ? "-" : "";
    if (num_bytes < 0)
    {
        num_bytes = -num_bytes;
    }

    // Special case for bytes.
    if (num_bytes < 1024)
    {
        // No fractions for bytes.
        char buf[8];  // Longest possible string is '-XXXXB'
        snprintf(buf, sizeof(buf), "%s%lldB", neg_str, static_cast<long long>(num_bytes));
        return {buf};
    }

    static const char units[] = "KMGTPE";  // int64 only goes up to E.
    const char*       unit    = units;
    while (num_bytes >= static_cast<int64_t>(1024) * 1024)
    {
        num_bytes /= 1024;
        ++unit;
    }

    // We use SI prefixes.
    char buf[16];
    snprintf(
        buf,
        sizeof(buf),
        ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
        neg_str,
        num_bytes / 1024.0,
        *unit);

    return {buf};
}
}  // namespace numbers

// C++17/C++20 compatibility utility for std::string::starts_with()
bool starts_with(const std::string& str, const std::string& prefix)
{
#if __cplusplus >= 202002L
    return str.starts_with(prefix);
#else
    return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
#endif
}

// C++17/C++20 compatibility utility for std::string_view::starts_with()
bool starts_with(std::string_view str, std::string_view prefix)
{
#if __cplusplus >= 202002L
    return str.starts_with(prefix);
#else
    return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
#endif
}

// C++17/C++20 compatibility utility for std::string::ends_with()
bool ends_with(const std::string& str, const std::string& suffix)
{
#if __cplusplus >= 202002L
    return str.ends_with(suffix);
#else
    return str.size() >= suffix.size() && str.substr(str.size() - suffix.size()) == suffix;
#endif
}

// C++17/C++20 compatibility utility for std::string_view::ends_with()
bool ends_with(std::string_view str, std::string_view suffix)
{
#if __cplusplus >= 202002L
    return str.ends_with(suffix);
#else
    return str.size() >= suffix.size() && str.substr(str.size() - suffix.size()) == suffix;
#endif
}

}  // namespace xsigma
