#ifndef CORE_UTIL_EXCEPTION_H
#define CORE_UTIL_EXCEPTION_H

#include <algorithm>
#include <exception>  // for exception
#include <string>     // for string
#include <vector>     // for vector

#include "common/macros.h"
#include "logging/logger.h"
#include "util/lazy.h"
#include "util/string_util.h"

#define XSIGMA_STRINGIZE_IMPL(x) #x
#define XSIGMA_STRINGIZE(x) XSIGMA_STRINGIZE_IMPL(x)

namespace xsigma
{
class XSIGMA_VISIBILITY Error : public std::exception
{
    // The actual error message.
    std::string msg_;

    // Context for the message (in order of decreasing specificity).  Context will
    // be automatically formatted appropriately, so it is not necessary to add
    // extra leading/trailing newlines to strings inside this vector
    std::vector<std::string> context_;

    // The C++ backtrace at the point when this exception was raised.  This
    // may be empty if there is no valid backtrace.  (We don't use optional
    // here to reduce the dependencies this file has.)
    std::string backtrace_;

    // These two are derived fields from msg_stack_ and backtrace_, but we need
    // fields for the strings so that we can return a const char* (as the
    // signature of std::exception requires).  Currently, the invariant
    // is that these fields are ALWAYS populated consistently with respect
    // to msg_stack_ and backtrace_.
    mutable optimistic_lazy<std::string> what_;
    std::string                          what_without_backtrace_;

    // This is a little debugging trick: you can stash a relevant pointer
    // in caller, and then when you catch the exception, you can compare
    // against pointers you have on hand to get more information about
    // where the exception came from.  In xsigma, this is used to figure
    // out which operator raised an exception.
    const void* caller_;

public:
    // Virtual destructor
    XSIGMA_API virtual ~Error();

    // xsigma-style Error constructor.
    // NB: the implementation of this is actually in Logging.cpp
    XSIGMA_API Error(SourceLocation source_location, std::string msg);

    // Base constructor
    XSIGMA_API Error(std::string msg, std::string backtrace, const void* caller = nullptr);

    // Add some new context to the message stack.  The last added context
    // will be formatted at the end of the context list upon printing.
    // WARNING: This method is O(n) in the size of the stack, so don't go
    // wild adding a ridiculous amount of context to error messages.
    XSIGMA_API void add_context(std::string msg);

    const std::string& msg() const { return msg_; }

    const std::vector<std::string>& context() const { return context_; }

    const std::string& backtrace() const { return backtrace_; }

    /// Returns the complete error message, including the source location.
    /// The returned pointer is invalidated if you call add_context() on
    /// this object.
    XSIGMA_API const char* what() const noexcept override;

    const void* caller() const noexcept { return caller_; }

    /// Returns only the error message string, without source location.
    /// The returned pointer is invalidated if you call add_context() on
    /// this object.
    const char* what_without_backtrace() const noexcept { return what_without_backtrace_.c_str(); }

private:
    void        refresh_what();
    std::string compute_what(bool include_backtrace) const;
};

class XSIGMA_VISIBILITY WarningHandler
{
public:
    virtual ~WarningHandler() noexcept(false) {};  // NOLINT

    // The default warning handler. Prints the message to stderr.
    XSIGMA_API virtual void process(
        const SourceLocation& source_location, const std::string& msg, bool verbatim);
};

namespace Warning
{
// Note: [Verbatim Warnings]
// Warnings originating in C++ code can appear out-of-place to Python users:
// a user runs a line in Python, but the warning references a line in C++.
// Some parts of xsigma, like the JIT, are cognizant of this mismatch
// and take care to map warnings back to the user's program, but most
// of xsigma simply throws a context-free warning. To allow warning
// handlers to add context where appropriate, warn takes the
// "verbatim" flag. When this is false a warning handler might append
// the C++ warning to a Python warning message that relates the warning
// back to the user's program. Callers who have already accounted for
// context in their warnings should set verbatim to true so their warnings
// appear without modification.

/// Issue a warning with a given message. Dispatched to the current
/// warning handler.
XSIGMA_API void warn(SourceLocation source_location, const std::string& msg, bool verbatim);

XSIGMA_API void set_warnAlways(bool) noexcept(true);  // NOLINT
XSIGMA_API bool get_warnAlways(void) noexcept(true);  // NOLINT
}  // namespace Warning

// Used in ATen for out-of-bound indices that can reasonably only be detected
// lazily inside a kernel (See: advanced indexing).  These turn into
// IndexError when they cross to Python.
class IndexError : public Error
{
    using Error::Error;
};

// Used in ATen for invalid values.  These turn into
// ValueError when they cross to Python.
class ValueError : public Error
{
    using Error::Error;
};

// Used in ATen for invalid types.  These turn into
// TypeError when they cross to Python.
class TypeError : public Error
{
    using Error::Error;
};

// Used in ATen for functionality that is not implemented.  These turn into
// NotImplementedError when they cross to Python.
class NotImplementedError : public Error
{
    using Error::Error;
};

// Used in ATen for non finite indices.  These turn into
// ExitException when they cross to Python.
class EnforceFiniteError : public Error
{
    using Error::Error;
};

// A utility function to return an exception std::string by prepending its
// exception type before its what() content
// XSIGMA_API std::string GetExceptionString(const std::exception& e);
}  // namespace xsigma

#define XSIGMA_THROW_ERROR(err_type, msg)                                              \
    do                                                                                 \
    {                                                                                  \
        xsigma::SourceLocation loc = {__func__, __FILE__, static_cast<int>(__LINE__)}; \
        throw xsigma::err_type(loc, msg);                                              \
    } while (0)

// ----------------------------------------------------------------------------
// Error reporting macros
#ifdef STRIP_ERROR_MESSAGES
#define XSIGMA_RETHROW(e, ...) throw
#else
#define XSIGMA_RETHROW(e, ...)                         \
    do                                                 \
    {                                                  \
        e.add_context(xsigma::to_string(__VA_ARGS__)); \
        throw e;                                       \
    } while (false)
#endif

//-----------------------------------------------------------------------------
// A utility macro to make it easier to test for error conditions from user
// input.  Like XSIGMA_CHECK, it supports an arbitrary number of extra
// arguments (evaluated only on failure), which will be printed in the error
// message using operator<< (e.g., you can pass any object which has
// operator<< defined.  Most objects in xsigma have these definitions!)
//
// Usage:
//    XSIGMA_CHECK(should_be_true); // A default error message will be provided
//                                 // in this case; but we recommend writing an
//                                 // explicit error message, as it is more
//                                 // user friendly.
//    XSIGMA_CHECK(x == 0, "Expected x to be 0, but got ", x);
//
// On failure, this macro will raise an exception.  If this exception propagates
// to Python, it will convert into a Python RuntimeError.
//
// NOTE: It is SAFE to use this macro in production code; on failure, this
// simply raises an exception, it does NOT unceremoniously quit the process
// (unlike XSIGMA_CHECK() from glog.)
//
#define XSIGMA_CHECK_WITH(error_t, cond, ...) XSIGMA_CHECK_WITH_MSG(error_t, cond, "", __VA_ARGS__)

#ifdef STRIP_ERROR_MESSAGES
#define XSIGMA_CHECK_MSG(cond, type, ...) \
    (#cond #type " XSIGMA_CHECK FAILED at " XSIGMA_STRINGIZE(__FILE__))
#define XSIGMA_CHECK_WITH_MSG(error_t, cond, type, ...)                       \
    if XSIGMA_UNLIKELY (!(cond))                                              \
    {                                                                         \
        XSIGMA_THROW_ERROR(Error, XSIGMA_CHECK_MSG(cond, type, __VA_ARGS__)); \
    }
#else
namespace xsigma
{
namespace details
{
template <typename... Args>
decltype(auto) check_msg_impl(const char* msg, const Args&... args)
{
    return xsigma::to_string(msg, args...);
}
// If there is just 1 user-provided C-string argument, use it.
inline const char* check_msg_impl(const char* /*msg*/, const char* args)
{
    return args;
}
}  // namespace details
}  // namespace xsigma

#define XSIGMA_CHECK_MSG(cond, type, ...) \
    xsigma::details::check_msg_impl(      \
        "Expected " #cond " to be true, but got false.  ", ##__VA_ARGS__)

#define XSIGMA_CHECK_WITH_MSG(error_t, cond, type, ...)                         \
    if XSIGMA_UNLIKELY (!(cond))                                                \
    {                                                                           \
        XSIGMA_THROW_ERROR(error_t, XSIGMA_CHECK_MSG(cond, type, __VA_ARGS__)); \
    }
#endif

namespace xsigma
{
namespace details
{
[[noreturn]] XSIGMA_API void check_fail(
    const char* func, const char* file, int line, const std::string& msg);

[[noreturn]] XSIGMA_API void check_fail(
    const char* func, const char* file, int line, const char* msg);

}  // namespace details
}  // namespace xsigma

//-----------------------------------------------------------------------------
#ifdef STRIP_ERROR_MESSAGES
#define XSIGMA_LOCAL_CHECK(cond, ...)                 \
    if XSIGMA_UNLIKELY (!(cond))                      \
    {                                                 \
        xsigma::details::check_fail(                  \
            __func__,                                 \
            __FILE__,                                 \
            static_cast<int>(__LINE__),               \
            XSIGMA_CHECK_MSG(cond, "", __VA_ARGS__)); \
    }
#else
#define XSIGMA_LOCAL_CHECK(cond, ...)                                                     \
    if XSIGMA_UNLIKELY (!(cond))                                                          \
    {                                                                                     \
        const std::string& msg = XSIGMA_CHECK_MSG(cond, "", ##__VA_ARGS__);               \
        XSIGMA_LOG_IF(WARNING, !(cond), "Check failed: {}", msg);                         \
        xsigma::details::check_fail(__func__, __FILE__, static_cast<int>(__LINE__), msg); \
    }
#endif

#define XSIGMA_CHECK(cond, ...) XSIGMA_LOCAL_CHECK(cond, ##__VA_ARGS__)  //NOLINT

#define XSIGMA_CHECK_ALL_POSITIVE(V)                                   \
    XSIGMA_CHECK(                                                      \
        std::all_of(V.begin(), V.end(), [](auto x) { return x > 0; }), \
        "All elements must be positive");

#define XSIGMA_CHECK_ALL_FINITE(V)                                                               \
    XSIGMA_CHECK(                                                                                \
        std::none_of(V.begin(), V.end(), [](auto x) { return std::isnan(x) || std::isinf(x); }), \
        "All elements must be finite numbers");

#define XSIGMA_CHECK_STRICTLY_INCREASING(V)                                                       \
    XSIGMA_CHECK(                                                                                 \
        std::adjacent_find(V.begin(), V.end(), [](auto a, auto b) { return a >= b; }) == V.end(), \
        "Elements must be in strictly increasing order");

#define XSIGMA_CHECK_STRICTLY_DECREASING(V)                                                       \
    XSIGMA_CHECK(                                                                                 \
        std::adjacent_find(V.begin(), V.end(), [](auto a, auto b) { return a <= b; }) == V.end(), \
        "Elements must be in strictly decreasing order");

#define XSIGMA_CHECK_STRICTLY_ORDERED(V)                                                   \
    XSIGMA_CHECK(                                                                          \
        ((std::adjacent_find(V.begin(), V.end(), [](auto a, auto b) { return a >= b; }) == \
          V.end()) ||                                                                      \
         (std::adjacent_find(V.begin(), V.end(), [](auto a, auto b) { return a <= b; }) == \
          V.end())),                                                                       \
        "Elements must be strictly ordered (increasing or decreasing)");

//-----------------------------------------------------------------------------
// An utility macro that does what `XSIGMA_CHECK` does if compiled in the host code,
// otherwise does nothing. Supposed to be used in the code shared between host and
// device code as an alternative for `XSIGMA_CHECK`.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define XSIGMA_CHECK_IF_NOT_ON_CUDA(cond, ...)
#else
#define XSIGMA_CHECK_IF_NOT_ON_CUDA(cond, ...) XSIGMA_CHECK(cond, ##__VA_ARGS__)
#endif

//-----------------------------------------------------------------------------
#define XSIGMA_CHECK_INDEX(cond, ...) XSIGMA_CHECK_WITH_MSG(IndexError, cond, "INDEX", __VA_ARGS__)

//-----------------------------------------------------------------------------
#define XSIGMA_CHECK_VALUE(cond, ...) XSIGMA_CHECK_WITH_MSG(ValueError, cond, "VALUE", __VA_ARGS__)

//-----------------------------------------------------------------------------
#define XSIGMA_CHECK_TYPE(cond, ...) XSIGMA_CHECK_WITH_MSG(TypeError, cond, "TYPE", __VA_ARGS__)

//-----------------------------------------------------------------------------
// Like XSIGMA_CHECK, but raises NotImplementedErrors instead of Errors.
#define XSIGMA_NOT_IMPLEMENTED(...) \
    XSIGMA_CHECK_WITH_MSG(NotImplementedError, false, "TYPE", __VA_ARGS__)

//-----------------------------------------------------------------------------
// Report a warning to the user.  Accepts an arbitrary number of extra
// arguments which are concatenated into the warning message using operator<<
#ifdef STRIP_ERROR_MESSAGES
#define XSIGMA_WARN(...) \
    xsigma::Warning::warn({__func__, __FILE__, static_cast<int>(__LINE__)}, {}, false)
#else
#define XSIGMA_WARN(...)   \
    xsigma::Warning::warn( \
        {__func__, __FILE__, static_cast<int>(__LINE__)}, xsigma::to_string(__VA_ARGS__), false)
#endif

//-----------------------------------------------------------------------------
// Report a warning to the user only once.  Accepts an arbitrary number of extra
// arguments which are concatenated into the warning message using operator<<
#ifdef STRIP_ERROR_MESSAGES
#define _XSIGMA_WARN_ONCE(...)                                                              \
    XSIGMA_UNUSED static const auto XSIGMA_ANONYMOUS_VARIABLE(xsigma_warn_once_) = [&]      \
    {                                                                                       \
        xsigma::Warning::warn({__func__, __FILE__, static_cast<int>(__LINE__)}, {}, false); \
        return true;                                                                        \
    }()
#else
#define _XSIGMA_WARN_ONCE(...)                                                         \
    XSIGMA_UNUSED static const auto XSIGMA_ANONYMOUS_VARIABLE(xsigma_warn_once_) = [&] \
    {                                                                                  \
        xsigma::Warning::warn(                                                         \
            {__func__, __FILE__, static_cast<int>(__LINE__)},                          \
            xsigma::to_string(__VA_ARGS__),                                            \
            false);                                                                    \
        return true;                                                                   \
    }()
#endif

//-----------------------------------------------------------------------------
#define XSIGMA_WARN_ONCE(...)              \
    if (xsigma::Warning::get_warnAlways()) \
    {                                      \
        XSIGMA_WARN(__VA_ARGS__);          \
    }                                      \
    else                                   \
    {                                      \
        _XSIGMA_WARN_ONCE(__VA_ARGS__);    \
    }

//-----------------------------------------------------------------------------
#ifdef NDEBUG
#define XSIGMA_CHECK_DEBUG(condition, ...)
#else
#define XSIGMA_CHECK_DEBUG(condition, ...)      \
    do                                          \
    {                                           \
        XSIGMA_CHECK(condition, ##__VA_ARGS__); \
    } while (0)
#endif

#define XSIGMA_CHECK_POSITIVE_DEBUG(x) \
    XSIGMA_CHECK_DEBUG(x >= 0, "element: ", x, " must be positive");

#define XSIGMA_CHECK_FINITE_DEBUG(x) \
    XSIGMA_CHECK_DEBUG(!std::isnan(x) && !std::isinf(x), "element: ", x, " must be finite");

#define XSIGMA_CHECK_ALL_POSITIVE_DEBUG(V)                              \
    XSIGMA_CHECK_DEBUG(                                                 \
        std::all_of(V.begin(), V.end(), [](auto x) { return x >= 0; }), \
        "All elements must be positive");

#define XSIGMA_CHECK_ALL_FINITE_DEBUG(V)                                                         \
    XSIGMA_CHECK_DEBUG(                                                                          \
        std::none_of(V.begin(), V.end(), [](auto x) { return std::isnan(x) || std::isinf(x); }), \
        "All elements must be finite numbers");

#define XSIGMA_CHECK_STRICTLY_INCREASING_DEBUG(V)                                                 \
    XSIGMA_CHECK_DEBUG(                                                                           \
        std::adjacent_find(V.begin(), V.end(), [](auto a, auto b) { return a >= b; }) == V.end(), \
        "Elements must be in strictly increasing order");

//-----------------------------------------------------------------------------
#define XSIGMA_THROW(...) XSIGMA_THROW_ERROR(Error, xsigma::to_string(__VA_ARGS__))

//-----------------------------------------------------------------------------
#ifdef NDEBUG
#define XSIGMA_DEBUG_WITH_OBJECT(self, x)
#else
#define XSIGMA_DEBUG_WITH_OBJECT(self, x)                         \
    do                                                            \
    {                                                             \
        std::ostringstream xsigmamsg;                             \
        if (self)                                                 \
        {                                                         \
            xsigmamsg << "fix me" /*self*/ << "): " << std::endl; \
        }                                                         \
        else                                                      \
        {                                                         \
            xsigmamsg << "(nullptr): " << std::endl;              \
        }                                                         \
        xsigmamsg << "" x;                                        \
    } while (false)
#endif
#endif  // XSIGMA_UTIL_EXCEPTION_H_

//-----------------------------------------------------------------------------
// CUDA: various checks for different function calls.
#define CUDA_ENFORCE(condition, ...)   \
    do                                 \
    {                                  \
        cudaError_t error = condition; \
        XSIGMA_CHECK(                  \
            error == cudaSuccess,      \
            "Error at: ",              \
            __FILE__,                  \
            ":",                       \
            __LINE__,                  \
            ": ",                      \
            cudaGetErrorString(error), \
            ##__VA_ARGS__);            \
    } while (0)

//-----------------------------------------------------------------------------
#define CUDA_CHECK(condition)                                          \
    do                                                                 \
    {                                                                  \
        cudaError_t error = condition;                                 \
        XSIGMA_CHECK(error == cudaSuccess, cudaGetErrorString(error)); \
    } while (0)

//-----------------------------------------------------------------------------
#define CUDA_DRIVERAPI_ENFORCE(condition)                                   \
    do                                                                      \
    {                                                                       \
        CUresult result = condition;                                        \
        if (result != CUDA_SUCCESS)                                         \
        {                                                                   \
            const char* msg;                                                \
            cuGetErrorName(result, &msg);                                   \
            XSIGMA_THROW("Error at: ", __FILE__, ":", __LINE__, ": ", msg); \
        }                                                                   \
    } while (0)

//-----------------------------------------------------------------------------
#define CUDA_DRIVERAPI_CHECK(condition)                                                     \
    do                                                                                      \
    {                                                                                       \
        CUresult result = condition;                                                        \
        if (result != CUDA_SUCCESS)                                                         \
        {                                                                                   \
            const char* msg;                                                                \
            cuGetErrorName(result, &msg);                                                   \
            XSIGMA_WARNING_LOG("Error at: " << __FILE__ << ":" << __LINE__ << ": " << msg); \
        }                                                                                   \
    } while (0)

//-----------------------------------------------------------------------------
#define CUBLAS_ENFORCE(condition)            \
    do                                       \
    {                                        \
        cublasStatus_t status = condition;   \
        XSIGMA_CHECK(                        \
            status == CUBLAS_STATUS_SUCCESS, \
            "Error at: ",                    \
            __FILE__,                        \
            ":",                             \
            __LINE__,                        \
            ": ",                            \
            cublasGetErrorString(status));   \
    } while (0)

//-----------------------------------------------------------------------------
#define CUBLAS_CHECK(condition)                                                      \
    do                                                                               \
    {                                                                                \
        cublasStatus_t status = condition;                                           \
        XSIGMA_CHECK(status == CUBLAS_STATUS_SUCCESS, cublasGetErrorString(status)); \
    } while (0)

//-----------------------------------------------------------------------------
#define CURAND_ENFORCE(condition)            \
    do                                       \
    {                                        \
        curandStatus_t status = condition;   \
        XSIGMA_CHECK(                        \
            status == CURAND_STATUS_SUCCESS, \
            "Error at: ",                    \
            __FILE__,                        \
            ":",                             \
            __LINE__,                        \
            ": ",                            \
            curandGetErrorString(status));   \
    } while (0)

//-----------------------------------------------------------------------------
// For CUDA Runtime API
#define XSIGMA_CUDA_CHECK(EXPR)                                             \
    do                                                                      \
    {                                                                       \
        cudaError_t __err = EXPR;                                           \
        if (__err != cudaSuccess)                                           \
        {                                                                   \
            cudaGetLastError();                                             \
            XSIGMA_CHECK(false, "CUDA error: ", cudaGetErrorString(__err)); \
        }                                                                   \
    } while (0)

//-----------------------------------------------------------------------------
#define XSIGMA_CUDA_CHECK_WARN(EXPR)                                  \
    do                                                                \
    {                                                                 \
        cudaError_t __err = EXPR;                                     \
        if (__err != cudaSuccess)                                     \
        {                                                             \
            cudaGetLastError();                                       \
            XSIGMA_WARN("CUDA warning: ", cudaGetErrorString(__err)); \
        }                                                             \
    } while (0)

//-----------------------------------------------------------------------------
// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define XSIGMA_CUDA_KERNEL_LAUNCH_CHECK() XSIGMA_CUDA_CHECK(cudaGetLastError())

//#define DCHECK_EQ(a, b) XSIGMA_CHECK_DEBUG((a) == (b))
//#define DCHECK_NE(a, b) XSIGMA_CHECK_DEBUG((a) != (b))
//#define DCHECK_GE(a, b) XSIGMA_CHECK_DEBUG((a) >= (b))
//#define DCHECK_LT(a, b) XSIGMA_CHECK_DEBUG((a) < (b))
//
//#define CHECK_EQ(a, b) XSIGMA_CHECK_DEBUG((a) == (b))
//#define CHECK_NE(a, b) XSIGMA_CHECK_DEBUG((a) != (b))
//#define CHECK_GE(a, b) XSIGMA_CHECK_DEBUG((a) >= (b))
//#define CHECK_LT(a, b) XSIGMA_CHECK_DEBUG((a) < (b))
//#define CHECK_LE(a, b) XSIGMA_CHECK_DEBUG((a) <= (b))
