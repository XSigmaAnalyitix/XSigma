#include "util/exception.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <utility>

#include "logging/logger.h"
#include "util/back_trace.h"
#include "util/string_util.h"

namespace xsigma
{
namespace
{
std::function<std::string(void)>* GetFetchStackTrace()  // NOLINT
{
    static std::function<std::string(void)> func = []()
    { return xsigma::back_trace::print(/*frames_to_skip=*/0, /*maximum_number_of_frames=*/32); };
    return &func;
};
}  // namespace

// Explicitly define the virtual destructor to ensure the vtable is generated
Error::~Error() = default;

//-----------------------------------------------------------------------------
Error::Error(std::string msg, std::string backtrace, const void* caller)
    : msg_(std::move(msg)), backtrace_(std::move(backtrace)), caller_(caller)
{
    refresh_what();
}

//-----------------------------------------------------------------------------
Error::Error(SourceLocation source_location, std::string msg)
    : Error(
          std::move(msg),
          xsigma::to_string(
              "Exception raised from ",
              source_location,
              " (most recent call first):\n",
              (*GetFetchStackTrace())()))
{
}

//-----------------------------------------------------------------------------
const char* Error::what() const noexcept
{
    return what_
        .ensure(
            [this]
            {
                try
                {
                    return compute_what(/*include_backtrace*/ true);
                }
                catch (...)
                {
                    // what() is noexcept, we need to return something here.
                    return std::string{"<Error computing Error::what()>"};
                }
            })
        .c_str();
}

//-----------------------------------------------------------------------------
std::string Error::compute_what(bool include_backtrace) const
{
    std::ostringstream oss;

    oss << msg();

    for (const auto& c : context())
    {
        oss << "\n  " << c;
    }

    if (include_backtrace)
    {
        oss << "\n" << backtrace();
    }

    return oss.str();
}

//-----------------------------------------------------------------------------
void Error::refresh_what()
{
    what_.reset();
    what_without_backtrace_ = compute_what(/*include_backtrace*/ true);
    XSIGMA_LOG_WARNING("Error message: " << what());
}

//-----------------------------------------------------------------------------
void Error::add_context(std::string msg)
{
    context_.push_back(std::move(msg));
    // TODO: Calling add_context O(n) times has O(n^2) cost.  We can fix
    // this perf problem by populating the fields lazily... if this ever
    // actually is a problem.
    // NB: If you do fix this, make sure you do it in a thread safe way!
    // what() is almost certainly expected to be thread safe even when
    // accessed across multiple threads
    refresh_what();
}

//-----------------------------------------------------------------------------
namespace details
{
void check_fail(const char* func, const char* file, int line, const std::string& msg)
{
    throw xsigma::Error({func, file, line}, msg);
}

//-----------------------------------------------------------------------------
void check_fail(const char* func, const char* file, int line, const char* msg)
{
    throw xsigma::Error({func, file, line}, msg);
}
}  // namespace details

//-----------------------------------------------------------------------------
namespace Warning
{
namespace
{
//-----------------------------------------------------------------------------
WarningHandler* getBaseHandler()
{
    static WarningHandler base_warning_handler_ = WarningHandler();
    return &base_warning_handler_;
};

//-----------------------------------------------------------------------------
class ThreadWarningHandler
{
public:
    ThreadWarningHandler() = delete;

    static WarningHandler* get_handler()
    {
        if (warning_handler_ == nullptr)
        {
            warning_handler_ = getBaseHandler();
        }
        return warning_handler_;
    }

    // static void set_handler(WarningHandler* handler) { warning_handler_ = handler; }

private:
    static thread_local WarningHandler* warning_handler_;
};

thread_local WarningHandler* ThreadWarningHandler::warning_handler_ = nullptr;

}  // namespace

//-----------------------------------------------------------------------------
void warn(SourceLocation source_location, const std::string& msg, const bool verbatim)
{
    ThreadWarningHandler::get_handler()->process(source_location, msg, verbatim);
}

//-----------------------------------------------------------------------------
// void set_warning_handler(WarningHandler* handler) noexcept(true)
//{
//    ThreadWarningHandler::set_handler(handler);
//}
//
// WarningHandler* get_warning_handler() noexcept(true)
//{
//    return ThreadWarningHandler::get_handler();
//}

bool warn_always = false;

void set_warnAlways(bool setting) noexcept(true)
{
    warn_always = setting;
}

bool get_warnAlways() noexcept(true)
{
    return warn_always;
}

}  // namespace Warning

void WarningHandler::process(
    const SourceLocation& source_location, const std::string& msg, bool /*verbatim*/)
{
    xsigma::logger::Log(
        logger_verbosity_enum::VERBOSITY_WARNING,
        source_location.file,
        source_location.line,
        xsigma::to_string("Warning: ", msg.c_str(), " (function ", source_location.function, ")")
            .c_str());
}

// std::string GetExceptionString(const std::exception& e)
//{
//#ifdef __GXX_RTTI
//    return demangle(typeid(e).name()) + ": " + e.what();
//#else
//    return std::string("Exception (no RTTI available): ") + e.what();
//#endif  // __GXX_RTTI
//}

}  // namespace xsigma
