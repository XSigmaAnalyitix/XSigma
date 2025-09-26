// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class logger
 * @brief logging framework for use in XSIGMA and in applications based on XSIGMA
 *
 * logger acts as the entry point to XSIGMA's logging framework. The
 * implementation uses the loguru (https://github.com/emilk/loguru). logger
 * provides some static API to initialize and configure logging together with a
 * collection of macros that can be used to add items to the generated log.
 *
 * The logging framework is based on verbosity levels. Level 0-9 are supported
 * in addition to named levels such as ERROR, WARNING, and INFO. When a log for
 * a particular verbosity level is being generated, all log additions issued
 * with verbosity level less than or equal to the requested verbosity level will
 * get logged.
 *
 * When using any of the logging macros, it must be noted that unless a log output
 * is requesting that verbosity provided (or higher), the call is a no-op and
 * the message stream or printf-style arguments will not be evaluated.
 *
 * @section Setup Setup
 *
 * To initialize logging, in your application's `main()` you may call
 * `logger::Init(argv, argc)`. This is totally optional but useful to
 * time-stamp the  start of the  log. Furthermore, it can optionally detect
 * verbosity level on the command line as `-v` (or any another string pass as
 * the optional argument to `Init`) that will be used as the verbosity level for
 * logging on to `stderr`. By default, it is set to `0` (or `INFO`) unless
 * changed by calling `logger::SetStderrVerbosity`.
 *
 * In additional to logging to `stderr`, one can accumulate logs to one or more files using
 * `logger::LogToFile`. Each log file can be given its own verbosity level.
 *
 * For multithreaded applications, you may want to name each of the threads so
 * that the generated log can use human readable names for the threads. For
 * that, use `logger::SetThreadName`. Calling `logger::Init` will set the name
 * for the main thread.
 *
 * You can choose to turn on signal handlers for intercepting signals. By default,
 * all signal handlers are disabled. The following is a list of signal handlers
 * and the corresponding static variable that can be used to enable/disable each
 * signal handler.
 *
 * - SIGABRT - `logger::EnableSigabrtHandler`
 * - SIGBUS - `logger::EnableSigbusHandler`
 * - SIGFPE - `logger::EnableSigfpeHandler`
 * - SIGILL - `logger::EnableSigillHandler`
 * - SIGINT - `logger::EnableSigintHandler`
 * - SIGSEGV - `logger::EnableSigsegvHandler`
 * - SIGTERM - `logger::EnableSigtermHandler`
 *
 * To enable any of these signal handlers, set their value to `true` prior to calling
 * `logger::Init(argc, argv)` or `logger::Init()`.
 *
 * When signal handlers are enabled,
 * to prevent the logging framework from intercepting signals from your application,
 * you can set the static variable `logger::EnableUnsafeSignalHandler` to `false`
 * prior to calling `logger::Init(argc, argv)` or `logger::Init()`.
 *
 * @section Logging Logging
 *
 * logger provides several macros (again, based on `loguru`) that can be
 * used to add the log. Both printf-style and stream-style is supported. All
 * printf-style macros are suffixed with `F` to distinguish them from the stream
 * macros. Another pattern in naming macros is the presence of `V`
 * e.g. `xsigmaVLog` vs `XSIGMA_LOG`. A macro with the `V` prefix takes a fully
 * qualified verbosity enum e.g. `logger::VERBOSITY_INFO` or
 * `logger::VERBOSITY_0`, while the non-`V` variant takes the verbosity
 * name e.g. `INFO` or `0`.
 *
 * Following code snippet provides an overview of the available macros and their
 * usage.
 *
 * @code{.cpp}
 *
 *  // Optional, leaving this as the default value `true` will let the logging
 *  // framework log signals such as segmentation faults.
 *
 *  logger::EnableUnsafeSignalHandler = false;
 *
 *  // Optional, but useful to time-stamp the start of the log.
 *  // Will also detect verbosity level on the command line as -v.
 *
 *  logger::Init(argc, argv);
 *
 *  // Put every log message in "everything.log":
 *  logger::LogToFile("everything.log", logger::APPEND, logger::VERBOSITY_MAX);
 *
 *  // Only log INFO, WARNING, ERROR to "latest_readable.log":
 *  logger::LogToFile("latest_readable.log", logger::TRUNCATE, logger::VERBOSITY_INFO);
 *
 *  // Only show most relevant things on stderr:
 *  logger::SetStderrVerbosity(logger::VERBOSITY_1);
 *
 *  // add a line to log using the verbosity name.
 *  xsigmaLogF(INFO, "I'm hungry for some %.3f!", 3.14159);
 *  xsigmaLogF(0, "same deal");
 *
 *  // add a line to log using the verbosity enum.
 *  xsigmaVLogF(logger::VERBOSITY_INFO, "I'm hungry for some %.3f!", 3.14159);
 *  xsigmaVLogF(logger::VERBOSITY_0, "same deal");
 *
 *  // to add an identifier for a xsigmaObjectBase or subclass
 *  xsigmaLogF(INFO, "The object is %s", xsigmaLogIdentifier(xsigmaobject));
 *
 *  // add a line conditionally to log if the condition succeeds:
 *  xsigmaLogIfF(INFO, ptr == nullptr, "ptr is nullptr (some number: %.3f)", *  3.14159);
 *
 *  xsigmaLogScopeF(INFO, "Will indent all log messages within this scope.");
 *  // in a function, you may use xsigmaLogScopeFunction(INFO)
 *
 *  // scope can be explicitly started and closed by xsigmaLogStartScope (or
 *  // xsigmaLogStartScopef) and xsigmaLogEndScope
 *  xsigmaLogStartScope(INFO, "id-used-as-message");
 *  xsigmaLogStartScopeF(INFO, "id", "message-%d", 1);
 *  xsigmaLogEndScope("id");
 *  xsigmaLogEndScope("id-used-as-message");
 *
 *  // alternatively, you can use streams instead of printf-style
 *  XSIGMA_INFO_LOG( "I'm hungry for some " << 3.14159 << "!");
 *  xsigmaLogIF(INFO, ptr == nullptr, "ptr is " << "nullptr");
 *
 * @endcode
 *
 * @section LoggingAndLegacyMacros Logging and XSIGMA error macros
 *
 * XSIGMA has long supported multiple macros to report errors, warnings and verbose
 * messages through `xsigmaErrorMacro`, `xsigmaWarningMacro`, `xsigmaDebugMacro`, etc.
 * In addition to performing the traditional message reporting via
 * `xsigmaOutputWindow`, these macros also log to the logging sub-system with
 * appropriate verbosity levels.
 *
 * To avoid the logger and xsigmaOutputWindow both posting the message to the
 * standard output streams, xsigmaOutputWindow now supports an ability to specify
 * terminal display mode, via `xsigmaOutputWindow::SetDisplayMode`. If display mode
 * is `xsigmaOutputWindow::DEFAULT` then the output window will not
 * post messages originating from the standard error/warning/verbose macros to the
 * standard output if XSIGMA is built with logging support. If XSIGMA is not built
 * with logging support, then xsigmaOutputWindow will post the messages to the
 * standard output streams, unless disabled explicitly.
 *
 * @section Callbacks Custom callbacks/handlers for log messages
 *
 * logger supports ability to register callbacks to call on each logged
 * message. This is useful to show the messages in application specific
 * viewports, e.g. a special message widget.
 *
 * To register a callback use `logger::AddCallback` and to remove a callback
 * use `logger::RemoveCallback` with the id provided when registering the
 * callback.
 *
 */

#pragma once

#include <string>  // needed for std::string


#include "common/macros.h"          // needed for macros
#include "common/wrapping_hints.h"  // for XSIGMA_FILEPATH

#if defined(_MSC_VER)
#include <sal.h>  // Needed for _In_z_ etc annotations
#endif

// this is copied from `loguru.hpp`
#if defined(__clang__) || defined(__GNUC__)
// Helper macro for declaring functions as having similar signature to printf.
// This allows the compiler to catch format errors at compile-time.
#define XSIGMA_PRINTF_LIKE(fmtarg, firstvararg) \
    __attribute__((__format__(__printf__, fmtarg, firstvararg)))
#define XSIGMA_FORMAT_STRING_TYPE const char*
#elif defined(_MSC_VER)
#define XSIGMA_PRINTF_LIKE(fmtarg, firstvararg)
#define XSIGMA_FORMAT_STRING_TYPE _In_z_ _Printf_format_string_ const char*
#else
#define XSIGMA_PRINTF_LIKE(fmtarg, firstvararg)
#define XSIGMA_FORMAT_STRING_TYPE const char*
#endif

#include "util/logger_verbosity_enum.h"

namespace xsigma
{
class XSIGMA_API logger
{
public:
    /**
   * Initializes logging. This should be called from the main thread, if at all.
   * Your application doesn't *need* to call this, but if you do:
   *  * signal handlers are installed
   *  * program arguments are logged
   *  * working directory is logged
   *  * optional -v verbosity flag is parsed
   *  * main thread name is set to "main thread"
   *  * explanation of the preamble (date, threadname, etc.) is logged.
   *
   * This method will look for arguments meant for logging subsystem and remove
   * them. Arguments meant for logging subsystem are:
   *
   * -v n Set stderr logging verbosity. Examples
   *    -v 3        Show verbosity level 3 and lower.
   *    -v 0        Only show INFO, WARNING, ERROR, FATAL (default).
   *    -v INFO     Only show INFO, WARNING, ERROR, FATAL (default).
   *    -v WARNING  Only show WARNING, ERROR, FATAL.
   *    -v ERROR    Only show ERROR, FATAL.
   *    -v FATAL    Only show FATAL.
   *    -v OFF      Turn off logging to stderr.
   *
   * You can set the default logging verbosity programmatically by calling
   * `logger::SetStderrVerbosity` before calling `logger::Init`. That
   * way, you can specify a default that the user can override using command
   * line arguments. Note that this does not affect file logging.
   *
   * You can also use something else instead of '-v' flag by the via
   * `verbosity_flag` argument. You can also set to nullptr to skip parsing
   * verbosity level from the command line arguments.
   *
   * For applications that do not want loguru to handle any signals, i.e.,
   * print a stack trace when a signal is intercepted, the
   * `logger::EnableUnsafeSignalHandler` static member variable
   * should be set to `false`.
   * @{
   */
    static void Init(int& argc, char* argv[], const char* verbosity_flag = "-v");
    static void Init();
    /** @} */

    /**
   * Set the verbosity level for the output logged to stderr. Everything with a
   * verbosity equal or less than the level specified will be written to
   * stderr. Set to `VERBOSITY_OFF` to write nothing to stderr.
   * Default is 0.
   */
    static void SetStderrVerbosity(logger_verbosity_enum level);

    /**
   * Set internal messages verbosity level. The library used by XSIGMA, `loguru`
   * generates log messages during initialization and at exit. These are logged
   * as log level VERBOSITY_1, by default. One can change that using this
   * method. Typically, you want to call this before `logger::Init`.
   */
    static void SetInternalVerbosityLevel(logger_verbosity_enum level);

    /**
   * Support log file modes: `TRUNCATE` truncates the file clearing any existing
   * contents while `APPEND` appends to the existing log file contents, if any.
   */
    enum FileMode
    {
        TRUNCATE,
        APPEND
    };

    /**
   * Enable logging to a file at the given path.
   * Any logging message with verbosity lower or equal to the given verbosity
   * will be included. This method will create all directories in the 'path' if
   * needed. To stop the file logging, call `EndLogToFile` with the same path.
   */
    static void LogToFile(const char* path, FileMode filemode, logger_verbosity_enum verbosity);

    /**
   * Stop logging to a file at the given path.
   */
    static void EndLogToFile(const char* path);

    ///@{
    /**
   * Get/Set the name to identify the current thread in the log output.
   */
    static void        SetThreadName(const std::string& name);
    static std::string GetThreadName();
    ///@}

    /**
   * The message structure that is passed to custom callbacks registered using
   * `logger::AddCallback`.
   */
    struct Message
    {
        // You would generally print a Message by just concatenating the buffers without spacing.
        // Optionally, ignore preamble and indentation.
        logger_verbosity_enum verbosity;    // Already part of preamble
        const char*           filename;     // Already part of preamble
        unsigned              line;         // Already part of preamble
        const char*           preamble;     // Date, time, uptime, thread, file:line, verbosity.
        const char*           indentation;  // Just a bunch of spacing.
        const char*           prefix;       // Assertion failure info goes here (or "").
        const char*           message;      // User message goes here.
    };

    ///@{
    /**
   * Callback handle types.
   */
    using LogHandlerCallbackT   = void (*)(void* user_data, const Message& message);
    using CloseHandlerCallbackT = void (*)(void* user_data);
    using FlushHandlerCallbackT = void (*)(void* user_data);
    ///@}

    /**
   * Add a callback to call on each log message with a  verbosity less or equal
   * to the given one.  Useful for displaying messages in an application output
   * window, for example. The given `on_close` is also expected to flush (if
   * desired).
   *
   * Note that if logging is disabled at compile time, then these callback will
   * never be called.
   */
#if !defined(__WRAP__)

    static void AddCallback(
        const char*           id,
        LogHandlerCallbackT   callback,
        void*                 user_data,
        logger_verbosity_enum verbosity,
        CloseHandlerCallbackT on_close = nullptr,
        FlushHandlerCallbackT on_flush = nullptr);

#endif  // #if !defined(__WRAP__)

    /**
   * Remove a callback using the id specified.
   * Returns true if and only if the callback was found (and removed).
   */
    static bool RemoveCallback(const char* id);

    /**
   * Returns true if XSIGMA is built with logging support enabled.
   */
    static bool IsEnabled();

    /**
   * Returns the maximum verbosity of all log outputs. A log item for a
   * verbosity higher than this will not be generated in any of the currently
   * active outputs.
   */
    static logger_verbosity_enum GetCurrentVerbosityCutoff();

    /**
   * Convenience function to convert an integer to matching verbosity level. If
   * val is less than or equal to logger::VERBOSITY_INVALID, then
   * logger::VERBOSITY_INVALID is returned. If value is greater than
   * logger::VERBOSITY_MAX, then logger::VERBOSITY_MAX is returned.
   */
    static logger_verbosity_enum ConvertToVerbosity(int value);

    /**
   * Convenience function to convert a string to matching verbosity level.
   * logger::VERBOSITY_INVALID will be return for invalid strings.
   * Accepted string values are OFF, ERROR, WARNING, INFO, TRACE, MAX, INVALID or ASCII
   * representation for an integer in the range [-9,9].
   */
    static logger_verbosity_enum ConvertToVerbosity(const char* text);

    ///@{
    /**
   * @internal
   *
   * Not intended for public use, please use the logging macros instead.
   */
    static void Log(
        logger_verbosity_enum       verbosity,
        XSIGMA_FILEPATH const char* fname,
        unsigned int                lineno,
        const char*                 txt);
    static void StartScope(
        logger_verbosity_enum       verbosity,
        const char*                 id,
        XSIGMA_FILEPATH const char* fname,
        unsigned int                lineno);
    static void EndScope(const char* id);
#if !defined(__WRAP__)
    static void LogF(
        logger_verbosity_enum       verbosity,
        XSIGMA_FILEPATH const char* fname,
        unsigned int                lineno,
        XSIGMA_FORMAT_STRING_TYPE   format,
        ...) XSIGMA_PRINTF_LIKE(4, 5);
    static void StartScopeF(
        logger_verbosity_enum       verbosity,
        const char*                 id,
        XSIGMA_FILEPATH const char* fname,
        unsigned int                lineno,
        XSIGMA_FORMAT_STRING_TYPE   format,
        ...) XSIGMA_PRINTF_LIKE(5, 6);

    class XSIGMA_API LogScopeRAII
    {
    public:
        LogScopeRAII();
        LogScopeRAII(
            logger_verbosity_enum     verbosity,
            const char*               fname,
            unsigned int              lineno,
            XSIGMA_FORMAT_STRING_TYPE format,
            ...) XSIGMA_PRINTF_LIKE(5, 6);
        ~LogScopeRAII();
#if defined(_MSC_VER) && _MSC_VER > 1800
        // see loguru.hpp for the reason why this is needed on MSVC
        LogScopeRAII(LogScopeRAII&& other) : Internals(other.Internals)
        {
            other.Internals = nullptr;
        }
#else
        LogScopeRAII(LogScopeRAII&&) = default;
#endif

    private:
        LogScopeRAII(const LogScopeRAII&)   = delete;
        void operator=(const LogScopeRAII&) = delete;
        class LSInternals;
        LSInternals* Internals = nullptr;
    };
#endif
    ///@}

    /**
   * Flag to enable/disable the logging frameworks printing of a stack trace
   * when catching signals, which could lead to crashes and deadlocks in
   * certain circumstances.
   */
    static bool EnableUnsafeSignalHandler;
    static bool EnableSigabrtHandler;
    static bool EnableSigbusHandler;
    static bool EnableSigfpeHandler;
    static bool EnableSigillHandler;
    static bool EnableSigintHandler;
    static bool EnableSigsegvHandler;
    static bool EnableSigtermHandler;

    XSIGMA_DELETE_COPY_AND_MOVE(logger)

protected:
    logger();
    ~logger();

private:
    static logger_verbosity_enum InternalVerbosityLevel;
};
}  // namespace xsigma
///@{
/**
 * Add to log given the verbosity level.
 * The text will be logged when the log verbosity is set to the specified level
 * or higher.
 *
 *     // using printf-style
 *     xsigmaLogF(INFO, "Hello %s", "world!");
 *     xsigmaVLogF( logger_verbosity_enum::VERBOSITY_INFO, "Hello %s", "world!");
 *
 *     // using streams
 *     XSIGMA_INFO_LOG( "Hello " << "world!");
 *     xsigmaVLog( logger_verbosity_enum::VERBOSITY_INFO, << "Hello world!");
 *
 */
#define XSIGMA_VLOGF(level, ...)                            \
    ((level) > xsigma::logger::GetCurrentVerbosityCutoff()) \
        ? (void)0                                           \
        : xsigma::logger::LogF(level, __FILE__, __LINE__, __VA_ARGS__)

#define XSIGMA_LOGF(verbosity_name, ...) \
    XSIGMA_VLOGF(xsigma::logger_verbosity_enum::VERBOSITY_##verbosity_name, __VA_ARGS__)

#define XSIGMA_VLOG(level, x)                                                    \
    if ((level) <= xsigma::logger::GetCurrentVerbosityCutoff())                  \
    {                                                                            \
        std::ostringstream xsigmamsg;                                            \
        xsigmamsg << " " << x;                                                   \
        xsigma::logger::Log(level, __FILE__, __LINE__, xsigmamsg.str().c_str()); \
    }
#define XSIGMA_LOG(verbosity_name, x) \
    XSIGMA_VLOG(xsigma::logger_verbosity_enum::VERBOSITY_##verbosity_name, x)

#ifndef NDEBUG
#define XSIGMA_LOG_DEBUG(verbosity_name, x) \
    XSIGMA_VLOG(xsigma::logger_verbosity_enum::VERBOSITY_##verbosity_name, x)
#else
#define XSIGMA_LOG_DEBUG(verbosity_name, x)
#endif
///@}

///@{
/**
 * Add to log only when the `cond` passes.
 *
 *     // using printf-style
 *     xsigmaLogIfF(ERROR, ptr == nullptr, "`ptr` cannot be null!");
 *     xsigmaVLogIfF( logger_verbosity_enum::VERBOSITY_ERROR, ptr == nullptr, "`ptr` cannot be null!");
 *
 *     // using streams
 *     xsigmaLogIf(ERROR, ptr == nullptr, "`ptr` cannot be null!");
 *     xsigmaVLogIf( logger_verbosity_enum::VERBOSITY_ERROR, ptr == nullptr, << "`ptr` cannot be null!");
 *
 */
#define XSIGMA_VLOG_IFF(level, cond, ...)                                      \
    ((level) > xsigma::logger::GetCurrentVerbosityCutoff() || (cond) == false) \
        ? (void)0                                                              \
        : xsigma::logger::LogF(level, __FILE__, __LINE__, __VA_ARGS__)

#define XSIGMA_LOG_IFF(verbosity_name, cond, ...) \
    XSIGMA_VLOG_IFF(xsigma::logger_verbosity_enum::VERBOSITY_##verbosity_name, cond, __VA_ARGS__)

#define XSIGMA_VLOG_IF(level, cond, x)                                           \
    if ((level) <= xsigma::logger::GetCurrentVerbosityCutoff() && (cond))        \
    {                                                                            \
        std::ostringstream xsigmamsg;                                            \
        xsigmamsg << "" x;                                                       \
        xsigma::logger::Log(level, __FILE__, __LINE__, xsigmamsg.str().c_str()); \
    }

#define XSIGMA_LOG_IF(verbosity_name, cond, x) \
    XSIGMA_VLOG_IF(xsigma::logger_verbosity_enum::VERBOSITY_##verbosity_name, cond, x)
///@}

#define XSIGMALOG_CONCAT_IMPL(s1, s2) s1##s2
#define XSIGMALOG_CONCAT(s1, s2) XSIGMALOG_CONCAT_IMPL(s1, s2)
#define XSIGMALOG_ANONYMOUS_VARIABLE(x) XSIGMALOG_CONCAT(x, __LINE__)

#define XSIGMA_VLOG_SCOPEF(level, ...)                          \
    auto XSIGMALOG_ANONYMOUS_VARIABLE(msg_context) =            \
        ((level) > xsigma::logger::GetCurrentVerbosityCutoff()) \
            ? xsigma::logger::LogScopeRAII()                    \
            : xsigma::logger::LogScopeRAII(level, __FILE__, __LINE__, __VA_ARGS__)

#define XSIGMA_LOG_SCOPEF(verbosity_name, ...) \
    XSIGMA_VLOG_SCOPEF(xsigma::logger_verbosity_enum::VERBOSITY_##verbosity_name, __VA_ARGS__)

#define XSIGMA_LOG_SCOPE_FUNCTION(verbosity_name) XSIGMA_LOG_SCOPEF(verbosity_name, "%s", __func__)
#define XSIGMA_VLOG_SCOPE_FUNCTION(level) XSIGMA_VLOG_SCOPEF(level, "%s", __func__)

///@{
/**
 * Explicitly mark start and end of log scope. This is useful in cases where the
 * start and end of the scope does not happen within the same C++ scope.
 */
#define XSIGMA_LOG_START_SCOPE(verbosity_name, id) \
    xsigma::logger::StartScope(                    \
        xsigma::logger_verbosity_enum::VERBOSITY_##verbosity_name, id, __FILE__, __LINE__)

#define XSIGMA_LOG_END_SCOPE(id) xsigma::logger::EndScope(id)

#define XSIGMA_LOG_START_SCOPEF(verbosity_name, id, ...)           \
    xsigma::logger::StartScopeF(                                   \
        xsigma::logger_verbosity_enum::VERBOSITY_##verbosity_name, \
        id,                                                        \
        __FILE__,                                                  \
        __LINE__,                                                  \
        __VA_ARGS__)

#define XSIGMA_VLOG_START_SCOPE(level, id) xsigma::logger::StartScope(level, id, __FILE__, __LINE__)

#define XSIGMA_VLOG_START_SCOPEF(level, id, ...) \
    xsigma::logger::StartScopeF(level, id, __FILE__, __LINE__, __VA_ARGS__)
///@}

/**
 * Convenience macro to generate an identifier string for any base_object subclass.
 * @note do not store the returned value as it returns a char* pointer to a
 * temporary std::string that will be released as soon as it goes out of scope.
 */
#define XSIGMA_LOG_IDENTIFIER(xsigmaobject) xsigma::logger::GetIdentifier(xsigmaobject).c_str()

#define XSIGMA_LOG_INFO(...) XSIGMA_LOG(INFO, __VA_ARGS__)

#ifndef NDEBUG
#define XSIGMA_LOG_INFO_DEBUG(...) XSIGMA_LOG_INFO(__VA_ARGS__)
#else
#define XSIGMA_LOG_INFO_DEBUG(...)
#endif  // !NDEBUG

#define XSIGMA_LOG_WARNING(...) XSIGMA_LOG(WARNING, __VA_ARGS__)

#define XSIGMA_LOG_ERROR(...) XSIGMA_LOG(ERROR, __VA_ARGS__)

/**
 * Convenience macros to start and end logging to a file. provide a file name
 * with the full path and extension.
 */
#define START_LOG_TO_FILE(file_name)                                                        \
    if (!file_name.empty())                                                                 \
    {                                                                                       \
        xsigma::logger::SetStderrVerbosity(xsigma::logger_verbosity_enum::VERBOSITY_TRACE); \
        xsigma::logger::LogToFile(                                                          \
            file_name.c_str(),                                                              \
            xsigma::logger::FileMode::TRUNCATE,                                             \
            xsigma::logger_verbosity_enum::VERBOSITY_TRACE);                                \
    }

#define END_LOG_TO_FILE(file_name)                       \
    if (!file_name.empty())                              \
    {                                                    \
        xsigma::logger::EndLogToFile(file_name.c_str()); \
    }

/**
 * Convenience macros to start and end logging to a file. provide a file name
 * without the extension. The extension `.log` will be added automatically.
 */
#define LOG_TO_FILE_NAME(file_name) std::string(std::string(#file_name) + ".log")
#define START_LOG_TO_FILE_NAME(file_name) START_LOG_TO_FILE(LOG_TO_FILE_NAME(file_name))
#define END_LOG_TO_FILE_NAME(file_name) END_LOG_TO_FILE(LOG_TO_FILE_NAME(file_name))
