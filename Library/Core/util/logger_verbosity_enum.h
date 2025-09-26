#pragma once



namespace xsigma
{
enum class logger_verbosity_enum : int
{
    // Used to mark an invalid verbosity. Do not log to this level.
    VERBOSITY_INVALID = -10,  // Never do LOG_F(INVALID)

    // You may use VERBOSITY_OFF on g_stderr_verbosity, but for nothing else!
    VERBOSITY_OFF = -9,  // Never do LOG_F(OFF)

    VERBOSITY_ERROR   = -2,
    VERBOSITY_WARNING = -1,

    // Normal messages. By default written to stderr.
    VERBOSITY_INFO = 0,

    // Same as VERBOSITY_INFO in every way.
    VERBOSITY_0 = 0,

    // logger_verbosity_enum levels 1-9 are generally not written to stderr, but are written to file.
    VERBOSITY_1 = +1,
    VERBOSITY_2 = +2,
    VERBOSITY_3 = +3,
    VERBOSITY_4 = +4,
    VERBOSITY_5 = +5,
    VERBOSITY_6 = +6,
    VERBOSITY_7 = +7,
    VERBOSITY_8 = +8,
    VERBOSITY_9 = +9,

    // trace level, same as VERBOSITY_9
    VERBOSITY_TRACE = +9,

    // Don not use higher verbosity levels, as that will make grepping log files harder.
    VERBOSITY_MAX = +9,
};
}