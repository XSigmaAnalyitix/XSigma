#pragma once

namespace xsigma
{
enum class logger_verbosity_enum : int
{
    VERBOSITY_INVALID = -5,  // Never do LOG_F(INVALID)
    VERBOSITY_OFF     = -4,  // Never do LOG_F(OFF)
    VERBOSITY_FATAL   = -3,
    VERBOSITY_ERROR   = -2,
    VERBOSITY_WARNING = -1,
    VERBOSITY_INFO = 0,
    VERBOSITY_TRACE = 1,
    VERBOSITY_MAX = 1,
};
}