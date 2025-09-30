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

#include <iostream>
#include <string>
#include <utility>

#include "logging/logger.h"
#include "util/exception.h"
#include "xsigmaTest.h"

namespace
{
XSIGMA_UNUSED void log_handler(void* user_data, const xsigma::logger::Message& message)
{
    auto* lines = reinterpret_cast<std::string*>(user_data);
    (*lines) += "\n";
    (*lines) += message.message;
}
}  // namespace

XSIGMATEST(Core, Logger)
{
    int    arg     = 0;
    char** arg_str = nullptr;

    xsigma::logger::Init(arg, arg_str);
    xsigma::logger::Init();

    xsigma::logger::ConvertToVerbosity(-100);
    xsigma::logger::ConvertToVerbosity(+100);
    xsigma::logger::ConvertToVerbosity("OFF");
    xsigma::logger::ConvertToVerbosity("ERROR");
    xsigma::logger::ConvertToVerbosity("WARNING");
    xsigma::logger::ConvertToVerbosity("INFO");
    xsigma::logger::ConvertToVerbosity("MAX");
    xsigma::logger::ConvertToVerbosity("NAN");

    XSIGMA_UNUSED auto v1 = xsigma::logger::ConvertToVerbosity(1);
    XSIGMA_UNUSED auto v2 = xsigma::logger::ConvertToVerbosity("TRACE");

    std::string lines;
    XSIGMA_LOGF(
        INFO,
        "changing verbosity to %d",
        static_cast<int>(xsigma::logger_verbosity_enum::VERBOSITY_TRACE));

    xsigma::logger::AddCallback(
        "sonnet-grabber", log_handler, &lines, xsigma::logger_verbosity_enum::VERBOSITY_INFO);

    xsigma::logger::SetStderrVerbosity(xsigma::logger_verbosity_enum::VERBOSITY_TRACE);

    XSIGMA_LOG_SCOPE_FUNCTION(TRACE);
    {
        XSIGMA_LOG_SCOPEF(TRACE, "Sonnet 18");
        const auto* whom = "thee";
        XSIGMA_LOG(TRACE, "Shall I compare " << whom << " to a summer's day?");

        const auto* what0 = "lovely";
        const auto* what1 = "temperate";
        XSIGMA_LOGF(TRACE, "Thou art more %s and more %s:", what0, what1);

        const auto* month = "May";
        XSIGMA_LOG_IF(TRACE, true, << "Rough winds do shake the darling buds of " << month << ",");
        XSIGMA_LOG_IFF(TRACE, true, "And %s’s lease hath all too short a date;", "summers");
    }

    std::cerr << "--------------------------------------------" << std::endl
              << lines << std::endl
              << std::endl
              << "--------------------------------------------" << std::endl;

    XSIGMA_WARN("testing generic warning -- should only show up in the log");

    // remove callback since the user-data becomes invalid out of this function.
    xsigma::logger::RemoveCallback("sonnet-grabber");

    // test out explicit scope start and end markers.
    {
        XSIGMA_LOG_START_SCOPE(INFO, "scope-0");
    }
    XSIGMA_LOG_START_SCOPEF(INFO, "scope-1", "scope %d", 1);
    XSIGMA_LOG_INFO("some text");
    XSIGMA_LOG_END_SCOPE("scope-1");
    {
        XSIGMA_LOG_END_SCOPE("scope-0");
    }

    xsigma::logger::SetInternalVerbosityLevel(v2);

    xsigma::logger::SetThreadName("ttq::worker");
    XSIGMA_UNUSED auto th = xsigma::logger::GetThreadName();

    xsigma::logger::LogScopeRAII obj;

    xsigma::logger::LogScopeRAII obj1 = std::move(obj);

    END_TEST();
}
