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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <gtest/gtest.h>  // for Test, TestInfo

#include <cstdio>    // for unlink
#include <iostream>  // for char_traits, basic_ostream, operator<<, endl, cerr, ostream, ostringstream

#include "util/timer_log.h"  // for timer_log
#include "xsigmaTest.h"      // for END_TEST, XSIGMATEST

#if defined(__CYGWIN__)
#include <sys/unistd.h>
#elif defined(_WIN32)
#include <io.h>
#endif

#ifdef _WIN32
#include <windows.h>  // for Sleep
#endif

using namespace xsigma;
int  max_iter = 10;
void TestTimerLogTest(std::ostream& strm)
{
    // actual test
    float a = 1.0;
    int   i;
    int   j;
    strm << "Test timer_log Start" << std::endl;

    timer_log::MarkEvent("bolb");
    auto* timer1 = new timer_log();
    timer_log::SetMaxEntries(timer_log::GetMaxEntries());
    timer_log::SetMaxEntries(8);
    timer_log::SetMaxEntries(18);

    timer_log::MarkEvent("bolb");

    timer1->start();
    for (j = 0; j < 4; j++)
    {
        timer_log::FormatAndMarkEvent("%s%d", "start", j);
        for (i = 0; i < max_iter; i++)
        {
            a *= a;
        }
#ifdef _WIN32
        Sleep(max_iter);
#endif
        timer_log::InsertTimedEvent("Timed Event", 1. / max_iter, 0);
        timer_log::FormatAndMarkEvent("%s%d", "end", j);
    }
    timer1->stop();
    strm << "GetElapsedTime: " << timer1->GetElapsedTime() << std::endl;
    strm << "GetCPUTime: " << timer_log::GetCPUTime() << std::endl;

    timer_log::DumpLog("timing");

    timer_log::DumpLogWithIndents(&std::cerr, 0);
    // std::ostream* os;
    // timer_log::dump_log_with_indents_and_percentages(&std::cerr);

    unlink("timing");

    std::cerr << "============== timer separator ================\n";

    timer_log::ResetLog();
    timer_log::SetMaxEntries(5);

    for (j = 0; j < 4; j++)
    {
        timer_log::MarkStartEvent("Other");
        for (i = 0; i < max_iter; i++)
        {
            a *= a;
        }
#ifdef _WIN32
        Sleep(max_iter);
#endif
        timer_log::InsertTimedEvent("Other Timed Event", 1. / max_iter, 0);
        timer_log::MarkEndEvent("Other");
    }
    timer1->stop();
    strm << "GetElapsedTime: " << timer1->GetElapsedTime() << std::endl;
    strm << "GetCPUTime: " << timer_log::GetCPUTime() << std::endl;
    timer_log::DumpLog("timing2");
    timer_log::DumpLogWithIndents(&std::cerr, 0);
    // timer_log::dump_log_with_indents_and_percentages(&std::cerr);

    unlink("timing2");

    timer_log::SetMaxEntries(50);

    delete timer1;  // timer1->Delete();
    strm << "Test timer_log End" << std::endl;
}

XSIGMATEST(Core, TimerLog)
{
    timer_log::InsertTimedEvent("Other Timed Event", 1., 0);

    timer_log::CleanupLog();

    timer_log::MarkEvent("bolb");

    std::ostringstream xsigmamsg_with_warning_C4701;

    timer_log::LoggingOff();
    TestTimerLogTest(xsigmamsg_with_warning_C4701);

    timer_log::ResetLog();
    timer_log::CleanupLog();

    timer_log::MarkEvent("next");
    timer_log::MarkEvent("bolb");
    timer_log::MarkEvent("biko");
    timer_log::LoggingOn();
    TestTimerLogTest(xsigmamsg_with_warning_C4701);
    END_TEST();
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif
