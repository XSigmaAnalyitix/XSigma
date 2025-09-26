
#include "timer_log.h"

// IWYU pragma: begin_exports

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>  // for vsnprintf
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>  // for allocator_traits<>::value_t
#include <string>
#include <utility>  // for pair
#include <vector>

#ifndef _WIN32
#include <sys/time.h>
#include <unistd.h>

#include <climits>  // for CLK_TCK
#endif

#ifndef _WIN32_WCE
#include <sys/types.h>

#include <ctime>
#endif
// IWYU pragma: end_exports

// initialize the class variables
bool                               xsigma::timer_log::Logging    = true;
int                                xsigma::timer_log::Indent     = 0;
int                                xsigma::timer_log::MaxEntries = 100;
int                                xsigma::timer_log::NextEntry  = 0;
int                                xsigma::timer_log::WrapFlag   = 0;
std::vector<xsigma::time_logEntry> xsigma::timer_log::TimerLog;

#ifdef CLK_TCK
int xsigma::timer_log::TicksPerSecond = CLK_TCK;
#else
int xsigma::timer_log::TicksPerSecond = 60;
#endif

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC (xsigma::timer_log::TicksPerSecond)
#endif

#ifdef _WIN32
#ifndef _WIN32_WCE
timeb xsigma::timer_log::first_wall_time_;
timeb xsigma::timer_log::current_wall_time_;
#else
FILETIME xsigma::timer_log::first_wall_time_;
FILETIME xsigma::timer_log::current_wall_time_;
#endif
#else
timeval xsigma::timer_log::first_wall_time_;
timeval xsigma::timer_log::current_wall_time_;
tms     xsigma::timer_log::FirstCpuTicks;
tms     xsigma::timer_log::CurrentCpuTicks;
#endif

namespace xsigma
{
//----------------------------------------------------------------------------
// Remove timer log.
void timer_log::CleanupLog()
{
    timer_log::TimerLog.clear();
}

//----------------------------------------------------------------------------
// Clear the timing table.  walltime and cputime will also be set
// to zero when the first new event is recorded.
void timer_log::ResetLog()
{
    timer_log::WrapFlag  = 0;
    timer_log::NextEntry = 0;
    // may want to free TimerLog to force realloc so
    // that user can resize the table by changing MaxEntries.
}

//----------------------------------------------------------------------------
// Record a timing event.  The event is represented by a formatted
// string.
void timer_log::FormatAndMarkEvent(const char* format, ...)
{
    if (!timer_log::Logging)
    {
        return;
    }

    static char event[4096];  // NOLINT
    va_list     var_args;
    va_start(var_args, format);
    vsnprintf(event, sizeof(event), format, var_args);
    va_end(var_args);

    timer_log::MarkEventInternal(event, time_logEntry::LogEntryType::STANDALONE);
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
void timer_log::MarkEvent(const char* event)
{
    timer_log::MarkEventInternal(event, time_logEntry::LogEntryType::STANDALONE);
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
void timer_log::MarkEventInternal(
    const char* event, time_logEntry::LogEntryType type, time_logEntry* entry)
{
    if (!timer_log::Logging)
    {
        return;
    }

    double time_diff;
    int    ticks_diff;

    // If this the first event we're recording, allocate the
    // internal timing table and initialize WallTime and CpuTicks
    // for this first event to zero.
    if (timer_log::NextEntry == 0 && timer_log::WrapFlag == 0)
    {
        if (timer_log::TimerLog.empty())
        {
            timer_log::TimerLog.resize(timer_log::MaxEntries);
        }

#ifdef _WIN32
#ifdef _WIN32_WCE
        SYSTEMTIME st;
        GetLocalTime(&st);
        SystemTimeToFileTime(&st, &(timer_log::first_wall_time_));
#else
        ::ftime(&(timer_log::first_wall_time_));
#endif
#else
        gettimeofday(&(timer_log::first_wall_time_), nullptr);
        times(&FirstCpuTicks);
#endif

        if (entry != nullptr)
        {
            timer_log::TimerLog[0] = *entry;
        }
        else
        {
            timer_log::TimerLog[0].Indent   = timer_log::Indent;
            timer_log::TimerLog[0].WallTime = 0.0;
            timer_log::TimerLog[0].CpuTicks = 0;
            if (event != nullptr)
            {
                timer_log::TimerLog[0].Event = event;
            }
            timer_log::TimerLog[0].Type = type;
            timer_log::NextEntry        = 1;
        }
        return;
    }

    if (entry != nullptr)
    {
        timer_log::TimerLog[timer_log::NextEntry] = *entry;
    }
    else
    {
#ifdef _WIN32
#ifdef _WIN32_WCE
        SYSTEMTIME st;
        GetLocalTime(&st);
        SystemTimeToFileTime(&st, &(timer_log::current_wall_time_));
        time_diff =
            (timer_log::current_wall_time_.dwHighDateTime -
             timer_log::first_wall_time_.dwHighDateTime);
        time_diff = time_diff * 429.4967296;
        time_diff = time_diff + ((timer_log::current_wall_time_.dwLowDateTime -
                                  timer_log::first_wall_time_.dwLowDateTime) /
                                 10000000.0);
#else
        static double scale = 1.0 / 1000.0;
        ::ftime(&(timer_log::current_wall_time_));
        time_diff = static_cast<double>(
            timer_log::current_wall_time_.time - timer_log::first_wall_time_.time);
        time_diff +=
            (timer_log::current_wall_time_.millitm - timer_log::first_wall_time_.millitm) * scale;
#endif
        ticks_diff = 0;
#else
        static double scale = 1.0 / 1000000.0;
        gettimeofday(&(timer_log::current_wall_time_), nullptr);
        time_diff =
            timer_log::current_wall_time_.tv_sec - timer_log::first_wall_time_.tv_sec;  // NOLINT
        time_diff +=
            static_cast<double>(
                timer_log::current_wall_time_.tv_usec - timer_log::first_wall_time_.tv_usec) *
            scale;

        times(&CurrentCpuTicks);
        ticks_diff = (CurrentCpuTicks.tms_utime + CurrentCpuTicks.tms_stime) -  // NOLINT
                     (FirstCpuTicks.tms_utime + FirstCpuTicks.tms_stime);
#endif

        timer_log::TimerLog[timer_log::NextEntry].Indent   = timer_log::Indent;
        timer_log::TimerLog[timer_log::NextEntry].WallTime = static_cast<double>(time_diff);
        timer_log::TimerLog[timer_log::NextEntry].CpuTicks = ticks_diff;
        if (event != nullptr)
        {
            timer_log::TimerLog[timer_log::NextEntry].Event = event;
        }
        timer_log::TimerLog[timer_log::NextEntry].Type = type;
    }

    timer_log::NextEntry++;
    if (timer_log::NextEntry == timer_log::MaxEntries)
    {
        timer_log::NextEntry = 0;
        timer_log::WrapFlag  = 1;
    }
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
// Increments indent after mark.
void timer_log::MarkStartEvent(const char* event)
{
    if (!timer_log::Logging)
    {
        return;
    }

    timer_log::MarkEventInternal(event, time_logEntry::LogEntryType::START);
    ++timer_log::Indent;
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
// Decrements indent after mark.
void timer_log::MarkEndEvent(const char* event)
{
    if (!timer_log::Logging)
    {
        return;
    }

    timer_log::MarkEventInternal(event, time_logEntry::LogEntryType::END);
    --timer_log::Indent;
}

//----------------------------------------------------------------------------
// Record a timing event with known walltime and cputicks.
void timer_log::InsertTimedEvent(const char* event, double time, int cpuTicks)
{
    if (!timer_log::Logging)
    {
        return;
    }

    // manually create both the start and end event and then
    // change the start events values to appear like other events
    time_logEntry entry;
    entry.WallTime = time;
    entry.CpuTicks = cpuTicks;
    if (event != nullptr)
    {
        entry.Event = event;
    }
    entry.Type   = time_logEntry::LogEntryType::INSERTED;
    entry.Indent = timer_log::Indent;

    timer_log::MarkEventInternal(event, time_logEntry::LogEntryType::INSERTED, &entry);
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
int timer_log::GetNumberOfEvents()
{
    return ((timer_log::WrapFlag != 0) ? timer_log::MaxEntries : timer_log::NextEntry);
}

//----------------------------------------------------------------------------
time_logEntry* timer_log::GetEvent(int idx)
{
    int start = 0;
    if (timer_log::WrapFlag != 0)
    {
        start = timer_log::NextEntry;
    }

    idx = (idx + start) % timer_log::MaxEntries;

    return &(timer_log::TimerLog[idx]);
}

//----------------------------------------------------------------------------
int timer_log::GetEventIndent(int idx)
{
    return timer_log::GetEvent(idx)->Indent;
}

//----------------------------------------------------------------------------
double timer_log::GetEventWallTime(int idx)
{
    return timer_log::GetEvent(idx)->WallTime;
}

//----------------------------------------------------------------------------
const char* timer_log::GetEventString(int idx)
{
    return timer_log::GetEvent(idx)->Event.c_str();
}

//----------------------------------------------------------------------------
time_logEntry::LogEntryType timer_log::GetEventType(int idx)
{
    return timer_log::GetEvent(idx)->Type;
}

//----------------------------------------------------------------------------
// Write the timing table out to a file.  Calculate some helpful
// statistics (deltas and percentages) in the process.
void timer_log::DumpLogWithIndents(std::ostream* os, double threshold)  // NOLINT
{
#ifndef _WIN32_WCE
    int               num = timer_log::GetNumberOfEvents();
    std::vector<bool> handledEvents(num, false);

    for (int w = 0; w < timer_log::WrapFlag + 1; w++)
    {
        int start = 0;
        int end   = timer_log::NextEntry;
        if (timer_log::WrapFlag != 0 && w == 0)
        {
            start = timer_log::NextEntry;
            end   = timer_log::MaxEntries;
        }
        for (int i1 = start; i1 < end; i1++)
        {
            int                         indent1   = timer_log::GetEventIndent(i1);
            time_logEntry::LogEntryType eventType = timer_log::GetEventType(i1);
            int                         endEvent  = -1;  // only Modified if this is a START event
            if (eventType == time_logEntry::LogEntryType::END && handledEvents[i1])
            {
                continue;  // this END event is handled by the corresponding START event
            }
            if (eventType == time_logEntry::LogEntryType::START)
            {
                // Search for an END event. it may be before the START event if we've
                // wrapped.
                int counter = 1;
                while (counter < num && timer_log::GetEventIndent((i1 + counter) % num) > indent1)
                {
                    counter++;
                }
                if (timer_log::GetEventIndent((i1 + counter) % num) == indent1)
                {
                    counter--;
                    endEvent                = (i1 + counter) % num;
                    handledEvents[endEvent] = true;
                }
            }
            double dtime = threshold;
            if (eventType == time_logEntry::LogEntryType::START)
            {
                dtime = timer_log::GetEventWallTime(endEvent) - timer_log::GetEventWallTime(i1);
            }
            if (dtime >= threshold)
            {
                int j = indent1;
                while (j-- > 0)
                {
                    *os << "    ";
                }
                *os << timer_log::GetEventString(i1);
                if (endEvent != -1)
                {  // Start event.
                    *os << ",  " << dtime << " seconds";
                }
                else if (eventType == time_logEntry::LogEntryType::INSERTED)
                {
                    *os << ",  " << timer_log::GetEventWallTime(i1) << " seconds (inserted time)";
                }
                else if (eventType == time_logEntry::LogEntryType::END)
                {
                    *os << " (END event without matching START event)";
                }
                *os << std::endl;
            }
        }
    }
#endif
}

//----------------------------------------------------------------------------
// Write the timing table out to a file. This is meant for non-timed events,
// i.e. event type = STANDALONE. All other event types besides the first
// are ignored.
void timer_log::DumpLog(const char* filename)
{
    //#ifndef _WIN32_WCE
    //    xsigmasys::ofstream os_with_warning_C4701(filename);
    //    int                 i;
    //
    //    if (timer_log::WrapFlag != 0)
    //    {
    //        timer_log::DumpEntry(
    //            os_with_warning_C4701,
    //            0,
    //            timer_log::TimerLog[timer_log::NextEntry].WallTime,
    //            0,
    //            timer_log::TimerLog[timer_log::NextEntry].CpuTicks,
    //            0,
    //            timer_log::TimerLog[timer_log::NextEntry].Event.c_str());
    //        int previousEvent = timer_log::NextEntry;
    //        for (i = timer_log::NextEntry + 1; i < timer_log::MaxEntries; i++)
    //        {
    //            if (timer_log::TimerLog[i].Type == time_logEntry::LogEntryType::STANDALONE)
    //            {
    //                timer_log::DumpEntry(
    //                    os_with_warning_C4701,
    //                    i - timer_log::NextEntry,
    //                    timer_log::TimerLog[i].WallTime,
    //                    timer_log::TimerLog[i].WallTime - timer_log::TimerLog[previousEvent].WallTime,
    //                    timer_log::TimerLog[i].CpuTicks,
    //                    timer_log::TimerLog[i].CpuTicks - timer_log::TimerLog[previousEvent].CpuTicks,
    //                    timer_log::TimerLog[i].Event.c_str());
    //                previousEvent = i;
    //            }
    //        }
    //        for (i = 0; i < timer_log::NextEntry; i++)
    //        {
    //            if (timer_log::TimerLog[i].Type == time_logEntry::LogEntryType::STANDALONE)
    //            {
    //                timer_log::DumpEntry(
    //                    os_with_warning_C4701,
    //                    timer_log::MaxEntries - timer_log::NextEntry + i,
    //                    timer_log::TimerLog[i].WallTime,
    //                    timer_log::TimerLog[i].WallTime - timer_log::TimerLog[previousEvent].WallTime,
    //                    timer_log::TimerLog[i].CpuTicks,
    //                    timer_log::TimerLog[i].CpuTicks - timer_log::TimerLog[previousEvent].CpuTicks,
    //                    timer_log::TimerLog[i].Event.c_str());
    //                previousEvent = i;
    //            }
    //        }
    //    }
    //    else
    //    {
    //        timer_log::DumpEntry(
    //            os_with_warning_C4701,
    //            0,
    //            timer_log::TimerLog[0].WallTime,
    //            0,
    //            timer_log::TimerLog[0].CpuTicks,
    //            0,
    //            timer_log::TimerLog[0].Event.c_str());
    //        int previousEvent = 0;
    //        for (i = 1; i < timer_log::NextEntry; i++)
    //        {
    //            if (timer_log::TimerLog[i].Type == time_logEntry::LogEntryType::STANDALONE)
    //            {
    //                timer_log::DumpEntry(
    //                    os_with_warning_C4701,
    //                    i,
    //                    timer_log::TimerLog[i].WallTime,
    //                    timer_log::TimerLog[i].WallTime - timer_log::TimerLog[previousEvent].WallTime,
    //                    timer_log::TimerLog[i].CpuTicks,
    //                    timer_log::TimerLog[i].CpuTicks - timer_log::TimerLog[previousEvent].CpuTicks,
    //                    timer_log::TimerLog[i].Event.c_str());
    //                previousEvent = i;
    //            }
    //        }
    //    }
    //
    //    os_with_warning_C4701.close();
    //#endif
}

// Methods to support simple timer functionality, separate from
// timer table logging.

//----------------------------------------------------------------------------
// Returns the elapsed number of seconds since 00:00:00 Coordinated Universal
// Time (UTC), Thursday, 1 January 1970. This is also called Unix Time.
double timer_log::GetUniversalTime()
{
    double currentTimeInSeconds;

#ifdef _WIN32
#ifdef _WIN32_WCE
    FILETIME   CurrentTime;
    SYSTEMTIME st;
    GetLocalTime(&st);
    SystemTimeToFileTime(&st, &CurrentTime);
    currentTimeInSeconds = CurrentTime.dwHighDateTime;
    currentTimeInSeconds *= 429.4967296;
    currentTimeInSeconds = currentTimeInSeconds + CurrentTime.dwLowDateTime / 10000000.0;
#else
    timeb         CurrentTime;
    static double scale = 1.0 / 1000.0;
    ::ftime(&CurrentTime);
    currentTimeInSeconds =
        static_cast<double>(CurrentTime.time) + scale * static_cast<double>(CurrentTime.millitm);
#endif
#else
    timeval       CurrentTime;
    static double scale = 1.0 / 1000000.0;
    gettimeofday(&CurrentTime, nullptr);
    currentTimeInSeconds = CurrentTime.tv_sec + scale * CurrentTime.tv_usec;
#endif

    return currentTimeInSeconds;
}

//----------------------------------------------------------------------------
double timer_log::GetCPUTime()
{
#ifndef _WIN32_WCE
    return static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC);
#else
    return 1.0;
#endif
}

//----------------------------------------------------------------------------
// Set the StartTime to the current time. Used with GetElapsedTime().
void timer_log::start()
{
    this->StartTime = timer_log::GetUniversalTime();
}

//----------------------------------------------------------------------------
// Sets EndTime to the current time. Used with GetElapsedTime().
void timer_log::stop()
{
    this->EndTime = timer_log::GetUniversalTime();
}

//----------------------------------------------------------------------------
// Returns the difference between StartTime and EndTime as
// a floating point value indicating the elapsed time in seconds.
double timer_log::GetElapsedTime() const
{
    return (this->EndTime - this->StartTime);
}

//----------------------------------------------------------------------------
void timer_log::DumpEntry(
    std::ostream& os,
    int           index,
    double        ttime,
    double        deltatime,
    int           tick,
    int           deltatick,
    const char*   event)
{
    os << index << "   " << ttime << "  " << deltatime << "   "
       << static_cast<double>(tick) / timer_log::TicksPerSecond << "  "
       << static_cast<double>(deltatick) / timer_log::TicksPerSecond << "  ";
    if (deltatime == 0.0)
    {
        os << "0.0   ";
    }
    else
    {
        os << 100.0 * deltatick / timer_log::TicksPerSecond / deltatime << "   ";
    }
    os << event << "\n";
}

//----------------------------------------------------------------------------
void timer_log::LoggingOn()
{
    timer_log::Logging = true;
}

//----------------------------------------------------------------------------
void timer_log::LoggingOff()
{
    timer_log::Logging = false;
}

//----------------------------------------------------------------------------
void timer_log::SetMaxEntries(int a)
{
    if (a == timer_log::MaxEntries)
    {
        return;
    }
    int numEntries = timer_log::GetNumberOfEvents();
    if (timer_log::WrapFlag != 0)
    {  // if we've wrapped events, reorder them
        std::vector<time_logEntry> tmp;
        tmp.reserve(timer_log::MaxEntries);
        std::copy(
            timer_log::TimerLog.begin() + timer_log::NextEntry,
            timer_log::TimerLog.end(),
            std::back_inserter(tmp));
        std::copy(
            timer_log::TimerLog.begin(),
            timer_log::TimerLog.begin() + timer_log::NextEntry,
            std::back_inserter(tmp));
        timer_log::TimerLog = tmp;
        timer_log::WrapFlag = 0;
    }
    if (numEntries <= a)
    {
        timer_log::TimerLog.resize(a);
        timer_log::NextEntry  = numEntries;
        timer_log::WrapFlag   = 0;
        timer_log::MaxEntries = a;
        return;
    }
    // Reduction so we need to get rid of the first bunch of events
    int offset = numEntries - a;
    assert(offset >= 0);
    timer_log::TimerLog.erase(timer_log::TimerLog.begin(), timer_log::TimerLog.begin() + offset);
    timer_log::MaxEntries = a;
    timer_log::NextEntry  = 0;
    timer_log::WrapFlag   = 1;
}

//----------------------------------------------------------------------------
int timer_log::GetMaxEntries()
{
    return timer_log::MaxEntries;
}
}  // namespace xsigma
