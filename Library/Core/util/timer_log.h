
#ifndef __timer_log_h__
#define __timer_log_h__

// IWYU pragma: begin_exports
#include <vector>

  // For export macro
#include "common/macros.h"

#ifdef _WIN32
#include <sys/timeb.h>  // Needed for Win32 implementation of timer
#include <sys/types.h>  // Needed for Win32 implementation of timer
#else
#include <sys/time.h>   // Needed for unix implementation of timer
#include <sys/times.h>  // Needed for unix implementation of timer
#include <sys/types.h>  // Needed for unix implementation of timer

#include <ctime>  // Needed for unix implementation of timer
#endif

// var args
#ifndef _WIN32
#include <unistd.h>  // Needed for unix implementation of timer
#endif
// IWYU pragma: end_exports

// select stuff here is for sleep method
#ifndef NO_FD_SET
#define SELECT_MASK fd_set
#else
#ifndef _AIX
using fd_mask = long;
#endif
#if defined(_IBMR2)
#define SELECT_MASK void
#else
#define SELECT_MASK int
#endif
#endif

namespace xsigma
{
struct time_logEntry
{
    enum LogEntryType
    {
        INVALID = -1,
        STANDALONE,  // an individual, marked event
        START,       // start of a timed event
        END,         // end of a timed event
        INSERTED     // externally timed value
    };
    double        WallTime = 0.;
    int           CpuTicks = 0;
    std::string   Event;
    LogEntryType  Type   = LogEntryType::INVALID;
    unsigned char Indent = 0;
    time_logEntry()      = default;
};

class XSIGMA_API timer_log
{
public:
    /**
     * This flag will turn logging of events off or on.
     * By default, logging is on.
     */
    static void LoggingOn();
    static void LoggingOff();

    //@{
    /**
     * Set/Get the maximum number of entries allowed in the timer log
     */
    static void SetMaxEntries(int a);
    static int  GetMaxEntries();
    //@}

    /**
     * Record a timing event.  The event is represented by a formatted
     * string.  The internal buffer is 4096 bytes and will truncate anything
     * longer.
     */
#ifndef __XSIGMA_WRAP__
    static void FormatAndMarkEvent(const char* format, ...)
        MACRO_CORE_PRINTF_FORMAT(1, 2);  // NOLINT
#endif

    //@{
    /**
     * Write the timing table out to a file.  Calculate some helpful
     * statistics (deltas and percentages) in the process.
     */
    static void DumpLog(const char* filename);
    //@}

    //@{
    /**
     * I want to time events, so I am creating this interface to
     * mark events that have a start and an end.  These events can be,
     * nested. The standard Dumplog ignores the indents.
     */
    static void MarkStartEvent(const char* event);
    static void MarkEndEvent(const char* event);
    //@}

    //@{
    /**
     * Insert an event with a known wall time value (in seconds)
     * and cpuTicks.
     */
    static void InsertTimedEvent(const char* event, double time, int cpuTicks);
    //@}

    static void DumpLogWithIndents(std::ostream* os, double threshold);
    // static void dump_log_with_indents_and_percentages(std::ostream* os);

    //@}

    /**
     * Record a timing event and capture wall time and cpu ticks.
     */
    static void MarkEvent(const char* event);

    /**
     * Clear the timing table.  walltime and cputime will also be set
     * to zero when the first new event is recorded.
     */
    static void ResetLog();

    /**
     * Remove timer log.
     */
    static void CleanupLog();

    /**
     * Returns the elapsed number of seconds since 00:00:00 Coordinated Universal
     * Time (UTC), Thursday, 1 January 1970. This is also called Unix Time.
     */
    static double GetUniversalTime();

    /**
     * Returns the CPU time for this process
     * On Win32 platforms this actually returns wall time.
     */
    static double GetCPUTime();

    /**
     * Set the StartTime to the current time. Used with GetElapsedTime().
     */
    void start();

    /**
     * Sets EndTime to the current time. Used with GetElapsedTime().
     */
    void stop();

    /**
     * Returns the difference between StartTime and EndTime as
     * a doubleing point value indicating the elapsed time in seconds.
     */
    double GetElapsedTime() const;

    timer_log()
    {
        this->StartTime = 0;
        this->EndTime   = 0;
    };  // insure constructor/destructor protected

    ~timer_log() = default;

protected:
    static bool                       Logging;
    static int                        Indent;
    static int                        MaxEntries;
    static int                        NextEntry;
    static int                        WrapFlag;
    static int                        TicksPerSecond;
    static std::vector<time_logEntry> TimerLog;

#ifdef _WIN32
#ifndef _WIN32_WCE
    static timeb first_wall_time_;
    static timeb current_wall_time_;
#else
    static FILETIME first_wall_time_;
    static FILETIME current_wall_time_;
#endif
#else
    static timeval first_wall_time_;
    static timeval current_wall_time_;
    static tms     FirstCpuTicks;
    static tms     CurrentCpuTicks;
#endif

    /**
     * Record a timing event and capture wall time and cpu ticks.
     */
    static void MarkEventInternal(
        const char* event, time_logEntry::LogEntryType type, time_logEntry* entry = nullptr);

    // instance variables to support simple timing functionality,
    // separate from timer table logging.
    double StartTime;
    double EndTime;

    static time_logEntry*              GetEvent(int idx);
    static int                         GetNumberOfEvents();
    static int                         GetEventIndent(int idx);
    static double                      GetEventWallTime(int idx);
    static const char*                 GetEventString(int idx);
    static time_logEntry::LogEntryType GetEventType(int idx);

    static void DumpEntry(
        std::ostream& os,
        int           index,
        double        ttime,
        double        deltatime,
        int           tick,
        int           deltatick,
        const char*   event);

    XSIGMA_DELETE_COPY(timer_log);
};

}  // namespace xsigma
//
// Set built-in type.  Creates member Set"name"() (e.g., SetVisibility());
//
#define time_logMacro(string)                     \
    {                                             \
        timer_log::FormatAndMarkEvent(            \
            "Mark: In %s, line %d, class %s: %s", \
            __FILE__,                             \
            __LINE__,                             \
            this->GetClassName(),                 \
            string);                              \
    }

#endif
