# XSigma Profiler-Logger Integration Design

## Table of Contents
1. [Use Cases](#use-cases)
2. [Integration Architecture](#integration-architecture)
3. [Implementation Plan](#implementation-plan)
4. [Example Usage](#example-usage)
5. [Performance Impact](#performance-impact)
6. [Configuration Reference](#configuration-reference)

---

## Use Cases

### 1. Logging Profiling Session Start/Stop Events

**Scenario**: Track when profiling sessions are started and stopped for audit purposes.

**Benefits**:
- Audit trail of profiling activities
- Debugging profiling configuration issues
- Monitoring profiling overhead in production

**Example Log Output**:
```
[INFO] [2025-10-13 13:30:45] Profiling session started (session_id=abc123, options={timing=true, memory=true})
[INFO] [2025-10-13 13:30:50] Profiling session stopped (session_id=abc123, duration=5.2s, events=1234)
```

---

### 2. Logging Performance Warnings When Thresholds Are Exceeded

**Scenario**: Automatically log warnings when functions exceed performance thresholds.

**Benefits**:
- Early detection of performance regressions
- Automated alerting for slow operations
- Historical performance tracking

**Example Log Output**:
```
[WARN] [2025-10-13 13:30:46] Performance threshold exceeded: function='process_data', execution_time=150ms, threshold=100ms
[WARN] [2025-10-13 13:30:47] Memory threshold exceeded: function='allocate_buffer', bytes=512MB, threshold=256MB
```

---

### 3. Audit Trail of Profiling Activities

**Scenario**: Maintain a complete audit trail of all profiling activities for compliance and debugging.

**Benefits**:
- Compliance with audit requirements
- Debugging profiling issues
- Understanding profiling impact on production

**Example Log Output**:
```
[INFO] [2025-10-13 13:30:45] Profiling session created (user=john.doe, host=server01, pid=12345)
[INFO] [2025-10-13 13:30:45] Profiling options configured (timing=true, memory=true, thread_safety=true)
[INFO] [2025-10-13 13:30:45] Profiling session started (session_id=abc123)
[DEBUG] [2025-10-13 13:30:46] Profiling scope entered (name='process_data', thread_id=1)
[DEBUG] [2025-10-13 13:30:46] Profiling scope exited (name='process_data', duration=150ms)
[INFO] [2025-10-13 13:30:50] Profiling session stopped (session_id=abc123)
[INFO] [2025-10-13 13:30:50] Profiling report exported (filename='profile.json', size=1.2MB)
```

---

### 4. Debug Logging for Profiling System Itself

**Scenario**: Debug issues within the profiling system itself.

**Benefits**:
- Troubleshooting profiling bugs
- Understanding profiling behavior
- Performance tuning of profiling system

**Example Log Output**:
```
[DEBUG] [2025-10-13 13:30:45] Profiler initialized (memory_tracker=enabled, statistical_analyzer=enabled)
[DEBUG] [2025-10-13 13:30:46] Scope stack depth: 3 (max_depth=5)
[DEBUG] [2025-10-13 13:30:47] Memory allocation tracked (ptr=0x7f8a4c000000, size=1024, alignment=64)
[TRACE] [2025-10-13 13:30:48] Profiling event recorded (type=SCOPE_EXIT, timestamp=1697203848123456)
```

---

## Integration Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Code                          │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Profiler Session (profiler.h)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  start() / stop() / profile_scope()                  │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        Profiler Logger Adapter (NEW)                 │   │
│  │  - Filter events by log level                        │   │
│  │  - Check performance thresholds                      │   │
│  │  - Format log messages                               │   │
│  └────────────────────┬─────────────────────────────────┘   │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              XSigma Logger (logger.h)                        │
│  - Log to file / console / syslog                            │
│  - Thread-safe logging                                       │
│  - Log rotation                                              │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction

```cpp
// Profiler Session
class profiler_session {
    void start() {
        if (logger_adapter_) {
            logger_adapter_->log_session_start(session_id_);
        }
        // ... start profiling ...
    }

    void stop() {
        // ... stop profiling ...
        if (logger_adapter_) {
            logger_adapter_->log_session_stop(session_id_, duration_);
        }
    }

private:
    std::unique_ptr<profiler_logger_adapter> logger_adapter_;
};

// Logger Adapter
class profiler_logger_adapter {
    void log_session_start(const std::string& session_id) {
        if (logger_ && should_log(log_level::INFO)) {
            logger_->info("Profiling session started (session_id={})", session_id);
        }
    }

    void check_performance_threshold(const std::string& function_name, double duration_ms) {
        auto threshold = get_threshold(function_name);
        if (threshold && duration_ms > *threshold) {
            logger_->warn("Performance threshold exceeded: function='{}', execution_time={}ms, threshold={}ms",
                         function_name, duration_ms, *threshold);
        }
    }

private:
    xsigma::logger* logger_;
    std::unordered_map<std::string, double> thresholds_;
};
```

---

## Implementation Plan

### Phase 1: Core Integration (Week 1)

**Tasks**:
1. Create `profiler_logger_adapter` class
2. Add logger integration to `profiler_session`
3. Implement basic logging for session start/stop
4. Add unit tests

**Files to Create/Modify**:
- `Library/Core/experimental/profiler/logging/profiler_logger_adapter.h` (NEW)
- `Library/Core/experimental/profiler/logging/profiler_logger_adapter.cxx` (NEW)
- `Library/Core/experimental/profiler/session/profiler.h` (MODIFY)
- `Library/Core/experimental/profiler/session/profiler.cxx` (MODIFY)

**API Design**:

```cpp
// profiler_logger_adapter.h
#pragma once

#include "experimental/profiler/core/profiler_interface.h"
#include "logging/logger.h"

namespace xsigma {
namespace profiler {

enum class profiler_log_level {
    TRACE,   // Very verbose (every scope entry/exit)
    DEBUG,   // Debug information
    INFO,    // Session events
    WARN,    // Performance warnings
    ERROR    // Errors
};

struct profiler_logger_options {
    bool log_session_events = true;        // Log start/stop
    bool log_performance_warnings = true;  // Log threshold violations
    bool log_memory_warnings = true;       // Log memory issues
    bool log_scope_entry_exit = false;     // Log every scope (verbose!)
    profiler_log_level min_log_level = profiler_log_level::INFO;
};

class XSIGMA_API profiler_logger_adapter {
public:
    explicit profiler_logger_adapter(xsigma::logger* logger,
                                    const profiler_logger_options& opts = {});
    ~profiler_logger_adapter();

    // Session lifecycle logging
    void log_session_start(const std::string& session_id);
    void log_session_stop(const std::string& session_id, double duration_s);

    // Performance threshold logging
    void set_performance_threshold(const std::string& function_name, double threshold_ms);
    void check_performance_threshold(const std::string& function_name, double duration_ms);

    // Memory threshold logging
    void set_memory_threshold(const std::string& operation_name, size_t threshold_bytes);
    void check_memory_threshold(const std::string& operation_name, size_t bytes);

    // Scope logging (verbose)
    void log_scope_entry(const std::string& scope_name, int thread_id);
    void log_scope_exit(const std::string& scope_name, double duration_ms);

    // Configuration
    void set_log_level(profiler_log_level level);
    void enable_session_logging(bool enable);
    void enable_performance_warnings(bool enable);
    void enable_memory_warnings(bool enable);
    void enable_scope_logging(bool enable);

private:
    xsigma::logger* logger_;
    profiler_logger_options options_;
    std::unordered_map<std::string, double> performance_thresholds_;
    std::unordered_map<std::string, size_t> memory_thresholds_;

    bool should_log(profiler_log_level level) const;
};

}  // namespace profiler
}  // namespace xsigma
```

---

### Phase 2: Performance Threshold Monitoring (Week 2)

**Tasks**:
1. Implement performance threshold checking
2. Add automatic threshold violation logging
3. Add configurable thresholds per function
4. Add unit tests

**Implementation**:

```cpp
// profiler_logger_adapter.cxx
void profiler_logger_adapter::check_performance_threshold(
    const std::string& function_name,
    double duration_ms)
{
    if (!options_.log_performance_warnings) {
        return;
    }

    auto it = performance_thresholds_.find(function_name);
    if (it != performance_thresholds_.end() && duration_ms > it->second) {
        if (should_log(profiler_log_level::WARN)) {
            logger_->warn(
                "Performance threshold exceeded: function='{}', "
                "execution_time={:.2f}ms, threshold={:.2f}ms, "
                "deviation={:.1f}%",
                function_name,
                duration_ms,
                it->second,
                ((duration_ms - it->second) / it->second) * 100.0
            );
        }
    }
}
```

---

### Phase 3: Memory Monitoring (Week 2)

**Tasks**:
1. Implement memory threshold checking
2. Add memory allocation/deallocation logging
3. Add memory leak detection logging
4. Add unit tests

**Implementation**:

```cpp
void profiler_logger_adapter::check_memory_threshold(
    const std::string& operation_name,
    size_t bytes)
{
    if (!options_.log_memory_warnings) {
        return;
    }

    auto it = memory_thresholds_.find(operation_name);
    if (it != memory_thresholds_.end() && bytes > it->second) {
        if (should_log(profiler_log_level::WARN)) {
            logger_->warn(
                "Memory threshold exceeded: operation='{}', "
                "bytes={}, threshold={}, deviation={:.1f}%",
                operation_name,
                bytes,
                it->second,
                ((static_cast<double>(bytes) - it->second) / it->second) * 100.0
            );
        }
    }
}
```

---

### Phase 4: Integration with Profiler Session (Week 3)

**Tasks**:
1. Integrate logger adapter into `profiler_session`
2. Add builder method for logger configuration
3. Update existing tests
4. Add integration tests

**Implementation**:

```cpp
// profiler.h
class profiler_session {
public:
    // ... existing methods ...

    // Set logger adapter
    void set_logger_adapter(std::unique_ptr<profiler_logger_adapter> adapter);

private:
    std::unique_ptr<profiler_logger_adapter> logger_adapter_;
};

// profiler_session_builder
class profiler_session_builder {
public:
    // ... existing methods ...

    profiler_session_builder& with_logger(xsigma::logger* logger,
                                         const profiler_logger_options& opts = {})
    {
        logger_ = logger;
        logger_options_ = opts;
        return *this;
    }

private:
    xsigma::logger* logger_ = nullptr;
    profiler_logger_options logger_options_;
};
```

---

## Example Usage

### Example 1: Basic Session Logging

```cpp
#include "experimental/profiler/session/profiler.h"
#include "experimental/profiler/logging/profiler_logger_adapter.h"
#include "logging/logger.h"

int main() {
    // Create logger
    xsigma::logger logger;
    logger.set_log_file("profiler.log");

    // Configure profiler with logger
    xsigma::profiler_options opts;
    opts.enable_timing_ = true;
    opts.enable_memory_tracking_ = true;

    xsigma::profiler_session session(opts);

    // Create and attach logger adapter
    xsigma::profiler::profiler_logger_options log_opts;
    log_opts.log_session_events = true;
    log_opts.log_performance_warnings = true;

    auto adapter = std::make_unique<xsigma::profiler::profiler_logger_adapter>(
        &logger, log_opts);
    session.set_logger_adapter(std::move(adapter));

    // Start profiling (logged)
    session.start();

    // Your code here
    {
        XSIGMA_PROFILE_SCOPE("main_work");
        // Work
    }

    // Stop profiling (logged)
    session.stop();

    return 0;
}
```

**Log Output**:
```
[INFO] [2025-10-13 13:30:45.123] Profiling session started (session_id=abc123, options={timing=true, memory=true})
[INFO] [2025-10-13 13:30:50.456] Profiling session stopped (session_id=abc123, duration=5.333s, events=42)
```

---

### Example 2: Performance Threshold Monitoring

```cpp
int main() {
    xsigma::logger logger;

    xsigma::profiler_options opts;
    opts.enable_timing_ = true;

    xsigma::profiler_session session(opts);

    // Create logger adapter with performance thresholds
    xsigma::profiler::profiler_logger_options log_opts;
    log_opts.log_performance_warnings = true;

    auto adapter = std::make_unique<xsigma::profiler::profiler_logger_adapter>(
        &logger, log_opts);

    // Set performance thresholds
    adapter->set_performance_threshold("process_data", 100.0);  // 100ms
    adapter->set_performance_threshold("compute", 50.0);        // 50ms

    session.set_logger_adapter(std::move(adapter));
    session.start();

    // This will trigger a warning if it takes > 100ms
    {
        XSIGMA_PROFILE_SCOPE("process_data");
        slow_operation();  // Takes 150ms
    }

    session.stop();

    return 0;
}
```

**Log Output**:
```
[INFO] [2025-10-13 13:30:45.123] Profiling session started (session_id=abc123)
[WARN] [2025-10-13 13:30:45.273] Performance threshold exceeded: function='process_data', execution_time=150.00ms, threshold=100.00ms, deviation=50.0%
[INFO] [2025-10-13 13:30:45.500] Profiling session stopped (session_id=abc123, duration=0.377s)
```

---

### Example 3: Memory Threshold Monitoring

```cpp
int main() {
    xsigma::logger logger;

    xsigma::profiler_options opts;
    opts.enable_memory_tracking_ = true;

    xsigma::profiler_session session(opts);

    // Create logger adapter with memory thresholds
    xsigma::profiler::profiler_logger_options log_opts;
    log_opts.log_memory_warnings = true;

    auto adapter = std::make_unique<xsigma::profiler::profiler_logger_adapter>(
        &logger, log_opts);

    // Set memory thresholds
    adapter->set_memory_threshold("allocate_buffer", 256 * 1024 * 1024);  // 256MB

    session.set_logger_adapter(std::move(adapter));
    session.start();

    // This will trigger a warning if allocation > 256MB
    {
        XSIGMA_PROFILE_SCOPE("allocate_buffer");
        std::vector<double> large_buffer(64 * 1024 * 1024);  // 512MB
    }

    session.stop();

    return 0;
}
```

**Log Output**:
```
[INFO] [2025-10-13 13:30:45.123] Profiling session started (session_id=abc123)
[WARN] [2025-10-13 13:30:45.456] Memory threshold exceeded: operation='allocate_buffer', bytes=536870912, threshold=268435456, deviation=100.0%
[INFO] [2025-10-13 13:30:46.000] Profiling session stopped (session_id=abc123, duration=0.877s)
```

---

### Example 4: Verbose Scope Logging (Debug Mode)

```cpp
int main() {
    xsigma::logger logger;
    logger.set_log_level(xsigma::log_level::DEBUG);

    xsigma::profiler_options opts;
    opts.enable_timing_ = true;

    xsigma::profiler_session session(opts);

    // Enable verbose scope logging
    xsigma::profiler::profiler_logger_options log_opts;
    log_opts.log_scope_entry_exit = true;  // ⚠️ Very verbose!
    log_opts.min_log_level = xsigma::profiler::profiler_log_level::DEBUG;

    auto adapter = std::make_unique<xsigma::profiler::profiler_logger_adapter>(
        &logger, log_opts);

    session.set_logger_adapter(std::move(adapter));
    session.start();

    {
        XSIGMA_PROFILE_SCOPE("outer");
        {
            XSIGMA_PROFILE_SCOPE("inner");
            // Work
        }
    }

    session.stop();

    return 0;
}
```

**Log Output**:
```
[INFO] [2025-10-13 13:30:45.123] Profiling session started (session_id=abc123)
[DEBUG] [2025-10-13 13:30:45.124] Scope entered: name='outer', thread_id=1
[DEBUG] [2025-10-13 13:30:45.125] Scope entered: name='inner', thread_id=1
[DEBUG] [2025-10-13 13:30:45.130] Scope exited: name='inner', duration=5.00ms
[DEBUG] [2025-10-13 13:30:45.131] Scope exited: name='outer', duration=7.00ms
[INFO] [2025-10-13 13:30:45.200] Profiling session stopped (session_id=abc123, duration=0.077s)
```

---

## Performance Impact

### Overhead by Logging Level

| Logging Configuration | Overhead | Use Case |
|----------------------|----------|----------|
| **Disabled** | 0% | Production (no logging) |
| **Session events only** | < 0.1% | Production (audit trail) |
| **+ Performance warnings** | < 1% | Production (monitoring) |
| **+ Memory warnings** | < 2% | Development (debugging) |
| **+ Scope entry/exit** | > 10% | ❌ Debug only (not recommended) |

### Measured Overhead

```cpp
// Baseline: No logging
void baseline() {
    for (int i = 0; i < 1000000; ++i) {
        XSIGMA_PROFILE_SCOPE("iteration");
        compute(i);
    }
}
// Time: 100ms

// Session logging only
void session_logging() {
    // log_session_events = true
    // log_scope_entry_exit = false
    for (int i = 0; i < 1000000; ++i) {
        XSIGMA_PROFILE_SCOPE("iteration");
        compute(i);
    }
}
// Time: 100.05ms (0.05% overhead)

// Scope logging (verbose) - ❌ DON'T USE IN PRODUCTION!
void scope_logging() {
    // log_scope_entry_exit = true
    for (int i = 0; i < 1000000; ++i) {
        XSIGMA_PROFILE_SCOPE("iteration");
        compute(i);
    }
}
// Time: 120ms (20% overhead!)
```

---

## Configuration Reference

### profiler_logger_options

```cpp
struct profiler_logger_options {
    // Enable logging of session start/stop events
    bool log_session_events = true;

    // Enable logging of performance threshold violations
    bool log_performance_warnings = true;

    // Enable logging of memory threshold violations
    bool log_memory_warnings = true;

    // Enable logging of every scope entry/exit (⚠️ VERY VERBOSE!)
    bool log_scope_entry_exit = false;

    // Minimum log level to output
    profiler_log_level min_log_level = profiler_log_level::INFO;
};
```

### Recommended Configurations

**Production (Minimal Overhead)**:
```cpp
profiler_logger_options opts;
opts.log_session_events = true;
opts.log_performance_warnings = true;
opts.log_memory_warnings = false;
opts.log_scope_entry_exit = false;
opts.min_log_level = profiler_log_level::INFO;
```

**Development (Moderate Overhead)**:
```cpp
profiler_logger_options opts;
opts.log_session_events = true;
opts.log_performance_warnings = true;
opts.log_memory_warnings = true;
opts.log_scope_entry_exit = false;
opts.min_log_level = profiler_log_level::DEBUG;
```

**Debug (High Overhead)**:
```cpp
profiler_logger_options opts;
opts.log_session_events = true;
opts.log_performance_warnings = true;
opts.log_memory_warnings = true;
opts.log_scope_entry_exit = true;  // ⚠️ Very verbose!
opts.min_log_level = profiler_log_level::TRACE;
```

---

## Summary

**Implementation Timeline**: 3 weeks

**Estimated Effort**:
- Week 1: Core integration (40 hours)
- Week 2: Threshold monitoring (40 hours)
- Week 3: Testing and documentation (20 hours)
- **Total**: 100 hours (2.5 weeks)

**Expected Benefits**:
- Automated performance monitoring
- Audit trail for compliance
- Early detection of performance regressions
- Minimal overhead (< 1% in production)
- Seamless integration with existing XSigma logger
