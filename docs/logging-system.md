# Logging System

XSigma provides a flexible logging system with three mutually-exclusive backends that can be selected at compile time. Each backend offers different trade-offs between features, performance, and dependencies.

## Table of Contents

- [Available Backends](#available-backends)
- [Quick Start](#quick-start)
- [Backend Feature Comparison](#backend-feature-comparison)
- [Usage Example](#usage-example)
- [Recommendations](#recommendations)
- [Performance Benchmarking](#performance-benchmarking)
- [Additional Documentation](#additional-documentation)

## Available Backends

| Backend | Description | Dependencies | Best For |
|---------|-------------|--------------|----------|
| **LOGURU** (default) | Full-featured logging with scopes and callbacks | Loguru library | Development, debugging, detailed logging |
| **GLOG** | Google's production-grade logging library | glog library | Production, log file management, Google ecosystem |
| **NATIVE** | Minimal native implementation | None | Embedded systems, minimal builds, no dependencies |

## Quick Start

### Using setup.py (Recommended)

```bash
cd Scripts

# Build with LOGURU (default)
python setup.py ninja.clang.python.build.test

# Build with GLOG
python setup.py ninja.clang.python.build.test --logging.backend=GLOG

# Build with NATIVE
python setup.py ninja.clang.python.build.test --logging.backend=NATIVE
```

### Using CMake Directly

```bash
# Configure with LOGURU (default)
cmake -B build -S . -DXSIGMA_LOGGING_BACKEND=LOGURU

# Configure with GLOG
cmake -B build -S . -DXSIGMA_LOGGING_BACKEND=GLOG

# Configure with NATIVE
cmake -B build -S . -DXSIGMA_LOGGING_BACKEND=NATIVE

# Build
cmake --build build
```

## Backend Feature Comparison

| Feature | LOGURU | GLOG | NATIVE |
|---------|--------|------|--------|
| **Severity Levels** | ✅ Full | ✅ Full | ✅ Basic |
| **Formatted Logging** | ✅ Yes | ✅ Yes | ✅ Yes |
| **RAII Scopes** | ✅ Full | ⚠️ Entry only | ❌ No |
| **Custom Callbacks** | ✅ Yes | ❌ No | ❌ No |
| **Signal Handlers** | ✅ Yes | ✅ Yes | ❌ No |
| **Thread Names** | ✅ Full | ✅ Stored | ✅ Stored |
| **File Logging** | ✅ Yes | ✅ Yes | ❌ Console only |
| **Verbosity Levels** | ✅ Flexible | ✅ FLAGS_v | ⚠️ Basic |
| **Build Time** | ~2 min | ~2.5 min | ~1.5 min |
| **Binary Size** | Medium | Large | Small |

## Usage Example

All backends support the same core API:

```cpp
#include "logging/logger.h"

void example_function() {
    // Initialize logging
    xsigma::logger::Init(argc, argv);

    // Simple logging
    xsigma::logger::Log(xsigma::logger::Severity::INFO, "Application started");

    // Formatted logging
    int value = 42;
    xsigma::logger::LogF(xsigma::logger::Severity::INFO, "Value: %d", value);

    // Scoped logging (RAII)
    {
        auto scope = xsigma::logger::StartScope("MyOperation");
        // Scope automatically logs entry and exit with timing (LOGURU)
        // or logs entry only (GLOG), or does nothing (NATIVE)

        xsigma::logger::Log(xsigma::logger::Severity::WARNING, "Processing...");
    }

    // Set thread name
    xsigma::logger::SetThreadName("WorkerThread");
}
```

## Recommendations

### For Development

Use **LOGURU** (default) for full-featured logging with minimal setup:
- Best debugging experience with scopes and callbacks
- Detailed timing information
- Easy to configure and use

### For Production

Use **GLOG** for Google-style logging and integration with Google tools:
- Well-tested, widely used in production environments
- Excellent log file management
- Robust and reliable

### For Minimal Builds

Use **NATIVE** for minimal dependencies and fastest build times:
- Suitable for embedded systems or resource-constrained environments
- No external dependencies
- Smallest binary size

### For Testing

Test with all three backends to ensure code doesn't depend on backend-specific features:
- Use CI/CD matrix builds to verify compatibility
- Ensures portability across different logging backends

## Performance Benchmarking

To benchmark logging performance across all backends:

```bash
cd Scripts
python run_logger_benchmarks.py
```

This will:
1. Build all three backends
2. Run comprehensive benchmarks
3. Generate a detailed comparison report in `benchmark_results/LOGGING_BACKEND_BENCHMARKS.md`

## Additional Documentation

For more detailed information, see:

- **Usage Guide**: [LOGGING_BACKEND_USAGE_GUIDE.md](../LOGGING_BACKEND_USAGE_GUIDE.md) - Detailed usage instructions and best practices
- **Implementation Details**: [LOGGING_BACKEND_IMPLEMENTATION.md](../LOGGING_BACKEND_IMPLEMENTATION.md) - Technical implementation details
- **Test Results**: [LOGGING_BACKEND_TEST_RESULTS.md](../LOGGING_BACKEND_TEST_RESULTS.md) - Comprehensive test results for all backends

## Related Documentation

- [Build Configuration](build-configuration.md) - Build system configuration
- [Third-Party Dependencies](third-party-dependencies.md) - Dependency management

