#pragma once

#include <string>

#include "common/export.h"

namespace xsigma::profiler::impl
{

// Adds the execution trace observer as a global callback function, the data
// will be written to output file path.
XSIGMA_API bool addExecutionTraceObserver(const std::string& output_file_path);

// Remove the execution trace observer from the global callback functions.
XSIGMA_API void removeExecutionTraceObserver();

// Enables execution trace observer.
XSIGMA_API void enableExecutionTraceObserver();

// Disables execution trace observer.
XSIGMA_API void disableExecutionTraceObserver();

}  // namespace xsigma::profiler::impl
