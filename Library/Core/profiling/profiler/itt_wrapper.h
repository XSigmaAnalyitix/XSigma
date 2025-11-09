#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H
#include "common/export.h"

namespace xsigma::profiler
{
XSIGMA_API bool itt_is_available();
XSIGMA_API void itt_range_push(const char* msg);
XSIGMA_API void itt_range_pop();
XSIGMA_API void itt_mark(const char* msg);
}  // namespace xsigma::profiler

#endif  // PROFILER_ITT_H
