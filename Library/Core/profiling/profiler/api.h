#pragma once

#include "profiling/profiler/orchestration/observer.h"

// There are some components which use these symbols. Until we migrate them
// we have to mirror them in the old autograd namespace.

// TODO: XSigma-specific types commented out
namespace xsigma::autograd::profiler
{
using xsigma::profiler::impl::ActivityType;
using xsigma::profiler::impl::getProfilerConfig;
using xsigma::profiler::impl::ProfilerConfig;
using xsigma::profiler::impl::profilerEnabled;
using xsigma::profiler::impl::ProfilerState;
}  // namespace xsigma::autograd::profiler
