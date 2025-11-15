#pragma once

#include "common/export.h"
#include "profiler/common/unwind/unwind.h"

namespace xsigma
{
XSIGMA_API bool get_cpp_stacktraces_enabled();
XSIGMA_API xsigma::unwind::Mode get_symbolize_mode();
}  // namespace xsigma
