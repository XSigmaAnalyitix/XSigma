#include "profiling/profiler/itt_wrapper.h"

#include <ittnotify.h>

#include "profiling/profiler/stubs/base.h"

namespace xsigma::profiler
{
static __itt_domain* _itt_domain = __itt_domain_create("XSigma");

bool itt_is_available()
{
    return xsigma::profiler::impl::ittStubs()->enabled();
}

void itt_range_push(const char* msg)
{
    __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
    __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
}

void itt_range_pop()
{
    __itt_task_end(_itt_domain);
}

void itt_mark(const char* msg)
{
    __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
    __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
    __itt_task_end(_itt_domain);
}
}  // namespace xsigma::profiler
