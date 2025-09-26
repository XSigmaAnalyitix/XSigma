#ifndef CORE_BACKTRACE_H
#define CORE_BACKTRACE_H

#include <cstddef>  // for size_t
#include <string>   // for string

     #include "common/macros.h"
#include "common/macros.h"  // for XSIGMA_DELETE_CLASS
//#include "common/pointer.h"
//#include "util/lazy.h"

namespace xsigma
{
class XSIGMA_API back_trace
{
public:
    XSIGMA_DELETE_CLASS(back_trace);

    //using back_trace_t = ptr_const<lazy_value<std::string>>;

    static std::string print(
        size_t frames_to_skip           = 0,
        size_t maximum_number_of_frames = 64,
        bool   skip_python_frames       = true);

    static void set_stack_trace_on_error(int enable);
};
}  // namespace xsigma

#endif  // CORE_BACKTRACE_H
