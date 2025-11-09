#pragma once

#if 0
#include <xsigma/custom_class.h>

#include <optional>

#include "record_function.h"

namespace xsigma::autograd::profiler
{

class XSIGMA_VISIBILITY PythonRecordFunction : public xsigma::CustomClassHolder
{
    xsigma::RecordFunction record;

    explicit PythonRecordFunction(xsigma::RecordScope scope = xsigma::RecordScope::FUNCTION) : record(scope)
    {
    }
};

// Creates a new profiling scope using RecordFunction and invokes its starting
// callbacks.
XSIGMA_API xsigma::intrusive_ptr<PythonRecordFunction> record_function_enter_new(
    const std::string& name, const std::optional<std::string>& args = std::nullopt);

// Schedules RecordFunction's end callbacks to be run on completion of a future.
XSIGMA_API xsigma::intrusive_ptr<xsigma::ivalue::Future> _call_end_callbacks_on_fut_new(
    const xsigma::intrusive_ptr<PythonRecordFunction>&   record,
    const xsigma::intrusive_ptr<xsigma::ivalue::Future>& fut);

}  // namespace xsigma::autograd::profiler
#endif
