#if 0
#include <ATen/ThreadLocalState.h>
#include <ATen/cpp_custom_type_hack.h>
#include <xsigma/csrc/autograd/record_function_ops.h>
#include <xsigma/csrc/jit/runtime/operator.h>
#include <xsigma/library.h>

#include "record_function.h"

namespace caffe2
{
// Required for cpp_custom_type_hack to work
// NOLINTNEXTLINE(bugprone-exception-escape)
CAFFE_KNOWN_TYPE(xsigma::RecordFunction)
}  // namespace caffe2

namespace xsigma::autograd::profiler
{

// Creates a new profiling scope using RecordFunction and invokes its starting
// callbacks.
static void record_function_enter(
    const std::string& name, const std::optional<std::string>& args, xsigma::RecordFunction& rec)
{
    if (rec.isActive())
    {
        if (rec.needsInputs() && args.has_value())
        {
            rec.before(name, xsigma::array_ref<const xsigma::IValue>{xsigma::IValue{args.value()}});
        }
        else
        {
            rec.before(name);
        }
    }
}

// Legacy signature using cpp_custom_type_hack
static xsigma::Tensor record_function_enter_legacy(
    const std::string& name, const std::optional<std::string>& args)
{
    auto rec = std::make_unique<xsigma::RecordFunction>(xsigma::RecordScope::USER_SCOPE);
    record_function_enter(name, args, *rec);
    return xsigma::cpp_custom_type_hack::create(std::move(rec), xsigma::TensorOptions());
}

// New signature using custom_class
xsigma::intrusive_ptr<PythonRecordFunction> record_function_enter_new(
    const std::string& name, const std::optional<std::string>& args)
{
    auto rec = xsigma::make_intrusive<PythonRecordFunction>(xsigma::RecordScope::USER_SCOPE);
    record_function_enter(name, args, rec->record);
    return rec;
}

static xsigma::RecordFunction& getRecordFunctionFromTensor(const xsigma::Tensor& handle)
{
    auto& rec = xsigma::cpp_custom_type_hack::cast<xsigma::RecordFunction>(handle);
    return rec;
}

// Ends the profiling scope created with record_function_enter.
static void record_function_exit(xsigma::RecordFunction& rec)
{
    rec.end();
}

// Legacy signature using cpp_custom_type_hack
static void record_function_exit_legacy(const xsigma::Tensor& handle)
{
    // We don't actually need to do anything with handle just need to persist the
    // lifetime until now.
    auto& rec = getRecordFunctionFromTensor(handle);
    record_function_exit(rec);
}

// New signature using custom_class
static void record_function_exit_new(const xsigma::intrusive_ptr<PythonRecordFunction>& record)
{
    record_function_exit(record->record);
}

template <typename Func>
static xsigma::intrusive_ptr<xsigma::ivalue::Future> _call_end_callbacks_on_fut(
    Func get_record, const xsigma::intrusive_ptr<xsigma::ivalue::Future>& fut)
{
    // Profiling callback that ends the associated record_function
    // and returns the value of the passed in future.
    auto futureProfilingFunc = [get_record = std::move(get_record)](xsigma::ivalue::Future& fut)
    {
        auto& rec = get_record();
        rec.end();
        // Note: this future is returned to the user to ensure that a call to
        // wait() ensures that profiling callbacks have ran. To ensure that this
        // is transparent, we must make this future propagate the value of the
        // RPC future. Use value() here instead of constValue() to ensure we
        // propagate errors.
        return fut.value();
    };
    // Define a future that completes after the profiling callbacks are run.
    auto profiledFut =
        fut->then(xsigma::wrapPropagateTLSState(std::move(futureProfilingFunc)), fut->elementType());
    return profiledFut;
}

// Legacy signature using cpp_custom_type_hack
static xsigma::intrusive_ptr<xsigma::ivalue::Future> _call_end_callbacks_on_fut_legacy(
    const xsigma::Tensor& handle, const xsigma::intrusive_ptr<xsigma::ivalue::Future>& fut)
{
    return _call_end_callbacks_on_fut(
        [handle]() -> xsigma::RecordFunction&
        {
            XSIGMA_CHECK(
                handle.defined(),
                "Undefined RecordFunction handle. This can happen if the handle is "
                "not correctly persisted and is destroyed before the future is "
                "realized.");

            return getRecordFunctionFromTensor(handle);
        },
        fut);
}

// New signature using custom_class
xsigma::intrusive_ptr<xsigma::ivalue::Future> _call_end_callbacks_on_fut_new(
    const xsigma::intrusive_ptr<PythonRecordFunction>&   record,
    const xsigma::intrusive_ptr<xsigma::ivalue::Future>& fut)
{
    return _call_end_callbacks_on_fut(
        [record]() -> xsigma::RecordFunction& { return record->record; }, fut);
}

// Internal only, do not use directly, use Python's record_function()
XSIGMA_LIBRARY_FRAGMENT(profiler, m)
{
    m.class_<PythonRecordFunction>("_RecordFunction");

    m.def(
        "_record_function_enter(str name, str? args=None) -> Tensor",
        &record_function_enter_legacy);
    m.def(
        "_record_function_enter_new(str name, str? args=None) -> "
        "__torch__.xsigma.classes.profiler._RecordFunction",
        &record_function_enter_new);
    m.def("_record_function_exit", &record_function_exit_legacy);
    m.def("_record_function_exit._RecordFunction", &record_function_exit_new);

    xsigma::jit::registerOperator(
        xsigma::jit::Operator(
            "profiler::_call_end_callbacks_on_jit_fut(Tensor x, Future(t) y) -> Future(t)",
            [](jit::Stack& stack)
            {
                // Pop inputs, which should be a future and a tensor
                auto fut         = jit::pop(stack).toFuture();
                auto tensor      = jit::pop(stack).toTensor();
                auto profiledFut = _call_end_callbacks_on_fut_legacy(tensor, fut);
                // return future that completes when profiling callbacks have run.
                jit::push(stack, std::move(profiledFut));
            },
            xsigma::AliasAnalysisKind::FROM_SCHEMA));
    xsigma::jit::registerOperator(
        xsigma::jit::Operator(
            "profiler::_call_end_callbacks_on_jit_fut._RecordFunction("
            "__torch__.xsigma.classes.profiler._RecordFunction x, Future(t) y) -> Future(t)",
            [](xsigma::Stack& stack)
            {
                // Pop inputs, which should be a future and a PythonRecordFunction
                auto fut         = xsigma::jit::pop(stack).toFuture();
                auto tensor      = xsigma::jit::pop(stack).toCustomClass<PythonRecordFunction>();
                auto profiledFut = _call_end_callbacks_on_fut_new(tensor, fut);
                // return future that completes when profiling callbacks have run.
                xsigma::jit::push(stack, std::move(profiledFut));
            },
            xsigma::AliasAnalysisKind::FROM_SCHEMA));
}

}  // namespace xsigma::autograd::profiler
#endif
