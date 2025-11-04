#pragma once

#include "common/export.h"
#include "core/impl/local_dispatch_key_set.h"
#include "core/impl/python_dispatcher_tls.h"
#include "core/impl/torch_dispatch_mode_tls.h"
#include "core/inference_mode.h"
// TODO: Files do not exist - need to be created or removed
// #include "experimental/xsigma_parallel/FuncTorchTLS.h"
// #include "experimental/xsigma_parallel/PythonTorchFunctionTLS.h"
// #include "experimental/xsigma_parallel/SavedTensorHooks.h"
// #include "experimental/xsigma_parallel/ThreadLocalPythonObjects.h"
// #include "experimental/xsigma_parallel/record_function.h"
#include "util/exception.h"
#include "util/thread_local_debug_info.h"

namespace at
{

// Thread local state contains values that are preserved across
// thread boundaries (e.g. at::launch/JIT fork, autograd).
// Note at::parallel_for doesn't preserve TLS across thread boundaries.
class XSIGMA_VISIBILITY thread_local_state
{
public:
    // Saves the thread local variables' values and
    // returns them as a thread_local_state
    thread_local_state();

    // set_grad_mode - force the value of the grad mode TLS in
    //  the current state object. This is used for example in the
    //  autograd engine.
    void set_grad_mode(bool enabled);

    // set_multithreading_enabled - force the value of the multithreadinmaximum
    // threads TLS in
    //  the current state object. This is used for example in the
    //  autograd engine.
    void set_multithreading_enabled(bool enabled);

    // Sets thread local variables in the current thread,
    // according to the thread boundary specified
    static void set_thread_local_state(const thread_local_state& state);

private:
    xsigma::impl::LocalDispatchKeySet dispatch_key_;

    // ThreadLocalDebugInfo does not change after being created
    // with DebugInfoGuard
    std::shared_ptr<xsigma::ThreadLocalDebugInfo> debug_info_;

    // RecordFunction TLS
    RecordFunctionTLS rf_tls_;

    // TLS for out-of-tree functorch
    // See NOTE [functorch TLS in pytorch/pytorch] for why this needs to be a
    // pointer (spoiler alert: it's due to the indirection)
    // This needs to be a shared_ptr instead of a unique_ptr because
    // ThreadLocalState is copy-able and does indeed get copied. Maybe we can
    // consider adding an explicit copy constructor for ThreadLocalState in the
    // future but I didn't want to add one just for this.
    std::shared_ptr<const functorch::FuncTorchTLSBase> functorch_tls_;

    // TLS for AutogradModes
    AutogradState autograd_tls_;

    // TLS for enable_torch_dispatch_mode
    xsigma::impl::TorchDispatchModeTLS torch_dispatch_mode_state_;

    // TLS for enable_python_dispatcher
    xsigma::impl::PyInterpreter* python_dispatcher_state_;

    // TLS for __torch_function__ (mode and disable_torch_function)
    at::impl::PythonTorchFunctionTLS python_torch_function_state_;

    // TLS for saved tensors default hooks
    at::impl::SavedTensorDefaultHooksTLS saved_tensors_default_hooks_state_;

    bool functionalization_reapply_views_state_;

    bool dtensor_allow_implicit_replication_;

    // TLS for arbitrary python objects that is registered via hooks
    at::impl::ThreadLocalPythonObjects saved_objects_;

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(XSIGMA_MOBILE) && !defined(BUILD_LITE_INTERPRETER)
    // TLS for autocast dtypes
    std::array<at::ScalarType, at::COMPILE_TIME_MAX_DEVICE_TYPES> autocast_dtypes_{};
#endif

    friend class thread_local_state_guard;
};

// Guard to set and reset the thread local state
class XSIGMA_VISIBILITY thread_local_state_guard
{
public:
    explicit thread_local_state_guard(const thread_local_state& state)
        : prev_state_(thread_local_state())
    {
        // set the given state across the thread boundary
        thread_local_state::set_thread_local_state(state);
    }
    thread_local_state_guard(thread_local_state_guard&& other)           = delete;
    thread_local_state_guard(const thread_local_state_guard&)            = delete;
    thread_local_state_guard& operator=(const thread_local_state_guard&) = delete;
    thread_local_state_guard& operator=(thread_local_state_guard&&)      = delete;

    ~thread_local_state_guard()
    {
        // restore previously set variables
        thread_local_state::set_thread_local_state(prev_state_);
    }

private:
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const thread_local_state prev_state_;
};

template <typename T>
auto wrap_propagate_tls_state(T callback)
{
    return [tls_state = thread_local_state(), callback = std::move(callback)](auto&&... args)
    {
        thread_local_state_guard g(tls_state);
        // Propagate value returned by callback().
        return callback(std::forward<decltype(args)>(args)...);
    };
}

}  // namespace at
