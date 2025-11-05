#include <atomic>
#include <functional>

#include "parallel/parallel.h"
#include "logging/logger.h"
#include "parallel/thread_pool.h"

namespace xsigma
{
namespace
{
const int NOT_SET  = -1;
const int CONSUMED = -2;

// Number of inter-op threads set by the user;
// NOT_SET -> positive value -> CONSUMED
// (CONSUMED - thread pool is initialized)
// or
// NOT_SET -> CONSUMED
std::atomic<int> num_interop_threads{NOT_SET};

// thread pool global instance is hidden,
// users should use xsigma::launch and get/set_num_interop_threads interface
xsigma::task_thread_pool_base& get_pool()
{
    static std::shared_ptr<xsigma::task_thread_pool_base> const pool = ThreadPoolRegistry()->run(
        "XSIGMA",
        /* device_id */ 0,
        /* pool_size */ num_interop_threads.exchange(CONSUMED),
        /* create_new */ true);
    return *pool;
}

// Factory function for ThreadPoolRegistry
std::shared_ptr<xsigma::task_thread_pool_base> create_xsigma_threadpool(
    int device_id, int pool_size, bool create_new)
{
    // For now, the only accepted device id is 0
    if (device_id != 0)
    {
        XSIGMA_LOG_ERROR("Invalid device_id: {}, expected 0", device_id);
        return nullptr;
    }
    // Create new thread pool
    if (!create_new)
    {
        XSIGMA_LOG_ERROR("create_new must be true");
        return nullptr;
    }
    return std::make_shared<xsigma::thread_pool>(pool_size);
}

}  // namespace

XSIGMA_REGISTER_CREATOR(ThreadPoolRegistry, XSIGMA, create_xsigma_threadpool)

void set_num_interop_threads(int nthreads)
{
    if (nthreads <= 0)
    {
        XSIGMA_LOG_ERROR("Expected positive number of threads, got {}", nthreads);
        return;
    }

    int no_value = NOT_SET;
    if (!num_interop_threads.compare_exchange_strong(no_value, nthreads))
    {
        XSIGMA_LOG_ERROR(
            "Error: cannot set number of interop threads after parallel work "
            "has started or set_num_interop_threads called");
    }
}

size_t get_num_interop_threads()
{
    xsigma::internal::lazy_init_num_threads();
    int const nthreads = num_interop_threads.load();
    if (nthreads > 0)
    {
        return nthreads;
    }
    if (nthreads == NOT_SET)
    {
        // return default value
        return xsigma::task_thread_pool_base::default_num_threads();
    }

    return get_pool().size();
}

namespace internal
{
void launch_no_thread_state(std::function<void()> fn)
{
#if XSIGMA_HAS_EXPERIMENTAL
    intraop_launch(std::move(fn));
#else
    get_pool().run(std::move(fn));
#endif
}
}  // namespace internal

void launch(std::function<void()> func)
{
    internal::launch_no_thread_state(std::move(func));
}

}  // namespace xsigma
