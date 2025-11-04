// TODO: File does not exist - needs to be created or removed
// #include "experimental/xsigma_parallel/Config.h"
#if AT_PARALLEL_NATIVE
#include <atomic>
#include <utility>

#include "experimental/xsigma_parallel/PTThreadPool.h"
#include "experimental/xsigma_parallel/Parallel.h"
#include "experimental/xsigma_parallel/ParallelFuture.h"
#include "logging/logger.h"

#ifndef XSIGMA_MOBILE
#include "experimental/xsigma_parallel/thread_pool.h"
#include "util/irange.h"
#else
#include "experimental/xsigma_parallel/pthreadpool-cpp.h"
#endif  // XSIGMA_MOBILE

#ifdef _OPENMP
#include <omp.h>
#endif

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

namespace at
{
namespace
{
// used with _set_in_parallel_region to mark master thread
// as in parallel region while executing parallel primitives
thread_local bool in_parallel_region_ = false;

// thread number (task_id) set by parallel primitive
thread_local int thread_num_ = 0;

void _set_in_parallel_region(bool in_region)
{
    in_parallel_region_ = in_region;
}

}  // namespace

namespace internal
{
void set_thread_num(int thread_num)
{
    thread_num_ = thread_num;
}
}  // namespace internal

namespace
{
void _unset_thread_num()
{
    thread_num_ = 0;
}

#ifndef XSIGMA_MOBILE

const int NOT_SET  = -1;
const int CONSUMED = -2;

// Number of threads set by the user
// NOT_SET -> positive value -> CONSUMED
// or
// NOT_SET -> CONSUMED
// Meaning:
//  - NOT_SET - pool not initialized, user value is not set
//  - positive value - pool not initialized, user value set
//  - CONSUMED - pool is initialized
std::atomic<int> num_intraop_threads{NOT_SET};

int _num_pool_threads(int nthreads)
{
    if (nthreads == NOT_SET)
    {
        nthreads = intraop_default_num_threads();
    }
    else
    {
        if (nthreads <= 0)
        {
            XSIGMA_LOG_ERROR("Invalid number of threads: {}", nthreads);
            nthreads = 1;
        }
    }
    // minus one because of the master thread
    return nthreads - 1;
}

xsigma::task_thread_pool_base& _get_intraop_pool()
{
    static std::shared_ptr<xsigma::task_thread_pool_base> pool = ThreadPoolRegistry()->Create(
        "C10",
        /* device_id */ 0,
        /* pool_size */ _num_pool_threads(num_intraop_threads.exchange(CONSUMED)),
        /* create_new */ true);  // create a separate thread pool for intra-op
    return *pool;
}

#endif  // XSIGMA_MOBILE

// Run lambda function `fn` over `task_id` in [0, `range`) with threadpool.
// `fn` will be called with params: task_id.
static void _run_with_pool(const std::function<void(size_t)>& fn, size_t range)
{
#ifndef XSIGMA_MOBILE
    for (const auto i : xsigma::irange(1, range))
    {
        _get_intraop_pool().run([fn, i]() { fn(i); });
    }
    // Run the first task on the current thread directly.
    fn(0);
#else
    caffe2::pthread_pool* const pool = caffe2::pthreadpool();
    if (!pool)
    {
        XSIGMA_LOG_ERROR("Invalid thread pool!");
        return;
    }

    pool->run(
        // pthread_pool::run() is blocking.  A std::function [const] reference to
        // this lambda cannot go out of scope before pthread_pool::run() returns.
        [&fn](const size_t task_id) { fn(task_id); },
        range);
#endif  // XSIGMA_MOBILE
}

// RAII guard helps to support in_parallel_region() and get_thread_num() API.
struct parallel_region_guard
{
    parallel_region_guard(int task_id)
    {
        internal::set_thread_num(task_id);
        _set_in_parallel_region(true);
    }
    parallel_region_guard(const parallel_region_guard&)            = delete;
    parallel_region_guard(parallel_region_guard&&)                 = delete;
    parallel_region_guard& operator=(const parallel_region_guard&) = delete;
    parallel_region_guard& operator=(parallel_region_guard&&)      = delete;

    ~parallel_region_guard()
    {
        _set_in_parallel_region(false);
        _unset_thread_num();
    }
};

}  // namespace

namespace internal
{

static std::tuple<size_t, size_t> calc_num_tasks_and_chunk_size(
    int64_t begin, int64_t end, int64_t grain_size)
{
    if ((end - begin) < grain_size)
    {
        return std::make_tuple(1, std::max((int64_t)0, end - begin));
    }
    // Choose number of tasks based on grain size and number of threads.
    int64_t chunk_size = divup((end - begin), get_num_threads());
    // Make sure each task is xsigma least grain_size size.
    chunk_size       = std::max(grain_size, chunk_size);
    size_t num_tasks = static_cast<size_t>(divup((end - begin), chunk_size));
    return std::make_tuple(num_tasks, chunk_size);
}

void invoke_parallel(
    const int64_t                                begin,
    const int64_t                                end,
    const int64_t                                grain_size,
    const std::function<void(int64_t, int64_t)>& f)
{
    xsigma::internal::lazy_init_num_threads();

    size_t num_tasks = 0, chunk_size = 0;
    std::tie(num_tasks, chunk_size) =
        internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);

    struct
    {
        std::atomic_flag        err_flag = ATOMIC_FLAG_INIT;
        std::exception_ptr      eptr;
        std::mutex              mutex;
        std::atomic_size_t      remaining{0};
        std::condition_variable cv;
    } state;

    auto task = [f, &state, begin, end, chunk_size](size_t task_id)
    {
        int64_t local_start = static_cast<int64_t>(begin + task_id * chunk_size);
        if (local_start < end)
        {
            int64_t local_end = std::min(end, static_cast<int64_t>(chunk_size + local_start));
            try
            {
                parallel_region_guard guard(static_cast<int>(task_id));
                f(local_start, local_end);
            }
            catch (...)
            {
                if (!state.err_flag.test_and_set())
                {
                    state.eptr = std::current_exception();
                }
            }
        }
        {
            std::unique_lock<std::mutex> lk(state.mutex);
            if (--state.remaining == 0)
            {
                state.cv.notify_one();
            }
        }
    };
    state.remaining = num_tasks;
    _run_with_pool(std::move(task), num_tasks);

    // Wait for all tasks to finish.
    {
        std::unique_lock<std::mutex> lk(state.mutex);
        if (state.remaining != 0)
        {
            state.cv.wait(lk);
        }
    }
    if (state.eptr)
    {
        std::rethrow_exception(state.eptr);
    }
}

}  // namespace internal

void init_num_threads()
{
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif

#if AT_MKL_ENABLED()
    mkl_set_num_threads(1);
#endif

#ifdef XSIGMA_MOBILE
    caffe2::pthreadpool();
#endif
}

void set_num_threads(int nthreads)
{
#ifndef XSIGMA_MOBILE
    if (nthreads <= 0)
    {
        XSIGMA_LOG_ERROR("Expected positive number of threads, got {}", nthreads);
        return;
    }
    int no_value = NOT_SET;
    if (!num_intraop_threads.compare_exchange_strong(no_value, nthreads))
    {
        // num_intraop_threads either stores a positive integer or CONSUMED,
        // check that requested size is the same as the current one
        int stored_nthreads = num_intraop_threads.load();
        if (stored_nthreads <= 0)
        {
            // plus one because of master thread
            stored_nthreads = static_cast<int>(_get_intraop_pool().size() + 1);
        }
        if (stored_nthreads != nthreads)
        {
            XSIGMA_LOG_WARNING(
                "Cannot set number of intraop threads "
                "after parallel work has started or after set_num_threads call "
                "when using native parallel backend");
        }
    }
#else
    caffe2::pthread_pool* const pool = caffe2::pthreadpool();
    if (!pool)
    {
        XSIGMA_LOG_ERROR("Invalid thread pool!");
        return;
    }
    pool->set_thread_count(nthreads);
#endif  // XSIGMA_MOBILE
}

int get_num_threads()
{
    at::internal::lazy_init_num_threads();
#ifndef XSIGMA_MOBILE
    // not initializing pool unnecessarily,
    // because pool cannot be resized after initialization
    int nthreads = num_intraop_threads.load();
    if (nthreads > 0)
    {
        return nthreads;
    }
    else if (nthreads == NOT_SET)
    {
        return intraop_default_num_threads();
    }
    else
    {
        if (nthreads != CONSUMED)
        {
            XSIGMA_LOG_ERROR("Unexpected thread count state: {}", nthreads);
        }
        return static_cast<int>(_get_intraop_pool().size() + 1);
    }
#else
    caffe2::pthread_pool* const pool = caffe2::pthreadpool();
    if (!pool)
    {
        XSIGMA_LOG_ERROR("Invalid thread pool!");
        return 1;
    }
    return in_parallel_region() ? 1 /* current thread */ : pool->get_thread_count();
#endif  // XSIGMA_MOBILE
}

int get_thread_num()
{
    return thread_num_;
}

bool in_parallel_region()
{
#ifndef XSIGMA_MOBILE
    return in_parallel_region_ || (num_intraop_threads.load() == CONSUMED &&
                                   // Needed as intraop_launch() doesn't set in_parallel_region().
                                   _get_intraop_pool().inThreadPool());
#else
    return in_parallel_region_;
#endif  // XSIGMA_MOBILE
}

void intraop_launch(const std::function<void()>& func)
{
#ifndef XSIGMA_MOBILE
    if (!in_parallel_region() && get_num_threads() > 1)
    {
        _get_intraop_pool().run(func);
    }
    else
    {
        // execute inline if we're in parallel region
        func();
    }
#else
    // TODO: caffe2::PThreadPool only provides a data-parallel API.
    // Task parallelism is not currently supported.
    func();
#endif  // XSIGMA_MOBILE
}

xsigma::intrusive_ptr<xsigma::ivalue::Future> intraop_launch_future(
    const std::function<void()>& func)
{
#ifndef XSIGMA_MOBILE
    auto future = xsigma::make_intrusive<xsigma::ivalue::Future>(xsigma::NoneType::get());
    if (!in_parallel_region() && get_num_threads() > 1)
    {
        _get_intraop_pool().run(
            [func, future]()
            {
                func();
                future->markCompleted();
            });
    }
    else
    {
        func();
        future->markCompleted();
    }
    return future;
#else
    // TODO: caffe2::PThreadPool only provides a data-parallel API.
    // Task parallelism is not currently supported.
    auto future = xsigma::make_intrusive<xsigma::ivalue::Future>(xsigma::dynT<NoneType>());
    func();
    future->markCompleted();
    return future;
#endif  // XSIGMA_MOBILE
}

}  // namespace at
#endif
