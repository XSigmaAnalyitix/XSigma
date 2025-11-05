// TODO: File does not exist - needs to be created or removed
// #include "Config.h"
#if !XSIGMA_HAS_OPENMP
#include <atomic>
#include <utility>

#include "logging/logger.h"
#include "parallel.h"
#include "thread_pool.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if XSIGMA_HAS_MKL
#include <mkl.h>
#endif

namespace xsigma
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
    static std::shared_ptr<xsigma::task_thread_pool_base> const pool = ThreadPoolRegistry()->run(
        "C10",
        /* device_id */ 0,
        /* pool_size */ _num_pool_threads(num_intraop_threads.exchange(CONSUMED)),
        /* create_new */ true);  // create a separate thread pool for intra-op
    return *pool;
}

// Run lambda function `fn` over `task_id` in [0, `range`) with threadpool.
// `fn` will be called with params: task_id.
void _run_with_pool(const std::function<void(size_t)>& fn, size_t range)
{
    for (size_t i = 1; i < range; ++i)
    {
        _get_intraop_pool().run([fn, i]() { fn(i); });
    }
    // Run the first task on the current thread directly.
    fn(0);
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
    auto const num_tasks = static_cast<size_t>(divup((end - begin), chunk_size));
    return std::make_tuple(num_tasks, chunk_size);
}

void invoke_parallel(
    const int64_t                                begin,
    const int64_t                                end,
    const int64_t                                grain_size,
    const std::function<void(int64_t, int64_t)>& f)
{
    xsigma::internal::lazy_init_num_threads();

    size_t num_tasks = 0;
    size_t chunk_size = 0;
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
        auto const local_start = static_cast<int64_t>(begin + (task_id * chunk_size));
        if (local_start < end)
        {
            int64_t const local_end = std::min(end, static_cast<int64_t>(chunk_size + local_start));
            try
            {
                parallel_region_guard const guard(static_cast<int>(task_id));
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
            std::unique_lock<std::mutex> const lk(state.mutex);
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

#if XSIGMA_HAS_MKL
    mkl_set_num_threads(1);
#endif
}

void set_num_threads(int nthreads)
{
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
}

int get_num_threads()
{
    xsigma::internal::lazy_init_num_threads();
    // not initializing pool unnecessarily,
    // because pool cannot be resized after initialization
    int nthreads = num_intraop_threads.load();
    if (nthreads > 0)
    {
        return nthreads;
    }
    if (nthreads == NOT_SET)
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
}

int get_thread_num()
{
    return thread_num_;
}

bool in_parallel_region()
{
    return in_parallel_region_ || (num_intraop_threads.load() == CONSUMED &&
                                   // Needed as intraop_launch() doesn't set in_parallel_region().
                                   _get_intraop_pool().in_thread_pool());
}

void intraop_launch(const std::function<void()>& func)
{
    if (!in_parallel_region() && get_num_threads() > 1)
    {
        _get_intraop_pool().run(func);
    }
    else
    {
        // execute inline if we're in parallel region
        func();
    }
}
}  // namespace xsigma
#endif
