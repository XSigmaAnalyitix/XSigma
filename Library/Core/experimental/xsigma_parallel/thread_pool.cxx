#include "experimental/xsigma_parallel/thread_pool.h"

#include "experimental/xsigma_parallel/thread_name.h"
#include "logging/logger.h"
#if !defined(__powerpc__) && !defined(__s390x__)
#include <cpuinfo.h>
#endif

namespace xsigma
{

size_t task_thread_pool_base::default_num_threads()
{
    size_t num_threads = 0;
#if !defined(__powerpc__) && !defined(__s390x__)
    if (cpuinfo_initialize())
    {
        // In cpuinfo parlance cores are physical ones and processors are virtual
        // thread_pool should be defaulted to number of physical cores
        size_t num_cores = cpuinfo_get_cores_count();
        num_threads      = cpuinfo_get_processors_count();
        if (num_cores > 0 && num_cores < num_threads)
        {
            return num_cores;
        }
        if (num_threads > 0)
        {
            return num_threads;
        }
    }
#endif
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
    {
        num_threads = 1;
    }
    return num_threads;
}

thread_pool::thread_pool(int pool_size, int numa_node_id, const std::function<void()>& init_thread)
    : threads_(pool_size < 0 ? default_num_threads() : pool_size),
      running_(true),
      complete_(true),
      available_(threads_.size()),
      total_(threads_.size()),
      numa_node_id_(numa_node_id)
{
    for (std::size_t i = 0; i < threads_.size(); ++i)
    {
        threads_[i] = std::thread(
            [this, i, init_thread]()
            {
                xsigma::set_thread_name("pt_thread_pool");
                if (init_thread)
                {
                    init_thread();
                }
                this->main_loop(i);
            });
    }
}

thread_pool::~thread_pool()
{
    // Set running flag to false then notify all threads.
    {
        std::unique_lock<std::mutex> lock(mutex_);
        running_ = false;
        condition_.notify_all();
    }

    for (auto& t : threads_)
    {
        try
        {
            t.join();
            // NOLINTNEXTLINE(bugprone-empty-catch)
        }
        catch (const std::exception&)
        {
        }
    }
}

size_t thread_pool::size() const
{
    return threads_.size();
}

size_t thread_pool::num_available() const
{
    std::unique_lock<std::mutex> lock(mutex_);
    return available_;
}

bool thread_pool::in_thread_pool() const
{
    for (auto& thread : threads_)
    {
        if (thread.get_id() == std::this_thread::get_id())
        {
            return true;
        }
    }
    return false;
}

void thread_pool::run(std::function<void()> func)
{
    if (threads_.size() == 0)
    {
        XSIGMA_LOG_ERROR("No threads to run a task");
        return;
    }
    std::unique_lock<std::mutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.emplace(std::move(func));
    complete_ = false;
    condition_.notify_one();
}

void thread_pool::wait_work_complete()
{
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [&]() { return complete_; });
}

void thread_pool::main_loop(std::size_t index)
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (running_)
    {
        // Wait on condition variable while the task is empty and
        // the pool is still running.
        condition_.wait(lock, [&]() { return !tasks_.empty() || !running_; });
        // If pool is no longer running, break out of loop.
        if (!running_)
        {
            break;
        }

        // Copy task locally and remove from the queue.  This is
        // done within its own scope so that the task object is
        // destructed immediately after running the task.  This is
        // useful in the event that the function contains
        // shared_ptr arguments bound via bind.
        {
            task_element_t tasks = std::move(tasks_.front());
            tasks_.pop();
            // Decrement count, indicating thread is no longer available.
            --available_;

            lock.unlock();

            // Run the task.
            try
            {
                if (tasks.run_with_id)
                {
                    tasks.with_id(index);
                }
                else
                {
                    tasks.no_id();
                }
            }
            catch (const std::exception& e)
            {
                XSIGMA_LOG_ERROR("Exception in thread pool task: {}", e.what());
            }
            catch (...)
            {
                XSIGMA_LOG_ERROR("Exception in thread pool task: unknown");
            }

            // Destruct tasks before taking the lock.  As tasks
            // are user provided std::function, they can run
            // arbitrary code during destruction, including code
            // that can reentrantly call into thread_pool (which would
            // cause a deadlock if we were holding the lock).
        }

        // Update status of empty, maybe
        // Need to recover the lock first
        lock.lock();

        // Increment count, indicating thread is available.
        ++available_;
        if (tasks_.empty() && available_ == total_)
        {
            complete_ = true;
            completed_.notify_one();
        }

        // Deliberately hold the lock on the backedge, so this thread has an
        // opportunity to acquire a new task before another thread acquires
        // the lock.
    }  // while running_
}

XSIGMA_DEFINE_SHARED_REGISTRY(ThreadPoolRegistry, task_thread_pool_base, int, int, bool)
}  // namespace xsigma
