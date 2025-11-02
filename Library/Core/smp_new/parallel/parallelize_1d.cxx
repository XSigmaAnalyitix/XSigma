#include "smp_new/parallel/parallelize_1d.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include "smp_new/core/thread_pool.h"

namespace xsigma::smp_new::parallel
{

namespace internal
{

// ============================================================================
// WorkStealingDeque Implementation
// ============================================================================

void WorkStealingDeque::push_back(const WorkItem& item)
{
    std::lock_guard<std::mutex> lock(mutex_);
    items_.push_back(item);
}

bool WorkStealingDeque::pop_back(WorkItem& item)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (items_.empty())
    {
        return false;
    }
    item = items_.back();
    items_.pop_back();
    return true;
}

bool WorkStealingDeque::steal_front(WorkItem& item)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (items_.empty())
    {
        return false;
    }
    item = items_.front();
    items_.pop_front();
    return true;
}

bool WorkStealingDeque::empty() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.empty();
}

size_t WorkStealingDeque::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.size();
}

// ============================================================================
// Parallelize1DCoordinator Implementation
// ============================================================================

Parallelize1DCoordinator::Parallelize1DCoordinator(
    const std::function<void(size_t)>& function,
    size_t range)
    : function_(function), range_(range)
{
    // Create work-stealing deques for each thread
    // Use default number of threads
    size_t num_threads = core::TaskThreadPoolBase::DefaultNumThreads();
    deques_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
    {
        deques_.push_back(std::make_unique<WorkStealingDeque>());
    }
    pending_work_.store(0);
    all_work_queued_.store(false);
}

void Parallelize1DCoordinator::execute()
{
    if (range_ == 0)
    {
        return;
    }

    size_t num_threads = deques_.size();

    // Distribute work evenly across threads
    // Each thread gets a chunk of work to start with
    size_t chunk_size = (range_ + num_threads - 1) / num_threads;

    for (size_t i = 0; i < num_threads; ++i)
    {
        size_t begin = i * chunk_size;
        if (begin >= range_)
        {
            break;
        }
        size_t end = std::min(begin + chunk_size, range_);
        deques_[i]->push_back({begin, end});
        pending_work_++;
    }

    all_work_queued_.store(true);

    // Get the thread pool and submit worker threads
    auto pool = core::CreateThreadPool(static_cast<int>(num_threads));

    for (size_t i = 0; i < num_threads; ++i)
    {
        pool->Run([this, i]() { worker_thread_func(*this, i); });
    }

    // Wait for all work to complete
    pool->WaitWorkComplete();

    // Check for exceptions
    if (exception_)
    {
        std::rethrow_exception(exception_);
    }
}

void Parallelize1DCoordinator::mark_complete()
{
    std::unique_lock<std::mutex> lock(completion_mutex_);
    pending_work_--;
    if (pending_work_ == 0)
    {
        completion_cv_.notify_all();
    }
}

void Parallelize1DCoordinator::wait_complete()
{
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [this]() { return pending_work_ == 0; });
}

void Parallelize1DCoordinator::set_exception(std::exception_ptr ex)
{
    std::lock_guard<std::mutex> lock(exception_mutex_);
    if (!exception_)
    {
        exception_ = ex;
    }
}

// ============================================================================
// Worker Thread Function
// ============================================================================

void worker_thread_func(
    Parallelize1DCoordinator& coordinator,
    size_t thread_id)
{
    const auto& function = coordinator.get_function();
    size_t num_threads = coordinator.get_num_threads();

    try
    {
        // Process work from own deque
        WorkItem item;
        while (coordinator.get_deque(thread_id).pop_back(item))
        {
            for (size_t i = item.begin; i < item.end; ++i)
            {
                function(i);
            }
            coordinator.mark_complete();
        }

        // Try to steal work from other threads
        bool found_work = true;
        while (found_work)
        {
            found_work = false;

            // Try to steal from each other thread
            for (size_t i = 0; i < num_threads; ++i)
            {
                if (i == thread_id)
                {
                    continue;
                }

                if (coordinator.get_deque(i).steal_front(item))
                {
                    found_work = true;

                    // Split the stolen work
                    size_t mid = (item.begin + item.end) / 2;
                    if (mid > item.begin)
                    {
                        // Put second half back for other threads to steal
                        coordinator.get_deque(i).push_back({mid, item.end});
                        item.end = mid;
                    }

                    // Process first half
                    for (size_t j = item.begin; j < item.end; ++j)
                    {
                        function(j);
                    }
                    coordinator.mark_complete();
                    break;
                }
            }
        }
    }
    catch (...)
    {
        coordinator.set_exception(std::current_exception());
    }
}

}  // namespace internal

// ============================================================================
// Public API
// ============================================================================

void parallelize_1d(
    const std::function<void(size_t)>& function,
    size_t range,
    uint32_t flags)
{
    if (range == 0)
    {
        return;
    }

    // Create coordinator and execute
    internal::Parallelize1DCoordinator coordinator(function, range);
    coordinator.execute();
}

}  // namespace xsigma::smp_new::parallel

