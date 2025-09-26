
// This file incorporates code from the Visualization Toolkit (VTK) and remains subject to the BSD-3-Clause VTK license.

/**
 * @class threaded_task_queue
 * @brief simple threaded task queue
 *
 * threaded_task_queue provides a simple task queue that can use threads to
 * execute individual tasks. It is intended for use applications such as data
 * compression, encoding etc. where the task may be completed concurrently
 * without blocking the main thread.
 *
 * threaded_task_queue's API is intended to called from the same main thread.
 * The constructor defines the work (or task) to be performed. `Push` allows the
 * caller to enqueue a task with specified input arguments. The call will return
 * immediately without blocking. The task is enqueued and will be executed
 * concurrently when resources become available.  `Pop` will block until the
 * result is available. To avoid waiting for results to be available, use
 * `TryPop`.
 *
 * The constructor allows mechanism to customize the queue. `strict_ordering`
 * implies that results should be popped in the same order that tasks were
 * pushed without dropping any task. If the caller is only concerned with
 * obtaining the latest available result where intermediate results that take
 * longer to compute may be dropped, then `strict_ordering` can be set to `false`.
 *
 * `max_concurrent_tasks` controls how many threads are used to process tasks in
 * the queue. Default is same as
 * `xsigmaMultiThreader::GetGlobalDefaultNumberOfThreads()`.
 *
 * `buffer_size` indicates how many tasks may be queued for processing. Default
 * is infinite size. If a positive number is provided, then pushing additional
 * tasks will result in discarding of older tasks that haven't begun processing
 * from the queue. Note, this does not impact tasks that may already be in
 * progress. Also, if `strict_ordering` is true, this is ignored; the
 * buffer_size will be set to unlimited.
 *
 */

#ifndef xsigmaThreadedTaskQueue_h
#define xsigmaThreadedTaskQueue_h

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#if !defined(__WRAP__)
namespace xsigma
{
template <typename R>
class TaskQueue;

template <typename R>
class ResultQueue;
}  // namespace xsigma

namespace xsigma
{
template <typename R, typename... Args>
class threaded_task_queue
{
public:
    threaded_task_queue(
        std::function<R(Args...)> worker,
        bool                      strict_ordering      = true,
        int                       buffer_size          = -1,
        int                       max_concurrent_tasks = -1);
    ~threaded_task_queue();

    /**
   * Push arguments for the work
   */
    void Push(Args&&... args);

    /**
   * Pop the last result. Returns true on success. May fail if called on an
   * empty queue. This will wait for result to be available.
   */
    bool Pop(R& result);

    /**
   * Attempt to pop without waiting. If not results are available, returns
   * false.
   */
    bool TryPop(R& result);

    /**
   * Returns false if there's some result that may be popped right now or in the
   * future.
   */
    bool IsEmpty() const;

    /**
   * Blocks till the queue becomes empty.
   */
    void Flush();

private:
    threaded_task_queue(const threaded_task_queue&) = delete;
    void operator=(const threaded_task_queue&)      = delete;

    std::function<R(Args...)> Worker;

    std::unique_ptr<xsigmaThreadedTaskQueueInternals::TaskQueue<R>>   Tasks;
    std::unique_ptr<xsigmaThreadedTaskQueueInternals::ResultQueue<R>> Results;

    int                            NumberOfThreads;
    std::unique_ptr<std::thread[]> Threads;
};

template <typename... Args>
class threaded_task_queue<void, Args...>
{
public:
    threaded_task_queue(
        std::function<void(Args...)> worker,
        bool                         strict_ordering      = true,
        int                          buffer_size          = -1,
        int                          max_concurrent_tasks = -1);
    ~threaded_task_queue();

    /**
   * Push arguments for the work
   */
    void Push(Args&&... args);

    /**
   * Returns false if there's some result that may be popped right now or in the
   * future.
   */
    bool IsEmpty() const;

    /**
   * Blocks till the queue becomes empty.
   */
    void Flush();

private:
    threaded_task_queue(const threaded_task_queue&) = delete;
    void operator=(const threaded_task_queue&)      = delete;

    std::function<void(Args...)> Worker;

    std::unique_ptr<xsigmaThreadedTaskQueueInternals::TaskQueue<void>> Tasks;

    std::condition_variable    ResultsCV;
    std::mutex                 NextResultIdMutex;
    std::atomic<std::uint64_t> NextResultId;

    int                            NumberOfThreads;
    std::unique_ptr<std::thread[]> Threads;
};
}  // namespace xsigma

#include "smp/threaded_task_queue.hxx"

#endif  // !defined(__WRAP__)

#endif
// XSIGMA-HeaderTest-Exclude: threaded_task_queue.h
