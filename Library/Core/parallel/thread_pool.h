#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "common/export.h"
#include "memory/numa.h"
#include "smp/Advanced/thread_name.h"
#include "util/registry.h"

namespace xsigma
{

class XSIGMA_VISIBILITY task_thread_pool_base
{
public:
    virtual void run(std::function<void()> func) = 0;

    virtual size_t size() const = 0;

    /**
   * The number of available (i.e. idle) threads in this thread pool.
   */
    virtual size_t num_available() const = 0;

    /**
   * Check if the current thread is from the thread pool.
   */
    virtual bool in_thread_pool() const = 0;

    virtual ~task_thread_pool_base() noexcept = default;

    static size_t default_num_threads();
};

class XSIGMA_VISIBILITY thread_pool : public xsigma::task_thread_pool_base
{
protected:
    struct task_element_t
    {
        bool run_with_id;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
        const std::function<void()> no_id;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
        const std::function<void(std::size_t)> with_id;

        explicit task_element_t(std::function<void()> f)
            : run_with_id(false), no_id(std::move(f)), with_id(nullptr)
        {
        }
        explicit task_element_t(std::function<void(std::size_t)> f)
            : run_with_id(true), no_id(nullptr), with_id(std::move(f))
        {
        }
    };

    std::queue<task_element_t> tasks_;
    std::vector<std::thread>   threads_;
    mutable std::mutex         mutex_;
    std::condition_variable    condition_;
    std::condition_variable    completed_;
    std::atomic_bool           running_;
    bool                       complete_;
    std::size_t                available_;
    std::size_t                total_;
    int                        numa_node_id_;

public:
    thread_pool() = delete;

    explicit thread_pool(
        int pool_size, int numa_node_id = -1, const std::function<void()>& init_thread = nullptr);

    ~thread_pool() override;

    size_t size() const override;

    size_t num_available() const override;

    bool in_thread_pool() const override;

    void run(std::function<void()> func) override;

    template <typename Task>
    void run_task_with_id(Task task)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        // Set task and signal condition variable so that a worker thread will
        // wake up and use the task.
        tasks_.emplace(static_cast<std::function<void(std::size_t)>>(task));
        complete_ = false;
        condition_.notify_one();
    }

    /// @brief Wait for queue to be empty
    void wait_work_complete();

private:
    // @brief Entry point for pool threads.
    void main_loop(std::size_t index);
};

class XSIGMA_VISIBILITY task_thread_pool : public xsigma::thread_pool
{
public:
    explicit task_thread_pool(int pool_size, int numa_node_id = -1)
        : thread_pool(
              pool_size,
              numa_node_id,
              [numa_node_id]()
              {
                  xsigma::detail::smp::Advanced::set_thread_name("XsigmaTaskThread");
#if XSIGMA_HAS_NUMA
                  NUMABind(numa_node_id);
#endif
              })
    {
    }
};

XSIGMA_DECLARE_SHARED_REGISTRY(ThreadPoolRegistry, task_thread_pool_base, int, int, bool);

}  // namespace xsigma
