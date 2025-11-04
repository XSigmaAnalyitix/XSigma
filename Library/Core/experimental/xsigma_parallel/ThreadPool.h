#ifndef CAFFE2_UTILS_THREADPOOL_H_
#define CAFFE2_UTILS_THREADPOOL_H_

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "common/export.h"
#include "experimental/xsigma_parallel/ThreadPoolCommon.h"
// TODO: File does not exist - needs to be created or removed
// #include "experimental/xsigma_parallel/core/common.h"
#include "util/flags.h"

//
// A work-stealing threadpool loosely based off of pthreadpool
//

namespace caffe2
{

struct task;
class workers_pool;

constexpr size_t k_cache_line_size = 64;

// A threadpool with the given number of threads.
// NOTE: the k_cache_line_size alignment is present only for cache
// performance, and is not strictly enforced (for example, when
// the object is created on the heap). Thus, in order to avoid
// misaligned intrinsics, no SSE instructions shall be involved in
// the thread_pool implementation.
// Note: alignas is disabled because some compilers do not deal with
// XSIGMA_API and alignas annotations xsigma the same time.
class XSIGMA_VISIBILITY /*alignas(k_cache_line_size)*/ thread_pool
{
public:
    static thread_pool*                 create_thread_pool(int num_threads);
    static std::unique_ptr<thread_pool> default_thread_pool();
    virtual ~thread_pool() = default;
    // Returns the number of threads currently in use
    virtual int  get_num_threads() const             = 0;
    virtual void set_num_threads(size_t num_threads) = 0;

    // Sets the minimum work size (range) for which to invoke the
    // threadpool; work sizes smaller than this will just be run on the
    // main (calling) thread
    void set_min_work_size(size_t size)
    {
        std::lock_guard<std::mutex> guard(execution_mutex_);
        min_work_size_ = size;
    }

    size_t       get_min_work_size() const { return min_work_size_; }
    virtual void run(const std::function<void(int, size_t)>& fn, size_t range) = 0;

    // Run an arbitrary function in a thread-safe manner accessing the Workers
    // Pool
    virtual void with_pool(const std::function<void(workers_pool*)>& fn) = 0;

protected:
    static size_t      default_num_threads_;
    mutable std::mutex execution_mutex_;
    size_t             min_work_size_;
};

size_t get_default_num_threads();
}  // namespace caffe2

XSIGMA_DECLARE_bool(caffe2_threadpool_force_inline);

// Whether or not threadpool caps apply to Android
XSIGMA_DECLARE_int(caffe2_threadpool_android_cap);

// Whether or not threadpool caps apply to iOS and MacOS
XSIGMA_DECLARE_int(caffe2_threadpool_ios_cap);
XSIGMA_DECLARE_int(caffe2_threadpool_macos_cap);

XSIGMA_DECLARE_int(pthreadpool_size);
#endif  // CAFFE2_UTILS_THREADPOOL_H_
