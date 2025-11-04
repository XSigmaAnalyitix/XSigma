#pragma once

#ifdef USE_PTHREADPOOL

#ifdef USE_INTERNAL_PTHREADPOOL_IMPL
// TODO: Update include path after flattening
// #include <caffe2/utils/threadpool/pthreadpool.h>
#include "experimental/xsigma_parallel/pthreadpool.h"
#else
#include <pthreadpool.h>
#endif

#include <functional>
#include <memory>
#include <mutex>

#include "common/export.h"

namespace caffe2 {

class XSIGMA_VISIBILITY pthread_pool final {
 public:
  explicit pthread_pool(size_t thread_count);
  ~pthread_pool() = default;

  pthread_pool(const pthread_pool&) = delete;
  pthread_pool& operator=(const pthread_pool&) = delete;

  pthread_pool(pthread_pool&&) = delete;
  pthread_pool& operator=(pthread_pool&&) = delete;

  size_t get_thread_count() const;
  void set_thread_count(size_t thread_count);

  // Run, in parallel, function fn(task_id) over task_id in range [0, range).
  // This function is blocking.  All input is processed by the time it returns.
  void run(const std::function<void(size_t)>& fn, size_t range);

 private:
  friend pthreadpool_t pthreadpool_();

  mutable std::mutex mutex_;
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_;
};

// Return a singleton instance of pthread_pool for ATen/TH multithreading.
XSIGMA_API pthread_pool* pthreadpool();
XSIGMA_API pthread_pool* pthreadpool(size_t thread_count);

// Exposes the underlying implementation of PThreadPool.
// Only for use in external libraries so as to unify threading across
// internal (i.e. ATen, etc.) and external (e.g. NNPACK, QNNPACK, XNNPACK)
// use cases.
pthreadpool_t pthreadpool_();

}  // namespace caffe2

#endif /* USE_PTHREADPOOL */
