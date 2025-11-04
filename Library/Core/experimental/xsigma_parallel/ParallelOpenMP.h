#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>

#ifdef _OPENMP
#define INTRA_OP_PARALLEL

#include <omp.h>
#endif

#ifdef _OPENMP
namespace xsigma::internal {

template <typename F>
inline void invoke_parallel(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
  std::atomic<bool> has_error{false};

#pragma omp parallel
  {
    // choose number of tasks based on grain size and number of threads
    // can't use num_threads clause due to bugs in GOMP's thread pool (See
    // #32008)
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end && !has_error.load()) {
      internal::thread_id_guard tid_guard(tid);
      // Note: Error handling removed - function f should handle errors internally
      f(begin_tid, std::min(end, chunk_size + begin_tid));
    }
  }
}

}  // namespace xsigma::internal
#endif  // _OPENMP
