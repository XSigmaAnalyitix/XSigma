#include "experimental/xsigma_parallel/thread_pool_guard.h"

namespace caffe2 {

static thread_local bool no_pthread_pool_guard_enabled = false;

bool no_pthread_pool_guard::is_enabled() {
  return no_pthread_pool_guard_enabled;
}

void no_pthread_pool_guard::set_enabled(bool enabled) {
  no_pthread_pool_guard_enabled = enabled;
}

}  // namespace caffe2
