#pragma once

#include "common/export.h"
#include "common/macros.h"

namespace caffe2 {

// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct XSIGMA_VISIBILITY no_pthread_pool_guard {
  static bool is_enabled();
  static void set_enabled(bool enabled);

  no_pthread_pool_guard() : prev_mode_(no_pthread_pool_guard::is_enabled()) {
    no_pthread_pool_guard::set_enabled(true);
  }
  ~no_pthread_pool_guard() { no_pthread_pool_guard::set_enabled(prev_mode_); }

 private:
  bool prev_mode_;
};

}  // namespace caffe2
