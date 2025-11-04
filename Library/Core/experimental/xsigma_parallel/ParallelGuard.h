#pragma once

#include "common/export.h"
#include "common/macros.h"

namespace xsigma
{

// RAII thread local guard that tracks whether code is being executed in
// `xsigma::parallel_for` or `xsigma::parallel_reduce` loop function.
class XSIGMA_VISIBILITY parallel_guard
{
public:
    static bool is_enabled();

    parallel_guard(bool state);
    ~parallel_guard();

private:
    bool previous_state_;
};

}  // namespace xsigma
