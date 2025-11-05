/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Implementation of parallel_guard for tracking parallel execution state
 */

#include "parallel_guard.h"

namespace xsigma
{

namespace
{
// Thread-local state tracking whether we're in a parallel region
thread_local bool in_parallel_region_ = false;
}  // namespace

bool parallel_guard::is_enabled()
{
    return in_parallel_region_;
}

parallel_guard::parallel_guard(bool state) : previous_state_(in_parallel_region_)
{
    in_parallel_region_ = state;
}

parallel_guard::~parallel_guard()
{
    in_parallel_region_ = previous_state_;
}

}  // namespace xsigma
