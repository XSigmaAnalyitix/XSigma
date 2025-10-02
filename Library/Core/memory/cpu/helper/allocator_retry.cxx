/*
 * XSigma: High-Performance Quantitative Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 XSigma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of XSigma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for XSigma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include "memory/cpu/helper/allocator_retry.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>

#include "common/macros.h"
#include "memory/cpu/helper/metrics.h"

namespace xsigma
{
namespace
{
class scoped_time_tracker
{
public:
    XSIGMA_DELETE_COPY(scoped_time_tracker);

    scoped_time_tracker()                                 = default;
    scoped_time_tracker(scoped_time_tracker&&)            = delete;
    scoped_time_tracker& operator=(scoped_time_tracker&&) = delete;

    // Start tracking time if not already tracking
    void enable()
    {
        if (!start_us_)
        {  // Only override start_us when not set yet
            start_us_ = std::chrono::steady_clock::now();
        }
    }

    // Destructor updates metrics with elapsed time
    ~scoped_time_tracker()
    {
        if (start_us_)
        {
            const auto end_us = std::chrono::steady_clock::now();
            const auto elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_us - *start_us_).count();

            metrics::update_allocator_bfc_delay_time(elapsed_us);
        }
    }

private:
    std::optional<std::chrono::steady_clock::time_point> start_us_;
};
}  // namespace

allocator_retry::allocator_retry() = default;

allocator_retry::~allocator_retry()
{
    // Lock the mutex to make sure that all memory effects are safely published
    // and available to a thread running the destructor.
    std::unique_lock<std::mutex> l(mu_);
}

void* allocator_retry::allocate_raw(
    std::function<void*(size_t alignment, size_t num_bytes, bool verbose_failure)> alloc_func,
    int    max_millis_to_wait,
    size_t alignment,
    size_t num_bytes)
{
    if (num_bytes == 0)
    {
        return nullptr;
    }

    scoped_time_tracker tracker;

    using Clock         = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::milliseconds(max_millis_to_wait);
    bool       first    = true;
    void*      ptr      = nullptr;

    while (ptr == nullptr)
    {
        // Try allocation without holding the lock
        ptr = alloc_func(alignment, num_bytes, /*verbose_failure=*/false);

        if (ptr == nullptr)
        {
            auto now = Clock::now();

            // Initialize deadline on first failure
            if (first)
            {
                first = false;
            }

            if (now < deadline)
            {
                tracker.enable();
                // Wait for memory to become available
                std::unique_lock<std::mutex> lock(mu_);

                memory_returned_.wait_until(lock, deadline);
            }
            else
            {
                // Final attempt with verbose failure
                return alloc_func(alignment, num_bytes, /*verbose_failure=*/true);
            }
        }
    }

    return ptr;
}
}  // namespace xsigma
