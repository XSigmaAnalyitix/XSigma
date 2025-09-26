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

#include "memory/cpu/helper/metrics.h"

#include <atomic>
#include <cstdint>
#include <string>

namespace xsigma::monitoring
{

// Simple counter class for metrics
template <size_t NumLabels = 0>
class Counter
{
public:
    static Counter* New(const char* name, const char* description)
    {
        return new Counter(name, description);
    }

    class Cell
    {
    public:
        void IncrementBy(int64_t value) { value_.fetch_add(value, std::memory_order_relaxed); }

    private:
        friend class Counter;
        std::atomic<int64_t> value_{0};
    };

    Cell* GetCell() { return &cell_; }

private:
    Counter(const char* name, const char* description) : name_(name), description_(description) {}

    const std::string name_;
    const std::string description_;
    Cell              cell_;
};

}  // namespace xsigma::monitoring

namespace xsigma::metrics
{
namespace
{

auto* allocator_bfc_delay = monitoring::Counter<0>::New(
    "/memory/cpu/allocator_bfc_delay",
    "The total time spent running each graph "
    "optimization pass in microseconds.");

}  // namespace

void update_allocator_bfc_delay_time(const uint64_t delay_usecs)
{
    static auto* allocator_bfc_delay_cell = allocator_bfc_delay->GetCell();
    if (delay_usecs > 0)
    {
        allocator_bfc_delay_cell->IncrementBy((int64_t)delay_usecs);
    }
}
}  // namespace xsigma::metrics