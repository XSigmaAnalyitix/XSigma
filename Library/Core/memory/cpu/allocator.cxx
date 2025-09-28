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

#include "memory/cpu/allocator.h"

#include <memory>  // for _Simple_types

#include "memory/cpu/allocator_tracking.h"         // for allocator_tracking
#include "memory/cpu/helper/allocator_registry.h"  // for allocator_factory_registry, process_st...
#include "util/strcat.h"                           // for StrCat

namespace xsigma
{

// Note: allocator_stats::debug_string() is now implemented in unified_memory_stats.cxx
// since allocator_stats is a type alias for unified_resource_stats

//constexpr size_t Allocator::kAllocatorAlignment;

// If true, cpu allocator collects full stats.
static bool cpu_allocator_collect_full_stats = false;

void EnableCPUAllocatorFullStats()
{
    cpu_allocator_collect_full_stats = true;
}
bool CPUAllocatorFullStatsEnabled()
{
    return cpu_allocator_collect_full_stats;
}

std::string allocator_attributes::debug_string() const
{
    return xsigma::strings::StrCat(
        "allocator_attributes(on_host=",
        on_host(),
        " nic_compatible=",
        nic_compatible(),
        " gpu_compatible=",
        gpu_compatible(),
        ")");
}

Allocator* allocator_cpu_base()
{
    static Allocator* cpu_alloc = allocator_factory_registry::singleton()->GetAllocator();
    // TODO(tucker): This really seems wrong.  It's only going to be effective on
    // the first call in a process (but the desired effect is associated with a
    // session), and we probably ought to be tracking the highest level Allocator,
    // not the lowest.  Revisit the advertised semantics of the triggering option.
    if (cpu_allocator_collect_full_stats && !cpu_alloc->TracksAllocationSizes())
    {
        cpu_alloc = new allocator_tracking(cpu_alloc, true);
    }
    return cpu_alloc;
}

Allocator* cpu_allocator(int numa_node)
{
    // Correctness relies on devices being created prior to the first call
    // to cpu_allocator, if devices are ever to be created in the process.
    // Device creation in turn triggers process_state creation and the availability
    // of the correct access pointer via this function call.
    static process_state_interface* ps = allocator_factory_registry::singleton()->process_state();
    if (ps != nullptr)
    {
        return ps->GetCPUAllocator(numa_node);
    }
    else
    {
        return allocator_cpu_base();
    }
}

sub_allocator::sub_allocator(
    const std::vector<Visitor>& alloc_visitors, const std::vector<Visitor>& free_visitors)
    : alloc_visitors_(alloc_visitors), free_visitors_(free_visitors)
{
}

void sub_allocator::VisitAlloc(void* ptr, int index, size_t num_bytes)
{
    for (const auto& v : alloc_visitors_)
    {
        v(ptr, index, num_bytes);
    }
}

void sub_allocator::VisitFree(void* ptr, int index, size_t num_bytes)
{
    // Although we don't guarantee any order of visitor application, strive
    // to apply free visitors in reverse order of alloc visitors.
    for (int i = (int)free_visitors_.size() - 1; i >= 0; --i)
    {
        free_visitors_[i](ptr, index, num_bytes);
    }
}
}  // namespace xsigma
