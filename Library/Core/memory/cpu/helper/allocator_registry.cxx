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

#include "memory/cpu/helper/allocator_registry.h"

#include <string>

#include "logging/logger.h"
#include "memory/numa.h"

namespace xsigma
{

// static
allocator_factory_registry* allocator_factory_registry::singleton()
{
    static auto* singleton = new allocator_factory_registry;
    return singleton;
}

const allocator_factory_registry::FactoryEntry* allocator_factory_registry::FindEntry(
    const std::string& name, int priority) const
{
    for (const auto& entry : factories_)
    {
        if (name == entry.name && priority == entry.priority)
        {
            return &entry;
        }
    }
    return nullptr;
}

XSIGMA_NO_SANITIZE_MEMORY
void allocator_factory_registry::Register(
    const char*        source_file,
    int                source_line,
    const std::string& name,
    int                priority,
    allocator_factory* factory)
{
    std::lock_guard<std::mutex> l(mu_);
    XSIGMA_CHECK(
        !first_alloc_made_,
        "Attempt to register an allocator_factory ",
        "after call to GetAllocator()");
    XSIGMA_CHECK(!name.empty(), "Need a valid name for Allocator");
    XSIGMA_CHECK(priority >= 0, "Priority needs to be non-negative");

    const FactoryEntry* existing = FindEntry(name, priority);
    if (existing != nullptr)
    {
        XSIGMA_LOG_ERROR(
            "New registration for allocator_factory with name={} priority={} at location {}:{} "
            "conflicts with previous registration at location {}:{}",
            name,
            priority,
            source_file,
            source_line,
            existing->source_file,
            existing->source_line);
    }
    FactoryEntry entry{};
    entry.source_file = source_file;
    entry.source_line = source_line;
    entry.name        = name;
    entry.priority    = priority;
    entry.factory.reset(factory);
    factories_.push_back(std::move(entry));
}

Allocator* allocator_factory_registry::GetAllocator()
{
    std::lock_guard<std::mutex> l(mu_);
    first_alloc_made_        = true;
    FactoryEntry* best_entry = nullptr;
    for (auto& entry : factories_)
    {
        if (best_entry == nullptr || entry.priority > best_entry->priority)
        {
            best_entry = &entry;
        }
    }

    if (best_entry)
    {
        if (!best_entry->allocator)
        {
            best_entry->allocator.reset(best_entry->factory->CreateAllocator());
        }

        return best_entry->allocator.get();
    }

    XSIGMA_LOG_ERROR("No registered CPU allocator_factory");
    return nullptr;
}

sub_allocator* allocator_factory_registry::GetSubAllocator(int numa_node)
{
    std::lock_guard<std::mutex> l(mu_);
    first_alloc_made_        = true;
    FactoryEntry* best_entry = nullptr;
    for (auto& entry : factories_)
    {
        if (best_entry == nullptr)
        {
            best_entry = &entry;  //NOLINT
        }
        else if (best_entry->factory->NumaEnabled())
        {
            if (entry.factory->NumaEnabled() && (entry.priority > best_entry->priority))
            {
                best_entry = &entry;  //NOLINT
            }
        }
        else
        {
            XSIGMA_CHECK_DEBUG(!best_entry->factory->NumaEnabled());
            if (entry.factory->NumaEnabled() || (entry.priority > best_entry->priority))
            {
                best_entry = &entry;  //NOLINT
            }
        }
    }
    if (best_entry)
    {
        int index = 0;
#ifdef XSIGMA_NUMA_ENABLED
        if (numa_node != -1)
        {
            XSIGMA_CHECK_DEBUG(numa_node <= xsigma::NUMANumNodes());  //NOLINT
            index = 1 + numa_node;
        }
#endif                                                                           // DEBUG
        if (best_entry->sub_allocators.size() < static_cast<size_t>(index + 1))  //NOLINT
        {
            best_entry->sub_allocators.resize(index + 1);
        }
        if (best_entry->sub_allocators[index] == nullptr)
        {
            best_entry->sub_allocators[index].reset(
                best_entry->factory->CreateSubAllocator(numa_node));
        }
        return best_entry->sub_allocators[index].get();
    }
    else
    {
        XSIGMA_LOG_ERROR("No registered CPU allocator_factory");
        return nullptr;
    }
}

}  // namespace xsigma
