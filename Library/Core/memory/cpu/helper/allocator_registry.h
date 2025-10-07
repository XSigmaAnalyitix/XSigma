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

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/macros.h"
#include "memory/cpu/allocator.h"
#include "memory/numa.h"

namespace xsigma
{
class process_state;

class allocator_factory
{
public:
    virtual ~allocator_factory() {}

    // Returns true if the factory will create a functionally different
    // sub_allocator for different (legal) values of numa_node.
    virtual bool NumaEnabled() { return false; }

    // Create an Allocator.
    virtual Allocator* CreateAllocator() = 0;

    // Create a sub_allocator. If NumaEnabled() is true, then returned sub_allocator
    // will allocate memory local to numa_node.  If numa_node == NUMANOAFFINITY
    // then allocated memory is not specific to any NUMA node.
    virtual sub_allocator* CreateSubAllocator(int numa_node) = 0;
};

// process_state is defined in a package that cannot be a dependency of
// framework.  This definition allows us to access the one method we need.
class process_state_interface
{
public:
    virtual ~process_state_interface() {}
    virtual Allocator* GetCPUAllocator(int numa_node) = 0;
};

// A singleton registry of AllocatorFactories.
//
// Allocators should be obtained through process_state or cpu_allocator()
// (deprecated), not directly through this interface.  The purpose of this
// registry is to allow link-time discovery of multiple AllocatorFactories among
// which process_state will obtain the best fit at startup.
class allocator_factory_registry
{
public:
    allocator_factory_registry() {}
    ~allocator_factory_registry() {}

    void Register(
        const char*        source_file,
        int                source_line,
        const std::string& name,
        int                priority,
        allocator_factory* factory);

    // Returns 'best fit' Allocator.  Find the factory with the highest priority
    // and return an allocator constructed by it.  If multiple factories have
    // been registered with the same priority, picks one by unspecified criteria.
    Allocator* GetAllocator();

    // Returns 'best fit' sub_allocator.  First look for the highest priority
    // factory that is NUMA-enabled.  If none is registered, fall back to the
    // highest priority non-NUMA-enabled factory.  If NUMA-enabled, return a
    // sub_allocator specific to numa_node, otherwise return a NUMA-insensitive
    // sub_allocator.
    sub_allocator* GetSubAllocator(int numa_node);

    // Returns the singleton value.
    static allocator_factory_registry* singleton();

    process_state_interface* process_state() const
    {
        std::lock_guard<std::mutex> ml(mu_);
        return process_state_;
    }

protected:
    friend class process_state;

    void SetProcessState(process_state_interface* val)
    {
        std::lock_guard<std::mutex> ml(mu_);
        process_state_ = val;
    }

private:
    mutable std::mutex                      mu_;
    process_state_interface* process_state_ XSIGMA_GUARDED_BY(mu_) = nullptr;
    bool                                    first_alloc_made_      = false;
    struct FactoryEntry
    {
        const char*                        source_file = nullptr;
        int                                source_line = 0;
        std::string                        name        = "";
        int                                priority    = 0;
        std::unique_ptr<allocator_factory> factory     = nullptr;
        std::unique_ptr<Allocator>         allocator   = nullptr;
        // Index 0 corresponds to NUMANOAFFINITY, other indices are (numa_node +
        // 1).
        std::vector<std::unique_ptr<xsigma::sub_allocator>> sub_allocators{};
    };
    std::vector<FactoryEntry> factories_ XSIGMA_GUARDED_BY(mu_);

    // Returns any FactoryEntry registered under 'name' and 'priority',
    // or 'nullptr' if none found.
    const FactoryEntry* FindEntry(const std::string& name, int priority) const
        XSIGMA_EXCLUSIVE_LOCKS_REQUIRED(mu_);

    allocator_factory_registry(const allocator_factory_registry&) = delete;
    void operator=(const allocator_factory_registry&)             = delete;
};

class allocator_factory_registration
{
public:
    allocator_factory_registration(
        const char*        file,
        int                line,
        const std::string& name,
        int                priority,
        allocator_factory* factory)
    {
        allocator_factory_registry::singleton()->Register(file, line, name, priority, factory);
    }
};

#define REGISTER_MEM_ALLOCATOR(name, priority, factory) \
    REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(__COUNTER__, __FILE__, __LINE__, name, priority, factory)

#define REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(ctr, file, line, name, priority, factory) \
    REGISTER_MEM_ALLOCATOR_UNIQ(ctr, file, line, name, priority, factory)

#define REGISTER_MEM_ALLOCATOR_UNIQ(ctr, file, line, name, priority, factory) \
    static allocator_factory_registration allocator_factory_reg_##ctr(        \
        file, line, name, priority, new factory)

}  // namespace xsigma
