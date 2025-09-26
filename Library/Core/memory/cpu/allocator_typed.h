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

#include <limits>
#include <string>

#include "memory/cpu/allocator.h"

namespace xsigma
{

//class Variant;

// Convenience functions to do typed allocation.  C++ constructors
// and destructors are invoked for complex types if necessary.
class allocator_typed
{
public:
    // May return NULL if the tensor has too many elements to represent in a
    // single allocation.
    template <typename T>
    static T* Allocate(
        Allocator* raw_allocator, size_t num_elements, const allocation_attributes& allocation_attr)
    {
        // TODO(jeff): Do we need to allow clients to pass in alignment
        // requirements?

        if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T)))
        {
            return nullptr;
        }

        void* p = raw_allocator->allocate_raw(
            Allocator::kAllocatorAlignment, sizeof(T) * num_elements, allocation_attr);
        T* typed_p = reinterpret_cast<T*>(p);
        if (typed_p)
            RunCtor<T>(raw_allocator, typed_p, num_elements);
        return typed_p;
    }

    template <typename T>
    static void Deallocate(Allocator* raw_allocator, T* ptr, size_t num_elements)
    {
        if (ptr)
        {
            RunDtor<T>(raw_allocator, ptr, num_elements);
            raw_allocator->deallocate_raw(
                ptr, Allocator::kAllocatorAlignment, sizeof(T) * num_elements);
        }
    }

private:
    // No constructors or destructors are run for simple types
    template <typename T>
    static void RunCtor(
        XSIGMA_UNUSED Allocator* raw_allocator, XSIGMA_UNUSED T* p, XSIGMA_UNUSED size_t n)
    {
        static_assert(std::is_trivial<T>::value, "T is not a simple type.");
    }

    template <typename T>
    static void RunDtor(
        XSIGMA_UNUSED Allocator* raw_allocator, XSIGMA_UNUSED T* p, XSIGMA_UNUSED size_t n)
    {
    }

    //static void RunVariantCtor(Variant* p, size_t n);

    //static void RunVariantDtor(Variant* p, size_t n);
};

template <>
/* static */
inline void allocator_typed::RunCtor(Allocator* raw_allocator, std::string* p, size_t n)
{
    if (!raw_allocator->AllocatesOpaqueHandle())
    {
        for (size_t i = 0; i < n; ++p, ++i)
            new (p) std::string();
    }
}

template <>
/* static */
inline void allocator_typed::RunDtor(Allocator* raw_allocator, std::string* p, size_t n)
{
    if (!raw_allocator->AllocatesOpaqueHandle())
    {
        using namespace std;
        for (size_t i = 0; i < n; ++p, ++i)
            p->~string();
    }
}

//template <>
///* static */
//inline void allocator_typed::RunCtor(Allocator* raw_allocator, ResourceHandle* p, size_t n)
//{
//    if (!raw_allocator->AllocatesOpaqueHandle())
//    {
//        for (size_t i = 0; i < n; ++p, ++i)
//            new (p) ResourceHandle();
//    }
//}
//
//template <>
///* static */
//inline void allocator_typed::RunDtor(Allocator* raw_allocator, ResourceHandle* p, size_t n)
//{
//    if (!raw_allocator->AllocatesOpaqueHandle())
//    {
//        for (size_t i = 0; i < n; ++p, ++i)
//            p->~ResourceHandle();
//    }
//}

//template <>
///* static */
//inline void allocator_typed::RunCtor(Allocator* raw_allocator, Variant* p, size_t n)
//{
//    if (!raw_allocator->AllocatesOpaqueHandle())
//    {
//        RunVariantCtor(p, n);
//    }
//}
//
//template <>
///* static */
//inline void allocator_typed::RunDtor(Allocator* raw_allocator, Variant* p, size_t n)
//{
//    if (!raw_allocator->AllocatesOpaqueHandle())
//    {
//        RunVariantDtor(p, n);
//    }
//}

}  // namespace xsigma
