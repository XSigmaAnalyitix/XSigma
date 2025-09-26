

// This file incorporates code from the Visualization Toolkit (VTK) and remains subject to the BSD-3-Clause VTK license.

#pragma once

#include <array>
#include <iterator>
#include <memory>

#include "smp/Common/thread_local_impl_abstract.h"
#include "smp/Common/tools_api.h"  // For GetBackendType(), DefaultBackend
#include "common/macros.h"


#if define(XSIGMA_ENABLE_TBB)
#include "smp/TBB/thread_local_impl.h"
#else
#include "smp/STDThread/thread_local_impl.h"
#endif

namespace xsigma
{
namespace detail
{
namespace smp
{

template <typename T>
class thread_local_api
{
#if XSIGMA_SMP_ENABLE_SEQUENTIAL
    using ThreadLocalSequential = thread_local_impl<BackendType::Sequential, T>;
#endif
#if XSIGMA_SMP_ENABLE_STDTHREAD
    using ThreadLocalSTDThread = thread_local_impl<BackendType::STDThread, T>;
#endif
#if XSIGMA_SMP_ENABLE_TBB
    using ThreadLocalTBB = thread_local_impl<BackendType::TBB, T>;
#endif
#if XSIGMA_SMP_ENABLE_OPENMP
    using ThreadLocalOpenMP = thread_local_impl<BackendType::OpenMP, T>;
#endif

    using ItImplAbstract = typename thread_local_impl_abstract<T>::ItImpl;

public:
    //--------------------------------------------------------------------------------
    thread_local_api()
    {
#if XSIGMA_SMP_ENABLE_SEQUENTIAL
        this->BackendsImpl[static_cast<int>(BackendType::Sequential)] =
            std::make_unique<ThreadLocalSequential>();
#endif
#if XSIGMA_SMP_ENABLE_STDTHREAD
        this->BackendsImpl[static_cast<int>(BackendType::STDThread)] =
            std::make_unique<ThreadLocalSTDThread>();
#endif
#if XSIGMA_SMP_ENABLE_TBB
        this->BackendsImpl[static_cast<int>(BackendType::TBB)] = std::make_unique<ThreadLocalTBB>();
#endif
#if XSIGMA_SMP_ENABLE_OPENMP
        this->BackendsImpl[static_cast<int>(BackendType::OpenMP)] =
            std::make_unique<ThreadLocalOpenMP>();
#endif
    }

    //--------------------------------------------------------------------------------
    explicit thread_local_api(const T& exemplar)
    {
#if XSIGMA_SMP_ENABLE_SEQUENTIAL
        this->BackendsImpl[static_cast<int>(BackendType::Sequential)] =
            std::make_unique<ThreadLocalSequential>(exemplar);
#endif
#if XSIGMA_SMP_ENABLE_STDTHREAD
        this->BackendsImpl[static_cast<int>(BackendType::STDThread)] =
            std::make_unique<ThreadLocalSTDThread>(exemplar);
#endif
#if XSIGMA_SMP_ENABLE_TBB
        this->BackendsImpl[static_cast<int>(BackendType::TBB)] =
            std::make_unique<ThreadLocalTBB>(exemplar);
#endif
#if XSIGMA_SMP_ENABLE_OPENMP
        this->BackendsImpl[static_cast<int>(BackendType::OpenMP)] =
            std::make_unique<ThreadLocalOpenMP>(exemplar);
#endif
    }

    //--------------------------------------------------------------------------------
    T& Local()
    {
        BackendType backendType = this->GetSMPBackendType();
        return this->BackendsImpl[static_cast<int>(backendType)]->Local();
    }

    //--------------------------------------------------------------------------------
    size_t size()
    {
        BackendType backendType = this->GetSMPBackendType();
        return this->BackendsImpl[static_cast<int>(backendType)]->size();
    }

    //--------------------------------------------------------------------------------
    class iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using reference         = T&;

        iterator() = default;

        iterator(const iterator& other) : ImplAbstract(other.ImplAbstract->Clone()) {}

        iterator& operator=(const iterator& other)
        {
            if (this != &other)
            {
                this->ImplAbstract = other.ImplAbstract->Clone();
            }
            return *this;
        }

        iterator& operator++()
        {
            this->ImplAbstract->Increment();
            return *this;
        }

        iterator operator++(int)
        {
            iterator copy = *this;
            this->ImplAbstract->Increment();
            return copy;
        }

        bool operator==(const iterator& other)
        {
            return this->ImplAbstract->Compare(other.ImplAbstract.get());
        }

        bool operator!=(const iterator& other)
        {
            return !this->ImplAbstract->Compare(other.ImplAbstract.get());
        }

        T& operator*() { return this->ImplAbstract->GetContent(); }

        T* operator->() { return this->ImplAbstract->GetContentPtr(); }

    private:
        std::unique_ptr<ItImplAbstract> ImplAbstract;

        friend class thread_local_api<T>;
    };

    //--------------------------------------------------------------------------------
    iterator begin()
    {
        BackendType backendType = this->GetSMPBackendType();
        iterator    iter;
        iter.ImplAbstract = this->BackendsImpl[static_cast<int>(backendType)]->begin();
        return iter;
    }

    //--------------------------------------------------------------------------------
    iterator end()
    {
        BackendType backendType = this->GetSMPBackendType();
        iterator    iter;
        iter.ImplAbstract = this->BackendsImpl[static_cast<int>(backendType)]->end();
        return iter;
    }

    // disable copying
    thread_local_api(const thread_local_api&)            = delete;
    thread_local_api& operator=(const thread_local_api&) = delete;

private:
    std::array<std::unique_ptr<thread_local_impl_abstract<T>>, XSIGMA_SMP_MAX_BACKENDS_NB>
        BackendsImpl;

    //--------------------------------------------------------------------------------
    BackendType GetSMPBackendType()
    {
        auto& SMPToolsAPI = tools_api::GetInstance();
        return SMPToolsAPI.GetBackendType();
    }
};

}  // namespace smp
}  // namespace detail
}  // namespace xsigma

/* XSIGMA-HeaderTest-Exclude: thread_local_api.h */
