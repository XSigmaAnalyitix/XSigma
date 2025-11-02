#pragma once

#include <algorithm>
#include <atomic>
#include <vector>

#include "smp_new/core/thread_pool.h"

namespace xsigma::smp_new::parallel
{

// Forward declarations for internal functions
namespace internal
{
void set_thread_num(int thread_id);
void set_in_parallel_region(bool in_region);
core::TaskThreadPoolBase& GetInteropPool();
}  // namespace internal

template <typename Functor>
void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const Functor& f)
{
    if (begin >= end)
    {
        return;
    }

    int64_t n = end - begin;

    // Determine grain size
    if (grain_size <= 0)
    {
        auto num_threads = static_cast<int64_t>(internal::GetInteropPool().Size());
        grain_size       = std::max(1LL, n / (num_threads * 4));
    }

    // If work is small enough, execute serially
    if (grain_size >= n)
    {
        internal::set_in_parallel_region(true);
        internal::set_thread_num(0);
        try
        {
            f(begin, end);
        }
        catch (...)
        {
            internal::set_in_parallel_region(false);
            throw;
        }
        internal::set_in_parallel_region(false);
        return;
    }

    // Parallel execution
    auto&            pool = internal::GetInteropPool();
    std::atomic<int> num_tasks{0};
    std::atomic<int> task_counter{0};

    for (int64_t i = begin; i < end; i += grain_size)
    {
        int64_t chunk_end = std::min(i + grain_size, end);
        int     task_id   = task_counter++;
        num_tasks++;

        pool.Run(
            [&f, i, chunk_end, task_id]()
            {
                internal::set_in_parallel_region(true);
                internal::set_thread_num(task_id);
                try
                {
                    f(i, chunk_end);
                }
                catch (...)
                {
                    internal::set_in_parallel_region(false);
                    throw;
                }
                internal::set_in_parallel_region(false);
            });
    }

    // Wait for all tasks to complete
    pool.WaitWorkComplete();
}

template <typename T, typename ReduceFunctor, typename CombineFunctor>
T parallel_reduce(
    int64_t               begin,
    int64_t               end,
    int64_t               grain_size,
    const T&              identity,
    const ReduceFunctor&  reduce_fn,
    const CombineFunctor& combine_fn)
{
    if (begin >= end)
    {
        return identity;
    }

    int64_t n = end - begin;

    // Determine grain size
    if (grain_size <= 0)
    {
        auto num_threads = static_cast<int64_t>(internal::GetInteropPool().Size());
        grain_size       = std::max(1LL, n / (num_threads * 4));
    }

    // If work is small enough, execute serially
    if (grain_size >= n)
    {
        internal::set_in_parallel_region(true);
        internal::set_thread_num(0);
        try
        {
            return reduce_fn(begin, end, identity);
        }
        catch (...)
        {
            internal::set_in_parallel_region(false);
            throw;
        }
        internal::set_in_parallel_region(false);
    }

    // Calculate number of chunks
    int64_t num_chunks = (n + grain_size - 1) / grain_size;
    std::vector<T> partial_results(num_chunks, identity);
    std::mutex results_mutex;

    // Parallel reduction
    auto&            pool = internal::GetInteropPool();
    std::atomic<int> task_counter{0};

    for (int64_t i = begin; i < end; i += grain_size)
    {
        int64_t chunk_end = std::min(i + grain_size, end);
        int     task_id   = task_counter++;

        pool.Run(
            [&reduce_fn, &partial_results, i, chunk_end, &identity, task_id]()
            {
                internal::set_in_parallel_region(true);
                internal::set_thread_num(task_id);
                try
                {
                    partial_results[task_id] = reduce_fn(i, chunk_end, identity);
                }
                catch (...)
                {
                    internal::set_in_parallel_region(false);
                    throw;
                }
                internal::set_in_parallel_region(false);
            });
    }

    // Wait for all tasks to complete
    pool.WaitWorkComplete();

    // Combine partial results
    T result = identity;
    for (const auto& partial : partial_results)
    {
        result = combine_fn(result, partial);
    }

    return result;
}

}  // namespace xsigma::smp_new::parallel
