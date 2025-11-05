#pragma once

#include "parallel/parallel_guard.h"
#include "util/exception.h"
#include "util/small_vector.h"

namespace xsigma
{

template <class F>
inline void parallel_for(
    const int64_t begin, const int64_t end, const int64_t grain_size, const F& f)
{
    XSIGMA_CHECK_DEBUG(grain_size >= 0);
    if (begin >= end)
    {
        return;
    }

#ifdef INTRA_OP_PARALLEL
    xsigma::internal::lazy_init_num_threads();
    const auto numiter = end - begin;
    const bool use_parallel =
        (numiter > grain_size && numiter > 1 && !xsigma::in_parallel_region() &&
         xsigma::get_num_threads() > 1);
    if (!use_parallel)
    {
        internal::thread_id_guard tid_guard(0);
        xsigma::parallel_guard    guard(true);
        f(begin, end);
        return;
    }

    internal::invoke_parallel(
        begin,
        end,
        grain_size,
        [&](int64_t begin, int64_t end)
        {
            xsigma::parallel_guard guard(true);
            f(begin, end);
        });
#else
    internal::thread_id_guard tid_guard(0);
    xsigma::parallel_guard    guard(true);
    f(begin, end);
#endif
}

template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t  begin,
    const int64_t  end,
    const int64_t  grain_size,
    const scalar_t ident,
    const F&       f,
    const SF&      sf)
{
    XSIGMA_CHECK(grain_size >= 0);
    if (begin >= end)
    {
        return ident;
    }

#ifdef INTRA_OP_PARALLEL
    xsigma::internal::lazy_init_num_threads();
    const auto max_threads = xsigma::get_num_threads();
    const bool use_parallel =
        ((end - begin) > grain_size && !xsigma::in_parallel_region() && max_threads > 1);
    if (!use_parallel)
    {
        internal::thread_id_guard tid_guard(0);
        xsigma::parallel_guard    guard(true);
        return f(begin, end, ident);
    }

    xsigma::small_vector<scalar_t, 64> results(max_threads, ident);
    internal::invoke_parallel(
        begin,
        end,
        grain_size,
        [&](const int64_t my_begin, const int64_t my_end)
        {
            const auto             tid = xsigma::get_thread_num();
            xsigma::parallel_guard guard(true);
            results[tid] = f(my_begin, my_end, ident);
        });

    scalar_t result = ident;
    for (auto partial_result : results)
    {
        result = sf(result, partial_result);
    }
    return result;
#else
    internal::thread_id_guard tid_guard(0);
    xsigma::parallel_guard    guard(true);
    return f(begin, end, ident);
#endif
}

}  // namespace xsigma
