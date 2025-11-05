#if XSIGMA_HAS_OPENMP
#include <atomic>

#include "logging/logger.h"
#include "parallel/parallel.h"
#include "parallel/thread_pool.h"

#if XSIGMA_HAS_MKL
#include <mkl.h>
#endif

namespace xsigma
{

namespace
{
// Number of threads set by the user
std::atomic<int> num_threads{-1};
thread_local int this_thread_id{0};

}  // namespace

void init_num_threads()
{
    auto nthreads = num_threads.load();
    if (nthreads > 0)
    {
        set_num_threads(nthreads);
    }
    else
    {
#if defined(_OPENMP) && XSIGMA_HAS_MKL && !XSIGMA_MKL_SEQUENTIAL
        // If we are using MKL an OpenMP make sure the number of threads match.
        // Otherwise, MKL and our OpenMP-enabled functions will keep changing the
        // size of the OpenMP thread pool, resulting in worse performance (and memory
        // leaks in GCC 5.4)
        omp_set_num_threads(mkl_get_max_threads());
#elif defined(_OPENMP)
        omp_set_num_threads(intraop_default_num_threads());
#endif
    }
}

void set_num_threads(int nthreads)
{
    if (nthreads <= 0)
    {
        XSIGMA_LOG_ERROR("Expected positive number of threads, got {}", nthreads);
        return;
    }
    num_threads.store(nthreads);
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
#if XSIGMA_HAS_MKL
    mkl_set_num_threads_local(nthreads);

    // because PyTorch uses OpenMP outside of MKL invocations
    // as well, we want this flag to be false, so that
    // threads aren't destroyed and recreated across every
    // MKL / non-MKL boundary of OpenMP usage
    // See https://github.com/pytorch/pytorch/issues/13757
    mkl_set_dynamic(false);
#endif
}

// Explicitly calling omp_get_max_threads() as the size of the parallel
// region might be different in the new thread;
// Use init_num_threads() during thread initialization to ensure
// consistent size of parallel region in different threads
int get_num_threads()
{
#ifdef _OPENMP
    xsigma::internal::lazy_init_num_threads();
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int get_thread_num()
{
    return this_thread_id;
}

namespace internal
{
void set_thread_num(int id)
{
    this_thread_id = id;
}
}  // namespace internal

bool in_parallel_region()
{
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return false;
#endif
}

void intraop_launch(const std::function<void()>& func)
{
    // execute inline in openmp case
    func();
}

}  // namespace xsigma
#endif
