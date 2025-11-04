// TODO: Files do not exist - need to be created or removed
// #include "experimental/xsigma_parallel/Config.h"
// #include "experimental/xsigma_parallel/core/jit_type.h"
#if AT_PARALLEL_OPENMP
#include <atomic>

#include "experimental/xsigma_parallel/Parallel.h"
#include "experimental/xsigma_parallel/ParallelFuture.h"
#include "logging/logger.h"

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#if AT_MKLDNN_ENABLED()
// #include "experimental/xsigma_parallel/native/mkldnn/IDeepRegistration.h"
#endif

#include "experimental/xsigma_parallel/pthreadpool-cpp.h"

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
#if defined(_OPENMP) && AT_MKL_ENABLED() && !AT_MKL_SEQUENTIAL()
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
#if AT_MKL_ENABLED()
    mkl_set_num_threads_local(nthreads);

    // because PyTorch uses OpenMP outside of MKL invocations
    // as well, we want this flag to be false, so that
    // threads aren't destroyed and recreated across every
    // MKL / non-MKL boundary of OpenMP usage
    // See https://github.com/pytorch/pytorch/issues/13757
    mkl_set_dynamic(false);
#endif
#ifdef USE_PTHREADPOOL
    // because PyTorch uses caffe2::pthreadpool() in QNNPACK
    caffe2::pthread_pool* const pool = caffe2::pthreadpool(nthreads);
    if (!pool)
    {
        XSIGMA_LOG_ERROR("Invalid thread pool!");
        return;
    }
#endif
#if AT_MKLDNN_ENABLED()
    xsigma::native::mkldnn::clear_computation_cache();
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

xsigma::intrusive_ptr<xsigma::ivalue::Future> intraop_launch_future(
    const std::function<void()>& func)
{
    func();
    auto future = xsigma::make_intrusive<xsigma::ivalue::Future>(NoneType::get());
    future->markCompleted();
    return future;
}

}  // namespace xsigma
#endif
