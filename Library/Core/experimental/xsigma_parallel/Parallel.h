#pragma once

#include <functional>
#include <string>

#include "common/export.h"
#include "common/macros.h"
// TODO: File does not exist - needs to be created or removed
// #include "experimental/xsigma_parallel/Config.h"

namespace xsigma
{

inline int64_t divup(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

// Called during new thread initialization
XSIGMA_API void init_num_threads();

// Sets the number of threads to be used in parallel region
XSIGMA_API void set_num_threads(int /*nthreads*/);

// Returns the maximum number of threads that may be used in a parallel region
XSIGMA_API int get_num_threads();

// Returns the current thread number (starting from 0)
// in the current parallel region, or 0 in the sequential region
XSIGMA_API int get_thread_num();

// Checks whether the code runs in parallel region
XSIGMA_API bool in_parallel_region();

namespace internal
{

// Initialise num_threads lazily xsigma first parallel call
inline void lazy_init_num_threads()
{
    thread_local bool init = false;
    if (XSIGMA_UNLIKELY(!init))
    {
        xsigma::init_num_threads();
        init = true;
    }
}

XSIGMA_API void set_thread_num(int /*id*/);

class XSIGMA_VISIBILITY thread_id_guard
{
public:
    thread_id_guard(int new_id) : old_id_(xsigma::get_thread_num()) { set_thread_num(new_id); }

    ~thread_id_guard() { set_thread_num(old_id_); }

private:
    int old_id_;
};

}  // namespace internal

/*
parallel_for

begin: index xsigma which to start applying user function

end: index xsigma which to stop applying user function

grain_size: number of elements per chunk. impacts the degree of parallelization

f: user function applied in parallel to the chunks, signature:
  void f(int64_t begin, int64_t end)

Warning: parallel_for does NOT copy thread local
states from the current thread to the worker threads.
This means for example that Tensor operations CANNOT be used in the
body of your function, only data pointers.
*/
template <class F>
inline void parallel_for(
    const int64_t begin, const int64_t end, const int64_t grain_size, const F& f);

/*
parallel_reduce

begin: index xsigma which to start applying reduction

end: index xsigma which to stop applying reduction

grain_size: number of elements per chunk. impacts number of elements in
intermediate results tensor and degree of parallelization.

ident: identity for binary combination function sf. sf(ident, x) needs to return
x.

f: function for reduction over a chunk. f needs to be of signature scalar_t
f(int64_t partial_begin, int64_t partial_end, scalar_t identify)

sf: function to combine two partial results. sf needs to be of signature
scalar_t sf(scalar_t x, scalar_t y)

For example, you might have a tensor of 10000 entries and want to sum together
all the elements. Parallel_reduce with a grain_size of 2500 will then allocate
an intermediate result tensor with 4 elements. Then it will execute the function
"f" you provide and pass the beginning and end index of these chunks, so
0-2499, 2500-4999, etc. and the combination identity. It will then write out
the result from each of these chunks into the intermediate result tensor. After
that it'll reduce the partial results from each chunk into a single number using
the combination function sf and the identity ident. For a total summation this
would be "+" and 0 respectively. This is similar to tbb's approach [1], where
you need to provide a function to accumulate a subrange, a function to combine
two partial results and an identity.

Warning: parallel_reduce does NOT copy thread local
states from the current thread to the worker threads.
This means for example that Tensor operations CANNOT be used in the
body of your function, only data pointers.

[1] https://software.intel.com/en-us/node/506154
*/
template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t  begin,
    const int64_t  end,
    const int64_t  grain_size,
    const scalar_t ident,
    const F&       f,
    const SF&      sf);

// Returns a detailed string describing parallelization settings
XSIGMA_API std::string get_parallel_info();

// Sets number of threads used for inter-op parallelism
XSIGMA_API void set_num_interop_threads(int /*nthreads*/);

// Returns the number of threads used for inter-op parallelism
XSIGMA_API size_t get_num_interop_threads();

// Launches inter-op parallel task
XSIGMA_API void launch(std::function<void()> func);
namespace internal
{
void launch_no_thread_state(std::function<void()> fn);
}  // namespace internal

// Launches intra-op parallel task
XSIGMA_API void intraop_launch(const std::function<void()>& func);

// Returns number of intra-op threads used by default
XSIGMA_API int intraop_default_num_threads();

}  // namespace xsigma

#if AT_PARALLEL_OPENMP
#include "experimental/xsigma_parallel/ParallelOpenMP.h"  // IWYU pragma: keep
#elif AT_PARALLEL_NATIVE
#include "experimental/xsigma_parallel/ParallelNative.h"  // IWYU pragma: keep
#endif

#include "experimental/xsigma_parallel/Parallel-inl.h"  // IWYU pragma: keep
