#pragma once

#include <cstdint>
#include <functional>

#include "common/macros.h"
#include "smp/xsigma_thread_local.h"
#include "smp_new/parallel/parallelize_1d.h"

namespace xsigma::smp_new::parallel
{

/**
 * @brief Parallel iteration over a range with lambda support.
 *
 * Executes a function in parallel over the range [begin, end).
 * The range is divided into chunks of size grain_size, and each chunk
 * is executed by a different thread.
 *
 * @tparam Functor Lambda or function object type
 * @param begin Start index (inclusive)
 * @param end End index (exclusive)
 * @param grain_size Minimum work per thread. If <= 0, automatically determined.
 * @param f Lambda: void(int64_t begin, int64_t end)
 *
 * @note Thread-safe. Can be called from multiple threads.
 * @note Nested parallelism is supported but may be limited by thread pool size.
 */
template <typename Functor>
void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const Functor& f);

/**
 * @brief Parallel reduction operation.
 *
 * Executes a reduction operation in parallel across multiple threads,
 * combining partial results with a combine function.
 *
 * @tparam T Result type
 * @tparam ReduceFunctor Reduction function type
 * @tparam CombineFunctor Combine function type
 *
 * @param begin Start index (inclusive)
 * @param end End index (exclusive)
 * @param grain_size Minimum work per thread. If <= 0, automatically determined.
 * @param identity Initial value for reduction
 * @param reduce_fn Function: T(int64_t begin, int64_t end, T ident) -> T
 * @param combine_fn Function: T(T a, T b) -> T
 *
 * @return Combined result from all threads
 *
 * @note Thread-safe. Can be called from multiple threads.
 * @note Each thread gets its own partial result initialized with identity.
 */
template <typename T, typename ReduceFunctor, typename CombineFunctor>
T parallel_reduce(
    int64_t               begin,
    int64_t               end,
    int64_t               grain_size,
    const T&              identity,
    const ReduceFunctor&  reduce_fn,
    const CombineFunctor& combine_fn);

/**
 * @brief Launches a task for inter-op parallelism.
 *
 * Executes a task asynchronously on the inter-op thread pool.
 * This is used for parallelism between independent operations.
 *
 * @param fn Task to execute
 *
 * @note Thread-safe.
 * @note The function returns immediately without waiting for the task to complete.
 */
XSIGMA_API void launch(std::function<void()> fn);

/**
 * @brief Launches a task for intra-op parallelism.
 *
 * Executes a task asynchronously on the intra-op thread pool.
 * This is used for parallelism within a single operation.
 *
 * If nested or pool exhausted, executes inline.
 *
 * @param fn Task to execute
 *
 * @note Thread-safe.
 * @note The function returns immediately without waiting for the task to complete.
 */
XSIGMA_API void intraop_launch(std::function<void()> fn);

/**
 * @brief Set the parallel backend.
 *
 * Selects which threading backend to use for parallel operations.
 * Must be called before any parallel work is started.
 *
 * @param backend The backend type (NATIVE, OPENMP, or AUTO)
 *
 * @note Thread-safe.
 * @throws std::runtime_error if the requested backend is not available.
 */
XSIGMA_API void set_backend(int backend);

/**
 * @brief Get the current parallel backend.
 *
 * @return The currently active backend type (0=NATIVE, 1=OPENMP, 2=AUTO)
 */
XSIGMA_API int get_backend();

/**
 * @brief Check if OpenMP backend is available.
 *
 * @return true if OpenMP support is compiled in, false otherwise.
 */
XSIGMA_API bool is_openmp_available();

/**
 * @brief Check if TBB backend is available.
 *
 * @return true if TBB support is compiled in, false otherwise.
 */
XSIGMA_API bool is_tbb_available();

/**
 * @brief Sets the number of threads for intra-op parallelism.
 *
 * This function must be called before any parallel work is started.
 * Once parallel work has started, the number of threads cannot be changed.
 *
 * @param nthreads The number of threads to use. Must be positive.
 *
 * @note Thread-safe. Uses atomic compare-and-swap to ensure thread safety.
 */
XSIGMA_API void set_num_intraop_threads(int nthreads);

/**
 * @brief Gets the number of threads for intra-op parallelism.
 *
 * If the number of threads has not been explicitly set, returns the default
 * number of threads based on the system's hardware concurrency.
 *
 * @return The number of threads in the intra-op thread pool.
 */
XSIGMA_API size_t get_num_intraop_threads();

/**
 * @brief Sets the number of threads for inter-op parallelism.
 *
 * This function must be called before any parallel work is started.
 * Once parallel work has started, the number of threads cannot be changed.
 *
 * @param nthreads The number of threads to use. Must be positive.
 *
 * @note Thread-safe. Uses atomic compare-and-swap to ensure thread safety.
 */
XSIGMA_API void set_num_interop_threads(int nthreads);

/**
 * @brief Gets the number of threads for inter-op parallelism.
 *
 * If the number of threads has not been explicitly set, returns the default
 * number of threads based on the system's hardware concurrency.
 *
 * @return The number of threads in the inter-op thread pool.
 */
XSIGMA_API size_t get_num_interop_threads();

/**
 * @brief Gets the current thread number in a parallel region.
 *
 * Returns the thread ID (0-based) of the current thread if executing
 * in a parallel region, or 0 if not in a parallel region.
 *
 * @return The thread ID (0-based) or 0 if not in parallel region.
 *
 * @note Thread-safe. Can be called from any thread.
 */
XSIGMA_API int get_thread_num();

/**
 * @brief Checks if the current thread is in a parallel region.
 *
 * Returns true if the current thread is executing within a parallel
 * operation (parallel_for, parallel_reduce, etc.), false otherwise.
 *
 * @return true if in a parallel region, false otherwise.
 *
 * @note Thread-safe. Can be called from any thread.
 */
XSIGMA_API bool in_parallel_region();

/**
 * @brief Gets detailed parallelization settings information.
 *
 * Returns a string containing information about the current parallelization
 * configuration, including backend type, thread counts, and other settings.
 *
 * @return A string describing the parallelization settings.
 *
 * @note Thread-safe. Can be called from any thread.
 */
XSIGMA_API std::string get_parallel_info();

}  // namespace xsigma::smp_new::parallel

// Include template implementations
#include "smp_new/parallel/parallel_api.hxx"
