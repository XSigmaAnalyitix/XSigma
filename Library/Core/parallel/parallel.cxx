#include <sstream>
#include <thread>


#include "parallel.h"
#include "thread_pool.h"
#include "logging/logger.h"
#include "util/env.h"

#if XSIGMA_HAS_MKL
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

namespace xsigma
{
namespace
{
std::string get_env_var(const char* var_name, const char* def_value = nullptr)
{
    auto env = xsigma::utils::get_env(var_name);
    return env.has_value() ? env.value() : def_value;
}

size_t get_env_num_threads(const char* var_name, size_t def_value = 0)
{
    try
    {
        if (auto value = xsigma::utils::get_env(var_name))
        {
            int nthreads = std::stoi(value.value());
            if (nthreads <= 0)
            {
                XSIGMA_LOG_WARNING("Invalid thread count: {}", nthreads);
                return def_value;
            }
            return nthreads;
        }
    }
    catch (const std::exception& e)
    {
        std::ostringstream oss;
        oss << "Invalid " << var_name << " variable value, " << e.what();
        XSIGMA_LOG_WARNING("{}", oss.str());
    }
    return def_value;
}

}  // namespace

std::string get_openmp_version()
{
#ifdef _OPENMP
    std::ostringstream ss;
    ss << "OpenMP " << _OPENMP;
    return ss.str();
#else
    return "OpenMP not available";
#endif
}

std::string get_mkl_version()
{
#if XSIGMA_HAS_MKL
    std::ostringstream ss;
    ss << "MKL Version";
    return ss.str();
#else
    return "MKL not available";
#endif
}

std::string get_parallel_info()
{
    std::ostringstream ss;

    ss << "ATen/Parallel:\n\tat::get_num_threads() : " << xsigma::get_num_threads() << '\n';
    ss << "\tat::get_num_interop_threads() : " << xsigma::get_num_interop_threads() << '\n';

#ifdef _OPENMP
    ss << xsigma::get_openmp_version() << '\n';
    ss << "\tomp_get_max_threads() : " << omp_get_max_threads() << '\n';
#endif

#if defined(__x86_64__) || defined(_M_X64)
    ss << xsigma::get_mkl_version() << '\n';
#endif
#if XSIGMA_HAS_MKL
    ss << "\tmkl_get_max_threads() : " << mkl_get_max_threads() << '\n';
#endif
    ss << "std::thread::hardware_concurrency() : " << std::thread::hardware_concurrency() << '\n';

    ss << "Environment variables:" << '\n';
    ss << "\tOMP_NUM_THREADS : " << get_env_var("OMP_NUM_THREADS", "[not set]") << '\n';
#if defined(__x86_64__) || defined(_M_X64)
    ss << "\tMKL_NUM_THREADS : " << get_env_var("MKL_NUM_THREADS", "[not set]") << '\n';
#endif

    ss << "ATen parallel backend: ";
#if XSIGMA_HAS_OPENMP
    ss << "OpenMP";
#elif !XSIGMA_HAS_OPENMP
    ss << "native thread pool";
#endif
    ss << '\n';

#if XSIGMA_HAS_EXPERIMENTAL
    ss << "Experimental: single thread pool" << std::endl;
#endif

    return ss.str();
}

int intraop_default_num_threads()
{
    size_t nthreads = get_env_num_threads("OMP_NUM_THREADS", 0);
    nthreads        = get_env_num_threads("MKL_NUM_THREADS", nthreads);
    if (nthreads == 0)
    {
#if defined(__aarch64__) && defined(__APPLE__)
        // On Apple Silicon there are efficient and performance core
        // Restrict parallel algorithms to performance cores by default
        int32_t num_cores     = -1;
        size_t  num_cores_len = sizeof(num_cores);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &num_cores, &num_cores_len, nullptr, 0) == 0)
        {
            if (num_cores > 1)
            {
                nthreads = num_cores;
                return num_cores;
            }
        }
#endif
        nthreads = xsigma::task_thread_pool_base::default_num_threads();
    }
    return static_cast<int>(nthreads);
}

}  // namespace xsigma
