#include "experimental/xsigma_parallel/thread_name.h"

#include <algorithm>
#include <array>

#ifndef __GLIBC_PREREQ
#define __GLIBC_PREREQ(x, y) 0
#endif

#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 12) && !defined(__APPLE__) && !defined(__ANDROID__)
#define XSIGMA_HAS_PTHREAD_SETNAME_NP
#endif

#ifdef XSIGMA_HAS_PTHREAD_SETNAME_NP
#include <pthread.h>
#endif

namespace xsigma
{

#ifdef XSIGMA_HAS_PTHREAD_SETNAME_NP
namespace
{
// pthreads has a limit of 16 characters including the null termination byte.
constexpr size_t k_max_thread_name = 15;
}  // namespace
#endif

void set_thread_name(std::string name)
{
#ifdef XSIGMA_HAS_PTHREAD_SETNAME_NP
    name.resize(std::min(name.size(), k_max_thread_name));

    pthread_setname_np(pthread_self(), name.c_str());
#endif
}

std::string get_thread_name()
{
#ifdef XSIGMA_HAS_PTHREAD_SETNAME_NP
    std::array<char, k_max_thread_name + 1> name{};
    pthread_getname_np(pthread_self(), name.data(), name.size());
    return name.data();
#else
    return "";
#endif
}

}  // namespace xsigma
