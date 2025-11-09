#pragma once

#include "common/macros.h"

/**
 * Android versions with libgnustl incorrectly handle thread_local C++
 * qualifier with composite types. NDK up to r17 version is affected.
 *
 * (A fix landed on Jun 4 2018:
 * https://android-review.googlesource.com/c/toolchain/gcc/+/683601)
 *
 * In such cases, use xsigma::ThreadLocal<T> wrapper
 * which is `pthread_*` based with smart pointer semantics.
 *
 * In addition, convenient macro XSIGMA_DEFINE_TLS_static is available.
 * To define static TLS variable of type std::string, do the following
 * ```
 *  XSIGMA_DEFINE_TLS_static(std::string, str_tls_);
 *  ///////
 *  {
 *    *str_tls_ = "abc";
 *    assert(str_tls_->length(), 3);
 *  }
 * ```
 *
 * (see xsigma/test/util/ThreadLocal_test.cpp for more examples)
 */
#if !defined(XSIGMA_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

#if defined(XSIGMA_ANDROID) && defined(__GLIBCXX__) && __GLIBCXX__ < 20180604
#define XSIGMA_PREFER_CUSTOM_THREAD_LOCAL_STORAGE
#endif  // defined(XSIGMA_ANDROID) && defined(__GLIBCXX__) && __GLIBCXX__ < 20180604

#endif  // !defined(XSIGMA_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

#if defined(XSIGMA_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
#include <errno.h>
#include <pthread.h>

#include <memory>

#include "util/exception.h"
namespace xsigma
{

/**
 * @brief Temporary thread_local C++ qualifier replacement for Android
 * based on `pthread_*`.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>
class ThreadLocal
{
public:
    ThreadLocal()
    {
        pthread_key_create(&key_, [](void* buf) { delete static_cast<Type*>(buf); });
    }

    ~ThreadLocal()
    {
        if (void* current = pthread_getspecific(key_))
        {
            delete static_cast<Type*>(current);
        }

        pthread_key_delete(key_);
    }

    ThreadLocal(const ThreadLocal&)            = delete;
    ThreadLocal& operator=(const ThreadLocal&) = delete;

    Type& get()
    {
        if (void* current = pthread_getspecific(key_))
        {
            return *static_cast<Type*>(current);
        }

        std::unique_ptr<Type> ptr = std::make_unique<Type>();
        if (0 == pthread_setspecific(key_, ptr.get()))
        {
            return *ptr.release();
        }

        int err = errno;
        XSIGMA_CHECK_DEBUG(false, "pthread_setspecific() failed, errno = ", err);
    }

    Type& operator*() { return get(); }

    Type* operator->() { return &get(); }

private:
    pthread_key_t key_;
};

}  // namespace xsigma

#define XSIGMA_DEFINE_TLS_static(Type, Name) static ::xsigma::ThreadLocal<Type> Name

#define XSIGMA_DECLARE_TLS_class_static(Class, Type, Name) static ::xsigma::ThreadLocal<Type> Name

#define XSIGMA_DEFINE_TLS_class_static(Class, Type, Name) ::xsigma::ThreadLocal<Type> Class::Name

#else  // defined(XSIGMA_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

namespace xsigma
{

/**
 * @brief Default thread_local implementation for non-Android cases.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>
class ThreadLocal
{
public:
    using Accessor = Type* (*)();
    explicit ThreadLocal(Accessor accessor) : accessor_(accessor) {}

    ThreadLocal(const ThreadLocal&)                = delete;
    ThreadLocal(ThreadLocal&&) noexcept            = default;
    ThreadLocal& operator=(const ThreadLocal&)     = delete;
    ThreadLocal& operator=(ThreadLocal&&) noexcept = default;
    ~ThreadLocal()                                 = default;

    Type& get() { return *accessor_(); }

    Type& operator*() { return get(); }

    Type* operator->() { return &get(); }

private:
    Accessor accessor_;
};

}  // namespace xsigma

#define XSIGMA_DEFINE_TLS_static(Type, Name) \
    static ::xsigma::ThreadLocal<Type> Name( \
        []()                                 \
        {                                    \
            static thread_local Type var;    \
            return &var;                     \
        })

#define XSIGMA_DECLARE_TLS_class_static(Class, Type, Name) static ::xsigma::ThreadLocal<Type> Name

#define XSIGMA_DEFINE_TLS_class_static(Class, Type, Name) \
    ::xsigma::ThreadLocal<Type> Class::Name(              \
        []()                                              \
        {                                                 \
            static thread_local Type var;                 \
            return &var;                                  \
        })

#endif  // defined(XSIGMA_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
