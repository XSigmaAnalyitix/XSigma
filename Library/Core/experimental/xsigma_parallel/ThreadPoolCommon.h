#ifndef CAFFE2_UTILS_THREADPOOL_COMMON_H_
#define CAFFE2_UTILS_THREADPOOL_COMMON_H_

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// caffe2 depends upon NNPACK, which depends upon this threadpool, so
// unfortunately we can't reference core/common.h here

// This is copied from core/common.h's definition of XSIGMA_MOBILE
// Define enabled when building for iOS or Android devices
#if defined(__ANDROID__)
#define XSIGMA_ANDROID 1
#elif (defined(__APPLE__) && (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define XSIGMA_IOS 1
#endif  // ANDROID / IOS

#endif  // CAFFE2_UTILS_THREADPOOL_COMMON_H_
