#include <sstream>
#if 0
#ifndef ROCM_ON_WINDOWS
#ifdef XSIGMA_CUDA_USE_NVTX3
#include <nvtx3/nvtx3.hpp>
#else
#include <nvToolsExt.h>
#endif
#else  // ROCM_ON_WINDOWS
#include "util/exception.h"
#endif  // ROCM_ON_WINDOWS
#include <xsigma/cuda/CUDAGuard.h>
#include <xsigma/util/ApproximateClock.h>

#include "profiling/profiler/stubs/base.h"
#include "profiling/util.h"
#include "util/irange.h"

namespace xsigma::profiler::impl {
namespace {

static void cudaCheck(cudaError_t result, const char* file, int line) {
  if (result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": ";
    if (result == cudaErrorInitializationError) {
      // It is common for users to use DataLoader with multiple workers
      // and the autograd profiler. Throw a nice error message here.
      ss << "CUDA initialization error. "
         << "This can occur if one runs the profiler in CUDA mode on code "
         << "that creates a DataLoader with num_workers > 0. This operation "
         << "is currently unsupported; potential workarounds are: "
         << "(1) don't use the profiler in CUDA mode or (2) use num_workers=0 "
         << "in the DataLoader or (3) Don't profile the data loading portion "
         << "of your code. https://github.com/pytorch/pytorch/issues/6313 "
         << "tracks profiler support for multi-worker DataLoader.";
    } else {
      ss << cudaGetErrorString(result);
    }
    XSIGMA_CHECK(false, ss.str());
  }
}
#define XSIGMA_CUDA_CHECK(result) cudaCheck(result, __FILE__, __LINE__);

struct CUDAMethods : public ProfilerStubs {
  void record(
      xsigma::device_option::int_t* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {
    if (device) {
      XSIGMA_CUDA_CHECK(xsigma::cuda::GetDevice(device));
    }
    CUevent_st* cuda_event_ptr{nullptr};
    XSIGMA_CUDA_CHECK(cudaEventCreate(&cuda_event_ptr));
    *event = std::shared_ptr<CUevent_st>(cuda_event_ptr, [](CUevent_st* ptr) {
      XSIGMA_CUDA_CHECK(cudaEventDestroy(ptr));
    });
    auto stream = xsigma::cuda::getCurrentCUDAStream();
    if (cpu_ns) {
      *cpu_ns = xsigma::getTime();
    }
    XSIGMA_CUDA_CHECK(cudaEventRecord(cuda_event_ptr, stream));
  }

  float elapsed(
      const ProfilerVoidEventStub* event_,
      const ProfilerVoidEventStub* event2_) const override {
    auto event = (const ProfilerEventStub*)(event_);
    auto event2 = (const ProfilerEventStub*)(event2_);
    XSIGMA_CUDA_CHECK(cudaEventSynchronize(event->get()));
    XSIGMA_CUDA_CHECK(cudaEventSynchronize(event2->get()));
    float ms = 0;
    XSIGMA_CUDA_CHECK(cudaEventElapsedTime(&ms, event->get(), event2->get()));
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions)
    return ms * 1000.0;
  }

#ifndef ROCM_ON_WINDOWS
  void mark(const char* name) const override {
    ::nvtxMark(name);
  }

  void rangePush(const char* name) const override {
    ::nvtxRangePushA(name);
  }

  void rangePop() const override {
    ::nvtxRangePop();
  }
#else  // ROCM_ON_WINDOWS
  static void printUnavailableWarning() {
    XSIGMA_LOG_WARNING("Warning: roctracer isn't available on Windows");
  }
  void mark(const char* name) const override {
    printUnavailableWarning();
  }
  void rangePush(const char* name) const override {
    printUnavailableWarning();
  }
  void rangePop() const override {
    printUnavailableWarning();
  }
#endif

  void onEachDevice(std::function<void(int)> op) const override {
    xsigma::cuda::OptionalCUDAGuard device_guard;
    for (const auto i : xsigma::irange(xsigma::cuda::device_count())) {
      device_guard.set_index(i);
      op(i);
    }
  }

  void synchronize() const override {
    XSIGMA_CUDA_CHECK(cudaDeviceSynchronize());
  }

  bool enabled() const override {
    return true;
  }
};

struct RegisterCUDAMethods {
  RegisterCUDAMethods() {
    static CUDAMethods methods;
    registerCUDAMethods(&methods);
  }
};
RegisterCUDAMethods reg;

} // namespace
} // namespace xsigma::profiler::impl
#endif  // 0
