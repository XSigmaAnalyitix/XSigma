#pragma once

#include <functional>
#include <memory>

#include "common/export.h"
#include "memory/device.h"
#include "util/strong_type.h"

struct CUevent_st;

namespace xsigma::profiler::impl
{

// ----------------------------------------------------------------------------
// -- Annotation --------------------------------------------------------------
// ----------------------------------------------------------------------------
using ProfilerEventStub     = std::shared_ptr<CUevent_st>;
using ProfilerVoidEventStub = std::shared_ptr<void>;

struct XSIGMA_VISIBILITY ProfilerStubs
{
    virtual void record(
        xsigma::device_option::int_t* device,
        ProfilerVoidEventStub*        event,
        int64_t*                      cpu_ns) const = 0;
    virtual float elapsed(
        const ProfilerVoidEventStub* event, const ProfilerVoidEventStub* event2) const = 0;
    virtual void mark(const char* name) const                                          = 0;
    virtual void rangePush(const char* name) const                                     = 0;
    virtual void rangePop() const                                                      = 0;
    virtual bool enabled() const { return false; }
    virtual void onEachDevice(std::function<void(int)> op) const = 0;
    virtual void synchronize() const                             = 0;
    virtual ~ProfilerStubs()                                     = default;
};

XSIGMA_API void                 registerCUDAMethods(ProfilerStubs* stubs);
XSIGMA_API const ProfilerStubs* cudaStubs();
XSIGMA_API void                 registerITTMethods(ProfilerStubs* stubs);
XSIGMA_API const ProfilerStubs* ittStubs();
XSIGMA_API void                 registerPrivateUse1Methods(ProfilerStubs* stubs);
XSIGMA_API const ProfilerStubs* privateuse1Stubs();

using vulkan_id_t = strong::type<
    int64_t,
    struct _VulkanID,
    strong::regular,
    strong::convertible_to<int64_t>,
    strong::hashable>;

}  // namespace xsigma::profiler::impl
