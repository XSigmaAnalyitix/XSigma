#include "base.h"
#include "itt_wrapper.h"
#include "util/irange.h"

//XSIGMA_DIAGNOSTIC_PUSH("-Wunused-parameter")

namespace xsigma::profiler::impl
{
namespace
{

struct ITTMethods : public ProfilerStubs
{
    void record(
        XSIGMA_UNUSED xsigma::device_enum*   device,
        XSIGMA_UNUSED ProfilerVoidEventStub* event,
        XSIGMA_UNUSED int64_t*               cpu_ns) const override
    {
    }

    float elapsed(
        XSIGMA_UNUSED const ProfilerVoidEventStub* event,
        XSIGMA_UNUSED const ProfilerVoidEventStub* event2) const override
    {
        return 0;
    }

    void mark(XSIGMA_UNUSED const char* name) const override { xsigma::profiler::itt_mark(name); }

    void rangePush(XSIGMA_UNUSED const char* name) const override
    {
        xsigma::profiler::itt_range_push(name);
    }

    void rangePop() const override { xsigma::profiler::itt_range_pop(); }

    void onEachDevice(XSIGMA_UNUSED std::function<void(int)> op) const override {}

    void synchronize() const override {}

    bool enabled() const override { return true; }
};

struct RegisterITTMethods
{
    RegisterITTMethods()
    {
        static ITTMethods methods;
        registerITTMethods(&methods);
    }
};
RegisterITTMethods reg;

}  // namespace
}  // namespace xsigma::profiler::impl
//XSIGMA_DIAGNOSTIC_POP()
