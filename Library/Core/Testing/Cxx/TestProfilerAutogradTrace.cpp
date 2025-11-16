#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>

#include "Testing/xsigmaTest.h"
#include "profiler/common/api.h"
#include "profiler/common/record_function.h"

namespace
{
#if XSIGMA_HAS_KINETO

std::string makeTracePath()
{
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return "xsigma_autograd_trace_" + std::to_string(timestamp) + ".json";
}

void runSampleWork()
{
    RECORD_USER_SCOPE("autograd_profiler_sample_work");

    volatile int accumulator = 0;
    for (int i = 0; i < 10000; ++i)
    {
        accumulator += (i & 0xFF);
    }

    if (accumulator == -1)
    {
        accumulator = 0;
    }
}

#endif  // XSIGMA_HAS_KINETO
}  // namespace

#if XSIGMA_HAS_KINETO

XSIGMATEST(profiler, autograd_chrome_trace_export)
{
    const std::set<xsigma::autograd::profiler::ActivityType> activities{
        xsigma::autograd::profiler::ActivityType::CPU,
    };

    xsigma::autograd::profiler::ProfilerConfig config(
        xsigma::autograd::profiler::ProfilerState::KINETO,
        /*report_input_shapes=*/true,
        /*profile_memory=*/true,
        /*with_stack=*/false,
        /*with_flops=*/false,
        /*with_modules=*/false);

    xsigma::autograd::profiler::prepareProfiler(config, activities);
    xsigma::autograd::profiler::enableProfiler(config, activities);

    runSampleWork();

    auto result = xsigma::autograd::profiler::disableProfiler();
    EXPECT_NE(result, nullptr);

    const auto trace_path = makeTracePath();
    result->save(trace_path);

    std::ifstream trace_input(trace_path, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(trace_input.is_open());
    EXPECT_GT(trace_input.tellg(), 0);
    trace_input.close();

    std::remove(trace_path.c_str());
}

#endif  // XSIGMA_HAS_KINETO
