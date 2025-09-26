/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

//#include <functional>
//#include <iomanip>
//#include <iostream>
//#include <optional>
//#include <sstream>
//#include <utility>
//
//#include "common/configure.h"  // IWYU pragma: keep
//#include "experimental/profiler/env_time.h"
//#include "experimental/profiler/env_var.h"
//#include "experimental/profiler/profiler_factory.h"
//#include "experimental/profiler/profiler_interface.h"
//#include "experimental/profiler/profiler_lock.h"
//#include "experimental/profiler/profiler_options.h"
//#include "experimental/profiler/profiler_session.h"
//#include "experimental/profiler/traceme_encode.h"
//#include "experimental/profiler/xplane/xplane.h"
//#include "experimental/profiler/xplane/xplane_builder.h"
//#include "experimental/profiler/xplane/xplane_schema.h"
//#include "util/logging.h"
//#include "util/strcat.h"
#include "xsigmaTest.h"
//
//using namespace xsigma::profiler;
//using namespace xsigma;
//
//XSIGMATEST_VOID(ProfilerLockTest, DefaultConstructorCreatesInactiveInstance)
//{
//    ProfilerLock profiler_lock;
//    EXPECT_FALSE(profiler_lock.Active());
//}
//
//XSIGMATEST_VOID(ProfilerLockTest, AcquireAndReleaseExplicitly)
//{
//    std::optional<ProfilerLock> profiler_lock = ProfilerLock::Acquire();
//    EXPECT_TRUE(profiler_lock.has_value());
//    EXPECT_TRUE(profiler_lock->Active());
//    profiler_lock->ReleaseIfActive();
//    EXPECT_FALSE(profiler_lock->Active());
//}
//
//XSIGMATEST_VOID(ProfilerLockTest, AcquireAndReleaseOnDestruction)
//{
//    std::optional<ProfilerLock> profiler_lock = ProfilerLock::Acquire();
//    EXPECT_TRUE(profiler_lock.has_value());
//    EXPECT_TRUE(profiler_lock->Active());
//}
//
//XSIGMATEST_VOID(ProfilerLockTest, ReacquireWithoutReleaseFails)
//{
//    std::optional<ProfilerLock> profiler_lock_1 = ProfilerLock::Acquire();
//    std::optional<ProfilerLock> profiler_lock_2 = ProfilerLock::Acquire();
//    EXPECT_TRUE(profiler_lock_1.has_value());
//    EXPECT_TRUE(profiler_lock_1->Active());
//    EXPECT_FALSE(profiler_lock_2.has_value());
//}
//
//XSIGMATEST_VOID(ProfilerLockTest, ReacquireAfterReleaseSucceeds)
//{
//    auto profiler_lock_1 = ProfilerLock::Acquire();
//    EXPECT_TRUE(profiler_lock_1.has_value());
//    EXPECT_TRUE(profiler_lock_1->Active());
//    profiler_lock_1->ReleaseIfActive();
//    EXPECT_FALSE(profiler_lock_1->Active());
//    auto profiler_lock_2 = ProfilerLock::Acquire();
//    EXPECT_TRUE(profiler_lock_2.has_value());
//    EXPECT_TRUE(profiler_lock_2->Active());
//}
//
//XSIGMATEST_VOID(ProfilerLockTest, InactiveAfterMove)
//{
//    std::optional<ProfilerLock> profiler_lock_1 = ProfilerLock::Acquire();
//    EXPECT_TRUE(profiler_lock_1.has_value());
//    EXPECT_TRUE(profiler_lock_1->Active());
//    ProfilerLock profiler_lock_2 = std::move(*profiler_lock_1);
//    EXPECT_FALSE(profiler_lock_1->Active());
//    EXPECT_TRUE(profiler_lock_2.Active());
//}
//
///*
//namespace xsigma
//{
//class TestProfiler : public ProfilerInterface
//{
//public:
//    bool Start() override { return true; }
//    bool Stop() override { return true; }
//    bool CollectData(XSpace*) override { return true; }
//};
//std::unique_ptr<ProfilerInterface> TestFactoryFunction(const xsigma::ProfileOptions& options)
//{
//    return std::make_unique<xsigma::TestProfiler>();
//}
//class FactoryClass
//{
//public:
//    explicit FactoryClass(void* ptr) : ptr_(ptr) {}
//    FactoryClass(const FactoryClass&) = default;  // copyable
//    FactoryClass(FactoryClass&&)      = default;  // movable
//
//    std::unique_ptr<ProfilerInterface> CreateProfiler(const xsigma::ProfileOptions& options) const
//    {
//        return std::make_unique<TestProfiler>();
//    }
//
//private:
//    XSIGMA_UNUSED void* ptr_ = nullptr;
//};
//
//std::unique_ptr<ProfilerInterface> NullFactoryFunction(const xsigma::ProfileOptions& options)
//{
//    return nullptr;
//}
//}  // namespace xsigma::profiler
//*/
//XSIGMATEST_VOID(ProfilerFactoryTest, FactoryFunctionPointer)
//{
//    /*ClearRegisteredProfilersForTest();
//    RegisterProfilerFactory(&TestFactoryFunction);
//    auto profilers = CreateProfilers(xsigma::ProfileOptions());
//    EXPECT_EQ(profilers.size(), 1);*/
//}
//
//XSIGMATEST_VOID(ProfilerFactoryTest, FactoryLambda)
//{
//    /*ClearRegisteredProfilersForTest();
//    RegisterProfilerFactory([](const xsigma::ProfileOptions& options)
//                            { return std::make_unique<xsigma::profiler::TestProfiler>(); });
//    auto profilers = CreateProfilers(xsigma::ProfileOptions());
//    EXPECT_EQ(profilers.size(), 1);*/
//}
//
//XSIGMATEST_VOID(ProfilerFactoryTest, FactoryReturnsNull)
//{
//    /* ClearRegisteredProfilersForTest();
//    RegisterProfilerFactory(&NullFactoryFunction);
//    auto profilers = CreateProfilers(xsigma::ProfileOptions());
//    EXPECT_TRUE(profilers.empty());*/
//}
//
//XSIGMATEST_VOID(ProfilerFactoryTest, FactoryClassCapturedByLambda)
//{
//    /*   ClearRegisteredProfilersForTest();
//    static int   token = 42;
//    FactoryClass factory(&token);
//    RegisterProfilerFactory([factory = std::move(factory)](const xsigma::ProfileOptions& options)
//                            { return factory.CreateProfiler(options); });
//    auto profilers = CreateProfilers(xsigma::ProfileOptions());
//    EXPECT_EQ(profilers.size(), 1);*/
//}
//XSIGMATEST_VOID(TraceMeEncodeTest, NoArgTest)
//{
//    EXPECT_EQ(TraceMeEncode("Hello!", {}), "Hello!");
//}
//
//XSIGMATEST_VOID(TraceMeEncodeTest, OneArgTest)
//{
//    EXPECT_EQ(TraceMeEncode("Hello", {{"context", "World"}}), "Hello#context=World#");
//}
//
//XSIGMATEST_VOID(TraceMeEncodeTest, TwoArgsTest)
//{
//    EXPECT_EQ(
//        TraceMeEncode("Hello", {{"context", "World"}, {"request_id", 42}}),
//        "Hello#context=World,request_id=42#");
//}
//
//XSIGMATEST_VOID(TraceMeEncodeTest, ThreeArgsTest)
//{
//    std::stringstream ss;
//    ss << std::uppercase << std::hex << 0xdeadbeef;
//
//    EXPECT_EQ(
//        TraceMeEncode("Hello", {{"context", "World"}, {"request_id", 42}, {"addr", ss.str()}}),
//        "Hello#context=World,request_id=42,addr=DEADBEEF#");
//}
//
//XSIGMATEST_VOID(TraceMeEncodeTest, TemporaryStringTest)
//{
//    EXPECT_EQ(
//        TraceMeEncode("Hello", {{std::string("context"), xsigma::strings::StrCat("World:", 2020)}}),
//        "Hello#context=World:2020#");
//}
//
//XSIGMATEST_VOID(TraceMeEncodeTest, NoNameTest)
//{
//    EXPECT_EQ(
//        TraceMeEncode({{"context", "World"}, {"request_id", 42}}), "#context=World,request_id=42#");
//}
//
//XSIGMATEST_VOID(ProfilerSessionTest, run)
//{
//    // Custom computation class to profile
//    class ComputationLoop
//    {
//    public:
//        void ProcessData(int iterations)
//        {
//            std::vector<float> data(1000, 1.0f);
//            for (int i = 0; i < iterations; ++i)
//            {
//                // Simulate some computation
//                ProcessBatch(data, i);
//            }
//        }
//
//    private:
//        void ProcessBatch(std::vector<float>& data, int iteration)
//        {
//            // Simulate different processing times for different iterations
//            if (iteration % 2 == 0)
//            {
//                std::this_thread::sleep_for(std::chrono::milliseconds(20));
//            }
//            else
//            {
//                std::this_thread::sleep_for(std::chrono::milliseconds(10));
//            }
//
//            // Do some actual computation
//            for (size_t i = 0; i < data.size(); ++i)
//            {
//                data[i] = std::sin(data[i]) * std::cos(data[i]);
//            }
//        }
//    };
//
//    class LoopProfiler
//    {
//    public:
//        void ProfileLoop()
//        {
//            // Create XSpace for profiling data
//            XSpace  xspace;
//            XPlane* plane = xspace.add_planes();
//            plane->set_name("Custom Profiling");
//
//            // Create XPlane builder
//            XPlaneBuilder plane_builder(plane);
//
//            // Add a line for our loop operations
//            XLineBuilder line = plane_builder.GetOrCreateLine(0);
//            line.SetName("MainThread");
//
//            // Start profiling session
//            auto status = profiler_session::Create(CreateProfileOptions());
//            if (!status)
//            {
//                LOG(ERROR) << "Failed to create profiler session: ";
//                return;
//            }
//            std::unique_ptr<profiler_session> session = std::move(status);
//
//            // Create computation object
//            ComputationLoop computation;
//
//            // Profile the loop
//            const int kIterations = 10;
//            uint64_t  start_time  = EnvTime::NowNanos();
//
//            // Add overall loop event
//            XEventBuilder loop_event =
//                line.add_event(*plane_builder.GetOrCreateEventMetadata("Complete Loop"));
//            loop_event.SetTimestampNs(start_time);
//
//            // Run and profile each iteration
//            for (int i = 0; i < kIterations; ++i)
//            {
//                // Add event for this iteration
//                uint64_t      iter_start = EnvTime::NowNanos();
//                XEventBuilder iter_event =
//                    line.add_event(*plane_builder.GetOrCreateEventMetadata("Iteration"));
//                iter_event.SetTimestampNs(iter_start);
//
//                // Run the computation
//                computation.ProcessData(1);
//
//                // Set iteration event duration
//                uint64_t iter_end = EnvTime::NowNanos();
//                iter_event.SetDurationNs(iter_end - iter_start);
//
//                // Add stats for this iteration
//                iter_event.AddStatValue(
//                    *plane_builder.GetOrCreateStatMetadata("Iteration Number"), i);
//                iter_event.AddStatValue(
//                    *plane_builder.GetOrCreateStatMetadata("Processing Time (ms)"),
//                    (iter_end - iter_start) / 1000000);
//            }
//
//            // Set overall loop event duration
//            uint64_t end_time = EnvTime::NowNanos();
//            loop_event.SetDurationNs(end_time - start_time);
//
//            // Collect and save profiling data
//            SaveProfilingData(std::move(session), xspace);
//        }
//
//    private:
//        ProfileOptions CreateProfileOptions()
//        {
//            ProfileOptions options;
//            options.set_version(1);
//            options.set_duration_ms(5000);  // 5 second profiling window
//            //options.set_timestamp(true);//
//            options.set_host_tracer_level(0);
//            return options;
//        }
//
//        void SaveProfilingData(std::unique_ptr<profiler_session> session, const XSpace& xspace)
//        {
//            // Save TensorFlow profiler session data
//            XSpace tf_response;
//            auto   status = session->CollectData(&tf_response);
//
//            // Save XSpace data
//            std::string   xspace_path = "/tmp/loop_profile.xspace.pb";
//            std::ofstream xspace_file(xspace_path, std::ios::out | std::ios::binary);
//            /* if (!xspace.SerializeToOstream(&xspace_file))
//            {
//                LOG(ERROR) << "Failed to write XSpace data to " << xspace_path;
//                return;
//            }*/
//            LOG(INFO) << "Profile data saved to " << xspace_path;
//
//            // Print summary statistics
//            AnalyzeResults(xspace);
//        }
//
//        void AnalyzeResults(const XSpace& xspace)
//        {
//            for (const XPlane& plane : xspace.planes())
//            {
//                if (plane.name() == "Custom Profiling")
//                {
//                    for (const XLine& line : plane.lines())
//                    {
//                        for (const XEvent& event : line.events())
//                        {
//                            if (event.metadata_id() == 1)
//                            {  // Complete Loop event
//                                LOG(INFO) << "Total loop time: " << event.duration_ps() / 1e9
//                                          << " seconds";
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    };
//
//    LoopProfiler profiler;
//    profiler.ProfileLoop();
//}
//
XSIGMATEST(Core, Profiler)
{
    //    START_LOG_TO_FILE_NAME(Profiler);
    //
    //    XSIGMATEST_CALL(ProfilerSessionTest, run);
    //    // ProfilerLockTest cases
    //    XSIGMATEST_CALL(ProfilerLockTest, DefaultConstructorCreatesInactiveInstance);
    //    XSIGMATEST_CALL(ProfilerLockTest, AcquireAndReleaseExplicitly);
    //    XSIGMATEST_CALL(ProfilerLockTest, AcquireAndReleaseOnDestruction);
    //    XSIGMATEST_CALL(ProfilerLockTest, ReacquireWithoutReleaseFails);
    //    XSIGMATEST_CALL(ProfilerLockTest, ReacquireAfterReleaseSucceeds);
    //    XSIGMATEST_CALL(ProfilerLockTest, InactiveAfterMove);
    //
    //    // ProfilerFactoryTest cases
    //    XSIGMATEST_CALL(ProfilerFactoryTest, FactoryFunctionPointer);
    //    XSIGMATEST_CALL(ProfilerFactoryTest, FactoryLambda);
    //    XSIGMATEST_CALL(ProfilerFactoryTest, FactoryReturnsNull);
    //    XSIGMATEST_CALL(ProfilerFactoryTest, FactoryClassCapturedByLambda);
    //
    //    // TraceMeEncodeTest cases
    //    XSIGMATEST_CALL(TraceMeEncodeTest, NoArgTest);
    //    XSIGMATEST_CALL(TraceMeEncodeTest, OneArgTest);
    //    XSIGMATEST_CALL(TraceMeEncodeTest, TwoArgsTest);
    //    XSIGMATEST_CALL(TraceMeEncodeTest, ThreeArgsTest);
    //    XSIGMATEST_CALL(TraceMeEncodeTest, TemporaryStringTest);
    //    XSIGMATEST_CALL(TraceMeEncodeTest, NoNameTest);
    //
    //    END_LOG_TO_FILE_NAME(Profiler);
    //
    END_TEST();
}
