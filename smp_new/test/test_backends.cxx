#include <gtest/gtest.h>

#include <iostream>
#include <mutex>
#include <vector>

#include "smp_new/native/parallel_native.h"
#include "smp_new/parallel/parallel_api.h"

namespace xsigma::smp_new::test
{

class BackendTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Reset backend state before each test
    }

    void TearDown() override
    {
        // Cleanup after each test
    }
};

// Test 1: Check OpenMP availability
TEST_F(BackendTest, CheckOpenMPAvailability)
{
    bool available = parallel::is_openmp_available();
    std::cout << "OpenMP Available: " << (available ? "Yes" : "No") << std::endl;
    // This test just checks if the function works, doesn't assert availability
    SUCCEED();
}

// Test 2: Set and get native backend
TEST_F(BackendTest, SetNativeBackend)
{
    parallel::set_backend(0);  // NATIVE
    int backend = parallel::get_backend();
    EXPECT_EQ(backend, 0);
}

// Test 3: Set and get OpenMP backend (if available)
TEST_F(BackendTest, SetOpenMPBackend)
{
    if (parallel::is_openmp_available())
    {
        parallel::set_backend(1);  // OPENMP
        int backend = parallel::get_backend();
        EXPECT_EQ(backend, 1);
    }
    else
    {
        SKIP() << "OpenMP not available";
    }
}

// Test 4: Set auto backend
TEST_F(BackendTest, SetAutoBackend)
{
    parallel::set_backend(2);  // AUTO
    int backend = parallel::get_backend();
    // AUTO should resolve to either NATIVE (0) or OPENMP (1)
    EXPECT_TRUE(backend == 0 || backend == 1);
}

// Test 5: Get backend info
TEST_F(BackendTest, GetBackendInfo)
{
    parallel::set_backend(0);  // NATIVE
    std::string info = native::GetBackendInfo();
    EXPECT_FALSE(info.empty());
    std::cout << "Backend Info:\n" << info << std::endl;
}

// Test 6: Get native backend info
TEST_F(BackendTest, GetNativeBackendInfo)
{
    std::string info = native::GetNativeBackendInfo();
    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(info.find("Native") != std::string::npos);
    std::cout << "Native Backend Info:\n" << info << std::endl;
}

// Test 7: Get OpenMP backend info
TEST_F(BackendTest, GetOpenMPBackendInfo)
{
    std::string info = native::GetOpenMPBackendInfo();
    EXPECT_FALSE(info.empty());
    std::cout << "OpenMP Backend Info:\n" << info << std::endl;
}

// Test 8: Parallel for with native backend
TEST_F(BackendTest, ParallelForNativeBackend)
{
    parallel::set_backend(0);  // NATIVE

    std::vector<int> results(100, 0);
    parallel::parallel_for(
        0,
        100,
        10,
        [&](int64_t begin, int64_t end)
        {
            for (int64_t i = begin; i < end; ++i)
            {
                results[i] = i * 2;
            }
        });

    // Verify results
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(results[i], i * 2);
    }
}

// Test 9: Parallel reduce with native backend
TEST_F(BackendTest, ParallelReduceNativeBackend)
{
    parallel::set_backend(0);  // NATIVE

    int sum = parallel::parallel_reduce(
        0,
        100,
        10,
        0,
        [](int64_t begin, int64_t end, int ident)
        {
            int local_sum = ident;
            for (int64_t i = begin; i < end; ++i)
            {
                local_sum += i;
            }
            return local_sum;
        },
        [](int a, int b) { return a + b; });

    // Sum of 0 to 99 = 99*100/2 = 4950
    EXPECT_EQ(sum, 4950);
}

// Test 10: Parallel for with OpenMP backend (if available)
TEST_F(BackendTest, ParallelForOpenMPBackend)
{
    if (!parallel::is_openmp_available())
    {
        SKIP() << "OpenMP not available";
    }

    parallel::set_backend(1);  // OPENMP

    std::vector<int> results(100, 0);
    parallel::parallel_for(
        0,
        100,
        10,
        [&](int64_t begin, int64_t end)
        {
            for (int64_t i = begin; i < end; ++i)
            {
                results[i] = i * 2;
            }
        });

    // Verify results
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(results[i], i * 2);
    }
}

// Test 11: Thread count configuration
TEST_F(BackendTest, ThreadCountConfiguration)
{
    parallel::set_backend(0);  // NATIVE

    size_t intra_threads = parallel::get_num_intraop_threads();
    size_t inter_threads = parallel::get_num_interop_threads();

    EXPECT_GT(intra_threads, 0);
    EXPECT_GT(inter_threads, 0);

    std::cout << "Intra-op threads: " << intra_threads << std::endl;
    std::cout << "Inter-op threads: " << inter_threads << std::endl;
}

// Test 12: Backend initialization state
TEST_F(BackendTest, BackendInitializationState)
{
    parallel::set_backend(0);  // NATIVE

    bool initialized = native::IsBackendInitialized();
    EXPECT_TRUE(initialized);
}

// Test 13: Get thread number outside parallel region
TEST_F(BackendTest, GetThreadNumOutsideParallelRegion)
{
    int thread_id = parallel::get_thread_num();
    EXPECT_EQ(thread_id, 0);
}

// Test 14: Check not in parallel region initially
TEST_F(BackendTest, NotInParallelRegionInitially)
{
    bool in_region = parallel::in_parallel_region();
    EXPECT_FALSE(in_region);
}

// Test 15: Get thread number inside parallel_for
TEST_F(BackendTest, GetThreadNumInsideParallelFor)
{
    parallel::set_backend(0);  // NATIVE

    std::vector<int> thread_ids;
    std::mutex       mutex;

    parallel::parallel_for(
        0,
        4,
        1,
        [&](int64_t begin, int64_t end)
        {
            int tid = parallel::get_thread_num();
            {
                std::lock_guard<std::mutex> lock(mutex);
                thread_ids.push_back(tid);
            }
        });

    // Should have collected thread IDs
    EXPECT_GT(thread_ids.size(), 0);

    // Thread IDs should be non-negative
    for (int tid : thread_ids)
    {
        EXPECT_GE(tid, 0);
    }
}

// Test 16: Check in parallel region inside parallel_for
TEST_F(BackendTest, InParallelRegionInsideParallelFor)
{
    parallel::set_backend(0);  // NATIVE

    bool       found_in_region = false;
    std::mutex mutex;

    parallel::parallel_for(
        0,
        2,
        1,
        [&](int64_t begin, int64_t end)
        {
            bool in_region = parallel::in_parallel_region();
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (in_region)
                {
                    found_in_region = true;
                }
            }
        });

    EXPECT_TRUE(found_in_region);
}

// Test 17: Get thread number inside parallel_reduce
TEST_F(BackendTest, GetThreadNumInsideParallelReduce)
{
    parallel::set_backend(0);  // NATIVE

    std::vector<int> thread_ids;
    std::mutex       mutex;

    int sum = parallel::parallel_reduce(
        0,
        4,
        1,
        0,
        [&](int64_t begin, int64_t end, int ident)
        {
            int tid = parallel::get_thread_num();
            {
                std::lock_guard<std::mutex> lock(mutex);
                thread_ids.push_back(tid);
            }
            int local_sum = ident;
            for (int64_t i = begin; i < end; ++i)
            {
                local_sum += i;
            }
            return local_sum;
        },
        [](int a, int b) { return a + b; });

    EXPECT_GT(thread_ids.size(), 0);
    EXPECT_EQ(sum, 6);  // 0+1+2+3
}

// Test 18: Get parallel info string
TEST_F(BackendTest, GetParallelInfo)
{
    parallel::set_backend(0);  // NATIVE

    std::string info = parallel::get_parallel_info();

    // Should contain backend info
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("Backend:"), std::string::npos);
    EXPECT_NE(info.find("threads:"), std::string::npos);
}

// Test 19: Parallel info contains thread count
TEST_F(BackendTest, ParallelInfoContainsThreadCount)
{
    parallel::set_backend(0);  // NATIVE
    parallel::set_num_intraop_threads(4);

    std::string info = parallel::get_parallel_info();

    // Should contain thread information
    EXPECT_NE(info.find("Intra-op threads:"), std::string::npos);
    EXPECT_NE(info.find("Inter-op threads:"), std::string::npos);
}

// Test 20: Thread state cleanup after parallel_for
TEST_F(BackendTest, ThreadStateCleanupAfterParallelFor)
{
    parallel::set_backend(0);  // NATIVE

    parallel::parallel_for(
        0,
        2,
        1,
        [](int64_t begin, int64_t end)
        {
            // Just execute
        });

    // After parallel_for completes, should not be in parallel region
    bool in_region = parallel::in_parallel_region();
    EXPECT_FALSE(in_region);
}

}  // namespace xsigma::smp_new::test
