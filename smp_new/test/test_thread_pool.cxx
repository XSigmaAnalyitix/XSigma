#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "smp_new/core/thread_pool.h"

using namespace xsigma::smp_new::core;

class ThreadPoolTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ThreadPoolTest, CreateThreadPool)
{
    auto pool = CreateThreadPool(4);
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->Size(), 4);
}

TEST_F(ThreadPoolTest, DefaultThreadCount)
{
    auto default_count = TaskThreadPoolBase::DefaultNumThreads();
    EXPECT_GT(default_count, 0);
}

TEST_F(ThreadPoolTest, ExecuteSimpleTask)
{
    auto             pool = CreateThreadPool(2);
    std::atomic<int> counter{0};

    pool->Run([&counter]() { counter++; });

    pool->WaitWorkComplete();
    EXPECT_EQ(counter, 1);
}

TEST_F(ThreadPoolTest, ExecuteMultipleTasks)
{
    auto             pool = CreateThreadPool(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < 10; ++i)
    {
        pool->Run([&counter]() { counter++; });
    }

    pool->WaitWorkComplete();
    EXPECT_EQ(counter, 10);
}

TEST_F(ThreadPoolTest, InThreadPool)
{
    auto              pool = CreateThreadPool(2);
    std::atomic<bool> in_pool{false};

    pool->Run([&in_pool, &pool]() { in_pool = pool->InThreadPool(); });

    pool->WaitWorkComplete();
    EXPECT_TRUE(in_pool);
}

TEST_F(ThreadPoolTest, ExceptionHandling)
{
    auto pool = CreateThreadPool(2);

    pool->Run([]() { throw std::runtime_error("Test exception"); });

    EXPECT_THROW(pool->WaitWorkComplete(), std::runtime_error);
}

TEST_F(ThreadPoolTest, PoolSize)
{
    auto pool = CreateThreadPool(8);
    EXPECT_EQ(pool->Size(), 8);
}

TEST_F(ThreadPoolTest, NumAvailable)
{
    auto pool = CreateThreadPool(4);
    // Initially all threads should be available
    EXPECT_EQ(pool->NumAvailable(), 4);
}

TEST_F(ThreadPoolTest, SequentialExecution)
{
    auto             pool = CreateThreadPool(1);
    std::vector<int> results;

    for (int i = 0; i < 5; ++i)
    {
        pool->Run([&results, i]() { results.push_back(i); });
    }

    pool->WaitWorkComplete();
    EXPECT_EQ(results.size(), 5);
}

TEST_F(ThreadPoolTest, ParallelExecution)
{
    auto             pool = CreateThreadPool(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < 100; ++i)
    {
        pool->Run(
            [&counter]()
            {
                counter++;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            });
    }

    pool->WaitWorkComplete();
    EXPECT_EQ(counter, 100);
}

TEST_F(ThreadPoolTest, DestructorWaitsForTasks)
{
    std::atomic<int> counter{0};

    {
        auto pool = CreateThreadPool(2);
        for (int i = 0; i < 10; ++i)
        {
            pool->Run([&counter]() { counter++; });
        }
        // Destructor should wait for all tasks
    }

    EXPECT_EQ(counter, 10);
}

TEST_F(ThreadPoolTest, NegativePoolSize)
{
    // Negative pool size should use default
    auto pool = CreateThreadPool(-1);
    EXPECT_GT(pool->Size(), 0);
}
