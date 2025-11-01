#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "smp_new/parallel/parallel_api.h"

using namespace xsigma::smp_new::parallel;

class TaskExecutionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        set_num_intraop_threads(2);
        set_num_interop_threads(4);
    }
    void TearDown() override {}
};

TEST_F(TaskExecutionTest, LaunchInteropTask)
{
    std::atomic<int> counter{0};

    launch([&counter]() { counter++; });

    // Give task time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(counter, 1);
}

TEST_F(TaskExecutionTest, LaunchMultipleInteropTasks)
{
    std::atomic<int> counter{0};

    for (int i = 0; i < 10; ++i)
    {
        launch([&counter]() { counter++; });
    }

    // Give tasks time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_EQ(counter, 10);
}

TEST_F(TaskExecutionTest, LaunchIntraopTask)
{
    std::atomic<int> counter{0};

    intraop_launch([&counter]() { counter++; });

    // Give task time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(counter, 1);
}

TEST_F(TaskExecutionTest, GetIntraopThreadCount)
{
    auto num_threads = get_num_intraop_threads();
    EXPECT_EQ(num_threads, 2);
}

TEST_F(TaskExecutionTest, GetInteropThreadCount)
{
    auto num_threads = get_num_interop_threads();
    EXPECT_EQ(num_threads, 4);
}

TEST_F(TaskExecutionTest, SetIntraopThreadCount)
{
    set_num_intraop_threads(8);
    auto num_threads = get_num_intraop_threads();
    EXPECT_EQ(num_threads, 8);
}

TEST_F(TaskExecutionTest, SetInteropThreadCount)
{
    set_num_interop_threads(16);
    auto num_threads = get_num_interop_threads();
    EXPECT_EQ(num_threads, 16);
}

TEST_F(TaskExecutionTest, NestedTaskExecution)
{
    std::atomic<int> outer_counter{0};
    std::atomic<int> inner_counter{0};

    launch(
        [&outer_counter, &inner_counter]()
        {
            outer_counter++;
            intraop_launch([&inner_counter]() { inner_counter++; });
        });

    // Give tasks time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_EQ(outer_counter, 1);
    EXPECT_EQ(inner_counter, 1);
}

TEST_F(TaskExecutionTest, TaskWithCapture)
{
    int              value = 42;
    std::atomic<int> result{0};

    launch([&result, value]() { result = value; });

    // Give task time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(result, 42);
}

TEST_F(TaskExecutionTest, TaskWithMutableCapture)
{
    std::vector<int> data;

    launch([&data]() { data.push_back(1); });

    // Give task time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(data.size(), 1);
    EXPECT_EQ(data[0], 1);
}

TEST_F(TaskExecutionTest, ConcurrentTaskExecution)
{
    std::atomic<int> counter{0};
    const int        NUM_TASKS = 100;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        launch(
            [&counter]()
            {
                counter++;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            });
    }

    // Give tasks time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    EXPECT_EQ(counter, NUM_TASKS);
}

TEST_F(TaskExecutionTest, IntraopLaunchInline)
{
    std::atomic<int> counter{0};

    // First intraop_launch should execute
    intraop_launch([&counter]() { counter++; });

    // Nested intraop_launch should execute inline
    intraop_launch([&counter]() { intraop_launch([&counter]() { counter++; }); });

    // Give tasks time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(counter, 2);
}
